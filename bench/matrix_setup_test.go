package bench

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
)

// Widened benchmark matrix (Measure phase): size dimension S/M/L plus a
// join dimension, all under the qpx_bench_* prefix. M reuses the legacy
// qpx_bench_data table (1M rows, see doc.go); S/L/J are UNLOGGED tables
// built here so crash-safety fsyncs don't tax setup or reruns.
const (
	mxTableS = "qpx_bench_s"
	mxTableL = "qpx_bench_l"
	mxTableJ = "qpx_bench_j"

	mxRowsS = 100_000
	mxRowsL = 10_000_000
	mxRowsJ = 1_000_000
)

// Query shapes: identical SQL text on QPX, pgx, and DuckDB; only the
// table name varies by dataset. Join/sort/limit are M-only (see below).
func mxScanSQL(tbl string) string {
	return fmt.Sprintf(`SELECT id, f, t, ts FROM %s`, tbl)
}
func mxFilterSQL(tbl string) string {
	return fmt.Sprintf(`SELECT id, f FROM %s WHERE f > 0.5`, tbl)
}
func mxAggSQL(tbl string) string {
	return fmt.Sprintf(`SELECT count(*), sum(f), avg(f) FROM %s`, tbl)
}
func mxGlowSQL(tbl string) string {
	return fmt.Sprintf(`SELECT (id %% 128) AS g, count(*) AS n, avg(f) AS a FROM %s GROUP BY 1`, tbl)
}
func mxGHighSQL(tbl string) string {
	return fmt.Sprintf(`SELECT (id %% 100000) AS g, count(*) AS n, avg(f) AS a FROM %s GROUP BY 1`, tbl)
}

func mxSQLFor(tbl string) (scan, filter, agg, glow, ghigh string) {
	return mxScanSQL(tbl), mxFilterSQL(tbl), mxAggSQL(tbl), mxGlowSQL(tbl), mxGHighSQL(tbl)
}

// Prebuilt per-dataset SQL so every system under test shares one string.
var (
	mxScanS, mxFilterS, mxAggS, mxGlowS, mxGHighS = mxSQLFor(mxTableS)
	mxScanL, mxFilterL, mxAggL, mxGlowL, mxGHighL = mxSQLFor(mxTableL)
)

const (
	mxAggQM   = `SELECT count(*), sum(f), avg(f) FROM qpx_bench_data`
	mxGHighQM = `SELECT (id % 100000) AS g, count(*) AS n, avg(f) AS a FROM qpx_bench_data GROUP BY 1`
	mxJoinQM  = `SELECT d.id, d.f, j.payload FROM qpx_bench_data d JOIN qpx_bench_j j ON j.jid = d.id`
	mxSortQM  = `SELECT id, f FROM qpx_bench_data ORDER BY f`
	mxLimitQM = `SELECT id, f, t, ts FROM qpx_bench_data LIMIT 100`
)

var (
	mxOnce sync.Once
	mxErr  error
)

// ensureMatrixDataset ensures the legacy M dataset plus S/L/J, rebuilding
// any table whose row count mismatches. Runs once per test binary.
func ensureMatrixDataset(tb testing.TB) {
	tb.Helper()
	ensureDataset(tb)
	mxOnce.Do(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
		defer cancel()
		mxErr = mxBuild(ctx)
	})
	if mxErr != nil {
		tb.Fatalf("matrix setup: %v", mxErr)
	}
}

func mxBuild(ctx context.Context) error {
	dsn := resolveDSN()
	if dsn == "" {
		return fmt.Errorf("no working DSN")
	}
	conn, err := openDB(ctx, dsn)
	if err != nil {
		return fmt.Errorf("connect: %w", err)
	}
	defer func() { _ = conn.Close(ctx) }()
	base := `(id bigint NOT NULL, f double precision NOT NULL, t text NOT NULL, th text NOT NULL, ts timestamptz NOT NULL)`
	fill := func(tbl string, n int) string {
		return fmt.Sprintf(`INSERT INTO %s SELECT g, (g%%100000)::float8/100000.0, 'txt_'||(g%%1000), 'hk_'||g,`+
			` timestamptz '2020-01-01 UTC' + make_interval(secs => g) FROM generate_series(1,%d) g`, tbl, n)
	}
	for _, spec := range []struct {
		tbl, ddl, ins string
		rows          int64
	}{
		{mxTableS, `CREATE UNLOGGED TABLE ` + mxTableS + base, fill(mxTableS, mxRowsS), mxRowsS},
		{mxTableL, `CREATE UNLOGGED TABLE ` + mxTableL + base, fill(mxTableL, mxRowsL), mxRowsL},
	} {
		if err := mxRebuild(ctx, conn, spec.tbl, spec.rows, spec.ddl, spec.ins); err != nil {
			return err
		}
	}
	// qpx_bench_data.id is unique by construction (1..1M) but
	// unconstrained, so add the UNIQUE once to give J a real FK target.
	if err := mxEnsureUnique(ctx, conn); err != nil {
		return err
	}
	return mxBuildJoin(ctx, conn)
}

// mxRebuild no-ops when tbl already holds want rows; otherwise it drops,
// recreates (UNLOGGED), and refills deterministically via generate_series.
func mxRebuild(ctx context.Context, conn *pgx.Conn, tbl string, want int64, ddl, ins string) error {
	var n int64
	if err := conn.QueryRow(ctx, "SELECT count(*) FROM "+tbl).Scan(&n); err == nil && n == want {
		return nil
	}
	if _, err := conn.Exec(ctx, "DROP TABLE IF EXISTS "+tbl); err != nil {
		return fmt.Errorf("drop %s: %w", tbl, err)
	}
	if _, err := conn.Exec(ctx, ddl); err != nil {
		return fmt.Errorf("create %s: %w", tbl, err)
	}
	if _, err := conn.Exec(ctx, ins); err != nil {
		return fmt.Errorf("fill %s: %w", tbl, err)
	}
	return nil
}

func mxEnsureUnique(ctx context.Context, conn *pgx.Conn) error {
	var exists bool
	if err := conn.QueryRow(ctx,
		`SELECT EXISTS(SELECT 1 FROM pg_constraint WHERE conname='qpx_bench_data_id_uniq')`).Scan(&exists); err != nil {
		return fmt.Errorf("constraint check: %w", err)
	}
	if exists {
		return nil
	}
	if _, err := conn.Exec(ctx,
		`ALTER TABLE qpx_bench_data ADD CONSTRAINT qpx_bench_data_id_uniq UNIQUE (id)`); err != nil {
		return fmt.Errorf("add unique: %w", err)
	}
	return nil
}

// mxBuildJoin builds the 1M probe side: jid matches M ids 1..1M exactly,
// so the M⋈J equijoin yields 1M rows. Index on jid keeps nested-loop plans
// honest; the FK documents the relationship into M.
func mxBuildJoin(ctx context.Context, conn *pgx.Conn) error {
	var n int64
	if err := conn.QueryRow(ctx, "SELECT count(*) FROM "+mxTableJ).Scan(&n); err == nil && n == mxRowsJ {
		_, err := conn.Exec(ctx, `CREATE INDEX IF NOT EXISTS qpx_bench_j_jid_idx ON `+mxTableJ+`(jid)`)
		return err
	}
	if _, err := conn.Exec(ctx, "DROP TABLE IF EXISTS "+mxTableJ); err != nil {
		return fmt.Errorf("drop %s: %w", mxTableJ, err)
	}
	ddl := `CREATE UNLOGGED TABLE ` + mxTableJ + `(jid bigint NOT NULL, payload double precision NOT NULL,` +
		` CONSTRAINT qpx_bench_j_jid_fkey FOREIGN KEY (jid) REFERENCES qpx_bench_data(id))`
	if _, err := conn.Exec(ctx, ddl); err != nil {
		return fmt.Errorf("create %s: %w", mxTableJ, err)
	}
	ins := `INSERT INTO ` + mxTableJ +
		` SELECT g, (g%1000)::float8/1000.0 FROM generate_series(1,1000000) g`
	if _, err := conn.Exec(ctx, ins); err != nil {
		return fmt.Errorf("fill %s: %w", mxTableJ, err)
	}
	if _, err := conn.Exec(ctx, `CREATE INDEX qpx_bench_j_jid_idx ON `+mxTableJ+`(jid)`); err != nil {
		return fmt.Errorf("index %s: %w", mxTableJ, err)
	}
	return nil
}

// TestMxDatasetSanity guards matrix preconditions: exact row counts,
// filter selectivity, group cardinalities, join size, and LIMIT size.
// (M count/selectivity/glow are covered by TestDatasetSanity.)
func TestMxDatasetSanity(t *testing.T) {
	ensureMatrixDataset(t)
	ctx := context.Background()
	conn, err := openDB(ctx, resolveDSN())
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer func() { _ = conn.Close(ctx) }()
	checks := []struct {
		name string
		sql  string
		want int64
	}{
		{"s.rows", "SELECT count(*) FROM " + mxTableS, mxRowsS},
		{"s.filter", "SELECT count(*) FROM " + mxTableS + " WHERE f > 0.5", 49999},
		{"s.glow", "SELECT count(*) FROM (" + mxGlowS + ") s", 128},
		{"s.ghigh", "SELECT count(*) FROM (" + mxGHighS + ") s", 100000},
		{"l.rows", "SELECT count(*) FROM " + mxTableL, mxRowsL},
		{"l.filter", "SELECT count(*) FROM " + mxTableL + " WHERE f > 0.5", 4999900},
		{"l.glow", "SELECT count(*) FROM (" + mxGlowL + ") s", 128},
		{"l.ghigh", "SELECT count(*) FROM (" + mxGHighL + ") s", 100000},
		{"j.rows", "SELECT count(*) FROM " + mxTableJ, mxRowsJ},
		{"m.join", "SELECT count(*) FROM qpx_bench_data d JOIN " + mxTableJ + " j ON j.jid = d.id", 1000000},
		{"m.limit", "SELECT count(*) FROM (" + mxLimitQM + ") s", 100},
		{"m.agg", "SELECT count(*) FROM (" + mxAggQM + ") s", 1},
	}
	for _, c := range checks {
		var got int64
		if err := conn.QueryRow(ctx, c.sql).Scan(&got); err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if got != c.want {
			t.Fatalf("%s: got %d, want %d", c.name, got, c.want)
		}
	}
}
