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
