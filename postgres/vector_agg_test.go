package postgres_test

import (
	"context"
	"math"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/queryance/qpx"
	"github.com/queryance/qpx/engine/vector"
	"github.com/queryance/qpx/postgres"
)

// Agg parity: every engine-side agg operator must match PG's GROUP BY on
// the same data, including NULL keys (own group) and NULL values
// (counted, not summed). The table is self-contained (20K rows,
// deterministic NULL cycles) so these tests never depend on bench setup.

const aggParityTable = "qpx_vector_agg_parity"

func ensureAggParityTable(t *testing.T, ctx context.Context, dsn string) {
	t.Helper()
	conn, err := pgx.Connect(ctx, dsn)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer func() { _ = conn.Close(ctx) }()
	var n int64
	if err := conn.QueryRow(ctx, "SELECT count(*) FROM "+aggParityTable).Scan(&n); err == nil && n == 20000 {
		return
	}
	for _, stmt := range []string{
		`DROP TABLE IF EXISTS ` + aggParityTable,
		`CREATE TABLE ` + aggParityTable + `(id bigint NOT NULL, f double precision, t text)`,
		`INSERT INTO ` + aggParityTable + `
			SELECT g,
				CASE WHEN g % 7 = 0 THEN NULL ELSE (g % 100000)::float8/100000.0 END,
				CASE WHEN g % 11 = 0 THEN NULL ELSE 'txt_'||(g%50) END
			FROM generate_series(1,20000) g`,
	} {
		if _, err := conn.Exec(ctx, stmt); err != nil {
			t.Fatalf("setup %q: %v", stmt[:40], err)
		}
	}
}

// drainVector scans sql through ExecuteVector and returns all batches.
func drainVector(t *testing.T, ctx context.Context, dsn, sql string) []*vector.Batch {
	t.Helper()
	src := postgres.NewVectorSource(postgres.Config{ConnString: dsn, SQL: sql})
	it, err := qpx.ExecuteVector(ctx, src, qpx.DefaultVectorOptions())
	if err != nil {
		t.Fatalf("ExecuteVector: %v", err)
	}
	defer func() { _ = it.Close() }()
	var out []*vector.Batch
	for {
		b, ok, err := it.Next(ctx)
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if !ok {
			return out
		}
		out = append(out, b)
	}
}

func asFloat64(t *testing.T, v any) float64 {
	t.Helper()
	switch n := v.(type) {
	case nil:
		return math.NaN()
	case float64:
		return n
	case pgtype.Numeric:
		if !n.Valid {
			return math.NaN()
		}
		f, err := n.Float64Value()
		if err != nil {
			t.Fatalf("numeric decode: %v", err)
		}
		return f.Float64
	default:
		t.Fatalf("want float64/numeric/nil, got %T", v)
		return 0
	}
}

// floatEq compares with an absolute epsilon scaled to the magnitudes
// here (sums < 1e5): PG and the engine fold rows in different orders, so
// the last ulp may differ. NaN == NaN (both sides all-NULL).
func floatEq(a, b float64) bool {
	if math.IsNaN(a) && math.IsNaN(b) {
		return true
	}
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= 1e-6
}

func TestVectorGlobalAggParity(t *testing.T) {
	dsn := vectorDSN(t)
	ctx := context.Background()
	ensureAggParityTable(t, ctx, dsn)

	conn, err := pgx.Connect(ctx, dsn)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer func() { _ = conn.Close(ctx) }()
	var wantRows, wantCount int64
	var wantSum, wantAvg any
	row := conn.QueryRow(ctx, `SELECT count(*), count(f), sum(f), avg(f) FROM `+aggParityTable)
	if err := row.Scan(&wantRows, &wantCount, &wantSum, &wantAvg); err != nil {
		t.Fatalf("pg global agg: %v", err)
	}

	batches := drainVector(t, ctx, dsn, `SELECT f FROM `+aggParityTable+` ORDER BY id`)
	a := vector.NewGlobalFloatAgg(0)
	for _, b := range batches {
		if err := a.Add(b); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}
	got := a.Result()
	if got.Rows != wantRows || got.Count != wantCount {
		t.Fatalf("rows/count = %d/%d, want %d/%d", got.Rows, got.Count, wantRows, wantCount)
	}
	if !floatEq(got.Sum, asFloat64(t, wantSum)) || !floatEq(got.Avg, asFloat64(t, wantAvg)) {
		t.Fatalf("sum/avg = %v/%v, want %v/%v", got.Sum, got.Avg, wantSum, wantAvg)
	}
	// Parallel merge must match too.
	par, err := vector.ParallelGlobalAgg(batches, 4, 0)
	if err != nil {
		t.Fatalf("ParallelGlobalAgg: %v", err)
	}
	pr := par.Result()
	if pr.Rows != got.Rows || pr.Count != got.Count ||
		!floatEq(pr.Sum, got.Sum) || !floatEq(pr.Avg, got.Avg) {
		t.Fatalf("parallel %+v != sequential %+v", pr, got)
	}
}

type pgGroup struct {
	rows  int64
