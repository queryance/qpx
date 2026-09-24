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
	count int64
	sum   float64
	avg   float64
}

func pgGroupsInt(t *testing.T, ctx context.Context, dsn, sql string) map[int64]pgGroup {
	t.Helper()
	conn, err := pgx.Connect(ctx, dsn)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer func() { _ = conn.Close(ctx) }()
	rows, err := conn.Query(ctx, sql)
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	defer rows.Close()
	out := make(map[int64]pgGroup)
	for rows.Next() {
		var k int64
		var n, c int64
		var s, a any
		if err := rows.Scan(&k, &n, &c, &s, &a); err != nil {
			t.Fatalf("scan: %v", err)
		}
		out[k] = pgGroup{rows: n, count: c, sum: asFloat64(t, s), avg: asFloat64(t, a)}
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("rows: %v", err)
	}
	return out
}

func TestVectorGroupByInt64Parity(t *testing.T) {
	dsn := vectorDSN(t)
	ctx := context.Background()
	ensureAggParityTable(t, ctx, dsn)

	batches := drainVector(t, ctx, dsn, `SELECT id, f FROM `+aggParityTable+` ORDER BY id`)
	g := vector.NewGroupByInt64(0, 1, 9)
	for _, b := range batches {
		if err := g.Add(b); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}
	want := pgGroupsInt(t, ctx, dsn,
		`SELECT (id % 9) AS k, count(*), count(f), sum(f), avg(f) FROM `+aggParityTable+` GROUP BY 1`)
	got := g.SortedRows()
	if len(got) != len(want) {
		t.Fatalf("groups = %d, want %d", len(got), len(want))
	}
	for _, gr := range got {
		w, ok := want[gr.Key]
		if !ok {
			t.Fatalf("unexpected group %d", gr.Key)
		}
		if gr.Rows != w.rows || gr.Count != w.count ||
			!floatEq(gr.Sum, w.sum) || !floatEq(gr.Avg, w.avg) {
			t.Fatalf("group %d = %+v, want %+v", gr.Key, gr, w)
		}
	}
	if g.HasNullGroup() {
		t.Fatal("NOT NULL keys must not produce a NULL group")
	}
	// Parallel merge must match row-for-row.
	par, err := vector.ParallelGroupByInt64(batches, 4, 0, 1, 9)
	if err != nil {
		t.Fatalf("ParallelGroupByInt64: %v", err)
	}
	prows := par.SortedRows()
	if len(prows) != len(got) {
		t.Fatalf("parallel groups = %d, want %d", len(prows), len(got))
	}
	for i := range got {
		p := prows[i]
		if p.Key != got[i].Key || p.Rows != got[i].Rows || p.Count != got[i].Count ||
			!floatEq(p.Sum, got[i].Sum) || !floatEq(p.Avg, got[i].Avg) {
			t.Fatalf("parallel group %d = %+v, want %+v", i, p, got[i])
		}
	}
}

func TestVectorGroupByStringParity(t *testing.T) {
	dsn := vectorDSN(t)
	ctx := context.Background()
	ensureAggParityTable(t, ctx, dsn)

	batches := drainVector(t, ctx, dsn, `SELECT t, f FROM `+aggParityTable+` ORDER BY id`)
	g := vector.NewGroupByString(0, 1)
	for _, b := range batches {
		if err := g.Add(b); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}

	conn, err := pgx.Connect(ctx, dsn)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer func() { _ = conn.Close(ctx) }()
	rows, err := conn.Query(ctx,
		`SELECT t, count(*), count(f), sum(f), avg(f) FROM `+aggParityTable+` GROUP BY t`)
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	defer rows.Close()
	want := make(map[string]pgGroup)
	var wantNull *pgGroup
	for rows.Next() {
		var k *string
		var n, c int64
		var s, a any
		if err := rows.Scan(&k, &n, &c, &s, &a); err != nil {
			t.Fatalf("scan: %v", err)
		}
		gr := pgGroup{rows: n, count: c, sum: asFloat64(t, s), avg: asFloat64(t, a)}
		if k == nil {
			wantNull = &gr
		} else {
			want[*k] = gr
		}
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("rows: %v", err)
	}

	got := g.SortedRows()
	if len(got) != len(want) {
		t.Fatalf("groups = %d, want %d", len(got), len(want))
	}
	for _, gr := range got {
		w, ok := want[gr.Key]
		if !ok {
			t.Fatalf("unexpected group %q", gr.Key)
		}
		if gr.Rows != w.rows || gr.Count != w.count ||
			!floatEq(gr.Sum, w.sum) || !floatEq(gr.Avg, w.avg) {
			t.Fatalf("group %q = %+v, want %+v", gr.Key, gr, w)
		}
	}
	ng := g.NullGroup()
	if wantNull == nil || ng == nil {
		t.Fatalf("NULL group: got %v, want %v", ng, wantNull)
	}
	if ng.Rows != wantNull.rows || ng.Count != wantNull.count ||
		!floatEq(ng.Sum, wantNull.sum) || !floatEq(ng.Avg, wantNull.avg) {
		t.Fatalf("NULL group = %+v, want %+v", ng, wantNull)
	}

	par, err := vector.ParallelGroupByString(batches, 4, 0, 1)
	if err != nil {
		t.Fatalf("ParallelGroupByString: %v", err)
	}
	if par.Len() != g.Len() {
		t.Fatalf("parallel groups = %d, want %d", par.Len(), g.Len())
	}
}
