package postgres_test

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/queryance/qpx"
	"github.com/queryance/qpx/engine"
	"github.com/queryance/qpx/engine/vector"
	"github.com/queryance/qpx/postgres"
)

// vectorDSN mirrors the bench discovery without importing bench:
// explicit env first, then the local unix-socket/TCP candidates.
func vectorDSN(t *testing.T) string {
	t.Helper()
	if env := os.Getenv("TEST_POSTGRES_DSN"); env != "" {
		return env
	}
	if env := os.Getenv("QPX_BENCH_DSN"); env != "" {
		return env
	}
	for _, dsn := range []string{
		"host=/tmp port=5432 user=juwen dbname=qpx_bench",
		"host=localhost port=5432 user=juwen dbname=qpx_bench",
	} {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		conn, err := pgx.Connect(ctx, dsn)
		cancel()
		if err != nil {
			continue
		}
		ctx2, cancel2 := context.WithTimeout(context.Background(), 3*time.Second)
		err = conn.Ping(ctx2)
		_ = conn.Close(ctx2)
		cancel2()
		if err == nil {
			return dsn
		}
	}
	t.Skip("no postgres reachable (TEST_POSTGRES_DSN/QPX_BENCH_DSN unset, no local socket)")
	return ""
}

// TestVectorParity proves the vector path equals the legacy path and PG
// itself on all four Q1 column types (bigint, float8, text,
// timestamptz), including NULLs. Rows are canonicalized to strings so the
// three systems (vector batches, legacy chunks, raw pgx Values) compare
// exactly.
func TestVectorParity(t *testing.T) {
	dsn := vectorDSN(t)
	ctx := context.Background()

	conn, err := pgx.Connect(ctx, dsn)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer func() { _ = conn.Close(ctx) }()

	const tbl = "qpx_vector_parity"
	_, err = conn.Exec(ctx, `DROP TABLE IF EXISTS `+tbl)
	if err != nil {
		t.Fatalf("drop: %v", err)
	}
	_, err = conn.Exec(ctx, `CREATE TABLE `+tbl+`(
		id bigint, f double precision, t text, ts timestamptz,
		b boolean NOT NULL, n numeric NOT NULL)`)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	// 5K deterministic rows; NULLs cycled through the four nullable
	// columns so null handling is compared, not just happy paths.
	_, err = conn.Exec(ctx, `INSERT INTO `+tbl+`
		SELECT
			g,
			CASE WHEN g % 7 = 0 THEN NULL ELSE (g % 100000)::float8/100000.0 END,
			CASE WHEN g % 11 = 0 THEN NULL ELSE 'txt_'||(g%1000) END,
			CASE WHEN g % 13 = 0 THEN NULL ELSE timestamptz '2020-01-01 UTC' + make_interval(secs => g) END,
			(g % 2 = 0),
			(g % 1000)::numeric / 3
		FROM generate_series(1,5000) g`)
	if err != nil {
		t.Fatalf("fill: %v", err)
	}

	sql := `SELECT id, f, t, ts, b, n FROM ` + tbl + ` ORDER BY id`

	want := pgCanonical(t, ctx, dsn, sql)
	gotLegacy := legacyCanonical(t, ctx, dsn, sql)
	gotVector := vectorCanonical(t, ctx, dsn, sql)

	compareCanonical(t, "legacy vs pg", want, gotLegacy)
	compareCanonical(t, "vector vs pg", want, gotVector)
}

// TestVectorFilterParity proves the fused filter+project matches the
// PG-side WHERE on row identity, not just counts.
func TestVectorFilterParity(t *testing.T) {
	dsn := vectorDSN(t)
	ctx := context.Background()

	conn, err := pgx.Connect(ctx, dsn)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer func() { _ = conn.Close(ctx) }()

	const tbl = "qpx_vector_fparity"
	_, err = conn.Exec(ctx, `DROP TABLE IF EXISTS `+tbl)
	if err != nil {
		t.Fatalf("drop: %v", err)
	}
	_, err = conn.Exec(ctx, `CREATE TABLE `+tbl+`(id bigint NOT NULL, f double precision NOT NULL)`)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	_, err = conn.Exec(ctx, `INSERT INTO `+tbl+`
		SELECT g, (g % 100000)::float8/100000.0 FROM generate_series(1,20000) g`)
	if err != nil {
		t.Fatalf("fill: %v", err)
	}

	pgSQL := `SELECT id, f FROM ` + tbl + ` WHERE f > 0.5 ORDER BY id`
	want := pgCanonical(t, ctx, dsn, pgSQL)

	scanSQL := `SELECT id, f FROM ` + tbl
	src := postgres.NewVectorSource(postgres.Config{ConnString: dsn, SQL: scanSQL})
	opts := qpx.DefaultVectorOptions()
	opts.Ops = []vector.Operator{vector.NewFilterFloat64GTProject(1, 0.5, 0, 1)}
	it, err := qpx.ExecuteVector(ctx, src, opts)
	if err != nil {
		t.Fatalf("ExecuteVector: %v", err)
	}
	defer func() { _ = it.Close() }()
	var got []string
	for {
		b, ok, err := it.Next(ctx)
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if !ok {
			break
		}
		got = append(got, batchCanonical(b)...)
	}
	compareCanonical(t, "vector fused vs pg WHERE", want, got)
}

// canon formats one cell exactly: canonical form shared by all three paths.
func canon(v any) string {
	switch n := v.(type) {
	case nil:
		return "NULL"
	case int64:
		return strconv.FormatInt(n, 10)
	case int32:
		return strconv.FormatInt(int64(n), 10)
	case int16:
		return strconv.FormatInt(int64(n), 10)
	case int:
		return strconv.Itoa(n)
	case float64:
		return strconv.FormatFloat(n, 'g', -1, 64)
	case float32:
