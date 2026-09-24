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
		return strconv.FormatFloat(float64(n), 'g', -1, 32)
	case string:
		return strconv.Quote(n)
	case bool:
		if n {
			return "true"
		}
		return "false"
	case []byte:
		return fmt.Sprintf("%x", n)
	case time.Time:
		return n.UTC().Format(time.RFC3339Nano)
	case pgtype.Numeric:
		// Canonical float form: the vector path stores numeric text,
		// pgx Values() yields the struct — both reduce to the same
		// float64 so the three systems compare exactly.
		if !n.Valid {
			return "NULL"
		}
		f, err := n.Float64Value()
		if err != nil {
			return fmt.Sprintf("bad-numeric:%v", err)
		}
		return strconv.FormatFloat(f.Float64, 'g', -1, 64)
	default:
		return fmt.Sprintf("%v", v)
	}
}

func pgCanonical(t *testing.T, ctx context.Context, dsn, sql string) []string {
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
	var out []string
	for rows.Next() {
		vals, err := rows.Values()
		if err != nil {
			t.Fatalf("Values: %v", err)
		}
		s := ""
		for i, v := range vals {
			if i > 0 {
				s += "|"
			}
			s += canon(v)
		}
		out = append(out, s)
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("rows: %v", err)
	}
	return out
}

func legacyCanonical(t *testing.T, ctx context.Context, dsn, sql string) []string {
	t.Helper()
	src := postgres.NewSource(postgres.Config{ConnString: dsn, SQL: sql})
	it, err := qpx.Execute(ctx, src, qpx.DefaultExecuteOptions())
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	defer func() { _ = it.Close() }()
	var out []string
	for {
		c, ok, err := it.Next(ctx)
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if !ok {
			break
		}
		for i := 0; i < c.NumRows(); i++ {
			s := ""
			for j := range c.Columns {
				if j > 0 {
					s += "|"
				}
				s += canon(c.Columns[j][i])
			}
			out = append(out, s)
		}
	}
	return out
}

func batchCanonical(b *vector.Batch) []string {
	out := make([]string, 0, b.Len())
	for i := 0; i < b.Len(); i++ {
		r := i
		if b.Sel != nil {
			r = int(b.Sel[i])
		}
		s := ""
		for j := range b.Columns {
			if j > 0 {
				s += "|"
			}
			c := &b.Columns[j]
			if c.IsNull(r) {
				s += "NULL"
				continue
			}
			switch c.Type {
			case engine.Int64:
				s += canon(c.Ints[r])
			case engine.Float64:
				s += canon(c.Floats[r])
			case engine.Bool:
				s += canon(c.Bools[r])
			case engine.String:
				s += canon(c.Strings[r])
			case engine.Bytes:
				s += canon(c.Bytes[r])
			case engine.Time:
				s += canon(c.Times[r])
			case engine.Numeric:
				f, err := strconv.ParseFloat(c.Numerics[r], 64)
				if err != nil {
					s += "bad-numeric:" + c.Numerics[r]
				} else {
					s += strconv.FormatFloat(f, 'g', -1, 64)
				}
			default:
				s += canon(c.Strings[r])
			}
		}
		out = append(out, s)
	}
	return out
}

func vectorCanonical(t *testing.T, ctx context.Context, dsn, sql string) []string {
	t.Helper()
	src := postgres.NewVectorSource(postgres.Config{ConnString: dsn, SQL: sql})
	it, err := qpx.ExecuteVector(ctx, src, qpx.DefaultVectorOptions())
	if err != nil {
		t.Fatalf("ExecuteVector: %v", err)
	}
	defer func() { _ = it.Close() }()
	var out []string
	for {
		b, ok, err := it.Next(ctx)
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if !ok {
			break
		}
		out = append(out, batchCanonical(b)...)
	}
	return out
}

func compareCanonical(t *testing.T, name string, want, got []string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d rows, want %d", name, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s: row %d differs:\n got %s\nwant %s", name, i, got[i], want[i])
		}
	}
}
