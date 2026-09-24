package bench

import (
	"context"
	"fmt"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/queryance/qpx"
	"github.com/queryance/qpx/engine"
	"github.com/queryance/qpx/postgres"
)

// Shared SQL: byte-identical on every system under test.
const (
	sqlQ1 = `SELECT id, f, t, ts FROM qpx_bench_data`
	sqlQ2 = `SELECT id, f FROM qpx_bench_data WHERE f > 0.5`
	sqlQ3 = `SELECT (id % 128) AS g, count(*) AS n, avg(f) AS a FROM qpx_bench_data GROUP BY 1`
	// sqlQ2FullScan is the engine-side Q2: same columns as sqlQ2 without
	// the WHERE clause. Filtering and projection run in the engine via
	// FilterOp/ProjectOp (see q2Ops), so this variant exercises the
	// operator + worker-pool path that sqlQ2 (PG-side WHERE, straight
	// stream, no Ops) bypasses.
	sqlQ2FullScan = `SELECT id, f FROM qpx_bench_data`
)

const (
	benchDB    = "qpx_bench"
	benchTable = "qpx_bench_data"
	benchRows  = 1_000_000
)

var dsnCandidates = []string{
	"host=/tmp port=5432 user=juwen dbname=qpx_bench",
	"host=localhost port=5432 user=juwen dbname=qpx_bench",
}

var (
	resolveOnce sync.Once
	resolvedDSN string
	setupOnce   sync.Once
	setupErr    error
)

// resolveDSN returns the first candidate DSN that connects (or
// QPX_BENCH_DSN when set), dialing each with a short timeout.
func resolveDSN() string {
	resolveOnce.Do(func() {
		if env := os.Getenv("QPX_BENCH_DSN"); env != "" {
			resolvedDSN = env
			return
		}
		for _, dsn := range dsnCandidates {
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
				resolvedDSN = dsn
				return
			}
		}
	})
	return resolvedDSN
}

func openDB(ctx context.Context, dsn string) (*pgx.Conn, error) {
	return pgx.Connect(ctx, dsn)
}

// ensureDataset creates the database/table when missing and rebuilds the
// table when its row count differs from benchRows. Owner has DDL rights,
// so no superuser is needed. Runs once per test binary.
func ensureDataset(tb testing.TB) {
	tb.Helper()
	setupOnce.Do(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		defer cancel()
		dsn := resolveDSN()
		if dsn == "" {
			setupErr = fmt.Errorf("bench: no working DSN (tried %v, QPX_BENCH_DSN=%q)", dsnCandidates, os.Getenv("QPX_BENCH_DSN"))
			return
		}
		conn, err := openDB(ctx, dsn)
		if err != nil {
			// Database itself may be missing: create it via the postgres db.
			admin, aerr := pgx.Connect(ctx, "host=/tmp port=5432 user=juwen dbname=postgres")
			if aerr != nil {
				admin, aerr = pgx.Connect(ctx, "host=localhost port=5432 user=juwen dbname=postgres")
			}
			if aerr != nil {
				setupErr = fmt.Errorf("bench: connect %s: %w", dsn, err)
				return
			}
			_, _ = admin.Exec(ctx, "CREATE DATABASE "+benchDB)
			_ = admin.Close(ctx)
			conn, err = openDB(ctx, dsn)
			if err != nil {
				setupErr = fmt.Errorf("bench: connect %s: %w", dsn, err)
				return
			}
		}
		defer func() { _ = conn.Close(ctx) }()
		var n int
		err = conn.QueryRow(ctx, "SELECT count(*) FROM "+benchTable).Scan(&n)
		if err == nil && n == benchRows {
			return
		}
		_, err = conn.Exec(ctx, `DROP TABLE IF EXISTS `+benchTable)
		if err != nil {
			setupErr = fmt.Errorf("bench: drop: %w", err)
			return
		}
		_, err = conn.Exec(ctx, `CREATE TABLE `+benchTable+
			`(id bigint NOT NULL, f double precision NOT NULL, t text NOT NULL, ts timestamptz NOT NULL)`)
		if err != nil {
			setupErr = fmt.Errorf("bench: create: %w", err)
			return
		}
		_, err = conn.Exec(ctx, `INSERT INTO `+benchTable+
			` SELECT g, (g % 100000)::float8/100000.0, 'txt_'||(g%1000),`+
			` timestamptz '2020-01-01 UTC' + make_interval(secs => g)`+
			` FROM generate_series(1,1000000) g`)
		if err != nil {
			setupErr = fmt.Errorf("bench: fill: %w", err)
			return
		}
	})
	if setupErr != nil {
		tb.Fatalf("bench setup: %v", setupErr)
	}
}

// runQPX executes sql through qpx.Execute with default options and drains
// the iterator, returning result rows. It is the (a) end-to-end path.
func runQPX(ctx context.Context, sql string) (int64, error) {
	src := postgres.NewSource(postgres.Config{ConnString: resolveDSN(), SQL: sql})
	it, err := qpx.Execute(ctx, src, qpx.DefaultExecuteOptions())
	if err != nil {
		return 0, err
	}
	defer func() { _ = it.Close() }()
	var rows int64
	for {
		c, ok, err := it.Next(ctx)
		if err != nil {
			return rows, err
		}
		if !ok {
			return rows, nil
		}
		rows += int64(c.NumRows())
	}
}

// runPGX executes sql with raw pgx, decoding every row via Values and
// discarding it. It is the (b) baseline: same query, no engine overhead.
func runPGX(ctx context.Context, sql string) (int64, error) {
	conn, err := pgx.Connect(ctx, resolveDSN())
	if err != nil {
		return 0, err
	}
	defer func() { _ = conn.Close(ctx) }()
	rows, err := conn.Query(ctx, sql)
	if err != nil {
		return 0, err
	}
	defer rows.Close()
	var n int64
	for rows.Next() {
		if _, err := rows.Values(); err != nil {
			return n, err
		}
		n++
	}
	return n, rows.Err()
}

// drive runs fn once per iteration and reports rows/s (result rows) and
// scan_rows/s (benchRows scanned per iteration, comparable across Q1-Q3).
func drive(b *testing.B, sql string, fn func(context.Context, string) (int64, error)) {
	b.Helper()
	ensureDataset(b)
	if resolveDSN() == "" {
		b.Fatalf("bench: no working DSN (tried %v)", dsnCandidates)
	}
	// Warmup: shake out one-time costs (connection, plan cache) so the
	// timed iterations measure steady-state streaming.
	if _, err := fn(context.Background(), sql); err != nil {
		b.Fatalf("warmup: %v", err)
	}
	b.ResetTimer()
	var total int64
	for i := 0; i < b.N; i++ {
		n, err := fn(context.Background(), sql)
		if err != nil {
			b.Fatalf("iter %d: %v", i, err)
		}
		total += n
	}
	b.StopTimer()
	sec := b.Elapsed().Seconds()
	b.ReportMetric(float64(total)/sec, "rows/s")
	b.ReportMetric(float64(int64(b.N)*benchRows)/sec, "scan_rows/s")
	b.ReportMetric(float64(total)/float64(b.N), "rows/op")
}

func BenchmarkQPXQ1FullScan(b *testing.B) { drive(b, sqlQ1, runQPX) }
func BenchmarkPGXQ1FullScan(b *testing.B) { drive(b, sqlQ1, runPGX) }

func BenchmarkQPXQ2FilterProject(b *testing.B) { drive(b, sqlQ2, runQPX) }

// q2Ops filters f > 0.5 (column 1, float64 from pgx float8) and projects
// (id, f). The predicate returns an error instead of panicking on an
// unexpected type so a schema change surfaces as a bench failure.
func q2Ops() []engine.Operator {
	return []engine.Operator{
		engine.NewFilterOp(func(row []any) (bool, error) {
			f, ok := row[1].(float64)
			if !ok {
				return false, fmt.Errorf("bench: q2 filter: column 1 is %T, want float64", row[1])
			}
			return f > 0.5, nil
		}),
		engine.NewProjectOp(0, 1),
	}
}

// runQPXOps executes sql through qpx.Execute with the given operator
// pipeline and drains the iterator, returning result rows.
func runQPXOps(ctx context.Context, sql string, ops []engine.Operator) (int64, error) {
	src := postgres.NewSource(postgres.Config{ConnString: resolveDSN(), SQL: sql})
	opts := qpx.DefaultExecuteOptions()
	opts.Ops = ops
	it, err := qpx.Execute(ctx, src, opts)
	if err != nil {
		return 0, err
	}
	defer func() { _ = it.Close() }()
	var rows int64
	for {
		c, ok, err := it.Next(ctx)
		if err != nil {
			return rows, err
		}
		if !ok {
			return rows, nil
		}
		rows += int64(c.NumRows())
	}
}

// BenchmarkQPXQ2FilterProjectOps runs Q2 with engine-side FilterOp +
// ProjectOp over a full scan, exercising the operator path. Result rows
// must match sqlQ2 (499,990); scan work is 1M rows.
func BenchmarkQPXQ2FilterProjectOps(b *testing.B) {
	ops := q2Ops()
	drive(b, sqlQ2FullScan, func(ctx context.Context, _ string) (int64, error) {
		return runQPXOps(ctx, sqlQ2FullScan, ops)
	})
}
func BenchmarkPGXQ2FilterProject(b *testing.B) { drive(b, sqlQ2, runPGX) }

func BenchmarkQPXQ3GroupBy(b *testing.B) { drive(b, sqlQ3, runQPX) }
func BenchmarkPGXQ3GroupBy(b *testing.B) { drive(b, sqlQ3, runPGX) }

// TestQ2OpsParity guards that the engine-side Q2 (full scan +
// FilterOp/ProjectOp) returns the same row count as the PG-side WHERE.
func TestQ2OpsParity(t *testing.T) {
	ensureDataset(t)
	if resolveDSN() == "" {
		t.Fatalf("bench: no working DSN (tried %v)", dsnCandidates)
	}
	ctx := context.Background()
	got, err := runQPXOps(ctx, sqlQ2FullScan, q2Ops())
	if err != nil {
		t.Fatalf("q2-ops: %v", err)
	}
	if got != 499990 {
		t.Fatalf("q2-ops: got %d rows, want 499990", got)
	}
}

// TestDatasetSanity guards the benchmark preconditions: exact row count,
// Q2 selectivity, and Q3 group count.
func TestDatasetSanity(t *testing.T) {
	ensureDataset(t)
	ctx := context.Background()
	conn, err := pgx.Connect(ctx, resolveDSN())
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer func() { _ = conn.Close(ctx) }()
	checks := []struct {
		name string
		sql  string
		want int64
	}{
		{"rows", "SELECT count(*) FROM " + benchTable, benchRows},
		{"q2", "SELECT count(*) FROM " + benchTable + " WHERE f > 0.5", 499990},
		{"q3", "SELECT count(*) FROM (SELECT (id % 128) AS g FROM " + benchTable + " GROUP BY 1) s", 128},
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
