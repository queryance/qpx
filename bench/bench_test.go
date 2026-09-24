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
