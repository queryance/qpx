// Package bench holds the QPX baseline benchmarks (Measure phase).
//
// Dataset: dedicated database qpx_bench, table qpx_bench_data, exactly
// 1_000_000 rows built deterministically via generate_series:
//
//	CREATE TABLE qpx_bench_data(
//	  id bigint NOT NULL,
//	  f  double precision NOT NULL,  -- (g % 100000) / 100000.0, uniform [0,1)
//	  t  text NOT NULL,              -- 'txt_' || (g % 1000), 1000 distinct values
//	  ts timestamptz NOT NULL        -- 2020-01-01 UTC + g seconds
//	);
//
// Queries (identical SQL text on every system under test):
//
//	Q1 full scan:        SELECT id, f, t, ts FROM qpx_bench_data
//	Q2 filter + project: SELECT id, f FROM qpx_bench_data WHERE f > 0.5 (~499,990 rows)
//	Q3 group-by agg:     SELECT (id % 128) AS g, count(*) AS n, avg(f) AS a
//	                     FROM qpx_bench_data GROUP BY 1 (128 groups)
//
// Systems: (a) qpx.Execute end to end (postgres.Source + scheduler +
// iterator drain), (b) raw pgx baseline (same query, Values() decode,
// rows discarded), (c) DuckDB — see duckdb_test.go for availability.
//
// DSN discovery (first working wins, override with QPX_BENCH_DSN):
//
//  1. host=/tmp port=5432 user=juwen dbname=qpx_bench      (unix socket)
//  2. host=localhost port=5432 user=juwen dbname=qpx_bench  (TCP)
//
// Hardware/method: darwin/arm64, Apple Silicon, go1.27.1. Each
// benchmark iteration opens a fresh connection (QPX and pgx both
// connect per run, so connect cost is symmetric) and drains the full
// result. Reported metrics: rows/s (result rows) and scan_rows/s
// (1M scanned rows per iteration, comparable across Q1-Q3).
package bench
