package bench

// Widened benchmark matrix: documentation supplement to doc.go.
//
// The legacy Q1-Q3 baseline (M = 1M rows, QPX vs pgx vs DuckDB) is
// documented in doc.go and untouched. This file documents the widened
// matrix only: size dimension S/M/L, join dimension J, the full query
// catalog, the worker sweep, and the memory test.
//
// DSNs (same discovery as the legacy benches, see resolveDSN; override
// with QPX_BENCH_DSN):
//
//  1. host=/tmp port=5432 user=juwen dbname=qpx_bench      (unix socket)
//  2. host=localhost port=5432 user=juwen dbname=qpx_bench  (TCP)
//
// Tables (all in database qpx_bench, qpx_bench_* prefix):
//
//  qpx_bench_data  M, legacy, LOGGED, 1_000_000 rows.
//                  (id, f, t low-card 1000 distinct, ts). See doc.go.
//  qpx_bench_s     S, UNLOGGED, 100_000 rows.
//                  (id, f, t low-card 1000 distinct, th high-card 100K
//                  unique, ts).
//  qpx_bench_l     L, UNLOGGED, 10_000_000 rows. Same shape as S; th is
//                  10M unique.
//  qpx_bench_j     J, UNLOGGED, 1_000_000 rows. (jid FK into
//                  qpx_bench_data.id, payload). jid matches M ids 1..1M
//                  exactly, so the M⋈J equijoin yields 1M rows. The FK
//                  target is CONSTRAINT qpx_bench_data_id_uniq, added to
//                  the legacy table once by setup (ids are unique by
//                  construction, so the ALTER is a pure index build).
//
// Sizes, heap size via \dt+ on darwin/arm64 PG 18 (UNLOGGED tables
// carry no WAL weight), measured 2026-09-24:
//
//  qpx_bench_data  57 MB heap + 21 MB qpx_bench_data_id_uniq (legacy + new UNIQUE)
//  qpx_bench_s     7368 kB heap (100K rows)
//  qpx_bench_l     730 MB heap (10M rows)
//  qpx_bench_j     42 MB heap + 21 MB qpx_bench_j_jid_idx (1M rows)
//
// Setup: ensureMatrixDataset (matrix_setup_test.go) first runs the legacy
// ensureDataset, then rebuilds any S/L/J table whose count mismatches via
// deterministic generate_series INSERTs (L takes ~1 min once). UNLOGGED
// keeps setup and reruns free of WAL fsyncs; contents are reproducible,
// so durability is not needed.
//
// Query catalog (byte-identical SQL on QPX, pgx, and DuckDB; only the
// table name varies by dataset; join/sort/limit are M-only):
//
//  scan    SELECT id, f, t, ts FROM T
//  filter  SELECT id, f FROM T WHERE f > 0.5
//          (~50%: S 49,999 / M 499,990 / L 4,999,900 rows)
//  agg     SELECT count(*), sum(f), avg(f) FROM T          (1 row out)
//  glow    SELECT (id % 128) ..., GROUP BY 1               (128 groups)
//  ghigh   SELECT (id % 100000) ..., GROUP BY 1            (100K groups)
//  join    SELECT d.id, d.f, j.payload FROM qpx_bench_data d
//          JOIN qpx_bench_j j ON j.jid = d.id              (1M rows)
//  sort    SELECT id, f FROM qpx_bench_data ORDER BY f     (1M rows)
//  limit   SELECT id, f, t, ts FROM qpx_bench_data LIMIT 100 (100 rows)
//
// NOTE join/sort/group/agg compute inside Postgres (QPX drains the
// server-side result); only scan/filter move row volume through the
// engine. DuckDB reads via postgres_scan views (setup hoisted out of the
// timed loop; see mxOpenDuckDB).
//
// Systems: (a) qpx.Execute end to end, (b) raw pgx baseline, (c) DuckDB
// in-process via the postgres extension.
//
// Worker sweep: ExecuteOptions.Workers 1/2/4/8/16 (default 4, see
// engine.DefaultWorkers) on M scan and filter+project, QPX only —
// BenchmarkMxQPXW{1,2,4,8,16}{Scan,Filter}.
//
// Memory: TestMxMemory (matrix_memory_test.go) GCs, snapshots
// runtime.MemStats, drains the full M scan per system, and logs
// TotalAlloc delta (+ bytes/row), Sys, HeapAlloc, and GC count delta.
// Run benchmarks with -benchmem for allocs/op on every benchmark.
//
// Hardware/method: darwin/arm64, Apple Silicon, go1.27.1. Fresh
// connection per iteration (QPX and pgx symmetric); rows/s = result
// rows, scan_rows/s = scanned rows (comparable across queries).
//
// Run recipes:
//
//  Full matrix with allocs (S+M+legacy+workers; L excluded):
//    go test ./bench/ -run NONE \
//      -bench 'Mx.*(S|M)$|MxQPXW|QPXQ|PGXQ|DuckDBQ' -benchmem
//  L only (slow; single iteration, twice):
//    go test ./bench/ -run NONE -bench 'Mx.*L$' \
//      -benchtime=1x -count=2 -benchmem
//  Memory:
//    go test ./bench/ -run TestMxMemory -v
//  Sanity (dataset preconditions):
//    go test ./bench/ -run 'TestMxDatasetSanity|TestDatasetSanity|TestDuckDBSanity' -v
