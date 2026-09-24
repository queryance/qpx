# QPX

QPX is a parallel analytical query execution engine for PostgreSQL, written in Go.

You give it a SQL string, it streams the result back as columnar chunks (row batches in column-major layout) processed by a bounded worker pool. The default vector path decodes Postgres wire bytes straight into typed column slices, so filter, project, and aggregate operators run tight per-type loops with no interface dispatch in hot code.

## Engine-first principle

The engine (`engine/`) knows nothing about Postgres. It never imports pgx or any database driver. It only understands three interfaces: `Source` (open a scan), `Scanner` (produce chunks or batches), and `Operator` (transform one chunk or batch at a time).

`postgres/` is the first backend. It runs one query and packs rows into chunks. Later sources (JSON, Parquet, and similar) reuse the same operators, scheduler, and iterator unchanged. If you are adding a source, you implement `Open` plus a scanner, and the engine handles parallelism, ordering, and cancellation for you.

## Quick start

Requirements: Go 1.23 or later, and a reachable Postgres for live queries.

```go
import (
    "context"
    "os"

    "github.com/queryance/qpx"
    "github.com/queryance/qpx/postgres"
)

src := postgres.NewVectorSource(postgres.Config{
    ConnString: os.Getenv("TEST_POSTGRES_DSN"),
    SQL:        `SELECT id, name FROM users ORDER BY id`,
    ChunkSize:  4096,
})

it, err := qpx.ExecuteVector(ctx, src, qpx.DefaultVectorOptions())
if err != nil {
    // handle error
}
defer it.Close()

for {
    batch, ok, err := it.Next(ctx)
    if err != nil {
        // handle error, see Cancellation below
    }
    if !ok {
        break
    }
    _ = batch // batch.Columns[j], batch.NumRows(), batch.Sel
}
```

With operators (filter then project, fused into one pass):

```go
import (
    "github.com/queryance/qpx/engine/vector"
)

opts := qpx.DefaultVectorOptions()
opts.Ops = []vector.Operator{
    vector.FilterProject{
        Filter: func(b *vector.Batch, i int) bool {
            return b.Ints[0][i] > 100,
        },
        Keep: []int{0},
    },
}
it, err := qpx.ExecuteVector(ctx, src, opts)
```

Prefer the legacy row path only when a custom operator must see `[]any` cells:

```go
src := postgres.NewSource(postgres.Config{
    ConnString: os.Getenv("TEST_POSTGRES_DSN"),
    SQL:        `SELECT id, name FROM users ORDER BY id`,
    ChunkSize:  4096,
})

opts := qpx.DefaultExecuteOptions()
opts.Ops = []engine.Operator{
    engine.NewFilterOp(func(row []any) (bool, error) {
        return row[0].(int64) > 100, nil
    }),
    engine.NewProjectOp(0),
}
it, err := qpx.Execute(ctx, src, opts)
```

Run the example (needs a Postgres DSN):

```sh
TEST_POSTGRES_DSN=postgres://localhost:5432/mydb go run ./examples/basic
```

## Cancellation

Cancelling the context passed to `qpx.Execute` or `qpx.ExecuteVector` (or to `Scheduler.Run`) surfaces a wrapped `ctx.Err()` from `Iterator.Next` instead of a clean EOF. Check it with `errors.Is(err, context.Canceled)` or `errors.Is(err, context.DeadlineExceeded)`. Cancelling only the per-call context passed to `Next` returns its `ctx.Err()` without marking the iterator done. `Iterator.Close` is idempotent, never blocks on a concurrent `Next`, and hides the run's terminal error.

## Layout

| Path | Contents |
|------|----------|
| `qpx.go` | `Execute` entry point, `ExecuteOptions` defaults |
| `qpx_vector.go` | `ExecuteVector` entry point, `VectorOptions` defaults |
| `engine/scheduler.go` | `Scheduler`, worker pool, reader, straight-stream fast path |
| `engine/reassemble.go` | Ordered reassembly, terminal teardown (sole `out`/`errCh` closer) |
| `engine/iterator.go` | `Iterator` drain, per-call vs terminal cancellation |
| `engine/operator.go` | `FilterOp` (single-pass), `ProjectOp` (zero-copy) |
| `engine/chunk.go` | `Chunk`, rectangular validation |
| `engine/schema.go` | `DataType`, `Field`, `Schema` |
| `engine/source.go` | `Source` / `Scanner` / `ChunkSizer` interfaces |
| `engine/vector/` | Typed batches, selection vectors, vector operators, schedulers, aggregation |
| `postgres/config.go` | `Config`, `NewSource`, chunk-size knobs |
| `postgres/source.go` | `Source.Open` (connect plus query, exactly once) |
| `postgres/scanner.go` | Chunked row streaming, no per-row copy |
| `postgres/datatype.go` | OID mapping, schema describe |
| `postgres/pack.go` | Row-major to column-major transpose |
| `postgres/vector.go` | `VectorSource`, wire bytes to typed batches |
| `bench/` | Baseline benchmarks: QPX vs raw pgx vs DuckDB on 1M rows |

## Benchmarks

1M-row table, Apple M2, medians of 3x 1M-row iters (`-benchtime=1x`). Same SQL on both paths. QPX is `Execute` plus default options plus full drain, pgx is the same query plus `Values()` decode, rows discarded.

Q2 as tabulated pushes `WHERE f > 0.5` into Postgres, so both paths run the straight-stream path (no engine `Ops`). The engine-side equivalent, full scan plus `FilterOp`/`ProjectOp` through the worker pool, is covered by `BenchmarkQPXQ2FilterProjectOps` (result parity with the PG-side `WHERE` is guarded by `TestQ2OpsParity`). Its numbers are not in the table yet.

DuckDB is not in the table either. `bench/duckdb_test.go` measures it in-process via the `postgres` extension reading the same `qpx_bench` database, but its setup (open, `LOAD` extension, create view) is hoisted out of the timed loop while QPX and pgx reconnect per iteration, so per-iteration costs are not symmetric. Run it directly for DuckDB numbers: `go test ./bench/ -bench DuckDB -benchtime=1x`.

| query | before QPX | before pgx | before ratio | after QPX | after pgx | after ratio |
|---|---|---|---|---|---|---|
| Q1 full scan | 280ms (3.57M rows/s) | 226ms (4.43M/s) | 0.81x | 251ms (3.99M/s) | 230ms (4.34M/s) | **0.92x** |
| Q2 filter+project | 89ms (11.2M scan/s) | 71ms (14.1M scan/s) | 0.80x | 85ms (11.7M scan/s) | 73ms (13.8M scan/s) | **0.85x** |
| Q3 group-by (128) | 82ms (12.1M scan/s) | 82ms (12.2M scan/s) | 0.99x | 82ms (12.2M scan/s) | 83ms (12.2M scan/s) | **~1.0x** |

What changed: removed one alloc plus copy per row (the pgx `Values` slice is copied straight into columns, never retained), rectangular fast paths in the scanner and `PackChunk` (branch per row, not per cell), and a straight-stream fast path in the scheduler (empty `Ops` bypasses the worker pool and reassembly: one goroutine owns both channels, 3 channel hops plus map insert and delete per chunk eliminated). `FilterOp` is single-pass (no decision bitmap, no second sweep). It is not on the tabulated Q2 path (PG-side `WHERE`) but on the `BenchmarkQPXQ2FilterProjectOps` path. Q1 overhead fell 54ms to 20ms per 1M rows (63 percent removed).

Stop-rule verdict: Q1 is within about 10 percent of raw pgx (9 percent slower, was 24 percent) and Q3 is at parity (PG-bound, unchanged as expected). CPU profile of the Q1 path is about 58 percent socket read, QPX packing below profiling noise, so the path is network-receive-bound. Q2 residual (about 15 percent) is the inherent `[]any` packing cost (interface stores plus write barriers). Removing it needs typed-vector columns, a noted follow-up, not this step. Gains flattened, stopping here.

The vector path (default for new code) closes the remaining gap. `engine/vector` holds typed batches: one plain slice per column (`Ints`, `Floats`, `Strings`, `Times`, and similar), a selection vector instead of copies for filters, and per-type operator loops with no `any` in hot code. `postgres.NewVectorSource` decodes pgx wire bytes straight into those slices (batch size 4096). On the 1M-row bench this is about 10x fewer allocs on Q1 (10.0M down to 1.0M) and about 1300x fewer on Q2 (5.0M down to 0.004M), and faster than raw pgx `Values()` on both.

Engine-side aggregation (`engine/vector/agg.go`) folds batches into per-worker local state (`GlobalFloatAgg`, `GroupByInt64`, `GroupByString`) with a single-threaded `Merge`, so there are no shared maps and no locks. NULL values are counted but not summed (SQL semantics). NULL keys form their own group. `Parallel{GlobalAgg,GroupByInt64,GroupByString}` shard resident batches across workers and merge. PG `GROUP BY` pushdown stays the serving route: on the 1M-row bench, pushdown drains 128 rows at about 11.8M scanned rows/s while the engine path scans 1M rows at about 9.4M scanned rows/s, so less wire wins. The engine path exists for future non-PG sources and is parity-checked against PG group-by results cell for cell. Standalone micro-bench (resident 1M rows, Apple M2): global agg 1.16B rows/s at zero allocs, group-by-128 57M rows/s vs DuckDB 452M rows/s. The gap is Go map hash plus insert per row (profiled), and closing it needs a custom hash table, deferred until engine agg becomes a serving path. String keys hash directly (`map[string]`). Dictionary encoding was measured and rejected at the profiled shape (1K short keys, hashing is noise next to the scan). Worker scaling on live PG scans is flat 1 to 16 (single-connection scan floor, see below). On resident batches the parallel fold scales 57M (W1) to 98M (W2) to 155M (W4) to 158M (W8) to 188M (W16) rows/s: linear to core count, then core-bound, with no contention by construction.

## Tuning

| Option | Default | Effect |
|--------|---------|--------|
| `Workers` | 4 | Chunk-processing goroutines. Only used when `Ops` is non-empty (straight streams bypass the pool). Raise for CPU-heavy operators. |
| `QueueSize` | 16 | Depth of internal chunk queues. Raise to absorb bursty sources at the cost of memory. |
| `ChunkSize` / `BatchSize` | 1024 / 4096 | Rows per chunk or batch. Larger chunks amortize per-chunk channel handoff but increase latency to first results and memory per chunk. |

Start with defaults, then size `ChunkSize` so a chunk fits comfortably in L2/L3 cache for your row width. If a run has no operators, `Workers` and `QueueSize` are inert by construction, so the only knob that matters is `ChunkSize`. Set `Workers` to the number of cores doing operator work.

## Roadmap

- [ ] JSON and Parquet sources on the same engine
- [x] Engine-side vector aggregation (global, int64/string group-by, parallel merge)
- [ ] More operators (join, sort)
- [ ] Predicate and projection pushdown into backends
- [x] Vectorized (typed-batch) columns instead of `[]any`

## License

MIT, see [LICENSE](LICENSE).
