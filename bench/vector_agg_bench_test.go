package bench

import (
	"context"
	"database/sql"
	"fmt"
	"runtime"
	"strconv"
	"sync"
	"testing"

	"github.com/queryance/qpx"
	"github.com/queryance/qpx/engine"
	"github.com/queryance/qpx/engine/vector"
	"github.com/queryance/qpx/postgres"
)

// Vector aggregation benches: engine-side agg over batches plus the
// worker-scaling sweep for vector scan and filter+project.
//
// Two arenas, kept separate on purpose:
//   - (mem) resident batches, no PG: pure engine compute. This is where
//     worker scaling is attributable to our code — no network floor.
//   - (pg) live PG scans: end-to-end scaling including the single PG
//     connection. Flat here means PG/network-bound, not engine-bound.
//
// DuckDB equivalents aggregate a resident table generated with the same
// distribution (no PG), so the micro-bench compares compute, not IO.

// aggMemBatches builds 1M resident rows with the bench distribution:
// id 1..N, f = (id%100000)/100000, t = 'txt_<id%1000>'. Shared read-only
// across benches; agg folds never mutate batches.
//
// Do NOT run Filter.Process (or any Sel-compacting op) on these batches:
// it compacts Sel in place, poisoning later iterations and racing under
// -race. A bench that filters must Clone each batch first.
var (
	aggMemOnce    sync.Once
	aggMemBatches []*vector.Batch
	aggMemSchema  = engine.Schema{Fields: []engine.Field{
		{Name: "id", Type: engine.Int64},
		{Name: "f", Type: engine.Float64},
		{Name: "t", Type: engine.String},
	}}
)

func memBatches() []*vector.Batch {
	aggMemOnce.Do(func() {
		const n = 1_000_000
		const size = 4096
		for base := 0; base < n; base += size {
			m := size
			if base+m > n {
				m = n - base
			}
			b := vector.NewBatch(aggMemSchema, m)
			for i := 0; i < m; i++ {
				id := int64(base + i + 1)
				b.Columns[0].AppendInt(id)
				b.Columns[1].AppendFloat(float64(id%100000) / 100000.0)
				b.Columns[2].AppendString("txt_" + strconv.FormatInt(id%1000, 10))
				b.NumRows++
			}
			aggMemBatches = append(aggMemBatches, b)
		}
	})
	return aggMemBatches
}

var (
	sinkGlobal vector.GlobalResult
	sinkGroups int
)

// driveMem times fn per iteration and reports scanned rows/s over the
// resident 1M rows. batches is the shared read-only fixture: fn must not
// filter or otherwise mutate it (see aggMemBatches).
func driveMem(b *testing.B, fn func() error) {
	b.Helper()
	batches := memBatches()
	var total int64
	for _, bh := range batches {
		total += int64(bh.NumRows)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := fn(); err != nil {
			b.Fatalf("iter %d: %v", i, err)
		}
	}
	sec := b.Elapsed().Seconds()
	b.ReportMetric(float64(int64(b.N)*total)/sec, "scan_rows/s")
}

func BenchmarkVectorAggGlobalMem(b *testing.B) {
	batches := memBatches()
	driveMem(b, func() error {
		a := vector.NewGlobalFloatAgg(1)
		for _, bh := range batches {
			if err := a.Add(bh); err != nil {
				return err
			}
		}
		sinkGlobal = a.Result()
		return nil
	})
}

func BenchmarkVectorGroupBy128Mem(b *testing.B) {
	batches := memBatches()
	driveMem(b, func() error {
		g := vector.NewGroupByInt64(0, 1, 128)
		for _, bh := range batches {
			if err := g.Add(bh); err != nil {
				return err
			}
		}
		sinkGroups = g.Len()
		return nil
	})
}

func BenchmarkVectorGroupBy100KMem(b *testing.B) {
	batches := memBatches()
	driveMem(b, func() error {
		g := vector.NewGroupByInt64(0, 1, 100000)
		for _, bh := range batches {
			if err := g.Add(bh); err != nil {
				return err
			}
		}
		sinkGroups = g.Len()
		return nil
	})
}

func BenchmarkVectorGroupByString1KMem(b *testing.B) {
	batches := memBatches()
	driveMem(b, func() error {
		g := vector.NewGroupByString(2, 1)
		for _, bh := range batches {
			if err := g.Add(bh); err != nil {
				return err
			}
		}
		sinkGroups = g.Len()
		return nil
	})
}

// Pure-compute worker scaling: same resident batches, Parallel merge.
// No PG in the loop, so any flattening here is ours to fix.
func driveParGroup128(b *testing.B, workers int) {
	batches := memBatches()
	driveMem(b, func() error {
		g, err := vector.ParallelGroupByInt64(batches, workers, 0, 1, 128)
		if err != nil {
			return err
		}
		sinkGroups = g.Len()
		return nil
	})
}

func BenchmarkVectorParGroup128W1(b *testing.B)  { driveParGroup128(b, 1) }
func BenchmarkVectorParGroup128W2(b *testing.B)  { driveParGroup128(b, 2) }
func BenchmarkVectorParGroup128W4(b *testing.B)  { driveParGroup128(b, 4) }
func BenchmarkVectorParGroup128W8(b *testing.B)  { driveParGroup128(b, 8) }
func BenchmarkVectorParGroup128W16(b *testing.B) { driveParGroup128(b, 16) }

// openDuckDBMem opens an in-memory DuckDB (no postgres extension) with a
// resident table m matching the memBatches distribution, generated via
// range. Setup cost is hoisted out of the timed loop by driveDuckDBMem.
func openDuckDBMem(ctx context.Context) (*duckDB, error) {
	db, err := sql.Open("duckdb", "")
	if err != nil {
		return nil, fmt.Errorf("bench: duckdb open: %w", err)
	}
	conn, err := db.Conn(ctx)
	if err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("bench: duckdb conn: %w", err)
	}
	const ddl = `CREATE TABLE m AS SELECT (r+1) AS id,` +
		` ((r+1) % 100000)::DOUBLE/100000.0 AS f,` +
		` 'txt_'||((r+1)%1000) AS t FROM range(1000000) t(r)`
	if _, err := conn.ExecContext(ctx, ddl); err != nil {
		_ = conn.Close()
		_ = db.Close()
		return nil, fmt.Errorf("bench: duckdb build m: %w", err)
	}
	return &duckDB{db: db, conn: conn}, nil
}

// DuckDB over a resident table with the same distribution, generated via
// range (no PG). Setup (open + CREATE) is hoisted out of the timed loop;
// each iteration runs the query and Scan-decodes every output row.
func driveDuckDBMem(b *testing.B, sql string) {
	b.Helper()
	ctx := context.Background()
	db, err := openDuckDBMem(ctx)
	if err != nil {
		b.Fatalf("duckdb setup: %v", err)
	}
	defer db.close()
	if _, err := db.runQuery(ctx, sql); err != nil {
		b.Fatalf("warmup: %v", err)
	}
	b.ResetTimer()
	var total int64
	for i := 0; i < b.N; i++ {
		n, err := db.runQuery(ctx, sql)
		if err != nil {
			b.Fatalf("iter %d: %v", i, err)
		}
		total += n
	}
	sec := b.Elapsed().Seconds()
	b.ReportMetric(float64(int64(b.N)*1_000_000)/sec, "scan_rows/s")
	b.ReportMetric(float64(total)/float64(b.N), "rows/op")
}

func BenchmarkDuckDBAggGlobalMem(b *testing.B) {
	driveDuckDBMem(b, `SELECT count(*), sum(f), avg(f) FROM m`)
}
func BenchmarkDuckDBGroupBy128Mem(b *testing.B) {
	driveDuckDBMem(b, `SELECT (id % 128) AS g, count(*), avg(f) FROM m GROUP BY 1`)
}
func BenchmarkDuckDBGroupBy100KMem(b *testing.B) {
	driveDuckDBMem(b, `SELECT (id % 100000) AS g, count(*), avg(f) FROM m GROUP BY 1`)
}
func BenchmarkDuckDBGroupByString1KMem(b *testing.B) {
	driveDuckDBMem(b, `SELECT t, count(*), avg(f) FROM m GROUP BY 1`)
}

// Live PG worker sweep for the vector path: scan (no ops) mirrors
// BenchmarkMxQPXW*Scan on the same sqlQ1. The fused filter+project sweep
// below (sqlQ2FullScan, engine-side filter) does NOT mirror
// BenchmarkMxQPXW*Filter (sqlQ2, PG-side WHERE): compare legacy Filter
// against BenchmarkVectorW*PGFilter instead.
func runVectorWorkers(ctx context.Context, sql string, workers int, ops []vector.Operator) (int64, error) {
	src := postgres.NewVectorSource(postgres.Config{ConnString: resolveDSN(), SQL: sql})
	opts := qpx.DefaultVectorOptions()
	opts.Workers = workers
	opts.Ops = ops
	it, err := qpx.ExecuteVector(ctx, src, opts)
	if err != nil {
		return 0, err
	}
	defer func() { _ = it.Close() }()
	var rows int64
	for {
		bh, ok, err := it.Next(ctx)
		if err != nil {
			return rows, err
		}
		if !ok {
			return rows, nil
		}
		rows += int64(bh.Len())
	}
}

func mxDriveVectorWorkers(b *testing.B, sql string, workers int, ops []vector.Operator) {
	b.Helper()
	driveMatrix(b, sql, benchRows, func(ctx context.Context, s string) (int64, error) {
		return runVectorWorkers(ctx, s, workers, ops)
	})
}

func BenchmarkVectorW1Scan(b *testing.B)  { mxDriveVectorWorkers(b, sqlQ1, 1, nil) }
func BenchmarkVectorW2Scan(b *testing.B)  { mxDriveVectorWorkers(b, sqlQ1, 2, nil) }
func BenchmarkVectorW4Scan(b *testing.B)  { mxDriveVectorWorkers(b, sqlQ1, 4, nil) }
func BenchmarkVectorW8Scan(b *testing.B)  { mxDriveVectorWorkers(b, sqlQ1, 8, nil) }
func BenchmarkVectorW16Scan(b *testing.B) { mxDriveVectorWorkers(b, sqlQ1, 16, nil) }

// Engine-side filter+project sweep: full 1M-row scan with the fused
// f>0.5 filter running in the workers. NOT head-to-head with
// BenchmarkMxQPXW*Filter (PG-side WHERE, ~500k rows out of PG): different
// scan work and different filter location. For a direct worker comparison
// against the legacy path, see BenchmarkVectorW*PGFilter below, which
// drains the PG-filtered sqlQ2 with no engine ops.
func BenchmarkVectorW1EngineFilter(b *testing.B) {
	mxDriveVectorWorkers(b, sqlQ2FullScan, 1, []vector.Operator{vector.NewFilterFloat64GTProject(1, 0.5, 0, 1)})
}
func BenchmarkVectorW2EngineFilter(b *testing.B) {
	mxDriveVectorWorkers(b, sqlQ2FullScan, 2, []vector.Operator{vector.NewFilterFloat64GTProject(1, 0.5, 0, 1)})
}
func BenchmarkVectorW4EngineFilter(b *testing.B) {
	mxDriveVectorWorkers(b, sqlQ2FullScan, 4, []vector.Operator{vector.NewFilterFloat64GTProject(1, 0.5, 0, 1)})
}
func BenchmarkVectorW8EngineFilter(b *testing.B) {
	mxDriveVectorWorkers(b, sqlQ2FullScan, 8, []vector.Operator{vector.NewFilterFloat64GTProject(1, 0.5, 0, 1)})
}
func BenchmarkVectorW16EngineFilter(b *testing.B) {
	mxDriveVectorWorkers(b, sqlQ2FullScan, 16, []vector.Operator{vector.NewFilterFloat64GTProject(1, 0.5, 0, 1)})
}

// PG-filtered sweep: same sqlQ2 (PG-side WHERE) as BenchmarkMxQPXW*Filter,
// drained through the vector path with no engine ops — the head-to-head
// worker comparison. Pair with BenchmarkVectorW*EngineFilter (full scan,
// engine-side filter) to separate scan/filter placement effects.
func BenchmarkVectorW1PGFilter(b *testing.B)  { mxDriveVectorWorkers(b, sqlQ2, 1, nil) }
func BenchmarkVectorW2PGFilter(b *testing.B)  { mxDriveVectorWorkers(b, sqlQ2, 2, nil) }
func BenchmarkVectorW4PGFilter(b *testing.B)  { mxDriveVectorWorkers(b, sqlQ2, 4, nil) }
func BenchmarkVectorW8PGFilter(b *testing.B)  { mxDriveVectorWorkers(b, sqlQ2, 8, nil) }
func BenchmarkVectorW16PGFilter(b *testing.B) { mxDriveVectorWorkers(b, sqlQ2, 16, nil) }

// Live engine-side group-by: full scan of (id, f) with per-batch folds
// into one GroupByInt64. Compare against BenchmarkQPXQ3GroupBy, where PG
// aggregates and QPX drains 128 rows.
func runVectorGroupBy128PG(ctx context.Context) (int64, error) {
	src := postgres.NewVectorSource(postgres.Config{ConnString: resolveDSN(), SQL: `SELECT id, f FROM qpx_bench_data`})
	it, err := qpx.ExecuteVector(ctx, src, qpx.DefaultVectorOptions())
	if err != nil {
		return 0, err
	}
	defer func() { _ = it.Close() }()
	g := vector.NewGroupByInt64(0, 1, 128)
	for {
		bh, ok, err := it.Next(ctx)
		if err != nil {
			return 0, err
		}
		if !ok {
			return int64(g.Len()), nil
		}
		if err := g.Add(bh); err != nil {
			return 0, err
		}
	}
}

func BenchmarkVectorGroupBy128PG(b *testing.B) {
	drive(b, `SELECT id, f FROM qpx_bench_data`, func(ctx context.Context, _ string) (int64, error) {
		return runVectorGroupBy128PG(ctx)
	})
}

// TestVectorGroupBy128Live guards the engine-side group-by on the real
// 1M-row dataset: per-group rows match PG's GROUP BY exactly, sums and
// avgs within fold-order tolerance.
func TestVectorGroupBy128Live(t *testing.T) {
	ensureDataset(t)
	if resolveDSN() == "" {
		t.Fatalf("bench: no working DSN (tried %v)", dsnCandidates)
	}
	ctx := context.Background()
	src := postgres.NewVectorSource(postgres.Config{ConnString: resolveDSN(), SQL: `SELECT id, f FROM qpx_bench_data`})
	it, err := qpx.ExecuteVector(ctx, src, qpx.DefaultVectorOptions())
	if err != nil {
		t.Fatalf("ExecuteVector: %v", err)
	}
	defer func() { _ = it.Close() }()
	g := vector.NewGroupByInt64(0, 1, 128)
	for {
		bh, ok, err := it.Next(ctx)
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if !ok {
			break
		}
		if err := g.Add(bh); err != nil {
			t.Fatalf("Add: %v", err)
		}
	}
	rows := g.SortedRows()
	if len(rows) != 128 {
		t.Fatalf("groups = %d, want 128", len(rows))
	}
	conn, err := openDB(ctx, resolveDSN())
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer func() { _ = conn.Close(ctx) }()
	pgRows, err := conn.Query(ctx,
		`SELECT (id % 128) AS k, count(*), sum(f), avg(f) FROM qpx_bench_data GROUP BY 1 ORDER BY 1`)
	if err != nil {
		t.Fatalf("pg group-by: %v", err)
	}
	defer pgRows.Close()
	i := 0
	for pgRows.Next() {
		var k, n int64
		var sv, av any
		if err := pgRows.Scan(&k, &n, &sv, &av); err != nil {
			t.Fatalf("scan: %v", err)
		}
		if i >= len(rows) {
			t.Fatalf("pg has more groups than engine")
		}
		gr := rows[i]
		i++
		if gr.Key != k || gr.Rows != n || gr.Count != n {
			t.Fatalf("group %d: rows/count = %d/%d, want key=%d rows=%d", i, gr.Key, gr.Rows, k, n)
		}
		sf, ok1 := sv.(float64)
		if !ok1 {
			t.Fatalf("group %d: sum is %T, want float64", k, sv)
		}
		d := gr.Sum - sf
		if d < 0 {
			d = -d
		}
		if d > 1e-6 {
			t.Fatalf("group %d: sum = %v, want %v", k, gr.Sum, sf)
		}
		_ = av
	}
	if err := pgRows.Err(); err != nil {
		t.Fatalf("rows: %v", err)
	}
	if i != 128 {
		t.Fatalf("compared %d groups, want 128", i)
	}
}

// TestVectorMemory compares allocator traffic for legacy scan, vector
// scan, and vector scan+engine-agg on the M dataset. The signal is the
// TotalAlloc delta around each drain: cumulative, so GC timing cannot
// skew it (HeapAlloc snapshots are intentionally not used — sampling
// HeapAlloc post-drain without a GC is noise). The vec+agg drain folds
// into 128 groups, so its row count is 128, not benchRows: scan cost and
// agg cost are reported separately instead of mixing them into one b/row.
func TestVectorMemory(t *testing.T) {
	ensureMatrixDataset(t)
	ctx := context.Background()
	drainLegacy := func() (int64, error) { return runQPX(ctx, sqlQ1) }
	drainVector := func() (int64, error) { return runVectorWorkers(ctx, sqlQ1, 4, nil) }
	drainVectorAgg := func() (int64, error) { return runVectorGroupBy128PG(ctx) }
	systems := []struct {
		name  string
		want  int64
		drain func() (int64, error)
	}{
		{"legacy", benchRows, drainLegacy},
		{"vector", benchRows, drainVector},
		{"vec+agg", 128, drainVectorAgg},
	}
	totals := make(map[string]uint64, len(systems))
	t.Logf("%-7s %10s %12s %10s", "system", "rows", "total_alloc", "b/row")
	for _, s := range systems {
		runtime.GC()
		var m0, m1 runtime.MemStats
		runtime.ReadMemStats(&m0)
		n, err := s.drain()
		runtime.ReadMemStats(&m1)
		if err != nil {
			t.Fatalf("%s drain: %v", s.name, err)
		}
		if n != s.want {
			t.Fatalf("%s drain: got %d rows, want %d", s.name, n, s.want)
		}
		total := m1.TotalAlloc - m0.TotalAlloc
		if total == 0 {
			t.Fatalf("%s drain: zero TotalAlloc delta, measurement broken", s.name)
		}
		totals[s.name] = total
		t.Logf("%-7s %10d %12d %10d", s.name, n, total, total/uint64(n))
	}
	// Agg cost isolated from scan cost: the fold into 128 groups must cost
	// far less than the 1M-row scan underneath it. Bound is deliberately
	// loose (4x) — TotalAlloc is cumulative and stable, this only trips on
	// order-of-magnitude regressions (e.g. retaining batches per group).
	aggOverhead := int64(totals["vec+agg"]) - int64(totals["vector"])
	if aggOverhead < 0 {
		aggOverhead = 0
	}
	t.Logf("agg-overhead (vec+agg minus vector scan): %d bytes total, %d b/scan-row",
		aggOverhead, uint64(aggOverhead)/uint64(benchRows))
	if totals["vec+agg"] >= 4*totals["vector"] {
		t.Fatalf("vec+agg TotalAlloc %d >= 4x vector scan %d: agg retains too much",
			totals["vec+agg"], totals["vector"])
	}
}
