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
