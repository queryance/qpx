package bench

import (
	"sync"
	"testing"

	"github.com/queryance/qpx/engine/vector"
)

// Extended group-by shapes for the ht.go tables: Sel-filtered,
// count-only, and raw-key (Mod=0, open-addressing at scale) folds, plus
// the DuckDB counterparts. The mod-128 / mod-100K / string-1K residents
// live in vector_agg_bench_test.go; the README comparison cites all of
// them together.

// selMemBatches mirrors memBatches but keeps every 2nd row via Sel
// (500K selected of 1M). Built once, read-only afterwards; unlike
// Filter.Process outputs these are never compacted in place.
var (
	selMemOnce    sync.Once
	selMemBatches []*vector.Batch
)

func selBatches() []*vector.Batch {
	selMemOnce.Do(func() {
		for _, b := range memBatches() {
			sel := make([]int32, 0, b.NumRows/2)
			for r := int32(0); r < int32(b.NumRows); r += 2 {
				sel = append(sel, r)
			}
			// Share backing arrays read-only (agg never mutates batches);
			// only Sel is new per batch.
			nb := *b
			nb.Sel = sel
			selMemBatches = append(selMemBatches, &nb)
		}
	})
	return selMemBatches
}

func BenchmarkVectorGroupBy128SelMem(b *testing.B) {
	batches := selBatches()
	var total int64
	for _, bh := range batches {
		total += int64(len(bh.Sel))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g := vector.NewGroupByInt64(0, 1, 128)
		for _, bh := range batches {
			if err := g.Add(bh); err != nil {
				b.Fatal(err)
			}
		}
		sinkGroups = g.Len()
	}
	sec := b.Elapsed().Seconds()
	b.ReportMetric(float64(int64(b.N)*total)/sec, "scan_rows/s")
}

func BenchmarkVectorGroupBy128CountOnlyMem(b *testing.B) {
	batches := memBatches()
	driveMem(b, func() error {
		g := vector.NewGroupByInt64(0, -1, 128)
		for _, bh := range batches {
			if err := g.Add(bh); err != nil {
				return err
			}
		}
		sinkGroups = g.Len()
		return nil
	})
}

// BenchmarkVectorGroupByRaw1MMem folds raw ids (1M distinct groups) with
// Mod=0 — the open-addressing table at scale, including growth from a
// 1024-slot start. No DuckDB perfect-hash equivalent exists (range ≫
// threshold); compare against BenchmarkDuckDBGroupByRaw1MMem below.
func BenchmarkVectorGroupByRaw1MMem(b *testing.B) {
	batches := memBatches()
	driveMem(b, func() error {
		g := vector.NewGroupByInt64(0, 1, 0)
		for _, bh := range batches {
			if err := g.Add(bh); err != nil {
				return err
			}
		}
		sinkGroups = g.Len()
		if g.Len() != 1_000_000 {
			b.Fatalf("groups = %d, want 1000000", g.Len())
		}
		return nil
	})
}

func BenchmarkDuckDBGroupByRaw1MMem(b *testing.B) {
	driveDuckDBMem(b, `SELECT id, count(*), avg(f) FROM m GROUP BY 1`)
}
