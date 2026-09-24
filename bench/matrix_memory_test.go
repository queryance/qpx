package bench

import (
	"context"
	"runtime"
	"testing"
)

// TestMxMemory drains the full M scan per system and reports allocator
// traffic (TotalAlloc delta) plus process size (Sys) around the drain —
// the memory half of the matrix. GC runs first so the delta reflects the
// drain, not prior garbage. Always run alongside -benchmem benchmarks;
// this test covers peak-RSS shape that allocs/op alone hides.
func TestMxMemory(t *testing.T) {
	ensureMatrixDataset(t)
	ctx := context.Background()
	d, err := mxOpenDuckDB(ctx)
	if err != nil {
		t.Fatalf("duckdb setup: %v", err)
	}
	defer d.close()
	systems := []struct {
		name  string
		drain func() (int64, error)
	}{
		{"qpx", func() (int64, error) { return runQPX(ctx, sqlQ1) }},
		{"pgx", func() (int64, error) { return runPGX(ctx, sqlQ1) }},
		{"duckdb", func() (int64, error) { return d.runQuery(ctx, sqlQ1) }},
	}
	t.Logf("%-6s %10s %12s %10s %12s %12s %6s", "system", "rows", "total_alloc", "b/row", "sys", "heap", "gcs")
	for _, s := range systems {
		runtime.GC()
		var m0, m1 runtime.MemStats
		runtime.ReadMemStats(&m0)
		n, err := s.drain()
		runtime.ReadMemStats(&m1)
		if err != nil {
			t.Fatalf("%s drain: %v", s.name, err)
		}
		if n != benchRows {
			t.Fatalf("%s drain: got %d rows, want %d", s.name, n, benchRows)
		}
		total := m1.TotalAlloc - m0.TotalAlloc
		t.Logf("%-6s %10d %12d %10d %12d %12d %6d",
			s.name, n, total, total/uint64(n), m1.Sys, m1.HeapAlloc, m1.NumGC-m0.NumGC)
	}
}
