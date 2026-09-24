// Micro-benchmarks for the ht.go tables, one per structure, plus the
// global fold. These isolate table cost from batch decoding: tight loops
// over synthetic key streams at the bench shapes (128 / 100K groups,
// 1K strings). Facade-level before/after numbers live in
// bench/vector_agg_bench_test.go; the README comparison cites those.
package vector

import (
	"strconv"
	"testing"
)

// benchKeys returns n int64 keys cycling over groups distinct values.
func benchKeys(n, groups int64) []int64 {
	out := make([]int64, n)
	for i := range out {
		out[i] = int64(i) % groups
	}
	return out
}

func BenchmarkDenseAdd128(b *testing.B) {
	keys := benchKeys(4096, 128)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t := newDenseIntAgg(128)
		for _, k := range keys {
			t.addIdx(k, k, float64(k)/10.0, true)
		}
		if t.used != 128 {
			b.Fatalf("groups = %d, want 128", t.used)
		}
	}
	b.ReportMetric(float64(4096*b.N)/b.Elapsed().Seconds(), "rows/s")
}

func BenchmarkDenseAdd100K(b *testing.B) {
	keys := benchKeys(100000, 100000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t := newDenseIntAgg(100000)
		for _, k := range keys {
			t.addIdx(k, k, float64(k)/10.0, true)
		}
		if t.used != 100000 {
			b.Fatalf("groups = %d, want 100000", t.used)
		}
	}
	b.ReportMetric(float64(100000*b.N)/b.Elapsed().Seconds(), "rows/s")
}

func BenchmarkOAAdd128(b *testing.B) {
	keys := benchKeys(4096, 128)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t := newInt64HashAgg(128)
		for _, k := range keys {
			t.add(k, float64(k)/10.0, true)
		}
		if t.used != 128 {
			b.Fatalf("groups = %d, want 128", t.used)
		}
	}
	b.ReportMetric(float64(4096*b.N)/b.Elapsed().Seconds(), "rows/s")
}

// BenchmarkOAAdd100K exercises the growth path: 100K distinct keys from
// a 1024-slot start (7 doublings) plus steady-state probing at 67% load.
func BenchmarkOAAdd100K(b *testing.B) {
	keys := benchKeys(200000, 100000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t := newInt64HashAgg(1024)
		for _, k := range keys {
			t.add(k, float64(k)/10.0, true)
		}
		if t.used != 100000 {
			b.Fatalf("groups = %d, want 100000", t.used)
		}
	}
	b.ReportMetric(float64(200000*b.N)/b.Elapsed().Seconds(), "rows/s")
}

func BenchmarkDictAdd1K(b *testing.B) {
	strs := make([]string, 4096)
	for i := range strs {
		strs[i] = "txt_" + strconv.Itoa(i%1000)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t := newStrDictAgg(1024)
		for _, s := range strs {
			if id, ok := t.ids[s]; ok {
				p := &t.pay[id]
				p.rows++
				p.count++
				p.sum += 1.0
			} else {
				t.addCold(s, 1.0, true)
			}
		}
		if len(t.keys) != 1000 {
			b.Fatalf("groups = %d, want 1000", len(t.keys))
		}
	}
	b.ReportMetric(float64(4096*b.N)/b.Elapsed().Seconds(), "rows/s")
}

func BenchmarkMergeOA8x100K(b *testing.B) {
	var locals []*int64HashAgg
	for w := 0; w < 8; w++ {
		t := newInt64HashAgg(1024)
		for k := int64(0); k < 100000; k++ {
			t.foldPayload(k, 1, 1, float64(k))
		}
		locals = append(locals, t)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := newInt64HashAgg(1024)
		mergeOA(out, locals, 8)
		if out.used != 100000 {
			b.Fatalf("groups = %d, want 100000", out.used)
		}
	}
}

func BenchmarkMergeDense8x100K(b *testing.B) {
	var locals []*denseIntAgg
	for w := 0; w < 8; w++ {
		t := newDenseIntAgg(100000)
		for k := int64(0); k < 100000; k++ {
			t.foldIdx(k, k, 1, 1, float64(k))
		}
		locals = append(locals, t)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := newDenseIntAgg(100000)
		mergeDense(out, locals, 8)
		if out.used != 100000 {
			b.Fatalf("groups = %d, want 100000", out.used)
		}
	}
}

func BenchmarkGlobalAdd(b *testing.B) {
	vals := make([]float64, 4096)
	for i := range vals {
		vals[i] = float64(i) / 100.0
	}
	batch := NewBatch(htSchema, 4096)
	for i := range vals {
		batch.Columns[0].AppendInt(int64(i))
		batch.Columns[1].AppendFloat(vals[i])
		batch.Columns[2].AppendString("a")
		batch.NumRows++
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a := NewGlobalFloatAgg(1)
		if err := a.Add(batch); err != nil {
			b.Fatal(err)
		}
		if a.Result().Count != 4096 {
			b.Fatalf("count = %d, want 4096", a.Result().Count)
		}
	}
	b.ReportMetric(float64(4096*b.N)/b.Elapsed().Seconds(), "rows/s")
}
