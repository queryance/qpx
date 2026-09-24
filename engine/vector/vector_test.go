package vector_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/queryance/qpx/engine"
	"github.com/queryance/qpx/engine/vector"
)

var testSchema = engine.Schema{Fields: []engine.Field{
	{Name: "id", Type: engine.Int64},
	{Name: "f", Type: engine.Float64},
	{Name: "t", Type: engine.String},
	{Name: "ts", Type: engine.Time},
}}

func testBatch() *vector.Batch {
	b := vector.NewBatch(testSchema, 8)
	for i := 1; i <= 6; i++ {
		b.Columns[0].Ints = append(b.Columns[0].Ints, int64(i))
		b.Columns[1].Floats = append(b.Columns[1].Floats, float64(i)/10.0)
		b.Columns[2].Strings = append(b.Columns[2].Strings, string(rune('a'+i)))
		b.Columns[3].Times = append(b.Columns[3].Times, time.Unix(int64(i), 0))
		b.NumRows++
	}
	return b
}

func selectedInts(b *vector.Batch) []int64 {
	var out []int64
	col := &b.Columns[0]
	if b.Sel == nil {
		return append(out, col.Ints[:b.NumRows]...)
	}
	for _, s := range b.Sel {
		out = append(out, col.Ints[int(s)])
	}
	return out
}

func TestNewBatchAndReset(t *testing.T) {
	b := testBatch()
	if b.Len() != 6 {
		t.Fatalf("Len = %d, want 6", b.Len())
	}
	if err := b.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	b.Reset()
	if b.Len() != 0 || b.NumRows != 0 || b.Sel != nil {
		t.Fatalf("after Reset: Len=%d NumRows=%d Sel=%v", b.Len(), b.NumRows, b.Sel)
	}
	if err := b.Validate(); err != nil {
		t.Fatalf("Validate after Reset: %v", err)
	}
	// Backing arrays retained: capacity survives Reset.
	if cap(b.Columns[0].Ints) < 6 {
		t.Fatalf("Reset dropped backing array: cap=%d", cap(b.Columns[0].Ints))
	}
}

func TestValidateRejects(t *testing.T) {
	b := testBatch()
	b.Columns[0].Ints = b.Columns[0].Ints[:2]
	if err := b.Validate(); err == nil {
		t.Fatal("ragged batch must fail validation")
	}
	b = testBatch()
	b.Sel = []int32{99}
	if err := b.Validate(); err == nil {
		t.Fatal("out-of-range selection must fail validation")
	}
}

func TestFloatFilterDense(t *testing.T) {
	b := testBatch() // f = 0.1..0.6
	got, err := vector.NewFloat64Filter(1, vector.Gt, 0.35).Process(b)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if ids := selectedInts(got); len(ids) != 3 || ids[0] != 4 || ids[2] != 6 {
		t.Fatalf("unexpected survivors: %v", ids)
	}
	if got.Len() != 3 {
		t.Fatalf("Len = %d, want 3", got.Len())
	}
	if err := got.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
}

func TestFilterFullMatchKeepsDense(t *testing.T) {
	b := testBatch()
	got, err := vector.NewInt64Filter(0, vector.Ge, 1).Process(b)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if got.Sel != nil {
		t.Fatalf("full match must keep Sel nil, got %v", got.Sel)
	}
	if got.Len() != 6 {
		t.Fatalf("Len = %d, want 6", got.Len())
	}
}

func TestFilterFullMatchZeroAllocs(t *testing.T) {
	// The dense all-match path must not materialize a selection: it scans
	// for the first mismatch and allocates only once a drop is found. The
	// batch and filter are hoisted: a full match leaves the batch
	// untouched, so reuse across runs is safe and only Process is measured.
	b := testBatch()
	f := vector.NewInt64Filter(0, vector.Ge, 1)
	if n := testing.AllocsPerRun(20, func() {
		got, err := f.Process(b)
		if err != nil {
			t.Fatalf("Process: %v", err)
		}
		if got.Sel != nil {
			t.Fatalf("full match must keep Sel nil")
		}
	}); n != 0 {
		t.Fatalf("full-match filter allocs = %v, want 0", n)
	}
}

func TestFilterEmptyMatch(t *testing.T) {
	b := testBatch()
	got, err := vector.NewInt64Filter(0, vector.Gt, 100).Process(b)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if got.Sel == nil || len(got.Sel) != 0 {
		t.Fatalf("empty match must be empty non-nil Sel, got %v", got.Sel)
	}
	if got.Len() != 0 {
		t.Fatalf("Len = %d, want 0", got.Len())
	}
}

func TestFilterChainedCompacts(t *testing.T) {
	b := testBatch()
	f1 := vector.NewInt64Filter(0, vector.Gt, 1) // 2..6
	f2 := vector.NewInt64Filter(0, vector.Lt, 6) // 2..5
	mid, err := f1.Process(b)
	if err != nil {
		t.Fatalf("f1: %v", err)
	}
	got, err := f2.Process(mid)
	if err != nil {
		t.Fatalf("f2: %v", err)
	}
	if ids := selectedInts(got); len(ids) != 4 || ids[0] != 2 || ids[3] != 5 {
		t.Fatalf("unexpected survivors: %v", ids)
	}
}

func TestFilterNullsNeverMatch(t *testing.T) {
	b := testBatch()
	b.Columns[1].Floats[2] = 0 // would match, but mark NULL
	b.Columns[1].Nulls = make([]bool, 6)
	b.Columns[1].Nulls[2] = true
	b.Columns[1].HasNulls = true
	got, err := vector.NewFloat64Filter(1, vector.Gt, -1).Process(b)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	// Row id=3 (NULL f) must be dropped even though threshold is -1.
	for _, s := range got.Sel {
		if int(s) == 2 {
			t.Fatalf("NULL row survived filter: %v", got.Sel)
		}
	}
	if got.Len() != 5 {
		t.Fatalf("Len = %d, want 5", got.Len())
	}
}

func TestStringAndBoolFilters(t *testing.T) {
	b := testBatch()
	got, err := vector.NewStringFilter(2, vector.Eq, "b").Process(b)
	if err != nil {
		t.Fatalf("string filter: %v", err)
	}
	if ids := selectedInts(got); len(ids) != 1 || ids[0] != 1 {
		t.Fatalf("unexpected survivors: %v", ids)
	}
	if _, err := vector.NewStringFilter(2, vector.Gt, "b").Process(testBatch()); err == nil {
		t.Fatal("string Gt must fail (Eq/Ne only)")
	}

	bschema := engine.Schema{Fields: []engine.Field{{Name: "ok", Type: engine.Bool}}}
	bb := vector.NewBatch(bschema, 4)
	bb.Columns[0].Bools = append(bb.Columns[0].Bools, true, false, true)
	bb.NumRows = 3
	bgot, err := vector.NewBoolFilter(0, true).Process(bb)
	if err != nil {
		t.Fatalf("bool filter: %v", err)
	}
	if bgot.Len() != 2 {
		t.Fatalf("Len = %d, want 2", bgot.Len())
	}
}

func TestFilterTypeMismatch(t *testing.T) {
	if _, err := vector.NewInt64Filter(2, vector.Gt, 1).Process(testBatch()); err == nil {
		t.Fatal("int filter on string column must fail")
	}
	if _, err := vector.NewFloat64Filter(9, vector.Gt, 1).Process(testBatch()); err == nil {
		t.Fatal("out-of-range column must fail")
	}
	if _, err := vector.NewFloat64Filter(1, vector.Gt, 1).Process(nil); err == nil {
		t.Fatal("nil batch must fail")
	}
}

func TestProject(t *testing.T) {
	b := testBatch()
