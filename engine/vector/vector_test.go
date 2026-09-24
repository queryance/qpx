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
	got, err := vector.NewProject(3, 0).Process(b)
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if len(got.Schema.Fields) != 2 || got.Schema.Fields[0].Name != "ts" {
		t.Fatalf("unexpected schema: %+v", got.Schema.Fields)
	}
	if got.Columns[1].Ints[0] != 1 {
		t.Fatal("projected column data mismatch (must alias input)")
	}
	if _, err := vector.NewProject().Process(testBatch()); err == nil {
		t.Fatal("empty projection must fail")
	}
	if _, err := vector.NewProject(7).Process(testBatch()); err == nil {
		t.Fatal("out-of-range projection must fail")
	}
	// Selection survives projection.
	filt, _ := vector.NewInt64Filter(0, vector.Gt, 4).Process(testBatch())
	pgot, err := vector.NewProject(0).Process(filt)
	if err != nil {
		t.Fatalf("project after filter: %v", err)
	}
	if ids := selectedInts(pgot); len(ids) != 2 || ids[0] != 5 {
		t.Fatalf("unexpected survivors: %v", ids)
	}
}

func TestFusedEqualsSequential(t *testing.T) {
	runSeq := func() []int64 {
		b := testBatch()
		f, err := vector.NewFloat64Filter(1, vector.Gt, 0.25).Process(b)
		if err != nil {
			t.Fatalf("seq filter: %v", err)
		}
		p, err := vector.NewProject(0, 1).Process(f)
		if err != nil {
			t.Fatalf("seq project: %v", err)
		}
		return selectedInts(p)
	}
	b := testBatch()
	fused, err := vector.NewFilterFloat64GTProject(1, 0.25, 0, 1).Process(b)
	if err != nil {
		t.Fatalf("fused: %v", err)
	}
	seq, fus := runSeq(), selectedInts(fused)
	if len(seq) != len(fus) {
		t.Fatalf("seq=%v fused=%v", seq, fus)
	}
	for i := range seq {
		if seq[i] != fus[i] {
			t.Fatalf("seq=%v fused=%v", seq, fus)
		}
	}
	if len(fused.Schema.Fields) != 2 {
		t.Fatalf("fused schema fields = %d, want 2", len(fused.Schema.Fields))
	}
	if _, err := (&vector.FilterProject{Indices: []int{0}}).Process(testBatch()); err == nil {
		t.Fatal("fused with nil filter must fail")
	}
}

func TestChunkRoundTrip(t *testing.T) {
	b := testBatch()
	// NULL in every type.
	b.Columns[0].Ints[1] = 0
	b.Columns[0].Nulls = make([]bool, 6)
	b.Columns[0].Nulls[1] = true
	b.Columns[0].HasNulls = true
	c, err := b.ToChunk()
	if err != nil {
		t.Fatalf("ToChunk: %v", err)
	}
	if c.NumRows() != 6 {
		t.Fatalf("chunk rows = %d, want 6", c.NumRows())
	}
	if c.Columns[0][1] != nil {
		t.Fatalf("NULL did not survive ToChunk: %v", c.Columns[0][1])
	}
	back, err := vector.FromChunk(c)
	if err != nil {
		t.Fatalf("FromChunk: %v", err)
	}
	if err := back.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if !back.Columns[0].IsNull(1) {
		t.Fatal("NULL did not survive round trip")
	}
	if back.Columns[0].Ints[0] != 1 || back.Columns[1].Floats[5] != 0.6 ||
		back.Columns[2].Strings[0] != "b" || !back.Columns[3].Times[0].Equal(time.Unix(1, 0)) {
		t.Fatal("values did not survive round trip")
	}
	// Selection honored.
	filt, _ := vector.NewInt64Filter(0, vector.Gt, 4).Process(testBatch())
	fc, err := filt.ToChunk()
	if err != nil {
		t.Fatalf("ToChunk selected: %v", err)
	}
	if fc.NumRows() != 2 || fc.Columns[0][0].(int64) != 5 {
		t.Fatalf("selected chunk wrong: %v", fc.Columns[0])
	}
}

func TestFromChunkRejects(t *testing.T) {
	schema := engine.Schema{Fields: []engine.Field{{Name: "id", Type: engine.Int64}}}
	c, err := engine.NewChunk(schema, [][]any{{"not-an-int"}})
	if err != nil {
		t.Fatalf("NewChunk: %v", err)
	}
	if _, err := vector.FromChunk(c); err == nil {
		t.Fatal("mistyped cell must fail, not coerce")
	}
	bad := engine.Chunk{Schema: schema, Columns: [][]any{{int64(1)}, {"x"}}}
	if _, err := vector.FromChunk(bad); err == nil {
		t.Fatal("ragged chunk must fail")
	}
}

func TestSchedulerStraightAndOps(t *testing.T) {
	src := &fakeBatchSource{schema: testSchema, batches: []*vector.Batch{testBatch(), testBatch()}}
	sched := vector.Scheduler{Workers: 2, QueueSize: 4}
	ctx := context.Background()
	chunks, errCh, err := sched.Run(ctx, src, nil)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	it := vector.NewIterator(chunks, errCh, nil)
	total := 0
	for {
		b, ok, err := it.Next(ctx)
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if !ok {
			break
		}
		total += b.Len()
	}
	if total != 12 {
		t.Fatalf("total = %d, want 12", total)
	}

	// Ops path: filter + project, order preserved across batches.
	src2 := &fakeBatchSource{schema: testSchema, batches: []*vector.Batch{testBatch(), testBatch()}}
	chunks2, errCh2, err := sched.Run(ctx, src2, []vector.Operator{
		vector.NewInt64Filter(0, vector.Gt, 4),
		vector.NewProject(0),
	})
	if err != nil {
		t.Fatalf("Run ops: %v", err)
	}
	it2 := vector.NewIterator(chunks2, errCh2, nil)
	var ids []int64
	for {
		b, ok, err := it2.Next(ctx)
		if err != nil {
			t.Fatalf("Next ops: %v", err)
		}
		if !ok {
			break
		}
		if len(b.Schema.Fields) != 1 {
			t.Fatalf("projected fields = %d, want 1", len(b.Schema.Fields))
		}
		ids = append(ids, selectedInts(b)...)
	}
	if len(ids) != 4 || ids[0] != 5 || ids[1] != 6 || ids[2] != 5 || ids[3] != 6 {
		t.Fatalf("ordered survivors wrong: %v", ids)
	}
}

func TestSchedulerOpError(t *testing.T) {
	src := &fakeBatchSource{schema: testSchema, batches: []*vector.Batch{testBatch()}}
	sched := vector.Scheduler{}
	wantErr := errors.New("boom")
	chunks, errCh, err := sched.Run(context.Background(), src, []vector.Operator{&failOp{err: wantErr}})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	it := vector.NewIterator(chunks, errCh, nil)
	for {
		_, ok, err := it.Next(context.Background())
		if err != nil {
			if !errors.Is(err, wantErr) {
				t.Fatalf("want wrapped op error, got %v", err)
			}
			return
		}
		if !ok {
			t.Fatal("want op error, got clean EOF")
		}
	}
}

type failOp struct{ err error }

func (o *failOp) Process(b *vector.Batch) (*vector.Batch, error) { return nil, o.err }

type fakeBatchSource struct {
	schema  engine.Schema
	batches []*vector.Batch
}

func (f *fakeBatchSource) OpenBatch(ctx context.Context) (vector.Scanner, error) {
	return &fakeBatchScanner{src: f}, nil
}

type fakeBatchScanner struct {
	src *fakeBatchSource
	pos int
}

func (s *fakeBatchScanner) Schema() engine.Schema { return s.src.schema }
func (s *fakeBatchScanner) Next() bool {
	if s.pos >= len(s.src.batches) {
		return false
	}
	s.pos++
	return true
}
func (s *fakeBatchScanner) Batch() *vector.Batch { return s.src.batches[s.pos-1] }
func (s *fakeBatchScanner) Err() error           { return nil }
func (s *fakeBatchScanner) Close() error         { return nil }
