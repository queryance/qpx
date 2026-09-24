package engine_test

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/queryance/qpx/engine"
)

var testSchema = engine.Schema{Fields: []engine.Field{
	{Name: "id", Type: engine.Int64},
	{Name: "name", Type: engine.String},
}}

func makeChunk(ids ...int) engine.Chunk {
	cols := [][]any{make([]any, len(ids)), make([]any, len(ids))}
	for i, id := range ids {
		cols[0][i] = int64(id)
		cols[1][i] = fmt.Sprintf("n%d", id)
	}
	c, err := engine.NewChunk(testSchema, cols)
	if err != nil {
		panic(err)
	}
	return c
}

func mustChunk(schema engine.Schema, cols [][]any) engine.Chunk {
	c, err := engine.NewChunk(schema, cols)
	if err != nil {
		panic(err)
	}
	return c
}

func col0(c engine.Chunk) []int64 {
	out := make([]int64, c.NumRows())
	for i := range out {
		out[i] = c.Columns[0][i].(int64)
	}
	return out
}

// fakeSource serves canned chunks; errAfter injects a scan error after
// that many chunks (negative disables).
type fakeSource struct {
	schema   engine.Schema
	chunks   []engine.Chunk
	errAfter int
	openErr  error
}

func (f *fakeSource) Open(context.Context) (engine.Scanner, error) {
	if f.openErr != nil {
		return nil, f.openErr
	}
	return &fakeScanner{src: f}, nil
}

type fakeScanner struct {
	src *fakeSource
	pos int
	cur engine.Chunk
	err error
}

func (s *fakeScanner) Schema() engine.Schema { return s.src.schema }

func (s *fakeScanner) Next() bool {
	if s.err != nil {
		return false
	}
	if s.src.errAfter >= 0 && s.pos >= s.src.errAfter {
		s.err = errors.New("fake scan boom")
		return false
	}
	if s.pos >= len(s.src.chunks) {
		return false
	}
	s.cur = s.src.chunks[s.pos]
	s.pos++
	return true
}

func (s *fakeScanner) Chunk() engine.Chunk { return s.cur }
func (s *fakeScanner) Err() error          { return s.err }
func (s *fakeScanner) Close() error        { return nil }

func drain(t *testing.T, it *engine.Iterator) ([]engine.Chunk, error) {
	t.Helper()
	ctx := context.Background()
	var out []engine.Chunk
	for {
		c, ok, err := it.Next(ctx)
		if err != nil {
			return out, err
		}
		if !ok {
			return out, nil
		}
		out = append(out, c)
	}
}

func run(t *testing.T, src engine.Source, ops []engine.Operator) ([]engine.Chunk, error) {
	t.Helper()
	sched := engine.Scheduler{Workers: 4, QueueSize: 8}
	ctx := context.Background()
	chunks, errCh, err := sched.Run(ctx, src, ops)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	it := engine.NewIterator(chunks, errCh, nil)
	return drain(t, it)
}

func TestChunkNumRows(t *testing.T) {
	if (engine.Chunk{}).NumRows() != 0 {
		t.Fatal("empty chunk should have 0 rows")
	}
	if got := makeChunk(1, 2, 3).NumRows(); got != 3 {
		t.Fatalf("got %d rows, want 3", got)
	}
}

func TestNewChunkRejectsRagged(t *testing.T) {
	if _, err := engine.NewChunk(testSchema, [][]any{
		{int64(1), int64(2)},
		{"only-one"},
	}); err == nil {
		t.Fatal("unequal column lengths must fail")
	}
	if _, err := engine.NewChunk(testSchema, [][]any{{int64(1)}}); err == nil {
		t.Fatal("column count != field count must fail")
	}
	if _, err := engine.NewChunk(testSchema, [][]any{
		{int64(1)}, {"a"}, {"extra"},
	}); err == nil {
		t.Fatal("too many columns must fail")
	}
	// Valid rectangular chunks still pass, including empty.
	if _, err := engine.NewChunk(testSchema, [][]any{{}, {}}); err != nil {
		t.Fatalf("empty rectangular chunk must pass: %v", err)
	}
	if _, err := engine.NewChunk(engine.Schema{}, nil); err != nil {
		t.Fatalf("zero-column chunk must pass: %v", err)
	}
}

func TestOperatorsRejectRagged(t *testing.T) {
	ragged := engine.Chunk{
		Schema:  testSchema,
		Columns: [][]any{{int64(1), int64(2)}, {"only-one"}},
	}
	filter := engine.NewFilterOp(func(row []any) (bool, error) { return true, nil })
	if _, err := filter.Process(ragged); err == nil {
		t.Fatal("filter on ragged chunk must fail, not panic")
	}
	if _, err := engine.NewProjectOp(0).Process(ragged); err == nil {
		t.Fatal("project on ragged chunk must fail, not panic")
	}
	wrongCount := engine.Chunk{
		Schema:  testSchema,
		Columns: [][]any{{int64(1)}},
	}
	if _, err := filter.Process(wrongCount); err == nil {
		t.Fatal("filter on wrong column count must fail")
	}
	if _, err := engine.NewProjectOp(0).Process(wrongCount); err == nil {
		t.Fatal("project on wrong column count must fail")
	}
}

func TestFilterOp(t *testing.T) {
	op := engine.NewFilterOp(func(row []any) (bool, error) {
		return row[0].(int64)%2 == 0, nil
	})
	got, err := op.Process(makeChunk(1, 2, 3, 4))
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if ids := col0(got); len(ids) != 2 || ids[0] != 2 || ids[1] != 4 {
		t.Fatalf("unexpected rows: %v", ids)
	}
	if got.Schema.Fields[1].Name != "name" {
		t.Fatal("filter must preserve schema")
	}
	// Empty input stays empty.
	emptyChunk := mustChunk(testSchema, [][]any{{}, {}})
	empty, err := op.Process(emptyChunk)
	if err != nil || empty.NumRows() != 0 {
		t.Fatalf("empty filter: rows=%d err=%v", empty.NumRows(), err)
	}
	// Predicate errors propagate wrapped.
	wantErr := errors.New("nope")
	opErr := engine.NewFilterOp(func([]any) (bool, error) { return false, wantErr })
	if _, err := opErr.Process(makeChunk(1)); !errors.Is(err, wantErr) {
		t.Fatalf("want wrapped predicate error, got %v", err)
	}
}

func TestProjectOp(t *testing.T) {
	op := engine.NewProjectOp(1, 0)
	got, err := op.Process(makeChunk(7))
	if err != nil {
		t.Fatalf("Process: %v", err)
	}
	if len(got.Schema.Fields) != 2 || got.Schema.Fields[0].Name != "name" {
		t.Fatalf("unexpected schema: %+v", got.Schema.Fields)
	}
	if got.Columns[0][0].(string) != "n7" || got.Columns[1][0].(int64) != 7 {
		t.Fatal("columns not reordered")
	}
	if _, err := engine.NewProjectOp().Process(makeChunk(1)); err == nil {
		t.Fatal("empty projection must fail")
	}
	if _, err := engine.NewProjectOp(5).Process(makeChunk(1)); err == nil {
		t.Fatal("out-of-range index must fail")
	}
}

func TestSchedulerOrderedMerge(t *testing.T) {
	// One row per chunk; a delay op completes later chunks first, so
	// output order proves sequence-numbered reassembly.
	var chunks []engine.Chunk
	for i := 0; i < 64; i++ {
		chunks = append(chunks, makeChunk(i))
	}
	src := &fakeSource{schema: testSchema, chunks: chunks, errAfter: -1}
	var maxInFlight atomic.Int64
	var inFlight atomic.Int64
	delay := &funcOp{fn: func(c engine.Chunk) (engine.Chunk, error) {
		id := c.Columns[0][0].(int64)
		cur := inFlight.Add(1)
		for {
			m := maxInFlight.Load()
			if cur <= m || maxInFlight.CompareAndSwap(m, cur) {
				break
			}
		}
		defer inFlight.Add(-1)
		time.Sleep(time.Duration(63-id) * 100 * time.Microsecond)
		return c, nil
	}}
	got, err := run(t, src, []engine.Operator{delay})
	if err != nil {
		t.Fatalf("drain: %v", err)
	}
	if maxInFlight.Load() < 2 {
		t.Fatalf("expected concurrent processing, max in flight %d", maxInFlight.Load())
	}
	if len(got) != 64 {
		t.Fatalf("got %d chunks, want 64", len(got))
	}
	for i, c := range got {
		if id := c.Columns[0][0].(int64); id != int64(i) {
			t.Fatalf("chunk %d out of order: id=%d", i, id)
		}
	}
}

func TestSchedulerFilterProjectPipeline(t *testing.T) {
	src := &fakeSource{schema: testSchema, chunks: []engine.Chunk{makeChunk(1, 2, 3, 4)}, errAfter: -1}
	filter := engine.NewFilterOp(func(row []any) (bool, error) { return row[0].(int64) > 1, nil })
	got, err := run(t, src, []engine.Operator{filter, engine.NewProjectOp(0)})
	if err != nil {
		t.Fatalf("drain: %v", err)
	}
	if len(got) != 1 || len(got[0].Schema.Fields) != 1 {
		t.Fatalf("unexpected pipeline output: %+v", got)
	}
	if ids := col0(got[0]); len(ids) != 3 || ids[0] != 2 {
		t.Fatalf("unexpected rows: %v", ids)
	}
}

func TestSchedulerErrorPropagation(t *testing.T) {
	src := &fakeSource{schema: testSchema, chunks: []engine.Chunk{makeChunk(1), makeChunk(2)}, errAfter: -1}
	wantErr := errors.New("op boom")
	bad := &funcOp{fn: func(c engine.Chunk) (engine.Chunk, error) {
		if c.Columns[0][0].(int64) == 2 {
			return engine.Chunk{}, wantErr
		}
		return c, nil
	}}
	_, err := run(t, src, []engine.Operator{bad})
	if !errors.Is(err, wantErr) {
		t.Fatalf("want wrapped op error, got %v", err)
	}
}

func TestSchedulerScanError(t *testing.T) {
	src := &fakeSource{schema: testSchema, chunks: []engine.Chunk{makeChunk(1)}, errAfter: 1}
	_, err := run(t, src, nil)
	if err == nil || err.Error() == "" {
		t.Fatal("want scan error, got nil")
	}
}

func TestSchedulerOpenError(t *testing.T) {
	src := &fakeSource{openErr: errors.New("cannot open")}
	sched := engine.Scheduler{}
	if _, _, err := sched.Run(context.Background(), src, nil); err == nil {
		t.Fatal("want open error, got nil")
	}
}

func TestSchedulerCancellation(t *testing.T) {
	src := &fakeSource{schema: testSchema, chunks: []engine.Chunk{makeChunk(1)}, errAfter: -1}
	sched := engine.Scheduler{Workers: 2, QueueSize: 1}
	ctx, cancel := context.WithCancel(context.Background())
	chunks, errCh, err := sched.Run(ctx, src, []engine.Operator{&funcOp{fn: func(c engine.Chunk) (engine.Chunk, error) {
		time.Sleep(50 * time.Millisecond)
		return c, nil
	}}})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	cancel()
	timeout := time.After(5 * time.Second)
	for {
		select {
		case _, ok := <-chunks:
			if !ok {
				select {
				case <-errCh:
				case <-timeout:
					t.Fatal("errCh not closed after cancel")
				}
				return
			}
		case <-timeout:
			t.Fatal("chunk channel not closed after cancel: goroutine leak")
		}
	}
}

func TestSchedulerCancellationSurfacesError(t *testing.T) {
	var chunks []engine.Chunk
	for i := 0; i < 64; i++ {
		chunks = append(chunks, makeChunk(i))
	}
	src := &fakeSource{schema: testSchema, chunks: chunks, errAfter: -1}
	sched := engine.Scheduler{Workers: 2, QueueSize: 1}
	ctx, cancel := context.WithCancel(context.Background())
	block := &funcOp{fn: func(c engine.Chunk) (engine.Chunk, error) {
		time.Sleep(50 * time.Millisecond)
		return c, nil
	}}
	out, errCh, err := sched.Run(ctx, src, []engine.Operator{block})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	it := engine.NewIterator(out, errCh, cancel)
	// Cancel the run context externally; the drained Iterator must surface
	// a wrapped context.Canceled instead of a clean EOF.
	cancel()
	var term error
	for {
		_, ok, err := it.Next(context.Background())
		if err != nil {
			term = err
			break
		}
		if !ok {
			break
		}
	}
	if term == nil {
		// The error may arrive as the terminal Next error after chunks drain;
		// drain once more to be sure we observed it.
		_, _, term = it.Next(context.Background())
	}
	if !errors.Is(term, context.Canceled) {
		t.Fatalf("cancelled run must surface context.Canceled, got %v", term)
	}
}

func TestIteratorCloseIdempotent(t *testing.T) {
	src := &fakeSource{schema: testSchema, chunks: []engine.Chunk{makeChunk(1)}, errAfter: -1}
	sched := engine.Scheduler{}
	chunks, errCh, err := sched.Run(context.Background(), src, nil)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	it := engine.NewIterator(chunks, errCh, nil)
	if err := it.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := it.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if _, ok, _ := it.Next(context.Background()); ok {
		t.Fatal("Next after Close should report done")
	}
}

func TestIteratorCloseDoesNotStarve(t *testing.T) {
	chunks := make(chan engine.Chunk)
	errCh := make(chan error)
	it := engine.NewIterator(chunks, errCh, func() {})
	nextDone := make(chan struct{})
	go func() {
		defer close(nextDone)
		_, _, _ = it.Next(context.Background())
	}()
	// Let Next block in select (outside the lock after the fix).
	time.Sleep(50 * time.Millisecond)
	closeDone := make(chan struct{})
	go func() {
		defer close(closeDone)
		_ = it.Close()
	}()
	select {
	case <-closeDone:
	case <-time.After(2 * time.Second):
		t.Fatal("Close starved by blocked Next: holds mu across select")
	}
	// Unblock the waiting Next and let it finish.
	close(chunks)
	close(errCh)
	select {
	case <-nextDone:
	case <-time.After(2 * time.Second):
		t.Fatal("blocked Next did not finish after channels closed")
	}
}

type funcOp struct {
	fn func(engine.Chunk) (engine.Chunk, error)
}

func (o *funcOp) Process(c engine.Chunk) (engine.Chunk, error) { return o.fn(c) }
