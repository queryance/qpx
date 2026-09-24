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
