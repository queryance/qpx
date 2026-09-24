package qpx_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/queryance/qpx"
	"github.com/queryance/qpx/engine"
)

var testSchema = engine.Schema{Fields: []engine.Field{
	{Name: "id", Type: engine.Int64},
	{Name: "name", Type: engine.String},
}}

func mustChunk(ids ...int) engine.Chunk {
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

type fakeSource struct {
	chunks    []engine.Chunk
	chunkSize int
}

func (f *fakeSource) SetChunkSize(n int) { f.chunkSize = n }

func (f *fakeSource) Open(context.Context) (engine.Scanner, error) {
	return &fakeScanner{src: f}, nil
}

type fakeScanner struct {
	src *fakeSource
	pos int
	cur engine.Chunk
}

func (s *fakeScanner) Schema() engine.Schema { return testSchema }
func (s *fakeScanner) Next() bool {
	if s.pos >= len(s.src.chunks) {
		return false
	}
	s.cur = s.src.chunks[s.pos]
	s.pos++
	return true
}
func (s *fakeScanner) Chunk() engine.Chunk { return s.cur }
func (s *fakeScanner) Err() error          { return nil }
func (s *fakeScanner) Close() error {
	return nil
}

func TestExecuteAppliesOpsAndChunkSize(t *testing.T) {
	src := &fakeSource{chunks: []engine.Chunk{mustChunk(1, 2, 3, 4)}}
	opts := qpx.DefaultExecuteOptions()
	opts.ChunkSize = 2
	opts.Ops = []engine.Operator{
		engine.NewFilterOp(func(row []any) (bool, error) { return row[0].(int64) > 1, nil }),
		engine.NewProjectOp(0),
	}
	ctx := context.Background()
	it, err := qpx.Execute(ctx, src, opts)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	defer func() { _ = it.Close() }()
	if src.chunkSize != 2 {
		t.Fatalf("ChunkSize not applied via engine.ChunkSizer: got %d, want 2", src.chunkSize)
	}
	var ids []int64
	for {
		c, ok, err := it.Next(ctx)
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if !ok {
			break
		}
		if len(c.Schema.Fields) != 1 {
			t.Fatalf("projected schema fields = %d, want 1", len(c.Schema.Fields))
		}
		for i := 0; i < c.NumRows(); i++ {
			ids = append(ids, c.Columns[0][i].(int64))
		}
	}
	if len(ids) != 3 || ids[0] != 2 || ids[1] != 3 || ids[2] != 4 {
		t.Fatalf("Ops not applied via Execute: got %v, want [2 3 4]", ids)
	}
}
