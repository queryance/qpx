package qpx_test

import (
	"context"
	"errors"
	"testing"

	"github.com/queryance/qpx"
	"github.com/queryance/qpx/engine"
)

func TestDefaultExecuteOptionsValues(t *testing.T) {
	opts := qpx.DefaultExecuteOptions()
	if opts.Workers != engine.DefaultWorkers || opts.Workers <= 0 {
		t.Fatalf("Workers = %d, want %d", opts.Workers, engine.DefaultWorkers)
	}
	if opts.QueueSize != engine.DefaultQueueSize || opts.QueueSize <= 0 {
		t.Fatalf("QueueSize = %d, want %d", opts.QueueSize, engine.DefaultQueueSize)
	}
	if opts.ChunkSize != engine.DefaultChunkSize || opts.ChunkSize <= 0 {
		t.Fatalf("ChunkSize = %d, want %d", opts.ChunkSize, engine.DefaultChunkSize)
	}
	if len(opts.Ops) != 0 {
		t.Fatalf("default Ops len = %d, want 0", len(opts.Ops))
	}
}

func TestExecuteZeroOptionsApplyDefaults(t *testing.T) {
	src := &fakeSource{chunks: []engine.Chunk{mustChunk(1, 2)}}
	ctx := context.Background()
	it, err := qpx.Execute(ctx, src, qpx.ExecuteOptions{})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	defer func() { _ = it.Close() }()
	if src.chunkSize != engine.DefaultChunkSize {
		t.Fatalf("zero ChunkSize not defaulted via ChunkSizer: got %d, want %d", src.chunkSize, engine.DefaultChunkSize)
	}
	var total int
	for {
		c, ok, err := it.Next(ctx)
		if err != nil {
			t.Fatalf("Next: %v", err)
		}
		if !ok {
			break
		}
		total += c.NumRows()
	}
	if total != 2 {
		t.Fatalf("total rows = %d, want 2", total)
	}
}

func TestExecuteNegativeOptionsApplyDefaults(t *testing.T) {
	src := &fakeSource{chunks: []engine.Chunk{mustChunk(9)}}
	opts := qpx.ExecuteOptions{Workers: -1, QueueSize: -5, ChunkSize: -10}
	it, err := qpx.Execute(context.Background(), src, opts)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	defer func() { _ = it.Close() }()
	if src.chunkSize != engine.DefaultChunkSize {
		t.Fatalf("negative ChunkSize not defaulted: got %d", src.chunkSize)
	}
	c, ok, err := it.Next(context.Background())
	if err != nil || !ok || c.NumRows() != 1 {
		t.Fatalf("Next = (%v, %v, %v), want 1 row", c, ok, err)
	}
}

func TestExecutePartialDefaults(t *testing.T) {
	// Only ChunkSize set: workers/queue still default, run succeeds.
	src := &fakeSource{chunks: []engine.Chunk{mustChunk(3)}}
	opts := qpx.ExecuteOptions{ChunkSize: 7}
	it, err := qpx.Execute(context.Background(), src, opts)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	defer func() { _ = it.Close() }()
	if src.chunkSize != 7 {
		t.Fatalf("ChunkSize = %d, want 7", src.chunkSize)
	}
	if _, ok, err := it.Next(context.Background()); err != nil || !ok {
		t.Fatalf("Next = (ok=%v, err=%v), want chunk", ok, err)
	}
}

// plainSource does not implement engine.ChunkSizer: Execute must skip the
// SetChunkSize assertion and still stream.
type plainSource struct {
	chunks []engine.Chunk
}

func (p *plainSource) Open(context.Context) (engine.Scanner, error) {
