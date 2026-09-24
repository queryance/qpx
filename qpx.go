// Package qpx is the entry point for running queries through the columnar
// engine. Postgres is the first backend; the same engine will later serve
// other sources such as JSON and Parquet.
package qpx

import (
	"context"
	"fmt"

	"github.com/queryance/qpx/engine"
)

// ExecuteOptions tunes a run. Zero values select sane defaults; use
// DefaultExecuteOptions as a starting point.
type ExecuteOptions struct {
	// Workers is the number of chunk-processing goroutines.
	Workers int
	// QueueSize bounds the internal chunk queues.
	QueueSize int
	// ChunkSize hints row counts per chunk to sources that support
	// chunk sizing (applied via engine.ChunkSizer).
	ChunkSize int
	// Ops is the operator pipeline applied per chunk in order.
	// Nil or empty means a straight source-to-chunk stream.
	Ops []engine.Operator
}

// DefaultExecuteOptions returns the recommended starting point.
func DefaultExecuteOptions() ExecuteOptions {
	return ExecuteOptions{
		Workers:   engine.DefaultWorkers,
		QueueSize: engine.DefaultQueueSize,
		ChunkSize: engine.DefaultChunkSize,
	}
}

// Execute opens src and returns an Iterator over its chunks, processed by a
// bounded worker pool. Operators from opts.Ops run inside the engine's
// scheduler; pass no ops for a straight source-to-chunk stream.
// The caller must Close the Iterator when done.
//
// Cancellation: cancelling the ctx passed to Execute (or to the Iterator's
// Next) surfaces a wrapped ctx.Err() from Next instead of a clean EOF;
// check it with errors.Is against context.Canceled or
// context.DeadlineExceeded. Iterator.Close hides the run's terminal error.
func Execute(ctx context.Context, src engine.Source, opts ExecuteOptions) (*engine.Iterator, error) {
	if opts.Workers <= 0 {
		opts.Workers = engine.DefaultWorkers
	}
	if opts.QueueSize <= 0 {
		opts.QueueSize = engine.DefaultQueueSize
	}
	if opts.ChunkSize <= 0 {
		opts.ChunkSize = engine.DefaultChunkSize
	}
	if s, ok := src.(engine.ChunkSizer); ok {
		s.SetChunkSize(opts.ChunkSize)
	}
	sched := engine.Scheduler{Workers: opts.Workers, QueueSize: opts.QueueSize}
	ctx, cancel := context.WithCancel(ctx)
	chunks, errCh, err := sched.Run(ctx, src, opts.Ops)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("qpx: execute: %w", err)
	}
	return engine.NewIterator(chunks, errCh, cancel), nil
}
