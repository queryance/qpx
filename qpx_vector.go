package qpx

import (
	"context"
	"fmt"

	"github.com/queryance/qpx/engine"
	"github.com/queryance/qpx/engine/vector"
)

// VectorOptions tunes a vector run. Zero values select sane defaults; use
// DefaultVectorOptions as a starting point.
type VectorOptions struct {
	// Workers is the number of batch-processing goroutines.
	Workers int
	// QueueSize bounds the internal batch queues.
	QueueSize int
	// BatchSize hints rows per batch to sources that support chunk
	// sizing (applied via engine.ChunkSizer).
	BatchSize int
	// Ops is the operator pipeline applied per batch in order.
	// Nil or empty means a straight source-to-batch stream.
	Ops []vector.Operator
}

// DefaultVectorOptions returns the recommended starting point.
func DefaultVectorOptions() VectorOptions {
	return VectorOptions{
		Workers:   engine.DefaultWorkers,
		QueueSize: engine.DefaultQueueSize,
		BatchSize: vector.DefaultBatchSize,
	}
}

// ExecuteVector is the default serving path for queries that need no
// legacy row ops: it opens src as a vector.Source and returns an Iterator
// over typed batches, processed by a bounded worker pool. Use Execute
// only when custom Operators must see legacy []any cells.
//
// Cancellation mirrors Execute: cancelling ctx surfaces a wrapped
// ctx.Err() from Next instead of a clean EOF.
func ExecuteVector(ctx context.Context, src vector.Source, opts VectorOptions) (*vector.Iterator, error) {
	if opts.Workers <= 0 {
		opts.Workers = engine.DefaultWorkers
	}
	if opts.QueueSize <= 0 {
		opts.QueueSize = engine.DefaultQueueSize
	}
	if opts.BatchSize <= 0 {
		opts.BatchSize = vector.DefaultBatchSize
	}
	if s, ok := src.(engine.ChunkSizer); ok {
		s.SetChunkSize(opts.BatchSize)
	}
	sched := vector.Scheduler{Workers: opts.Workers, QueueSize: opts.QueueSize}
	ctx, cancel := context.WithCancel(ctx)
	batches, errCh, err := sched.Run(ctx, src, opts.Ops)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("qpx: execute vector: %w", err)
	}
	return vector.NewIterator(batches, errCh, cancel), nil
}
