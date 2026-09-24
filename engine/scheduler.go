package engine

import (
	"context"
	"fmt"
	"sync"
)

// DefaultWorkers is the Scheduler worker count used when none is given.
const DefaultWorkers = 4

// DefaultQueueSize is the Scheduler queue depth used when none is given.
const DefaultQueueSize = 16

// Scheduler runs one Source through a chain of Operators on a bounded
// worker pool. Chunks are processed concurrently but reassembled in source
// order via sequence numbers before being emitted.
//
// When ops is empty the worker pool is bypassed entirely: chunks stream
// straight from the scanner to the output in source order, so Workers and
// QueueSize only tune runs that actually process chunks.
type Scheduler struct {
	Workers   int
	QueueSize int
}

func (s Scheduler) workers() int {
	if s.Workers <= 0 {
		return DefaultWorkers
	}
	return s.Workers
}

func (s Scheduler) queueSize() int {
	if s.QueueSize <= 0 {
		return DefaultQueueSize
	}
	return s.QueueSize
}

type seqChunk struct {
	seq   int
	chunk Chunk
}

type seqResult struct {
	seq   int
	chunk Chunk
	err   error
}

func applyOps(c Chunk, ops []Operator) (Chunk, error) {
	var err error
	for _, op := range ops {
		c, err = op.Process(c)
		if err != nil {
			return Chunk{}, err
		}
	}
	return c, nil
}

// Run opens src and streams its chunks through ops, returning an ordered
// chunk channel and an error channel. The error channel carries at most one
// error (nil means clean end of stream); both channels are closed when the
// run finishes. Cancelling ctx stops the run and releases all goroutines;
// the caller must drain the chunk channel (or cancel) for workers to exit.
//
// Error-channel ownership: the reader (scan/close errors) and the
// reassembler (worker/cancellation errors) are concurrent senders, so every
// send goes through sendErr (mutex plus closed flag) and errCh is closed
// only after the reader signals readerDone. Sends happen-before the close,
// so send-on-closed-channel is impossible and a close error is never lost
// to an early close.
//
// Cancellation semantics: external cancellation of ctx is recorded into the
// error channel as a wrapped ctx.Err(), so a drained Iterator surfaces it
// (check with errors.Is against context.Canceled or
// context.DeadlineExceeded) instead of reporting a clean EOF.
func (s Scheduler) Run(ctx context.Context, src Source, ops []Operator) (<-chan Chunk, <-chan error, error) {
	nw, qs := s.workers(), s.queueSize()
	// runCtx drives internal teardown; parent reports external
	// cancellation (our own cancel must not look like one).
	runCtx, cancel := context.WithCancel(ctx)

	scanner, err := src.Open(runCtx)
	if err != nil {
		cancel()
		return nil, nil, fmt.Errorf("scheduler: open source: %w", err)
	}

	out := make(chan Chunk, qs)
	errCh := make(chan error, 1)

	// Straight stream: no operators, no pool, no reassembly. One
	// goroutine owns both channels, so no coordination is needed.
	if len(ops) == 0 {
		go runStraight(ctx, runCtx, cancel, scanner, out, errCh)
		return out, errCh, nil
	}

	jobs := make(chan seqChunk, qs)
	results := make(chan seqResult, qs)

	var errMu sync.Mutex
	errClosed := false
	sendErr := func(err error) {
		errMu.Lock()
		defer errMu.Unlock()
		if errClosed {
			return
		}
		select {
		case errCh <- err:
		default:
		}
	}
	closeErrCh := func() {
		errMu.Lock()
		defer errMu.Unlock()
