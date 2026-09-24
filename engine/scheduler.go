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
		if !errClosed {
			errClosed = true
			close(errCh)
		}
	}
	readerDone := make(chan struct{})

	var wg sync.WaitGroup
	for range nw {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-runCtx.Done():
					return
				case j, ok := <-jobs:
					if !ok {
						return
					}
					c, werr := applyOps(j.chunk, ops)
					if werr != nil {
						werr = fmt.Errorf("scheduler: worker: %w", werr)
					}
					select {
					case <-runCtx.Done():
						return
					case results <- seqResult{seq: j.seq, chunk: c, err: werr}:
					}
				}
			}
		}()
	}

	go runReader(runCtx, cancel, scanner, jobs, sendErr, readerDone)

	// Closer: no more results once workers drain.
	go func() {
		wg.Wait()
		close(results)
	}()

	go reassemble(ctx, runCtx, cancel, results, out, sendErr, closeErrCh, readerDone)

	return out, errCh, nil
}

// runStraight streams scanner chunks directly to out in source order.
// It owns both channels, so sends and closes need no synchronization.
// parent reports external cancellation; runCtx drives teardown.
func runStraight(parent, runCtx context.Context, cancel context.CancelFunc, scanner Scanner, out chan<- Chunk, errCh chan<- error) {
	defer func() {
		if cerr := scanner.Close(); cerr != nil {
			select {
			case errCh <- fmt.Errorf("scheduler: close source: %w", cerr):
			default:
			}
		}
		if parent.Err() != nil {
			select {
			case errCh <- fmt.Errorf("scheduler: %w", parent.Err()):
			default:
			}
		}
		cancel()
		close(out)
		close(errCh)
	}()
	for scanner.Next() {
		select {
		case <-runCtx.Done():
			return
		case out <- scanner.Chunk():
		}
	}
	if serr := scanner.Err(); serr != nil {
		select {
		case errCh <- fmt.Errorf("scheduler: scan: %w", serr):
		default:
		}
		cancel()
	}
}

// runReader packs sequence numbers onto queued chunks. Teardown order:
// close(jobs) first so a hung scanner.Close cannot wedge workers, then
// close the scanner and report its error, then close readerDone so the
// reassembler knows no more sends are coming.
func runReader(ctx context.Context, cancel context.CancelFunc, scanner Scanner, jobs chan<- seqChunk, sendErr func(error), readerDone chan struct{}) {
	defer close(readerDone)
	defer func() {
		close(jobs)
		if cerr := scanner.Close(); cerr != nil {
			sendErr(fmt.Errorf("scheduler: close source: %w", cerr))
		}
	}()
	seq := 0
	for scanner.Next() {
		j := seqChunk{seq: seq, chunk: scanner.Chunk()}
		seq++
		select {
		case <-ctx.Done():
			return
		case jobs <- j:
		}
	}
	if serr := scanner.Err(); serr != nil {
		sendErr(fmt.Errorf("scheduler: scan: %w", serr))
		cancel()
	}
}
