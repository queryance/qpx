package vector

import (
	"context"
	"fmt"
	"sync"

	"github.com/queryance/qpx/engine"
)

// Source produces a stream of batches. It mirrors engine.Source: opening
// runs the underlying query exactly once; parallelism happens downstream.
type Source interface {
	OpenBatch(ctx context.Context) (Scanner, error)
}

// Scanner streams batches from an opened Source. Schema is valid
// immediately after OpenBatch. Batch returns the current batch, owned by
// the caller until the next Next call — the same ownership rule as
// engine.Scanner, except the unit is a *Batch.
type Scanner interface {
	Schema() engine.Schema
	Next() bool
	Batch() *Batch
	Err() error
	Close() error
}

// Scheduler runs one Source through a chain of Operators on a bounded
// worker pool. Batches are processed concurrently but reassembled in
// source order. When ops is empty the pool is bypassed: batches stream
// straight from the scanner in source order.
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

// DefaultWorkers and DefaultQueueSize track the engine scheduler so
// vector and legacy runs tune identically.
const (
	DefaultWorkers   = 4
	DefaultQueueSize = 16
)

type seqBatch struct {
	seq   int
	batch *Batch
}

type seqResult struct {
	seq   int
	batch *Batch
	err   error
}

func applyOps(b *Batch, ops []Operator) (*Batch, error) {
	var err error
	for _, op := range ops {
		b, err = op.Process(b)
		if err != nil {
			return nil, err
		}
	}
	return b, nil
}

// Run opens src and streams its batches through ops, returning an ordered
// batch channel and an error channel with the same ownership and
// cancellation semantics as engine.Scheduler.Run: errCh carries at most
// one error, both channels close at run end, cancelling ctx stops the run
// and surfaces a wrapped ctx.Err() instead of a clean EOF.
func (s Scheduler) Run(ctx context.Context, src Source, ops []Operator) (<-chan *Batch, <-chan error, error) {
	nw, qs := s.workers(), s.queueSize()
	runCtx, cancel := context.WithCancel(ctx)

	scanner, err := src.OpenBatch(runCtx)
	if err != nil {
		cancel()
		return nil, nil, fmt.Errorf("vector: scheduler: open source: %w", err)
	}

	out := make(chan *Batch, qs)
	errCh := make(chan error, 1)

	if len(ops) == 0 {
		go runStraight(ctx, runCtx, cancel, scanner, out, errCh)
		return out, errCh, nil
	}

	jobs := make(chan seqBatch, qs)
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
					b, werr := applyOps(j.batch, ops)
					if werr != nil {
						werr = fmt.Errorf("vector: scheduler: worker: %w", werr)
					}
					select {
					case <-runCtx.Done():
						return
					case results <- seqResult{seq: j.seq, batch: b, err: werr}:
					}
				}
			}
		}()
	}

	go runReader(runCtx, cancel, scanner, jobs, sendErr, readerDone)

	go func() {
		wg.Wait()
		close(results)
	}()

	go reassemble(ctx, runCtx, cancel, results, out, sendErr, closeErrCh, readerDone)

	return out, errCh, nil
}

func runStraight(parent, runCtx context.Context, cancel context.CancelFunc, scanner Scanner, out chan<- *Batch, errCh chan<- error) {
	defer func() {
		if cerr := scanner.Close(); cerr != nil {
			select {
			case errCh <- fmt.Errorf("vector: scheduler: close source: %w", cerr):
			default:
			}
		}
		if parent.Err() != nil {
			select {
			case errCh <- fmt.Errorf("vector: scheduler: %w", parent.Err()):
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
		case out <- scanner.Batch():
		}
	}
	if serr := scanner.Err(); serr != nil {
		select {
		case errCh <- fmt.Errorf("vector: scheduler: scan: %w", serr):
		default:
		}
		cancel()
	}
}

func runReader(ctx context.Context, cancel context.CancelFunc, scanner Scanner, jobs chan<- seqBatch, sendErr func(error), readerDone chan struct{}) {
	defer close(readerDone)
	defer func() {
		close(jobs)
		if cerr := scanner.Close(); cerr != nil {
			sendErr(fmt.Errorf("vector: scheduler: close source: %w", cerr))
		}
	}()
	seq := 0
	for scanner.Next() {
		j := seqBatch{seq: seq, batch: scanner.Batch()}
		seq++
		select {
		case <-ctx.Done():
			return
		case jobs <- j:
		}
	}
	if serr := scanner.Err(); serr != nil {
		sendErr(fmt.Errorf("vector: scheduler: scan: %w", serr))
		cancel()
	}
}

func reassemble(parent, runCtx context.Context, cancel context.CancelFunc, results <-chan seqResult, out chan<- *Batch, sendErr func(error), closeErrCh func(), readerDone <-chan struct{}) {
	defer func() {
		cancel()
		<-readerDone
		if parent.Err() != nil {
			sendErr(fmt.Errorf("vector: scheduler: %w", parent.Err()))
		}
		close(out)
		closeErrCh()
	}()
	pending := make(map[int]*Batch)
	next := 0
	emit := func() {
		for {
			b, ok := pending[next]
			if !ok {
				return
			}
			delete(pending, next)
			next++
			select {
			case <-runCtx.Done():
				return
			case out <- b:
			}
		}
	}
	for {
		select {
		case <-runCtx.Done():
			return
		case r, ok := <-results:
			if !ok {
				emit()
				return
			}
			if r.err != nil {
				sendErr(r.err)
				return
			}
			pending[r.seq] = r.batch
			emit()
		}
	}
}

// Iterator drains the ordered batch channel of a Scheduler run. Semantics
// mirror engine.Iterator: after exhaustion Next keeps returning the
// terminal error; Close is idempotent and never starves a blocked Next.
type Iterator struct {
	batches <-chan *Batch
	errCh   <-chan error
	cancel  context.CancelFunc

	cancelOnce sync.Once
	mu         sync.Mutex
	done       bool
	err        error
}

// NewIterator builds an Iterator over a Scheduler run.
func NewIterator(batches <-chan *Batch, errCh <-chan error, cancel context.CancelFunc) *Iterator {
	return &Iterator{batches: batches, errCh: errCh, cancel: cancel}
}

// Next returns the next batch. ok is false at end of stream, in which
// case err carries the terminal error. The passed ctx bounds only this
// call: cancelling it returns ctx.Err() without marking done.
func (it *Iterator) Next(ctx context.Context) (b *Batch, ok bool, err error) {
	it.mu.Lock()
	if it.done {
		term := it.err
		it.mu.Unlock()
		return nil, false, term
	}
	batches := it.batches
	errCh := it.errCh
	it.mu.Unlock()

	select {
	case <-ctx.Done():
		return nil, false, ctx.Err()
	case batch, open := <-batches:
		if !open {
			var term error
			select {
			case e, has := <-errCh:
				if has {
					term = e
				}
			default:
			}
			it.mu.Lock()
			if !it.done {
				it.done = true
				it.err = term
			}
			term = it.err
			it.mu.Unlock()
			return nil, false, term
		}
		return batch, true, nil
	}
}

// Close stops the underlying run. It never blocks on a concurrent Next.
func (it *Iterator) Close() error {
	it.cancelOnce.Do(func() {
		if it.cancel != nil {
			it.cancel()
		}
	})
	it.mu.Lock()
	it.done = true
	it.mu.Unlock()
	return nil
}
