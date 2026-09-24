package engine

import (
	"context"
	"sync"
)

// Iterator drains the ordered chunk channel of a Scheduler run. After the
// channel is exhausted, Next returns the run's terminal error (nil on a
// clean run) and keeps returning it on subsequent calls.
//
// Cancellation semantics: external cancellation of the run context surfaces
// as a wrapped ctx.Err() (use errors.Is with context.Canceled or
// context.DeadlineExceeded) instead of a clean EOF. Calling Close hides the
// run's terminal error: Next after Close reports done with the error
// published before Close, or nil if none was published yet.
type Iterator struct {
	chunks <-chan Chunk
	errCh  <-chan error
	cancel context.CancelFunc

	cancelOnce sync.Once
	mu         sync.Mutex
	done       bool
	err        error
}

// NewIterator builds an Iterator over a Scheduler run. cancel stops the
// run; it may be nil if the caller owns cancellation another way.
func NewIterator(chunks <-chan Chunk, errCh <-chan error, cancel context.CancelFunc) *Iterator {
	return &Iterator{chunks: chunks, errCh: errCh, cancel: cancel}
}

// Next returns the next chunk. ok is false at end of stream, in which case
// err carries the terminal error. The passed ctx bounds only this call:
// cancelling it returns ctx.Err() without marking the Iterator done.
func (it *Iterator) Next(ctx context.Context) (c Chunk, ok bool, err error) {
	it.mu.Lock()
	if it.done {
		term := it.err
		it.mu.Unlock()
		return Chunk{}, false, term
	}
	chunks := it.chunks
	errCh := it.errCh
	it.mu.Unlock()

	select {
	case <-ctx.Done():
		return Chunk{}, false, ctx.Err()
	case chunk, open := <-chunks:
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
			return Chunk{}, false, term
		}
		return chunk, true, nil
	}
}

// Close stops the underlying run. It never blocks on a concurrent Next:
// cancellation is issued via sync.Once without holding mu. It is idempotent.
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
