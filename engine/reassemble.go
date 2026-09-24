package engine

import (
	"context"
	"fmt"
)

// reassemble emits worker results in source order and owns the terminal
// teardown: it is the only goroutine that closes out and errCh.
//
// parent reports external cancellation (runCtx is already cancelled by our
// own teardown, so it cannot); runCtx drives the select loop.
//
// Teardown order (no LIFO dependence): cancel first to unblock a reader
// stuck on jobs-send, then wait for readerDone so every sendErr
// happens-before the close, then record a wrapped parent.Err() on
// cancellation, then close out and errCh. Waiting is bounded by the
// source's Close; the postgres backend caps it with a timeout.
func reassemble(parent, runCtx context.Context, cancel context.CancelFunc, results <-chan seqResult, out chan<- Chunk, sendErr func(error), closeErrCh func(), readerDone <-chan struct{}) {
	defer func() {
		cancel()
		<-readerDone
		if parent.Err() != nil {
			sendErr(fmt.Errorf("scheduler: %w", parent.Err()))
		}
		close(out)
		closeErrCh()
	}()
	pending := make(map[int]Chunk)
	next := 0
	emit := func() {
		for {
			c, ok := pending[next]
			if !ok {
				return
			}
			delete(pending, next)
			next++
			select {
			case <-runCtx.Done():
				return
			case out <- c:
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
			pending[r.seq] = r.chunk
			emit()
		}
	}
}
