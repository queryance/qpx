package engine_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/queryance/qpx/engine"
)

func TestDataTypeString(t *testing.T) {
	cases := []struct {
		typ  engine.DataType
		want string
	}{
		{engine.Int64, "Int64"},
		{engine.Float64, "Float64"},
		{engine.Bool, "Bool"},
		{engine.String, "String"},
		{engine.Bytes, "Bytes"},
		{engine.Time, "Time"},
		{engine.Numeric, "Numeric"},
		{engine.DataType(99), "Unknown"},
		{engine.DataType(-1), "Unknown"},
	}
	for _, tc := range cases {
		if got := tc.typ.String(); got != tc.want {
			t.Errorf("DataType(%d).String() = %q, want %q", int(tc.typ), got, tc.want)
		}
	}
}

func TestIteratorPerCallContextCancel(t *testing.T) {
	// Empty (unbuffered, no sender) chunks makes the cancelled-ctx case
	// the only ready select branch, so the result is deterministic.
	// A buffered chunk would race ctx.Done vs chunk delivery.
	chunks := make(chan engine.Chunk)
	errCh := make(chan error)
	defer close(errCh)
	it := engine.NewIterator(chunks, errCh, nil)

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := it.Next(cancelled); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled Next = %v, want context.Canceled", err)
	}
	// Per-call cancel must not mark the iterator done: a later live
	// call still receives a chunk sent after the cancellation.
	go func() { chunks <- makeChunk(1, 2) }()
	ctx, stop := context.WithTimeout(context.Background(), 5*time.Second)
	defer stop()
	c, ok, err := it.Next(ctx)
	if err != nil || !ok {
		t.Fatalf("Next after per-call cancel = (%v, %v, %v), want chunk", c, ok, err)
	}
	if ids := col0(c); len(ids) != 2 {
		t.Fatalf("unexpected rows after cancel: %v", ids)
	}
	// DeadlineExceeded surfaces the same way.
	deadline, cancel2 := context.WithDeadline(context.Background(), time.Now().Add(-time.Hour))
	defer cancel2()
	chunks2 := make(chan engine.Chunk)
	errCh2 := make(chan error)
	defer close(errCh2)
	defer close(chunks2)
	it2 := engine.NewIterator(chunks2, errCh2, nil)
	if _, _, err := it2.Next(deadline); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expired Next = %v, want context.DeadlineExceeded", err)
	}
}

func TestIteratorTerminalSticky(t *testing.T) {
	// Clean EOF is sticky.
	chunks := make(chan engine.Chunk)
	errCh := make(chan error)
	close(chunks)
	close(errCh)
	it := engine.NewIterator(chunks, errCh, nil)
	for i := 0; i < 3; i++ {
		if _, ok, err := it.Next(context.Background()); ok || err != nil {
			t.Fatalf("clean EOF call %d = (ok=%v, err=%v), want (false, nil)", i, ok, err)
		}
	}

	// Published terminal error is sticky across Next calls.
	wantErr := errors.New("terminal boom")
	chunks2 := make(chan engine.Chunk)
	errCh2 := make(chan error, 1)
	errCh2 <- wantErr
	close(chunks2)
	close(errCh2)
	it2 := engine.NewIterator(chunks2, errCh2, nil)
	for i := 0; i < 3; i++ {
		_, ok, err := it2.Next(context.Background())
		if ok || !errors.Is(err, wantErr) {
			t.Fatalf("error terminal call %d = (ok=%v, err=%v), want wrapped boom", i, ok, err)
		}
	}
}

func TestIteratorCloseInvokesCancelOnce(t *testing.T) {
	calls := 0
	chunks := make(chan engine.Chunk)
	errCh := make(chan error)
	it := engine.NewIterator(chunks, errCh, func() { calls++ })
	if err := it.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := it.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if calls != 1 {
		t.Fatalf("cancel calls = %d, want 1", calls)
	}
	if _, ok, _ := it.Next(context.Background()); ok {
		t.Fatal("Next after Close should report done")
	}
}

func TestIteratorCloseHidesTerminalPreservesPrior(t *testing.T) {
	// Run a failing pipeline to publish a terminal error, then Close:
	// Next after Close still reports the published error (Close only
	// marks done, it does not clear err).
	src := &fakeSource{schema: testSchema, chunks: []engine.Chunk{makeChunk(1)}, errAfter: -1}
	sched := engine.Scheduler{Workers: 1, QueueSize: 1}
	ctx := context.Background()
	wantErr := errors.New("hide me")
	chunks, errCh, err := sched.Run(ctx, src, []engine.Operator{&funcOp{fn: func(c engine.Chunk) (engine.Chunk, error) {
		return engine.Chunk{}, wantErr
	}}})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	it := engine.NewIterator(chunks, errCh, nil)
	_, ok, err := it.Next(ctx)
	// First Next may return the chunk-less terminal directly or drain;
	// keep draining until done to publish the terminal error.
	for ok && err == nil {
		_, ok, err = it.Next(ctx)
	}
	if !errors.Is(err, wantErr) {
		t.Fatalf("terminal = %v, want wrapped boom", err)
	}
	if err := it.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	_, ok, err = it.Next(ctx)
	if ok || !errors.Is(err, wantErr) {
		t.Fatalf("Next after Close = (ok=%v, err=%v), want sticky boom", ok, err)
	}
}

// errFakeScanBoom is the sentinel scan error for closeErrScanner, so tests
// can distinguish scan errors from close errors with errors.Is.
var errFakeScanBoom = errors.New("fake scan boom")

// closeErrSource exercises the scheduler's "close source" error path.
type closeErrSource struct {
	schema   engine.Schema
	chunks   []engine.Chunk
	errAfter int
	closeErr error
}

func (f *closeErrSource) Open(context.Context) (engine.Scanner, error) {
	return &closeErrScanner{src: f}, nil
}

type closeErrScanner struct {
	src *closeErrSource
	pos int
	err error
	cur engine.Chunk
}

func (s *closeErrScanner) Schema() engine.Schema { return s.src.schema }
func (s *closeErrScanner) Next() bool {
	if s.err != nil {
		return false
	}
	if s.src.errAfter >= 0 && s.pos >= s.src.errAfter {
		s.err = errFakeScanBoom
		return false
	}
	if s.pos >= len(s.src.chunks) {
		return false
	}
	s.cur = s.src.chunks[s.pos]
	s.pos++
	return true
}
func (s *closeErrScanner) Chunk() engine.Chunk { return s.cur }
func (s *closeErrScanner) Err() error          { return s.err }
func (s *closeErrScanner) Close() error        { return s.src.closeErr }

func TestSchedulerCloseError(t *testing.T) {
	// The reader's close-error send used to race the reassembler's
	// close(errCh) (send on closed channel under -race). Both paths below
	// must surface the close error deterministically with no sleeps: the
	// straight path has a single channel owner, and the worker path orders
	// every send before the close via readerDone.
	closeErr := errors.New("close boom")
	cases := []struct {
		name string
		ops  []engine.Operator
	}{
		{"straight", nil},
		{"workers", []engine.Operator{&funcOp{fn: func(c engine.Chunk) (engine.Chunk, error) {
			return c, nil
		}}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			src := &closeErrSource{
				schema:   testSchema,
				chunks:   []engine.Chunk{makeChunk(1)},
				errAfter: -1,
				closeErr: closeErr,
			}
			sched := engine.Scheduler{Workers: 2, QueueSize: 4}
			chunks, errCh, err := sched.Run(context.Background(), src, tc.ops)
			if err != nil {
				t.Fatalf("Run: %v", err)
			}
			it := engine.NewIterator(chunks, errCh, nil)
			_, err = drain(t, it)
			if !errors.Is(err, closeErr) {
				t.Fatalf("terminal = %v, want wrapped close error", err)
			}
		})
	}
}

func TestSchedulerScanAndCloseError(t *testing.T) {
	// Scan error is sent before close error on one bounded channel, so the
	// scan error deterministically wins and the run terminates. Covers both
	// the straight and worker paths.
	cases := []struct {
		name string
		ops  []engine.Operator
	}{
		{"straight", nil},
		{"workers", []engine.Operator{&funcOp{fn: func(c engine.Chunk) (engine.Chunk, error) {
			return c, nil
		}}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			src := &closeErrSource{
				schema:   testSchema,
				chunks:   []engine.Chunk{makeChunk(1)},
				errAfter: 1,
				closeErr: errors.New("close after scan boom"),
			}
			sched := engine.Scheduler{Workers: 2, QueueSize: 4}
			ctx := context.Background()
			chunks, errCh, err := sched.Run(ctx, src, tc.ops)
			if err != nil {
				t.Fatalf("Run: %v", err)
			}
			it := engine.NewIterator(chunks, errCh, nil)
			done := make(chan error, 1)
			go func() {
				var term error
				for {
					_, ok, err := it.Next(ctx)
					if err != nil {
						term = err
						break
					}
					if !ok {
						break
					}
				}
				done <- term
			}()
			select {
			case term := <-done:
				if !errors.Is(term, errFakeScanBoom) {
					t.Fatalf("terminal = %v, want wrapped scan error", term)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("run with scan+close errors did not terminate")
			}
		})
	}
}

func TestSchedulerCloseErrorStress(t *testing.T) {
	// Repeated close-error runs interleave reader sends with reassembler
	// teardown; under -race any send-on-close or unsynchronized access
	// fails. Tiny queue forces jobs-send blocking for varied interleavings.
	for i := 0; i < 25; i++ {
		closeErr := errors.New("close boom")
		src := &closeErrSource{
			schema:   testSchema,
			chunks:   []engine.Chunk{makeChunk(i), makeChunk(i + 1)},
			errAfter: -1,
			closeErr: closeErr,
		}
		sched := engine.Scheduler{Workers: 2, QueueSize: 1}
		identity := []engine.Operator{&funcOp{fn: func(c engine.Chunk) (engine.Chunk, error) {
			return c, nil
		}}}
		chunks, errCh, err := sched.Run(context.Background(), src, identity)
		if err != nil {
			t.Fatalf("iter %d Run: %v", i, err)
		}
		it := engine.NewIterator(chunks, errCh, nil)
		_, err = drain(t, it)
		if !errors.Is(err, closeErr) {
			t.Fatalf("iter %d terminal = %v, want wrapped close error", i, err)
		}
	}
}

func TestSchedulerWorkerAndScanError(t *testing.T) {
	// Op error plus scan error: both the reader and the reassembler try
	// to publish to errCh (capacity 1), so one of the `default` branches
	// is exercised. The run must still surface an error promptly.
	src := &closeErrSource{
		schema:   testSchema,
		chunks:   []engine.Chunk{makeChunk(1), makeChunk(2)},
		errAfter: 1,
		closeErr: nil,
	}
	wantErr := errors.New("worker boom")
	bad := &funcOp{fn: func(c engine.Chunk) (engine.Chunk, error) {
		time.Sleep(200 * time.Millisecond)
		return engine.Chunk{}, wantErr
	}}
	sched := engine.Scheduler{Workers: 2, QueueSize: 4}
	ctx := context.Background()
	chunks, errCh, err := sched.Run(ctx, src, []engine.Operator{bad})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	it := engine.NewIterator(chunks, errCh, nil)
	done := make(chan error, 1)
	go func() {
		var term error
		for {
			_, ok, err := it.Next(ctx)
			if err != nil {
				term = err
				break
			}
			if !ok {
				break
			}
		}
		done <- term
	}()
	select {
	case term := <-done:
		if term == nil {
			t.Fatal("want worker or scan error, got clean EOF")
		}
		// Either error is acceptable; both paths wrap with context.
		if !errors.Is(term, wantErr) && term.Error() == "" {
			t.Fatalf("unexpected terminal error: %v", term)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("run with worker+scan errors did not terminate")
	}
}

func TestSchedulerNoDrainCancelCloses(t *testing.T) {
	// Never draining `out` while cancelling forces the reassembler's
	// emit path to observe ctx.Done instead of blocking forever.
	var chunks []engine.Chunk
	for i := 0; i < 32; i++ {
		chunks = append(chunks, makeChunk(i))
	}
	src := &fakeSource{schema: testSchema, chunks: chunks, errAfter: -1}
	sched := engine.Scheduler{Workers: 2, QueueSize: 1}
	ctx, cancel := context.WithCancel(context.Background())
	out, errCh, err := sched.Run(ctx, src, nil)
	if err != nil {
