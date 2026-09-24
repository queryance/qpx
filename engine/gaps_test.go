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
