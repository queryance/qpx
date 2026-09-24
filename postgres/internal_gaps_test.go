package postgres

import (
	"context"
	"errors"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/queryance/qpx/engine"
)

func TestInternalChunkSize(t *testing.T) {
	s := NewSource(Config{})
	if got := s.chunkSize(); got != engine.DefaultChunkSize {
		t.Fatalf("zero ChunkSize = %d, want default %d", got, engine.DefaultChunkSize)
	}
	s = NewSource(Config{ChunkSize: 64})
	if got := s.chunkSize(); got != 64 {
		t.Fatalf("ChunkSize = %d, want 64", got)
	}
	// Negative falls back to default.
	s = NewSource(Config{ChunkSize: -5})
	if got := s.chunkSize(); got != engine.DefaultChunkSize {
		t.Fatalf("negative ChunkSize = %d, want default", got)
	}
}

func TestInternalSetChunkSize(t *testing.T) {
	s := NewSource(Config{})
	s.SetChunkSize(128)
	if got := s.chunkSize(); got != 128 {
		t.Fatalf("after SetChunkSize(128) = %d, want 128", got)
	}
	// Non-positive values are ignored.
	s.SetChunkSize(0)
	if got := s.chunkSize(); got != 128 {
		t.Fatalf("SetChunkSize(0) changed size to %d, want 128", got)
	}
	s.SetChunkSize(-10)
	if got := s.chunkSize(); got != 128 {
		t.Fatalf("SetChunkSize(-10) changed size to %d, want 128", got)
	}
	s.SetChunkSize(256)
	if got := s.chunkSize(); got != 256 {
		t.Fatalf("after SetChunkSize(256) = %d, want 256", got)
	}
}

func TestInternalDescribeSchema(t *testing.T) {
	fds := []pgconn.FieldDescription{
		{Name: "id", DataTypeOID: pgtype.Int8OID},
		{Name: "name", DataTypeOID: pgtype.TextOID},
		{Name: "when", DataTypeOID: pgtype.TimestamptzOID},
		{Name: "mystery", DataTypeOID: 999999},
	}
	schema := describeSchema(fds)
	if len(schema.Fields) != 4 {
		t.Fatalf("fields = %d, want 4", len(schema.Fields))
	}
	if schema.Fields[0] != (engine.Field{Name: "id", Type: engine.Int64}) {
		t.Fatalf("field 0 = %+v", schema.Fields[0])
	}
	if schema.Fields[1].Type != engine.String {
		t.Fatalf("field 1 type = %v, want String", schema.Fields[1].Type)
	}
	if schema.Fields[2].Type != engine.Time {
		t.Fatalf("field 2 type = %v, want Time", schema.Fields[2].Type)
	}
	if schema.Fields[3].Type != engine.String {
		t.Fatalf("unknown OID must map to String, got %v", schema.Fields[3].Type)
	}
	if got := describeSchema(nil); len(got.Fields) != 0 {
		t.Fatalf("nil input fields = %d, want 0", len(got.Fields))
	}
}

// fakeRows is a minimal pgx.Rows mock for scanner unit tests (no DB).
type fakeRows struct {
	vals      [][]any
	idx       int
	valuesErr error
	rowsErr   error
	fds       []pgconn.FieldDescription
	closed    bool
}

func newFakeRows(vals [][]any, fds []pgconn.FieldDescription) *fakeRows {
	return &fakeRows{vals: vals, idx: -1, fds: fds}
}

func (f *fakeRows) Close()                                       { f.closed = true }
func (f *fakeRows) Err() error                                   { return f.rowsErr }
func (f *fakeRows) CommandTag() pgconn.CommandTag                { return pgconn.CommandTag{} }
func (f *fakeRows) FieldDescriptions() []pgconn.FieldDescription { return f.fds }
func (f *fakeRows) Next() bool {
	if f.idx+1 >= len(f.vals) {
		return false
	}
	f.idx++
	return true
}
func (f *fakeRows) Scan(...any) error { return nil }
func (f *fakeRows) Values() ([]any, error) {
	if f.valuesErr != nil {
		return nil, f.valuesErr
	}
	if f.idx < 0 || f.idx >= len(f.vals) {
		return nil, errors.New("no current row")
	}
	return f.vals[f.idx], nil
}
func (f *fakeRows) RawValues() [][]byte  { return nil }
func (f *fakeRows) Conn() *pgx.Conn      { return nil }
func (f *fakeRows) TypeMap() *pgtype.Map { return nil }

var internalTestSchema = engine.Schema{Fields: []engine.Field{
	{Name: "id", Type: engine.Int64},
	{Name: "name", Type: engine.String},
}}

func TestInternalScannerNextChunks(t *testing.T) {
	rows := newFakeRows([][]any{
		{int64(1), "a"},
		{int64(2), "b"},
		{int64(3), "c"},
	}, nil)
	sc := &scanner{ctx: context.Background(), rows: rows, schema: internalTestSchema, size: 2}
	if got := sc.Schema(); len(got.Fields) != 2 {
		t.Fatalf("Schema fields = %d, want 2", len(got.Fields))
	}
	if !sc.Next() {
		t.Fatal("first Next = false, want true")
	}
	if n := sc.Chunk().NumRows(); n != 2 {
		t.Fatalf("first chunk rows = %d, want 2", n)
	}
	if err := sc.Chunk().Validate(); err != nil {
		t.Fatalf("first chunk invalid: %v", err)
	}
	if got := sc.Chunk().Columns[0][0].(int64); got != 1 {
		t.Fatalf("row 0 = %v, want 1", got)
	}
	if !sc.Next() {
		t.Fatal("second Next = false, want true")
	}
	if n := sc.Chunk().NumRows(); n != 1 {
		t.Fatalf("second chunk rows = %d, want 1", n)
	}
	if got := sc.Chunk().Columns[0][0].(int64); got != 3 {
		t.Fatalf("row 0 of second chunk = %v, want 3", got)
	}
	if sc.Next() {
		t.Fatal("third Next = true, want false (exhausted)")
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("Err = %v, want nil", err)
	}
	// Exhausted Next stays false.
	if sc.Next() {
		t.Fatal("Next after exhaustion must stay false")
	}
}

func TestInternalScannerNextEmpty(t *testing.T) {
	rows := newFakeRows(nil, nil)
	sc := &scanner{ctx: context.Background(), rows: rows, schema: internalTestSchema, size: 2}
	if sc.Next() {
		t.Fatal("Next on empty rows = true, want false")
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("Err = %v, want nil", err)
	}
}

func TestInternalScannerNextValuesError(t *testing.T) {
	wantErr := errors.New("decode boom")
	rows := newFakeRows([][]any{{int64(1), "a"}}, nil)
	rows.valuesErr = wantErr
	sc := &scanner{ctx: context.Background(), rows: rows, schema: internalTestSchema, size: 10}
	if sc.Next() {
		t.Fatal("Next with Values error = true, want false")
	}
	if !errors.Is(sc.Err(), wantErr) {
		t.Fatalf("Err = %v, want wrapped decode boom", sc.Err())
	}
	// Scan error is sticky: further Next calls stay false.
	if sc.Next() {
		t.Fatal("Next after Values error must stay false")
	}
}

func TestInternalScannerNextRowsErr(t *testing.T) {
	wantErr := errors.New("rows boom")
	rows := newFakeRows([][]any{{int64(1), "a"}}, nil)
	rows.rowsErr = wantErr
	sc := &scanner{ctx: context.Background(), rows: rows, schema: internalTestSchema, size: 10}
	if sc.Next() {
		t.Fatal("Next with rows.Err set = true, want false")
	}
	if !errors.Is(sc.Err(), wantErr) {
		t.Fatalf("Err = %v, want wrapped rows boom", sc.Err())
	}
}

func TestInternalScannerShortAndLongRows(t *testing.T) {
	// Short rows nil-fill, long rows ignore extras (matches PackChunk).
	rows := newFakeRows([][]any{
		{int64(1)},
		{int64(2), "b", "EXTRA", 999},
	}, nil)
	sc := &scanner{ctx: context.Background(), rows: rows, schema: internalTestSchema, size: 10}
	if !sc.Next() {
		t.Fatal("Next = false, want true")
	}
	c := sc.Chunk()
	if err := c.Validate(); err != nil {
		t.Fatalf("chunk invalid: %v", err)
	}
	if c.Columns[1][0] != nil {
		t.Fatalf("short row cell = %v, want nil", c.Columns[1][0])
	}
	if got := c.Columns[1][1].(string); got != "b" {
		t.Fatalf("row 1 col 1 = %v, want b", got)
	}
	if got := c.Columns[0][1].(int64); got != 2 {
		t.Fatalf("row 1 col 0 = %v, want 2", got)
	}
}

func TestInternalScannerClosedAndErrSticky(t *testing.T) {
	rows := newFakeRows([][]any{{int64(1), "a"}}, nil)
	sc := &scanner{ctx: context.Background(), rows: rows, schema: internalTestSchema, size: 10, closed: true}
	if sc.Next() {
		t.Fatal("Next when closed = true, want false")
	}
	sc2 := &scanner{ctx: context.Background(), rows: rows, schema: internalTestSchema, size: 10}
	sc2.scanErr = errors.New("prior")
	if sc2.Next() {
		t.Fatal("Next with prior scanErr = true, want false")
	}
	if sc2.Err() == nil {
		t.Fatal("Err must report prior error")
	}
}

func TestInternalScannerCloseIdempotentWhenClosed(t *testing.T) {
	// Pre-closed scanner with nil conn must not touch the connection.
	sc := &scanner{closed: true}
	if err := sc.Close(); err != nil {
		t.Fatalf("Close on pre-closed = %v, want nil", err)
	}
	if err := sc.Close(); err != nil {
		t.Fatalf("second Close = %v, want nil", err)
	}
}
