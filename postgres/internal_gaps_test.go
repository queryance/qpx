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
