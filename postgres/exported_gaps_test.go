package postgres_test

import (
	"testing"

	"github.com/jackc/pgx/v5/pgtype"
	"github.com/queryance/qpx/engine"
	"github.com/queryance/qpx/postgres"
)

func TestMapDataTypeTimeAndBinaryVariants(t *testing.T) {
	cases := []struct {
		oid  uint32
		want engine.DataType
	}{
		{pgtype.DateOID, engine.Time},
		{pgtype.TimeOID, engine.Time},
		{pgtype.TimetzOID, engine.Time},
		{pgtype.TimestampOID, engine.Time},
		{pgtype.TimestamptzOID, engine.Time},
		{pgtype.IntervalOID, engine.Time},
		{pgtype.ByteaOID, engine.Bytes},
		{pgtype.NumericOID, engine.Numeric},
		{pgtype.BoolOID, engine.Bool},
		{pgtype.Int2OID, engine.Int64},
		{pgtype.Float8OID, engine.Float64},
		// Text-like and misc fall back to String.
		{pgtype.TextOID, engine.String},
		{pgtype.VarcharOID, engine.String},
		{pgtype.BPCharOID, engine.String},
		{pgtype.NameOID, engine.String},
		{pgtype.UUIDOID, engine.String},
		{pgtype.JSONOID, engine.String},
		{pgtype.JSONBOID, engine.String},
		{pgtype.XMLOID, engine.String},
		{pgtype.PointOID, engine.String},
		{0, engine.String},
		{4294967295, engine.String},
	}
	for _, tc := range cases {
		if got := postgres.MapDataType(tc.oid); got != tc.want {
			t.Errorf("MapDataType(%d) = %v, want %v", tc.oid, got, tc.want)
		}
	}
}

func TestPackChunkEdges(t *testing.T) {
	schema := engine.Schema{Fields: []engine.Field{
		{Name: "id", Type: engine.Int64},
		{Name: "name", Type: engine.String},
	}}

	// Extra values beyond the schema are ignored.
	c, err := postgres.PackChunk(schema, [][]any{
		{int64(1), "a", "EXTRA", 999},
		{int64(2), "b", nil},
	})
	if err != nil {
		t.Fatalf("PackChunk: %v", err)
	}
	if err := c.Validate(); err != nil {
		t.Fatalf("extra-value pack invalid: %v", err)
	}
	if c.NumRows() != 2 {
		t.Fatalf("NumRows = %d, want 2", c.NumRows())
	}
	if got := c.Columns[0][0].(int64); got != 1 {
		t.Errorf("row 0 col 0 = %v, want 1", got)
	}
	if got := c.Columns[1][1].(string); got != "b" {
		t.Errorf("row 1 col 1 = %v, want b", got)
	}
	if len(c.Columns) != 2 {
		t.Fatalf("columns = %d, want 2 (extras dropped)", len(c.Columns))
	}

	// Empty rows produce a valid 0-row chunk preserving schema.
	empty, err := postgres.PackChunk(schema, [][]any{})
	if err != nil {
		t.Fatalf("empty PackChunk: %v", err)
	}
	if empty.NumRows() != 0 {
		t.Fatalf("empty NumRows = %d, want 0", empty.NumRows())
	}
	if len(empty.Schema.Fields) != 2 {
		t.Fatalf("empty schema fields = %d, want 2", len(empty.Schema.Fields))
	}
	if err := empty.Validate(); err != nil {
		t.Fatalf("empty chunk invalid: %v", err)
	}

	// Zero-column schema packs to a valid 0-column chunk.
	zeroSchema := engine.Schema{}
	z, err := postgres.PackChunk(zeroSchema, [][]any{{int64(1)}, {}})
	if err != nil {
		t.Fatalf("zero-col PackChunk: %v", err)
	}
	if z.NumRows() != 0 {
		t.Fatalf("zero-col NumRows = %d, want 0 (no columns)", z.NumRows())
	}
	if err := z.Validate(); err != nil {
		t.Fatalf("zero-col chunk invalid: %v", err)
	}

	// Nil values are preserved by reference (shared, not copied).
	nilRows := [][]any{{nil, nil}}
	n, err := postgres.PackChunk(schema, nilRows)
	if err != nil {
		t.Fatalf("PackChunk: %v", err)
	}
	if n.Columns[0][0] != nil || n.Columns[1][0] != nil {
		t.Fatalf("nil cells not preserved: %v", n.Columns)
	}
}

func TestNewSourceChunkSizeViaExecute(t *testing.T) {
	// NewSource + SetChunkSize are the exported knobs qpx.Execute uses.
	src := postgres.NewSource(postgres.Config{SQL: "SELECT 1"})
	if src == nil {
		t.Fatal("NewSource returned nil")
	}
	// Must not panic on edge values; behavior verified in internal tests.
	src.SetChunkSize(0)
	src.SetChunkSize(-1)
	src.SetChunkSize(512)
}
