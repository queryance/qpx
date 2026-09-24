package postgres_test

import (
	"context"
	"os"
	"testing"

	"github.com/jackc/pgx/v5/pgtype"
	"github.com/queryance/qpx/engine"
	"github.com/queryance/qpx/postgres"
)

func TestMapDataType(t *testing.T) {
	cases := []struct {
		oid  uint32
		want engine.DataType
	}{
		{pgtype.Int2OID, engine.Int64},
		{pgtype.Int4OID, engine.Int64},
		{pgtype.Int8OID, engine.Int64},
		{pgtype.Float4OID, engine.Float64},
		{pgtype.Float8OID, engine.Float64},
		{pgtype.BoolOID, engine.Bool},
		{pgtype.TextOID, engine.String},
		{pgtype.VarcharOID, engine.String},
		{pgtype.NameOID, engine.String},
		{pgtype.ByteaOID, engine.Bytes},
		{pgtype.TimestampOID, engine.Time},
		{pgtype.TimestamptzOID, engine.Time},
		{pgtype.DateOID, engine.Time},
		{pgtype.NumericOID, engine.Numeric},
		{0, engine.String},      // unknown OID falls back to String
		{999999, engine.String}, // unmapped OID falls back to String
		{pgtype.JSONBOID, engine.String},
		{pgtype.UUIDOID, engine.String},
	}
	for _, tc := range cases {
		if got := postgres.MapDataType(tc.oid); got != tc.want {
			t.Errorf("MapDataType(%d) = %v, want %v", tc.oid, got, tc.want)
		}
	}
}

func TestPackChunk(t *testing.T) {
	schema := engine.Schema{Fields: []engine.Field{
		{Name: "id", Type: engine.Int64},
		{Name: "name", Type: engine.String},
	}}
	rows := [][]any{{int64(1), "a"}, {int64(2), "b"}, {int64(3), "c"}}
	c, err := postgres.PackChunk(schema, rows)
	if err != nil {
		t.Fatalf("PackChunk: %v", err)
	}
	if c.NumRows() != 3 {
		t.Fatalf("NumRows = %d, want 3", c.NumRows())
	}
	for i, want := range []int64{1, 2, 3} {
		if got := c.Columns[0][i].(int64); got != want {
			t.Errorf("row %d col 0 = %v, want %v", i, got, want)
		}
	}
	if got := c.Columns[1][2].(string); got != "c" {
		t.Errorf("row 2 col 1 = %v, want c", got)
	}
	if len(c.Schema.Fields) != 2 || c.Schema.Fields[0].Name != "id" {
		t.Errorf("schema not preserved: %+v", c.Schema.Fields)
	}
	empty, err := postgres.PackChunk(schema, nil)
	if err != nil {
		t.Fatalf("empty PackChunk: %v", err)
	}
	if n := empty.NumRows(); n != 0 {
		t.Errorf("empty pack NumRows = %d, want 0", n)
	}
}

func TestPackChunkNilFillsShortRows(t *testing.T) {
	schema := engine.Schema{Fields: []engine.Field{
		{Name: "id", Type: engine.Int64},
		{Name: "name", Type: engine.String},
	}}
	// Short rows must be nil-filled, not emitted as ragged chunks.
	c, err := postgres.PackChunk(schema, [][]any{{int64(1)}, {int64(2), "b"}})
	if err != nil {
		t.Fatalf("PackChunk: %v", err)
	}
	if err := c.Validate(); err != nil {
		t.Fatalf("short-row pack must stay rectangular: %v", err)
	}
	if c.NumRows() != 2 {
		t.Fatalf("NumRows = %d, want 2", c.NumRows())
	}
	if c.Columns[1][0] != nil {
		t.Errorf("missing cell should be nil-filled, got %v", c.Columns[1][0])
	}
	if got := c.Columns[1][1].(string); got != "b" {
		t.Errorf("row 1 col 1 = %v, want b", got)
	}
}

func TestLiveQuery(t *testing.T) {
	dsn := os.Getenv("TEST_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("TEST_POSTGRES_DSN not set")
	}
	ctx := context.Background()
	src := postgres.NewSource(postgres.Config{
		ConnString: dsn,
		SQL:        `SELECT g AS id FROM generate_series(1, 2500) g`,
		ChunkSize:  512,
	})
	sc, err := src.Open(ctx)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() {
		_ = sc.Close()
	}()
	schema := sc.Schema()
	if len(schema.Fields) != 1 || schema.Fields[0].Name != "id" {
		t.Fatalf("unexpected schema: %+v", schema.Fields)
	}
	if schema.Fields[0].Type != engine.Int64 {
		t.Fatalf("id type = %v, want Int64", schema.Fields[0].Type)
	}
	total := 0
	var next int64 = 1
	for sc.Next() {
		c := sc.Chunk()
		if err := c.Validate(); err != nil {
			t.Fatalf("ragged chunk from scanner: %v", err)
		}
		for i := 0; i < c.NumRows(); i++ {
			v, ok := c.Columns[0][i].(int32)
			if !ok {
				t.Fatalf("row value type %T, want int32", c.Columns[0][i])
			}
			if int64(v) != next {
				t.Fatalf("row %d out of order: got %d", next, v)
			}
			next++
		}
		total += c.NumRows()
		if c.NumRows() > 512 {
			t.Fatalf("chunk of %d rows exceeds ChunkSize 512", c.NumRows())
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("Err: %v", err)
	}
	if total != 2500 {
		t.Fatalf("total = %d, want 2500", total)
	}
}
