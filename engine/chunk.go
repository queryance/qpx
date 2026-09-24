package engine

import "fmt"

// DefaultChunkSize is the number of rows per chunk when a source does not
// specify its own size.
const DefaultChunkSize = 1024

// Chunk is a batch of rows stored column-major: Columns[i][j] is row j of
// column i. All columns hold the same number of cells.
type Chunk struct {
	Schema  Schema
	Columns [][]any
}

// Validate enforces the rectangular invariant: len(Columns) must equal
// len(Schema.Fields) and every column must hold the same number of rows.
func (c Chunk) Validate() error {
	if len(c.Columns) != len(c.Schema.Fields) {
		return fmt.Errorf("chunk: column count %d != field count %d", len(c.Columns), len(c.Schema.Fields))
	}
	if len(c.Columns) == 0 {
		return nil
	}
	n := len(c.Columns[0])
	for i := 1; i < len(c.Columns); i++ {
		if len(c.Columns[i]) != n {
			return fmt.Errorf("chunk: ragged columns: column 0 has %d rows, column %d has %d rows", n, i, len(c.Columns[i]))
		}
	}
	return nil
}

// NewChunk builds a Chunk from per-column cell slices. The caller retains
// ownership of columns; it must not mutate them after this call.
// It returns an error if the rectangular invariant does not hold.
func NewChunk(schema Schema, columns [][]any) (Chunk, error) {
	c := Chunk{Schema: schema, Columns: columns}
	if err := c.Validate(); err != nil {
		return Chunk{}, err
	}
	return c, nil
}

// NumRows reports the number of rows in the chunk.
func (c Chunk) NumRows() int {
	if len(c.Columns) == 0 {
		return 0
	}
	return len(c.Columns[0])
}
