package vector

import (
	"fmt"
	"time"

	"github.com/queryance/qpx/engine"
)

// ToChunk boxes the batch's effective rows (honoring Sel) into a legacy
// engine.Chunk. It exists for parity tests and as an escape hatch to
// legacy row ops; the serving path never calls it (boxing is exactly the
// cost the vector path removes). Byte cells share backing with the batch.
func (b *Batch) ToChunk() (engine.Chunk, error) {
	if err := b.Validate(); err != nil {
		return engine.Chunk{}, err
	}
	n := b.Len()
	cols := make([][]any, len(b.Columns))
	for j := range b.Columns {
		cols[j] = make([]any, n)
	}
	for i := 0; i < n; i++ {
		r := i
		if b.Sel != nil {
			r = int(b.Sel[i])
		}
		for j := range b.Columns {
			c := &b.Columns[j]
			if c.IsNull(r) {
				cols[j][i] = nil
				continue
			}
			switch c.Type {
			case engine.Int64:
				cols[j][i] = c.Ints[r]
			case engine.Float64:
				cols[j][i] = c.Floats[r]
			case engine.Bool:
				cols[j][i] = c.Bools[r]
			case engine.String:
				cols[j][i] = c.Strings[r]
			case engine.Bytes:
				cols[j][i] = c.Bytes[r]
			case engine.Time:
				cols[j][i] = c.Times[r]
			case engine.Numeric:
				cols[j][i] = c.Numerics[r]
			default:
				cols[j][i] = c.Strings[r]
			}
		}
	}
	return engine.NewChunk(b.Schema, cols)
}

// FromChunk unboxes a legacy chunk into a typed Batch (dense, Sel == nil).
// Cell Go types must match the schema: Int64 accepts int/int16/int32/int64,
// Float64 accepts float32/float64, Time accepts time.Time, Numeric accepts
// string. Anything else (including unexpected nil handling beyond NULL) is
// an error, never a silent zero — parity tests rely on this strictness.
func FromChunk(c engine.Chunk) (*Batch, error) {
	if err := c.Validate(); err != nil {
		return nil, fmt.Errorf("vector: from chunk: %w", err)
	}
	b := NewBatch(c.Schema, c.NumRows())
	for j := range c.Columns {
		col := &b.Columns[j]
		for i := 0; i < c.NumRows(); i++ {
			v := c.Columns[j][i]
			if v == nil {
				col.AppendNull()
				continue
			}
			var aerr error
			switch col.Type {
			case engine.Int64:
				aerr = appendIntCell(col, v, j, i)
			case engine.Float64:
				aerr = appendFloatCell(col, v, j, i)
