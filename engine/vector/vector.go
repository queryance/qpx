// Package vector is the typed columnar core of the engine: batches of
// plain Go slices with no any/interface in hot loops.
//
// A Batch holds one slice per column selected by its engine.DataType
// (Int64s, Floats, Strings, ...). Only the slice matching the column's
// type is populated. Filters do not move data: they build Sel, a
// selection vector of physical row indexes; Sel == nil means every row
// (0..NumRows) is selected. Outputs alias their input's backing arrays,
// so a Batch must not be reused after it has been passed to an Operator.
package vector

import (
	"fmt"
	"time"

	"github.com/queryance/qpx/engine"
)

// DefaultBatchSize is the rows-per-batch hint for vector scanners. 4096
// rows × narrow columns fits comfortably in L2 while amortizing per-batch
// channel handoffs and backing-array allocations (~8 allocs per batch, or
// ~0.002 allocs/row — negligible next to per-row string bytes).
const DefaultBatchSize = 4096

// Batch is a column-major set of typed rows with an optional selection
// vector. Columns[i] holds NumRows values; Sel holds the selected
// physical indexes (nil = all rows dense).
type Batch struct {
	Schema  engine.Schema
	Columns []Column
	NumRows int
	Sel     []int32
}

// Column is a single typed column. Exactly one data slice matches Type;
// Nulls parallels it (true = SQL NULL, value undefined) and is nil until
// the first null is appended. HasNulls caches len(Nulls) > 0 for the
// decode path; operators check len(Nulls) directly.
type Column struct {
	Type     engine.DataType
	Ints     []int64
	Floats   []float64
	Bools    []bool
	Strings  []string
	Bytes    [][]byte
	Times    []time.Time
	Numerics []string
	Nulls    []bool
	HasNulls bool
}

// NewBatch builds an empty Batch for schema with per-column backing
// arrays pre-sized to cap rows. cap <= 0 selects a small default.
func NewBatch(schema engine.Schema, cap int) *Batch {
	if cap <= 0 {
		cap = 64
	}
	b := &Batch{Schema: schema, Columns: make([]Column, len(schema.Fields))}
	for i, f := range schema.Fields {
		b.Columns[i].Type = f.Type
		switch f.Type {
		case engine.Int64:
			b.Columns[i].Ints = make([]int64, 0, cap)
		case engine.Float64:
			b.Columns[i].Floats = make([]float64, 0, cap)
		case engine.Bool:
			b.Columns[i].Bools = make([]bool, 0, cap)
		case engine.String:
			b.Columns[i].Strings = make([]string, 0, cap)
		case engine.Bytes:
			b.Columns[i].Bytes = make([][]byte, 0, cap)
		case engine.Time:
			b.Columns[i].Times = make([]time.Time, 0, cap)
		case engine.Numeric:
			b.Columns[i].Numerics = make([]string, 0, cap)
		default:
			b.Columns[i].Strings = make([]string, 0, cap)
		}
	}
	return b
}

// Len reports effective rows: len(Sel) when a selection is present,
// NumRows otherwise.
func (b *Batch) Len() int {
	if b.Sel != nil {
		return len(b.Sel)
	}
	return b.NumRows
}

// Reset truncates every column and drops the selection, keeping backing
// arrays for reuse. The caller must own the Batch exclusively (it must
// not be in flight in a scheduler) — this is what makes Reset safe
// without a pool: reuse is explicit at the owner, never shared.
func (b *Batch) Reset() {
	for i := range b.Columns {
		c := &b.Columns[i]
		switch c.Type {
		case engine.Int64:
			c.Ints = c.Ints[:0]
		case engine.Float64:
			c.Floats = c.Floats[:0]
		case engine.Bool:
			c.Bools = c.Bools[:0]
		case engine.String:
			c.Strings = c.Strings[:0]
		case engine.Bytes:
			// Clear references so reused backing does not pin old values.
			for j := range c.Bytes {
				c.Bytes[j] = nil
			}
			c.Bytes = c.Bytes[:0]
		case engine.Time:
			c.Times = c.Times[:0]
		case engine.Numeric:
			c.Numerics = c.Numerics[:0]
		default:
			c.Strings = c.Strings[:0]
		}
