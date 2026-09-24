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
		// Same pin-avoidance for strings: headers are values, but the
		// backing bytes are referenced by them; dropping the headers is
		// enough, no per-element clearing needed.
		c.Nulls = c.Nulls[:0]
		c.HasNulls = false
	}
	b.NumRows = 0
	// Drop, not [:0]: Len must return NumRows (not 0) for a refilled
	// dense batch, so empty-but-non-nil would be ambiguous. The next
	// filter allocates a fresh Sel; at ~1 alloc per batch this is noise.
	b.Sel = nil
}

// Validate enforces the rectangular invariant: one column per field,
// every active slice (and non-empty Nulls) exactly NumRows long, Sel
// indexes in range.
func (b *Batch) Validate() error {
	if len(b.Columns) != len(b.Schema.Fields) {
		return fmt.Errorf("vector: column count %d != field count %d", len(b.Columns), len(b.Schema.Fields))
	}
	for i := range b.Columns {
		c := &b.Columns[i]
		want := c.Type
		if got := b.Schema.Fields[i].Type; got != want {
			return fmt.Errorf("vector: column %d type %v != schema %v", i, want, got)
		}
		if n := c.Len(); n != b.NumRows {
			return fmt.Errorf("vector: column %d has %d rows, want %d", i, n, b.NumRows)
		}
		if len(c.Nulls) != 0 && len(c.Nulls) != b.NumRows {
			return fmt.Errorf("vector: column %d null bitmap %d != %d rows", i, len(c.Nulls), b.NumRows)
		}
	}
	for _, s := range b.Sel {
		if s < 0 || int(s) >= b.NumRows {
			return fmt.Errorf("vector: selection index %d out of range (%d rows)", s, b.NumRows)
		}
	}
	return nil
}

// Len reports the physical row count of the column's active slice.
func (c *Column) Len() int {
	switch c.Type {
	case engine.Int64:
		return len(c.Ints)
	case engine.Float64:
		return len(c.Floats)
	case engine.Bool:
		return len(c.Bools)
	case engine.String:
		return len(c.Strings)
	case engine.Bytes:
		return len(c.Bytes)
	case engine.Time:
		return len(c.Times)
	case engine.Numeric:
		return len(c.Numerics)
	default:
		return len(c.Strings)
	}
}

// AppendNull records a SQL NULL: a zero value on the active slice plus a
// true bit. Nulls stays nil until the first null so dense non-null data
// pays no bitmap cost at all.
func (c *Column) AppendNull() {
	n := c.Len()
	if len(c.Nulls) == 0 {
		c.Nulls = make([]bool, n+1)
	} else {
		c.Nulls = append(c.Nulls, false)
	}
	c.Nulls[n] = true
	c.HasNulls = true
	switch c.Type {
	case engine.Int64:
		c.Ints = append(c.Ints, 0)
	case engine.Float64:
		c.Floats = append(c.Floats, 0)
	case engine.Bool:
		c.Bools = append(c.Bools, false)
	case engine.String:
		c.Strings = append(c.Strings, "")
	case engine.Bytes:
		c.Bytes = append(c.Bytes, nil)
	case engine.Time:
		c.Times = append(c.Times, time.Time{})
	case engine.Numeric:
		c.Numerics = append(c.Numerics, "")
	default:
		c.Strings = append(c.Strings, "")
	}
}

// IsNull reports whether physical row r is SQL NULL.
func (c *Column) IsNull(r int) bool {
	return len(c.Nulls) > 0 && c.Nulls[r]
}

// Typed appends maintain the null bitmap: once a column HasNulls every
// subsequent value — null or not — extends Nulls, so len(Nulls) is always
// 0 or the row count (Validate enforces this). Prefer these over raw
// slice appends whenever a column may hold NULLs; the branch is
// perfectly predicted for dense NOT NULL data.
func (c *Column) AppendInt(v int64)      { c.Ints = append(c.Ints, v); c.extendNull() }
func (c *Column) AppendFloat(v float64)  { c.Floats = append(c.Floats, v); c.extendNull() }
func (c *Column) AppendBool(v bool)      { c.Bools = append(c.Bools, v); c.extendNull() }
func (c *Column) AppendString(v string)  { c.Strings = append(c.Strings, v); c.extendNull() }
func (c *Column) AppendBytes(v []byte)   { c.Bytes = append(c.Bytes, v); c.extendNull() }
func (c *Column) AppendTime(v time.Time) { c.Times = append(c.Times, v); c.extendNull() }
func (c *Column) AppendNumeric(v string) { c.Numerics = append(c.Numerics, v); c.extendNull() }
func (c *Column) extendNull() {
	if c.HasNulls {
		c.Nulls = append(c.Nulls, false)
	}
}
