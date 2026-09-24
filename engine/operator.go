package engine

import "fmt"

// Operator transforms one chunk into another. Implementations must be safe
// for concurrent use: the Scheduler runs them from a worker pool.
type Operator interface {
	Process(Chunk) (Chunk, error)
}

// FilterOp keeps only rows for which Keep returns true. The row slice is
// reused across calls and must not be retained by Keep.
type FilterOp struct {
	Keep func(row []any) (bool, error)
}

// NewFilterOp builds a FilterOp around a row predicate.
func NewFilterOp(keep func(row []any) (bool, error)) *FilterOp {
	return &FilterOp{Keep: keep}
}

// Process filters the chunk, preserving column order and schema.
// Single pass: rows are tested and kept cells appended in one sweep, so no
// per-row decision bitmap is allocated. Output columns are capped at n and
// may over-allocate when selectivity is low; the waste is bounded by one
// chunk and avoids a second pass over every cell.
func (op *FilterOp) Process(c Chunk) (Chunk, error) {
	if err := c.Validate(); err != nil {
		return Chunk{}, fmt.Errorf("filter: %w", err)
	}
	n := c.NumRows()
	if n == 0 {
		return c, nil
	}
	ncols := len(c.Schema.Fields)
	row := make([]any, ncols)
	out := make([][]any, ncols)
	for j := 0; j < ncols; j++ {
		out[j] = make([]any, 0, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < ncols; j++ {
			row[j] = c.Columns[j][i]
		}
		ok, err := op.Keep(row)
		if err != nil {
			return Chunk{}, fmt.Errorf("filter: %w", err)
		}
		if !ok {
			continue
		}
		for j := 0; j < ncols; j++ {
			out[j] = append(out[j], c.Columns[j][i])
		}
	}
	return Chunk{Schema: c.Schema, Columns: out}, nil
}

// ProjectOp keeps a subset of columns, in the given order. Indices refer
// to positions in the input chunk; duplicates are allowed.
type ProjectOp struct {
	Indices []int
}

// NewProjectOp builds a ProjectOp keeping the given column indices.
func NewProjectOp(indices ...int) *ProjectOp {
	return &ProjectOp{Indices: indices}
}

// Process returns a chunk with only the projected columns.
func (op *ProjectOp) Process(c Chunk) (Chunk, error) {
	if err := c.Validate(); err != nil {
		return Chunk{}, fmt.Errorf("project: %w", err)
	}
	if len(op.Indices) == 0 {
		return Chunk{}, fmt.Errorf("project: no columns selected")
	}
	ncols := len(c.Schema.Fields)
	for _, idx := range op.Indices {
		if idx < 0 || idx >= ncols {
			return Chunk{}, fmt.Errorf("project: column index %d out of range (have %d columns)", idx, ncols)
		}
	}
	fields := make([]Field, len(op.Indices))
	cols := make([][]any, len(op.Indices))
	for i, idx := range op.Indices {
		fields[i] = c.Schema.Fields[idx]
		cols[i] = c.Columns[idx]
	}
	return Chunk{Schema: Schema{Fields: fields}, Columns: cols}, nil
}
