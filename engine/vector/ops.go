package vector

import (
	"fmt"

	"github.com/queryance/qpx/engine"
)

// Operator transforms one batch into another. Like engine.Operator,
// implementations must be safe for concurrent use. The input Batch must
// not be touched after Process returns: outputs alias its backing arrays.
type Operator interface {
	Process(*Batch) (*Batch, error)
}

// Cmp is a scalar comparison predicate.
type Cmp int

const (
	Eq Cmp = iota
	Ne
	Gt
	Ge
	Lt
	Le
)

func (c Cmp) String() string {
	switch c {
	case Eq:
		return "="
	case Ne:
		return "<>"
	case Gt:
		return ">"
	case Ge:
		return ">="
	case Lt:
		return "<"
	case Le:
		return "<="
	default:
		return "?"
	}
}

// Filter keeps rows where column Col compares Cmp against the threshold.
// Kind selects the specialized loop (Int64, Float64, String, Bool) and
// must match the column's type; only the threshold field for Kind is
// read. NULLs never match (SQL WHERE semantics: the row is dropped, not
// an error). String filters support Eq/Ne only.
type Filter struct {
	Col  int
	Kind engine.DataType
	Cmp  Cmp
	I64  int64
	F64  float64
	Str  string
	B    bool
}

// NewInt64Filter builds an integer comparison filter.
func NewInt64Filter(col int, cmp Cmp, v int64) *Filter {
	return &Filter{Col: col, Kind: engine.Int64, Cmp: cmp, I64: v}
}

// NewFloat64Filter builds a float comparison filter (e.g. Q2's f > 0.5).
func NewFloat64Filter(col int, cmp Cmp, v float64) *Filter {
	return &Filter{Col: col, Kind: engine.Float64, Cmp: cmp, F64: v}
}

// NewStringFilter builds a string equality filter.
func NewStringFilter(col int, cmp Cmp, v string) *Filter {
	return &Filter{Col: col, Kind: engine.String, Cmp: cmp, Str: v}
}

// NewBoolFilter keeps rows where the column equals v.
func NewBoolFilter(col int, v bool) *Filter {
	return &Filter{Col: col, Kind: engine.Bool, Cmp: Eq, B: v}
}

// Process filters in place: it builds (dense input) or compacts
// (chained input) the selection vector without moving column data. A
// fully matching dense batch keeps Sel == nil via a first-mismatch scan
// that allocates only once a dropped row is found — the all-match fast
// path costs zero allocs and leaves the batch untouched.
func (op *Filter) Process(b *Batch) (*Batch, error) {
	if err := checkColumn(b, op.Col, op.Kind, "filter"); err != nil {
		return nil, err
	}
	c := &b.Columns[op.Col]
	switch op.Kind {
	case engine.Int64:
		filterInt64Into(b, c, op.Cmp, op.I64)
	case engine.Float64:
		filterFloat64Into(b, c, op.Cmp, op.F64)
	case engine.String:
		if op.Cmp != Eq && op.Cmp != Ne {
			return nil, fmt.Errorf("vector: filter: string supports =/<> only, got %v", op.Cmp)
		}
		filterStringInto(b, c, op.Cmp, op.Str)
	case engine.Bool:
		filterBoolInto(b, c, op.B)
	default:
		return nil, fmt.Errorf("vector: filter: unsupported kind %v", op.Kind)
	}
	if b.Sel != nil && len(b.Sel) == b.NumRows {
		b.Sel = nil
	}
	return b, nil
}

func checkColumn(b *Batch, col int, kind engine.DataType, op string) error {
	if b == nil {
		return fmt.Errorf("vector: %s: nil batch", op)
	}
	if col < 0 || col >= len(b.Columns) {
		return fmt.Errorf("vector: %s: column %d out of range (have %d)", op, col, len(b.Columns))
	}
	if got := b.Columns[col].Type; got != kind {
		return fmt.Errorf("vector: %s: column %d is %v, filter is %v", op, col, got, kind)
	}
	if err := b.Validate(); err != nil {
		return fmt.Errorf("vector: %s: %w", op, err)
	}
	return nil
}

// The per-type loops below are intentionally separate, not one generic
// loop over any: each is a tight typed-slice scan with the null bitmap
// hoisted. The NOT NULL variant — which NOT NULL tables always take —
// has no per-row branch beyond the comparison itself.

func filterInt64Into(b *Batch, c *Column, cmp Cmp, v int64) {
	data := c.Ints
	if b.Sel == nil {
		// First-mismatch scan: rows before the first drop all match, so a
		// fully matching batch returns with Sel == nil and zero allocs.
		// Only after a mismatch do we materialize the selection, backfilling
		// the known-matching prefix with identity indices (no re-evaluation).
		if len(c.Nulls) == 0 {
			miss := -1
		for r := 0; r < b.NumRows; r++ {
			if !applyIntCmp(data[r], cmp, v) {
				miss = r
				break
			}
		}
		if miss < 0 {
			return
		}
		out := make([]int32, 0, b.NumRows)
		for r := 0; r < miss; r++ {
			out = append(out, int32(r))
		}
		for r := miss + 1; r < b.NumRows; r++ {
			if applyIntCmp(data[r], cmp, v) {
				out = append(out, int32(r))
			}
		}
		if len(out) == 0 {
			out = []int32{}
		}
		b.Sel = out
		return
		}
		nulls := c.Nulls
		miss := -1
		for r := 0; r < b.NumRows; r++ {
			if nulls[r] || !applyIntCmp(data[r], cmp, v) {
				miss = r
				break
			}
		}
		if miss < 0 {
			return
		}
		out := make([]int32, 0, b.NumRows)
		for r := 0; r < miss; r++ {
			out = append(out, int32(r))
		}
		for r := miss + 1; r < b.NumRows; r++ {
			if !nulls[r] && applyIntCmp(data[r], cmp, v) {
				out = append(out, int32(r))
			}
		}
		if len(out) == 0 {
			out = []int32{}
		}
		b.Sel = out
		return
	}
	sel := b.Sel
	n := 0
	if len(c.Nulls) == 0 {
		for _, s := range sel {
			if applyIntCmp(data[int(s)], cmp, v) {
				sel[n] = s
				n++
			}
		}
	} else {
		nulls := c.Nulls
		for _, s := range sel {
			if !nulls[int(s)] && applyIntCmp(data[int(s)], cmp, v) {
				sel[n] = s
				n++
			}
		}
	}
	b.Sel = sel[:n]
}

func applyIntCmp(x int64, cmp Cmp, v int64) bool {
	switch cmp {
	case Eq:
		return x == v
	case Ne:
		return x != v
	case Gt:
		return x > v
	case Ge:
		return x >= v
	case Lt:
		return x < v
	case Le:
		return x <= v
	}
	return false
}

func filterFloat64Into(b *Batch, c *Column, cmp Cmp, v float64) {
	data := c.Floats
	if b.Sel == nil {
		// First-mismatch scan, as in filterInt64Into: all-match costs zero
		// allocs and leaves Sel == nil.
		if len(c.Nulls) == 0 {
			miss := -1
			for r := 0; r < b.NumRows; r++ {
				if !applyFloatCmp(data[r], cmp, v) {
					miss = r
					break
				}
			}
			if miss < 0 {
				return
			}
			out := make([]int32, 0, b.NumRows)
			for r := 0; r < miss; r++ {
				out = append(out, int32(r))
			}
			for r := miss + 1; r < b.NumRows; r++ {
				if applyFloatCmp(data[r], cmp, v) {
					out = append(out, int32(r))
				}
			}
			if len(out) == 0 {
				out = []int32{}
			}
			b.Sel = out
			return
		}
		nulls := c.Nulls
		miss := -1
		for r := 0; r < b.NumRows; r++ {
			if nulls[r] || !applyFloatCmp(data[r], cmp, v) {
				miss = r
				break
			}
		}
		if miss < 0 {
			return
		}
		out := make([]int32, 0, b.NumRows)
		for r := 0; r < miss; r++ {
			out = append(out, int32(r))
		}
		for r := miss + 1; r < b.NumRows; r++ {
			if !nulls[r] && applyFloatCmp(data[r], cmp, v) {
				out = append(out, int32(r))
			}
		}
		if len(out) == 0 {
			out = []int32{}
		}
		b.Sel = out
		return
	}
	sel := b.Sel
	n := 0
	if len(c.Nulls) == 0 {
		for _, s := range sel {
			if applyFloatCmp(data[int(s)], cmp, v) {
				sel[n] = s
				n++
			}
		}
	} else {
		nulls := c.Nulls
		for _, s := range sel {
			if !nulls[int(s)] && applyFloatCmp(data[int(s)], cmp, v) {
				sel[n] = s
				n++
			}
		}
	}
	b.Sel = sel[:n]
}

func applyFloatCmp(x float64, cmp Cmp, v float64) bool {
	switch cmp {
	case Eq:
		return x == v
	case Ne:
		return x != v
	case Gt:
		return x > v
	case Ge:
		return x >= v
	case Lt:
		return x < v
	case Le:
		return x <= v
	}
	return false
}

func filterStringInto(b *Batch, c *Column, cmp Cmp, v string) {
	data := c.Strings
	eq := cmp == Eq
	if b.Sel == nil {
		// First-mismatch scan, as in filterInt64Into: all-match costs zero
		// allocs and leaves Sel == nil.
		match := func(r int) bool { return (data[r] == v) == eq }
		if len(c.Nulls) == 0 {
			miss := -1
			for r := 0; r < b.NumRows; r++ {
				if !match(r) {
					miss = r
					break
				}
			}
			if miss < 0 {
				return
			}
			out := make([]int32, 0, b.NumRows)
			for r := 0; r < miss; r++ {
				out = append(out, int32(r))
			}
			for r := miss + 1; r < b.NumRows; r++ {
				if match(r) {
					out = append(out, int32(r))
				}
			}
			if len(out) == 0 {
				out = []int32{}
			}
			b.Sel = out
			return
		}
		nulls := c.Nulls
		miss := -1
		for r := 0; r < b.NumRows; r++ {
			if nulls[r] || !match(r) {
				miss = r
				break
			}
		}
		if miss < 0 {
			return
		}
		out := make([]int32, 0, b.NumRows)
		for r := 0; r < miss; r++ {
			out = append(out, int32(r))
		}
		for r := miss + 1; r < b.NumRows; r++ {
			if !nulls[r] && match(r) {
				out = append(out, int32(r))
			}
		}
		if len(out) == 0 {
			out = []int32{}
		}
		b.Sel = out
		return
	}
	sel := b.Sel
	n := 0
	if len(c.Nulls) == 0 {
		for _, s := range sel {
			if (data[int(s)] == v) == eq {
				sel[n] = s
				n++
			}
		}
	} else {
		nulls := c.Nulls
		for _, s := range sel {
			if !nulls[int(s)] && (data[int(s)] == v) == eq {
				sel[n] = s
				n++
			}
		}
	}
	b.Sel = sel[:n]
}

func filterBoolInto(b *Batch, c *Column, v bool) {
	data := c.Bools
	if b.Sel == nil {
		// First-mismatch scan, as in filterInt64Into: all-match costs zero
		// allocs and leaves Sel == nil.
		if len(c.Nulls) == 0 {
			miss := -1
			for r := 0; r < b.NumRows; r++ {
				if data[r] != v {
					miss = r
					break
				}
			}
			if miss < 0 {
				return
			}
			out := make([]int32, 0, b.NumRows)
			for r := 0; r < miss; r++ {
				out = append(out, int32(r))
			}
			for r := miss + 1; r < b.NumRows; r++ {
				if data[r] == v {
					out = append(out, int32(r))
				}
			}
			if len(out) == 0 {
				out = []int32{}
			}
			b.Sel = out
			return
		}
		nulls := c.Nulls
		miss := -1
		for r := 0; r < b.NumRows; r++ {
			if nulls[r] || data[r] != v {
				miss = r
				break
			}
		}
		if miss < 0 {
			return
		}
		out := make([]int32, 0, b.NumRows)
		for r := 0; r < miss; r++ {
			out = append(out, int32(r))
		}
		for r := miss + 1; r < b.NumRows; r++ {
			if !nulls[r] && data[r] == v {
				out = append(out, int32(r))
			}
		}
		if len(out) == 0 {
			out = []int32{}
		}
		b.Sel = out
		return
	}
	sel := b.Sel
	n := 0
	if len(c.Nulls) == 0 {
		for _, s := range sel {
			if data[int(s)] == v {
				sel[n] = s
				n++
			}
		}
	} else {
		nulls := c.Nulls
		for _, s := range sel {
			if !nulls[int(s)] && data[int(s)] == v {
				sel[n] = s
				n++
			}
		}
	}
	b.Sel = sel[:n]
}

// Project keeps a subset of columns, in the given order. Indices refer to
// positions in the input batch; duplicates are allowed. Projection is a
// header-only operation — no per-row loop, typed or otherwise — so there
// is nothing to specialize: the output shares the input's backing arrays
// and selection.
type Project struct {
	Indices []int
}

// NewProject builds a Project keeping the given column indices.
func NewProject(indices ...int) *Project {
	return &Project{Indices: indices}
}

// Process returns a batch with only the projected columns.
func (op *Project) Process(b *Batch) (*Batch, error) {
	if b == nil {
		return nil, fmt.Errorf("vector: project: nil batch")
	}
	if len(op.Indices) == 0 {
		return nil, fmt.Errorf("vector: project: no columns selected")
	}
	if err := b.Validate(); err != nil {
		return nil, fmt.Errorf("vector: project: %w", err)
	}
	for _, idx := range op.Indices {
		if idx < 0 || idx >= len(b.Columns) {
			return nil, fmt.Errorf("vector: project: column index %d out of range (have %d columns)", idx, len(b.Columns))
		}
	}
	out := &Batch{
		Schema:  engine.Schema{Fields: make([]engine.Field, len(op.Indices))},
		Columns: make([]Column, len(op.Indices)),
		NumRows: b.NumRows,
		Sel:     b.Sel,
	}
	for i, idx := range op.Indices {
		out.Schema.Fields[i] = b.Schema.Fields[idx]
		out.Columns[i] = b.Columns[idx]
	}
	return out, nil
}

// FilterProject fuses a single Filter with a Project into one operator:
// one pass over the filter column, one batch header allocated. Use it
// wherever a query filters and then narrows columns (Q2's f > 0.5 plus
// project (id, f)) — the sequential Filter-then-Project would do two
// passes and two headers for the same result.
type FilterProject struct {
	Filter  *Filter
	Indices []int
}

// NewFilterFloat64GTProject builds the fused Q2-shaped operator: keep
// rows where float column filterCol > threshold, then keep indices.
func NewFilterFloat64GTProject(filterCol int, threshold float64, indices ...int) *FilterProject {
	return &FilterProject{Filter: NewFloat64Filter(filterCol, Gt, threshold), Indices: indices}
}

// Process runs the filter in place, then projects the survivors.
func (op *FilterProject) Process(b *Batch) (*Batch, error) {
	if op.Filter == nil {
		return nil, fmt.Errorf("vector: filter-project: nil filter")
	}
	filtered, err := op.Filter.Process(b)
	if err != nil {
		return nil, err
	}
	return (&Project{Indices: op.Indices}).Process(filtered)
}
