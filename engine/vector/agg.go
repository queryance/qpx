// Aggregation over typed batches: global and grouped float measures.
//
// These are sinks, not Operators: each Add folds one batch into local
// state, Merge combines per-worker states single-threaded. PG pushdown
// stays the default query route; these serve future non-PG sources and
// standalone engine-side benches. Every Add honors Sel and skips NULL
// values for sum/avg (SQL semantics); NULL keys form their own group,
// matching PG's GROUP BY NULL behavior.
//
// Parallelism: shard batches across workers, one agg per worker, then
// Merge. No shared maps, no locks — see Parallel* below. The int64 path
// is the fast path (single map load per row); the string path is a plain
// map[string] — dictionary encoding was considered and rejected (see
// GroupByString): at the profiled shape (short keys, ≤1K distinct) map
// hashing is noise next to the PG scan floor.
package vector

import (
	"fmt"
	"sort"

	"github.com/queryance/qpx/engine"
)

// GlobalResult is the outcome of a GlobalFloatAgg: Rows counts selected
// rows (count(*)), Count counts non-NULL values, Avg = Sum/Count.
type GlobalResult struct {
	Rows  int64
	Count int64
	Sum   float64
	Avg   float64
}

// GlobalFloatAgg folds one float column across batches: count(*),
// sum(v), avg(v). NULLs are counted in Rows but excluded from Sum/Avg.
type GlobalFloatAgg struct {
	Col   int
	rows  int64
	count int64
	sum   float64
}

// NewGlobalFloatAgg builds an aggregator over float column col.
func NewGlobalFloatAgg(col int) *GlobalFloatAgg {
	return &GlobalFloatAgg{Col: col}
}

// Add folds b into the aggregator, honoring its selection vector.
func (a *GlobalFloatAgg) Add(b *Batch) error {
	if err := checkAggColumn(b, a.Col, engine.Float64, "global-agg"); err != nil {
		return err
	}
	c := &b.Columns[a.Col]
	n := b.Len()
	a.rows += int64(n)
	if n == 0 {
		return nil
	}
	data := c.Floats
	if b.Sel == nil {
		if len(c.Nulls) == 0 {
			var sum float64
			for r := 0; r < n; r++ {
				sum += data[r]
			}
			a.count += int64(n)
			a.sum += sum
			return nil
		}
		nulls := c.Nulls
		var sum float64
		var cnt int64
		for r := 0; r < n; r++ {
			if !nulls[r] {
				sum += data[r]
				cnt++
			}
		}
		a.count += cnt
		a.sum += sum
		return nil
	}
	sel := b.Sel
	if len(c.Nulls) == 0 {
		var sum float64
		for _, s := range sel {
			sum += data[int(s)]
		}
		a.count += int64(len(sel))
		a.sum += sum
		return nil
	}
	nulls := c.Nulls
	var sum float64
	var cnt int64
	for _, s := range sel {
		if !nulls[int(s)] {
			sum += data[int(s)]
			cnt++
		}
	}
	a.count += cnt
	a.sum += sum
	return nil
}

// Merge folds other's state into a. Workers own their aggregators; the
// merge runs single-threaded after they finish, so no locks are needed.
func (a *GlobalFloatAgg) Merge(other *GlobalFloatAgg) {
	a.rows += other.rows
	a.count += other.count
	a.sum += other.sum
}

// Reset clears accumulated state for reuse.
func (a *GlobalFloatAgg) Reset() {
	a.rows, a.count, a.sum = 0, 0, 0
}

// Result returns the accumulated aggregate.
func (a *GlobalFloatAgg) Result() GlobalResult {
	r := GlobalResult{Rows: a.rows, Count: a.count, Sum: a.sum}
	if a.count > 0 {
		r.Avg = a.sum / float64(a.count)
	}
	return r
}

// groupState holds one group's measure: rows counts selected rows
// (count(*)), count counts non-NULL values, sum their total.
type groupState struct {
	rows  int64
	count int64
	sum   float64
}

func (s *groupState) add(v float64, valid bool) {
	s.rows++
	if valid {
		s.count++
		s.sum += v
	}
}

func (s *groupState) merge(o groupState) {
	s.rows += o.rows
	s.count += o.count
	s.sum += o.sum
}

// Int64Group is one group's finished row: Rows is count(*), Count the
// non-NULL values averaged into Avg.
type Int64Group struct {
	Key   int64
	Rows  int64
	Count int64
	Sum   float64
	Avg   float64
}

// GroupByInt64 groups by an int64 column with a float measure. Mod != 0
// groups by key%Mod instead (matrix glow/ghigh shape: id%128, id%100000);
// ids are positive so Go % matches PG % exactly. ValCol < 0 means
// count-only: no value is read, Sum/Avg stay zero.
type GroupByInt64 struct {
	KeyCol int
	ValCol int
	Mod    int64

	groups  map[int64]groupState
	nullKey groupState
	hasNull bool
}

// NewGroupByInt64 builds an int64 group-by. valCol < 0 selects count-only.
func NewGroupByInt64(keyCol, valCol int, mod int64) *GroupByInt64 {
	return &GroupByInt64{KeyCol: keyCol, ValCol: valCol, Mod: mod, groups: make(map[int64]groupState)}
}

// Add folds b into the group table, honoring its selection vector.
func (g *GroupByInt64) Add(b *Batch) error {
	if err := checkAggColumn(b, g.KeyCol, engine.Int64, "groupby-int64"); err != nil {
		return err
	}
	if g.ValCol >= 0 {
		if err := checkAggColumn(b, g.ValCol, engine.Float64, "groupby-int64"); err != nil {
			return err
		}
	}
	keys := b.Columns[g.KeyCol].Ints
	keyNulls := b.Columns[g.KeyCol].Nulls
	hasVal := g.ValCol >= 0
	var vals []float64
	var valNulls []bool
	if hasVal {
		vals = b.Columns[g.ValCol].Floats
		valNulls = b.Columns[g.ValCol].Nulls
	}
	n := b.Len()
	if n == 0 {
		return nil
	}
	if g.groups == nil {
		g.groups = make(map[int64]groupState)
	}
	keyOf := func(k int64) int64 {
		if g.Mod != 0 {
			return k % g.Mod
		}
		return k
	}
	if b.Sel == nil {
		for r := 0; r < n; r++ {
			if len(keyNulls) > 0 && keyNulls[r] {
				g.hasNull = true
				v, ok := measureAt(vals, valNulls, hasVal, r)
				g.nullKey.add(v, ok)
				continue
			}
			k := keyOf(keys[r])
			st := g.groups[k]
			v, ok := measureAt(vals, valNulls, hasVal, r)
			st.add(v, ok)
			g.groups[k] = st
		}
		return nil
	}
	for _, s := range b.Sel {
		r := int(s)
		if len(keyNulls) > 0 && keyNulls[r] {
			g.hasNull = true
			v, ok := measureAt(vals, valNulls, hasVal, r)
			g.nullKey.add(v, ok)
			continue
		}
		k := keyOf(keys[r])
		st := g.groups[k]
		v, ok := measureAt(vals, valNulls, hasVal, r)
		st.add(v, ok)
		g.groups[k] = st
	}
	return nil
}

// Merge folds other's groups into g, single-threaded after workers finish.
func (g *GroupByInt64) Merge(other *GroupByInt64) {
	if g.groups == nil {
		g.groups = make(map[int64]groupState, len(other.groups))
	}
	for k, o := range other.groups {
		st := g.groups[k]
		st.merge(o)
		g.groups[k] = st
	}
	if other.hasNull {
		g.hasNull = true
		g.nullKey.merge(other.nullKey)
	}
}

// Reset clears all groups for reuse.
func (g *GroupByInt64) Reset() {
	for k := range g.groups {
		delete(g.groups, k)
	}
	g.nullKey = groupState{}
	g.hasNull = false
}

// Len reports the number of groups, including the NULL group when present.
func (g *GroupByInt64) Len() int {
	n := len(g.groups)
	if g.hasNull {
		n++
	}
	return n
}

// HasNullGroup reports whether any NULL key was seen.
func (g *GroupByInt64) HasNullGroup() bool { return g.hasNull }

// SortedRows returns groups ordered by key. The NULL group, when present,
// sorts last; use NullGroup for an explicit handle instead.
func (g *GroupByInt64) SortedRows() []Int64Group {
	out := make([]Int64Group, 0, g.Len())
	for k, st := range g.groups {
		out = append(out, finishInt64Group(k, st))
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Key < out[j].Key })
	return out
}

// NullGroup returns the NULL-key group, or nil when no NULL key was seen.
func (g *GroupByInt64) NullGroup() *Int64Group {
	if !g.hasNull {
		return nil
	}
	gr := finishInt64Group(0, g.nullKey)
	return &gr
}

func finishInt64Group(k int64, st groupState) Int64Group {
	g := Int64Group{Key: k, Rows: st.rows, Count: st.count, Sum: st.sum}
	if st.count > 0 {
		g.Avg = st.sum / float64(st.count)
	}
	return g
}

// StringGroup is one string-keyed group's finished row.
type StringGroup struct {
	Key   string
	Rows  int64
	Count int64
	Sum   float64
	Avg   float64
}

// GroupByString groups by a string column with a float measure. Keys are
// hashed directly (map[string]): dictionary encoding would add a full
// pre-pass plus an int64 group-by for the same result, which only pays
// off at cardinalities far above the profiled shape (1K short keys — see
// the string micro-bench). ValCol < 0 means count-only.
type GroupByString struct {
	KeyCol int
	ValCol int

	groups  map[string]groupState
	nullKey groupState
	hasNull bool
}

// NewGroupByString builds a string group-by. valCol < 0 selects count-only.
func NewGroupByString(keyCol, valCol int) *GroupByString {
	return &GroupByString{KeyCol: keyCol, ValCol: valCol, groups: make(map[string]groupState)}
}

// Add folds b into the group table, honoring its selection vector.
func (g *GroupByString) Add(b *Batch) error {
	if err := checkAggColumn(b, g.KeyCol, engine.String, "groupby-string"); err != nil {
		return err
	}
	if g.ValCol >= 0 {
		if err := checkAggColumn(b, g.ValCol, engine.Float64, "groupby-string"); err != nil {
			return err
		}
	}
	keys := b.Columns[g.KeyCol].Strings
	keyNulls := b.Columns[g.KeyCol].Nulls
	hasVal := g.ValCol >= 0
	var vals []float64
	var valNulls []bool
	if hasVal {
		vals = b.Columns[g.ValCol].Floats
		valNulls = b.Columns[g.ValCol].Nulls
	}
	n := b.Len()
	if n == 0 {
		return nil
	}
	if g.groups == nil {
		g.groups = make(map[string]groupState)
	}
	if b.Sel == nil {
		for r := 0; r < n; r++ {
			if len(keyNulls) > 0 && keyNulls[r] {
				g.hasNull = true
				v, ok := measureAt(vals, valNulls, hasVal, r)
				g.nullKey.add(v, ok)
				continue
			}
			st := g.groups[keys[r]]
			v, ok := measureAt(vals, valNulls, hasVal, r)
			st.add(v, ok)
			g.groups[keys[r]] = st
		}
		return nil
	}
	for _, s := range b.Sel {
		r := int(s)
		if len(keyNulls) > 0 && keyNulls[r] {
			g.hasNull = true
			v, ok := measureAt(vals, valNulls, hasVal, r)
			g.nullKey.add(v, ok)
			continue
		}
		st := g.groups[keys[r]]
		v, ok := measureAt(vals, valNulls, hasVal, r)
		st.add(v, ok)
		g.groups[keys[r]] = st
	}
	return nil
}

// Merge folds other's groups into g, single-threaded after workers finish.
func (g *GroupByString) Merge(other *GroupByString) {
	if g.groups == nil {
		g.groups = make(map[string]groupState, len(other.groups))
	}
	for k, o := range other.groups {
		st := g.groups[k]
		st.merge(o)
		g.groups[k] = st
	}
	if other.hasNull {
		g.hasNull = true
		g.nullKey.merge(other.nullKey)
	}
}

