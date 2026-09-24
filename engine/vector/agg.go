// Aggregation over typed batches: global and grouped float measures.
//
// These are sinks, not Operators: each Add folds one batch into local
// state, Merge combines per-worker states single-threaded (or via a
// partitioned merge past the parthash.go gate). PG pushdown stays the
// default query route; these serve future non-PG sources and standalone
// engine-side benches. Every Add honors Sel and skips NULL values for
// sum/avg (SQL semantics); NULL keys form their own group, matching PG's
// GROUP BY NULL behavior.
//
// Hash structure (see ht.go for the DuckDB source mapping):
//   - GroupByInt64 with a small positive Mod folds into a dense array
//     (perfect hash: index IS the reduced key) — no hashing, no probes.
//   - GroupByInt64 otherwise folds into an open-addressing
//     linear-probing table with contiguous key/payload arrays.
//   - GroupByString folds into a dictionary-encoded table: one map load
//     per row to a dense id, payloads updated in place.
//
// In every case payload updates happen in place, so the old
// map[K]groupState two-probes-per-row (load + store-back with rehash and
// value copy) is gone.
//
// Parallelism: shard batches across workers, one agg per worker, then
// Merge. No shared tables, no locks — see Parallel* below.
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

// Add folds b into the aggregator, honoring its selection vector. The
// dense non-NULL path sums with 4 independent accumulators: the running
// total is a loop-carried FP dependency (latency-bound), and unrolling
// lets the core retire one add per cycle instead of one per latency.
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
			var s0, s1, s2, s3 float64
			m := n &^ 3
			i := 0
			for ; i < m; i += 4 {
				s0 += data[i]
				s1 += data[i+1]
				s2 += data[i+2]
				s3 += data[i+3]
			}
			for ; i < n; i++ {
				s0 += data[i]
			}
			a.count += int64(n)
			a.sum += s0 + s1 + s2 + s3
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
// (count(*)), count counts non-NULL values, sum their total. It survives
// only as the NULL-key accumulator; live groups live in the ht.go tables'
// SoA payload arrays (updated in place, never copied per row).
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
//
// Representation is chosen by Mod: a small positive Mod (<=
// denseMaxGroups) folds into a dense perfect-hash array keyed by the
// reduced key; anything else (raw keys, negative or huge Mod) folds into
// the open-addressing table. Either way Add is a single indexed update
// per row with payloads folded in place.
type GroupByInt64 struct {
	KeyCol int
	ValCol int
	Mod    int64

	dense   *denseIntAgg
	oa      *int64HashAgg
	spill   *int64HashAgg // negative reduced keys (see addDense)
	modMask int64         // Mod-1 when Mod is a power of two (see setupMod)
	modPow2 bool
	nullKey groupState
	hasNull bool
}

// useDense reports whether mod admits the perfect-hash array: positive
// (so idx needs only a sign bias, never a full range scan) and bounded
// (mod slots × ~33 B each: 8 key + 1 present + 8 rows + 8 count + 8 sum).
// The bound caps ONE table; ParallelGroupByInt64 holds W locals plus the
// output, so it applies the tighter wantDenseParallel gate (mod × (W+1)
// against denseParBudget) and falls back to the OA table past it.
func useDense(mod int64) bool {
	return mod > 0 && mod <= denseMaxGroups
}

// Memory guard for the parallel dense path: one dense table costs ~33 B
// per slot, and the parallel fold holds W locals plus the merged output
// (W+1 tables). Past this budget the merge falls back to the
// open-addressing table, which grows with distinct keys instead of Mod.
// 128 MB keeps Mod=100K dense through W16 (~56 MB total) while routing
// Mod=1M at W16 (~560 MB) to OA.
const (
	denseBytesPerSlot = 33
	denseParBudget    = 128 << 20
)

// wantDenseParallel reports whether the parallel dense fold fits the
// memory budget for mod over workers locals plus the output table.
func wantDenseParallel(mod int64, workers int) bool {
	if !useDense(mod) {
		return false
	}
	return mod*int64(workers+1)*denseBytesPerSlot <= denseParBudget
}

// newGroupByInt64OA builds an int64 group-by that folds into the
// open-addressing table even when Mod would admit the dense array: the
// parallel memory-guard fallback. Mod is preserved so addOA still
// reduces keys identically and Merge stays cross-representation safe.
func newGroupByInt64OA(keyCol, valCol int, mod int64) *GroupByInt64 {
	g := &GroupByInt64{KeyCol: keyCol, ValCol: valCol, Mod: mod}
	g.setupMod()
	g.oa = newInt64HashAgg(htInitialCap)
	return g
}

// NewGroupByInt64 builds an int64 group-by. valCol < 0 selects count-only.
func NewGroupByInt64(keyCol, valCol int, mod int64) *GroupByInt64 {
	g := &GroupByInt64{KeyCol: keyCol, ValCol: valCol, Mod: mod}
	g.setupMod()
	if useDense(mod) {
		g.dense = newDenseIntAgg(mod)
	} else {
		g.oa = newInt64HashAgg(htInitialCap)
	}
	return g
}

// setupMod precomputes the power-of-two fast path: when Mod is a power
// of two, rk = raw & (Mod-1) replaces a ~20-cycle SDIV with one AND —
// the same bit-trick DuckDB's perfect hash uses for binning. Negative
// raws keep Go-% semantics via the spillover (see addDense), so -1 and
// Mod-1 stay distinct groups exactly as PG reports them.

// ensureTable lazily builds the representation (New always does, but a
// zero-value GroupByInt64 must still Add safely).
func (g *GroupByInt64) setupMod() {
	g.modPow2 = g.Mod > 0 && g.Mod&(g.Mod-1) == 0
	if g.modPow2 {
		g.modMask = g.Mod - 1
	}
}

func (g *GroupByInt64) ensureTable() {
	if g.dense != nil || g.oa != nil {
		return
	}
	g.setupMod()
	if useDense(g.Mod) {
		g.dense = newDenseIntAgg(g.Mod)
	} else {
		g.oa = newInt64HashAgg(htInitialCap)
	}
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
	g.ensureTable()
	if g.dense != nil {
		g.addDense(keys, keyNulls, vals, valNulls, hasVal, b.Sel, n)
		return nil
	}
	g.addOA(keys, keyNulls, vals, valNulls, hasVal, b.Sel, n)
	return nil
}

// addDense folds rows into the perfect-hash array. Reduced key
// rk = raw % mod keeps Go-%-matches-PG-% semantics. Non-negative rk
// indexes the array directly; negative rk (negative raw keys: Go and PG
// % both keep the sign) collides with rk+mod in any dense mapping, so
// those rows spill to a small open-addressing table keyed by the true
// rk — same groups as the old map path, at negligible cost since bench
// and parity shapes never take this branch.
func (g *GroupByInt64) addDense(keys []int64, keyNulls []bool, vals []float64, valNulls []bool, hasVal bool, sel []int32, n int) {
	mod := g.Mod
	mask := g.modMask
	pow2 := g.modPow2
	// Hot loop, deliberately closure-free (a per-row closure call does
	// not inline and costs ~40% here): non-negative raws reduce by mask
	// (power-of-two Mod, one AND) or Go-% otherwise; negative raws take
	// Go-% for their true signed key and spill iff it stays negative, so
	// -Mod and 0 share group 0 while -1 stays its own — exactly the map
	// path's (and PG's) groups. All sign branches predict perfectly on
	// non-negative shapes.
	d := g.dense
	if sel == nil {
		// range over keys (len == NumRows == n when Sel is nil, per
		// Validate) gives check-free raw; the present-hot-path update
		// below is the inlined addIdx fast path — the call + cold
		// insert stay outlined in addIdx.
		for r, raw := range keys {
			if len(keyNulls) > 0 && keyNulls[r] {
				g.hasNull = true
				v, ok := measureAt(vals, valNulls, hasVal, r)
				g.nullKey.add(v, ok)
				continue
			}
			if raw < 0 {
				rk := raw % mod
				v, ok := measureAt(vals, valNulls, hasVal, r)
				if rk < 0 {
					g.ensureSpill().add(rk, v, ok)
					continue
				}
				d.addIdx(rk, rk, v, ok)
				continue
			}
			var rk int64
			if pow2 {
				rk = raw & mask
			} else {
				rk = raw % mod
			}
			v, ok := measureAt(vals, valNulls, hasVal, r)
			if d.present[rk] {
				d.rows[rk]++
				if ok {
					d.counts[rk]++
					d.sums[rk] += v
				}
			} else {
				d.addIdx(rk, rk, v, ok)
			}
		}
		return
	}
	for _, s := range sel {
		r := int(s)
		if len(keyNulls) > 0 && keyNulls[r] {
			g.hasNull = true
			v, ok := measureAt(vals, valNulls, hasVal, r)
			g.nullKey.add(v, ok)
			continue
		}
		raw := keys[r]
		if raw < 0 {
			rk := raw % mod
			v, ok := measureAt(vals, valNulls, hasVal, r)
			if rk < 0 {
				g.ensureSpill().add(rk, v, ok)
				continue
			}
			d.addIdx(rk, rk, v, ok)
			continue
		}
		var rk int64
		if pow2 {
			rk = raw & mask
		} else {
			rk = raw % mod
		}
		v, ok := measureAt(vals, valNulls, hasVal, r)
		if d.present[rk] {
			d.rows[rk]++
			if ok {
				d.counts[rk]++
				d.sums[rk] += v
			}
		} else {
			d.addIdx(rk, rk, v, ok)
		}
	}
}

// ensureSpill lazily builds the negative-key spillover (stays nil on
// non-negative shapes, so the hot path pays one predictable branch).

// addOA folds rows into the open-addressing table: one mix + one probe
// per row, payload updated in place. keyOf applies Mod when set.
func (g *GroupByInt64) addOA(keys []int64, keyNulls []bool, vals []float64, valNulls []bool, hasVal bool, sel []int32, n int) {
	keyOf := func(k int64) int64 {
		if g.Mod != 0 {
			return k % g.Mod
		}
		return k
	}
	if sel == nil {
		for r := 0; r < n; r++ {
			if len(keyNulls) > 0 && keyNulls[r] {
				g.hasNull = true
				v, ok := measureAt(vals, valNulls, hasVal, r)
				g.nullKey.add(v, ok)
				continue
			}
			v, ok := measureAt(vals, valNulls, hasVal, r)
			g.oa.add(keyOf(keys[r]), v, ok)
		}
		return
	}
	for _, s := range sel {
		r := int(s)
		if len(keyNulls) > 0 && keyNulls[r] {
			g.hasNull = true
			v, ok := measureAt(vals, valNulls, hasVal, r)
			g.nullKey.add(v, ok)
			continue
		}
		v, ok := measureAt(vals, valNulls, hasVal, r)
		g.oa.add(keyOf(keys[r]), v, ok)
	}
}

// foldPayload merges one aggregated triple into g's own representation,
// used by Merge when the two sides disagree on representation (only
// possible when Mod differs between the merged aggregators).
func (g *GroupByInt64) ensureSpill() *int64HashAgg {
	if g.spill == nil {
		g.spill = newInt64HashAgg(64)
	}
	return g.spill
}

func (g *GroupByInt64) foldPayload(k int64, rows, count int64, sum float64) {
	g.ensureTable()
	if g.dense != nil {
		// In-domain reduced keys fold into the array; anything else
		// (negative keys, or raw keys from a foreign-Mod aggregator on
		// this misuse-tolerant path) spills to the OA table keyed by
		// the true key — the same union the old map merge produced.
		if k < 0 || k >= g.Mod {
			g.ensureSpill().foldPayload(k, rows, count, sum)
			return
		}
		g.dense.foldIdx(k, k, rows, count, sum)
		return
	}
	g.oa.foldPayload(k, rows, count, sum)
}

// eachGroup visits every live group; used by the cross-representation
// Merge path and by SortedRows.
func (g *GroupByInt64) eachGroup(fn func(k int64, rows, count int64, sum float64)) {
	if g.dense != nil {
		for _, s := range g.dense.occupied {
			idx := int64(s)
			fn(g.dense.keys[idx], g.dense.rows[idx], g.dense.counts[idx], g.dense.sums[idx])
		}
		if g.spill != nil {
			for _, s := range g.spill.occupied {
				i := uint64(s)
				fn(g.spill.keys[i], g.spill.rows[i], g.spill.counts[i], g.spill.sums[i])
			}
		}
		return
	}
	if g.oa != nil {
		for _, s := range g.oa.occupied {
			i := uint64(s)
			fn(g.oa.keys[i], g.oa.rows[i], g.oa.counts[i], g.oa.sums[i])
		}
	}
}

// Merge folds other's groups into g, single-threaded after workers
// finish (Parallel* may fan the table fold out — see parthash.go).
func (g *GroupByInt64) Merge(other *GroupByInt64) {
	g.ensureTable()
	other.ensureTable()
	switch {
	case g.dense != nil && other.dense != nil && g.Mod == other.Mod:
		mergeDense(g.dense, []*denseIntAgg{other.dense}, 1)
		if other.spill != nil {
			mergeOA(g.ensureSpill(), []*int64HashAgg{other.spill}, 1)
		}
	case g.oa != nil && other.oa != nil:
		mergeOA(g.oa, []*int64HashAgg{other.oa}, 1)
	default:
		other.eachGroup(g.foldPayload)
	}
	if other.hasNull {
		g.hasNull = true
		g.nullKey.merge(other.nullKey)
	}
}

// Reset clears all groups for reuse, keeping backing arrays.
func (g *GroupByInt64) Reset() {
	if g.dense != nil {
		g.dense.reset()
	}
	if g.spill != nil {
		g.spill.reset()
	}
	if g.oa != nil {
		g.oa.reset()
	}
	g.nullKey = groupState{}
	g.hasNull = false
}

// Len reports the number of groups, including the NULL group when present.
func (g *GroupByInt64) Len() int {
	n := 0
	if g.dense != nil {
		n = g.dense.used
		if g.spill != nil {
			n += g.spill.used
		}
	} else if g.oa != nil {
		n = g.oa.used
	}
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
	g.eachGroup(func(k int64, rows, count int64, sum float64) {
		gr := Int64Group{Key: k, Rows: rows, Count: count, Sum: sum}
		if count > 0 {
			gr.Avg = sum / float64(count)
		}
		out = append(out, gr)
	})
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
// dictionary-encoded: one map load per row resolves the key to a dense
// id (stores only for unseen keys) and payloads fold in place, so the
// per-row store-back, rehash, and groupState copy of a plain
// map[string]groupState disappear. ValCol < 0 means count-only.
type GroupByString struct {
	KeyCol int
	ValCol int

	dict    *strDictAgg
	nullKey groupState
	hasNull bool
}

// NewGroupByString builds a string group-by. valCol < 0 selects count-only.
func NewGroupByString(keyCol, valCol int) *GroupByString {
	return &GroupByString{KeyCol: keyCol, ValCol: valCol, dict: newStrDictAgg(1024)}
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
	if g.dict == nil {
		g.dict = newStrDictAgg(1024)
	}
	d := g.dict
	if b.Sel == nil {
		// Inlined dict.add hot path (same rationale as addDense: the
		// call costs ~1ns/row against a ~6ns budget): single map load
		// to the id, payloads folded in place; the unseen-key insert
		// stays outlined in addCold.
		for r, key := range keys {
			if len(keyNulls) > 0 && keyNulls[r] {
				g.hasNull = true
				v, ok := measureAt(vals, valNulls, hasVal, r)
				g.nullKey.add(v, ok)
				continue
			}
			v, ok := measureAt(vals, valNulls, hasVal, r)
			if id, found := d.ids[key]; found {
				p := &d.pay[id]
				p.rows++
				if ok {
					p.count++
					p.sum += v
				}
			} else {
				d.addCold(key, v, ok)
			}
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
		v, ok := measureAt(vals, valNulls, hasVal, r)
		if id, found := d.ids[keys[r]]; found {
			p := &d.pay[id]
			p.rows++
			if ok {
				p.count++
				p.sum += v
			}
		} else {
			d.addCold(keys[r], v, ok)
		}
	}
	return nil
}

// Merge folds other's groups into g, single-threaded after workers
// finish (Parallel* may fan the table fold out — see parthash.go).
func (g *GroupByString) Merge(other *GroupByString) {
	if other == nil || (other.dict == nil && !other.hasNull) {
		return
	}
	if g.dict == nil {
		n := 0
		if other.dict != nil {
			n = len(other.dict.keys)
		}
		g.dict = newStrDictAgg(n)
	}
	if other.dict != nil {
		mergeDict(g.dict, []*strDictAgg{other.dict}, 1)
	}
	if other.hasNull {
		g.hasNull = true
		g.nullKey.merge(other.nullKey)
	}
}

// Reset clears all groups for reuse.
func (g *GroupByString) Reset() {
	if g.dict != nil {
		g.dict.reset()
	}
	g.nullKey = groupState{}
	g.hasNull = false
}

// Len reports the number of groups, including the NULL group when present.
func (g *GroupByString) Len() int {
	n := 0
	if g.dict != nil {
		n = len(g.dict.keys)
	}
	if g.hasNull {
		n++
	}
	return n
}

// HasNullGroup reports whether any NULL key was seen.
func (g *GroupByString) HasNullGroup() bool { return g.hasNull }

// SortedRows returns groups ordered by key; the NULL group sorts last.
func (g *GroupByString) SortedRows() []StringGroup {
	out := make([]StringGroup, 0, g.Len())
	if g.dict != nil {
		for i, k := range g.dict.keys {
			p := &g.dict.pay[i]
			gr := StringGroup{Key: k, Rows: p.rows, Count: p.count, Sum: p.sum}
			if p.count > 0 {
				gr.Avg = p.sum / float64(p.count)
			}
			out = append(out, gr)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Key < out[j].Key })
	return out
}

// NullGroup returns the NULL-key group, or nil when no NULL key was seen.
func (g *GroupByString) NullGroup() *StringGroup {
	if !g.hasNull {
		return nil
	}
	gr := StringGroup{Rows: g.nullKey.rows, Count: g.nullKey.count, Sum: g.nullKey.sum}
	if g.nullKey.count > 0 {
		gr.Avg = g.nullKey.sum / float64(g.nullKey.count)
	}
	return &gr
}

// measureAt reads the value/validity at physical row r. hasVal is false
// for count-only aggregations (no value column): always invalid, zero.
// NULL values are invalid (skipped by sum/avg) but still counted in rows.
func measureAt(vals []float64, valNulls []bool, hasVal bool, r int) (float64, bool) {
	if !hasVal {
		return 0, false
	}
	if len(valNulls) > 0 && valNulls[r] {
		return 0, false
	}
	return vals[r], true
}

// Parallel aggregation: shard batches across workers, one local agg per
// worker, single-threaded merge at the end (the table fold fans out past
// the parthash.go gate). Workers never share a table, so there is no
// lock contention by construction; sharding is contiguous (not
// round-robin) to keep each worker on adjacent batches. workers <= 1
// runs sequentially. The input batches are read-only and may be shared;
// each agg copies group keys into its own table.

// ParallelGlobalAgg folds batches with workers local GlobalFloatAggs and
// merges them into one result.
func ParallelGlobalAgg(batches []*Batch, workers, col int) (*GlobalFloatAgg, error) {
	w := clampWorkers(len(batches), workers)
	locals := make([]*GlobalFloatAgg, w)
	for i := range locals {
		locals[i] = NewGlobalFloatAgg(col)
	}
	err := parallelRun(batches, w, func(i int, shard []*Batch) error {
		for _, b := range shard {
			if err := locals[i].Add(b); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	out := NewGlobalFloatAgg(col)
	for _, l := range locals {
		out.Merge(l)
	}
	return out, nil
}

// ParallelGroupByInt64 folds batches with workers local GroupByInt64s
// (keyCol, valCol, mod as in NewGroupByInt64) and merges them. Past the
// denseParBudget gate (mod × (W+1) tables) locals and output fall back
// to the OA table so a near-cap Mod with many workers cannot OOM.
func ParallelGroupByInt64(batches []*Batch, workers, keyCol, valCol int, mod int64) (*GroupByInt64, error) {
	w := clampWorkers(len(batches), workers)
	dense := wantDenseParallel(mod, w)
	newAgg := NewGroupByInt64
	if !dense && useDense(mod) {
		newAgg = newGroupByInt64OA
	}
	locals := make([]*GroupByInt64, w)
	for i := range locals {
		locals[i] = newAgg(keyCol, valCol, mod)
	}
	err := parallelRun(batches, w, func(i int, shard []*Batch) error {
		for _, b := range shard {
			if err := locals[i].Add(b); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	out := newAgg(keyCol, valCol, mod)
	out.ensureTable()
	for _, l := range locals {
		l.ensureTable()
	}
	if out.dense != nil {
		tables := make([]*denseIntAgg, 0, len(locals))
		var spills []*int64HashAgg
		for _, l := range locals {
			tables = append(tables, l.dense)
			if l.spill != nil {
				spills = append(spills, l.spill)
			}
		}
		mergeDense(out.dense, tables, w)
		if len(spills) > 0 {
			mergeOA(out.ensureSpill(), spills, w)
		}
	} else {
		tables := make([]*int64HashAgg, 0, len(locals))
		for _, l := range locals {
			tables = append(tables, l.oa)
		}
		mergeOA(out.oa, tables, w)
	}
	for _, l := range locals {
		if l.hasNull {
			out.hasNull = true
			out.nullKey.merge(l.nullKey)
		}
	}
	return out, nil
}

// ParallelGroupByString folds batches with workers local GroupByStrings
// and merges them.
func ParallelGroupByString(batches []*Batch, workers, keyCol, valCol int) (*GroupByString, error) {
	w := clampWorkers(len(batches), workers)
	locals := make([]*GroupByString, w)
	for i := range locals {
		locals[i] = NewGroupByString(keyCol, valCol)
	}
	err := parallelRun(batches, w, func(i int, shard []*Batch) error {
		for _, b := range shard {
			if err := locals[i].Add(b); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	out := NewGroupByString(keyCol, valCol)
	tables := make([]*strDictAgg, 0, len(locals))
	for _, l := range locals {
		if l.dict == nil {
			l.dict = newStrDictAgg(0)
		}
		tables = append(tables, l.dict)
		if l.hasNull {
			out.hasNull = true
			out.nullKey.merge(l.nullKey)
		}
	}
	mergeDict(out.dict, tables, w)
	return out, nil
}

// clampWorkers bounds the worker count to [1, len(batches)] (empty input
// still yields one local so the merged result is well-defined).
func clampWorkers(n, workers int) int {
	if workers < 1 {
		workers = 1
	}
	if n > 0 && workers > n {
		workers = n
	}
	return workers
}

// parallelRun executes run over contiguous shards, one goroutine per
// worker. Each worker touches only its own shard and local state; errs
// are collected after the done barrier (channel receive happens-before
// the read, so no mutex is needed). The first shard error is returned.
func parallelRun(batches []*Batch, workers int, run func(worker int, shard []*Batch) error) error {
	if workers <= 1 {
		return run(0, batches)
	}
	n := len(batches)
	errs := make([]error, workers)
	done := make(chan struct{}, workers)
	for w := 0; w < workers; w++ {
		lo := w * n / workers
		hi := (w + 1) * n / workers
		go func(w int, shard []*Batch) {
			errs[w] = run(w, shard)
			done <- struct{}{}
		}(w, batches[lo:hi])
	}
	for range workers {
		<-done
	}
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

func checkAggColumn(b *Batch, col int, kind engine.DataType, op string) error {
	if b == nil {
		return fmt.Errorf("vector: %s: nil batch", op)
	}
	if col < 0 || col >= len(b.Columns) {
		return fmt.Errorf("vector: %s: column %d out of range (have %d)", op, col, len(b.Columns))
	}
	if got := b.Columns[col].Type; got != kind {
		return fmt.Errorf("vector: %s: column %d is %v, want %v", op, col, got, kind)
	}
	if err := b.Validate(); err != nil {
		return fmt.Errorf("vector: %s: %w", op, err)
	}
	return nil
}
