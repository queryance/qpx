// Oracle tests for the ht.go tables and their agg.go facades.
//
// Strategy: a dead-simple map-based reference (independent of the tables
// under test) folds the same deterministic fixtures; the tables must
// match it group-for-group. Single-threaded folds use the same row order
// as the reference, so sums compare EXACTLY; merged (parallel) results
// fold partitions in a different order and compare with a 1e-9 relative
// epsilon — the same allowance agg_test.go grants. Parity against PG
// GROUP BY uses an absolute 1e-6 epsilon instead (see
// postgres/vector_agg_test.go floatEq): PG folds in its own order, so
// the last ulp may differ there too.
package vector

import (
	"testing"

	"github.com/queryance/qpx/engine"
)

var htSchema = engine.Schema{Fields: []engine.Field{
	{Name: "id", Type: engine.Int64},
	{Name: "f", Type: engine.Float64},
	{Name: "t", Type: engine.String},
}}

type refGroup struct {
	rows  int64
	count int64
	sum   float64
}

// lcg is a deterministic key stream (no test-time rand dependency).
type lcg struct{ s uint64 }

func (l *lcg) next() uint64 {
	l.s = l.s*6364136223846793005 + 1442695040888963407
	return l.s >> 33
}

// htFixture builds a batch with adversarial-but-deterministic content:
// negative and duplicate int keys, NULL keys/values on fixed cycles,
// empty/long strings. n <= 0 yields a valid empty batch.
func htFixture(n int) *Batch {
	b := NewBatch(htSchema, htMax(n, 1))
	r := &lcg{s: 0x12345}
	for i := 0; i < n; i++ {
		k := int64(r.next()%5000) - 1000 // [-1000, 4000): negatives + dups
		b.Columns[0].AppendInt(k)
		b.Columns[1].AppendFloat(float64(int64(r.next()%100000)) / 100000.0)
		switch i % 7 {
		case 0:
			b.Columns[2].AppendString("")
		case 1:
			b.Columns[2].AppendString("a-much-longer-key-to-force-heap-compare-paths-" + string(rune('a'+i%26)))
		default:
			b.Columns[2].AppendString("txt_" + string(rune('a'+i%5)))
		}
		b.NumRows++
	}
	if n > 0 {
		nk := make([]bool, n)
		nv := make([]bool, n)
		nt := make([]bool, n)
		for i := 0; i < n; i++ {
			if i%11 == 0 {
				nk[i] = true
			}
			if i%13 == 0 {
				nv[i] = true
			}
			if i%17 == 0 {
				nt[i] = true
			}
		}
		b.Columns[0].Nulls = nk
		b.Columns[0].HasNulls = true
		b.Columns[1].Nulls = nv
		b.Columns[1].HasNulls = true
		b.Columns[2].Nulls = nt
		b.Columns[2].HasNulls = true
	}
	return b
}

// refFoldInt folds rows [0,n) (or sel) into a reference map, applying
// mod like GroupByInt64. Returns groups + the NULL-key group.
func refFoldInt(keys []int64, keyNulls []bool, vals []float64, valNulls []bool, hasVal bool, mod int64, sel []int32, n int) (map[int64]refGroup, refGroup, bool) {
	m := make(map[int64]refGroup)
	var ng refGroup
	var hasNg bool
	one := func(r int) {
		if len(keyNulls) > 0 && keyNulls[r] {
			hasNg = true
			ng.rows++
			if hasVal && !(len(valNulls) > 0 && valNulls[r]) {
				ng.count++
				ng.sum += vals[r]
			}
			return
		}
		k := keys[r]
		if mod != 0 {
			k = k % mod
		}
		g := m[k]
		g.rows++
		if hasVal && !(len(valNulls) > 0 && valNulls[r]) {
			g.count++
			g.sum += vals[r]
		}
		m[k] = g
	}
	if sel == nil {
		for r := 0; r < n; r++ {
			one(r)
		}
	} else {
		for _, s := range sel {
			one(int(s))
		}
	}
	return m, ng, hasNg
}

func stridedSel(n, stride, off int) []int32 {
	var out []int32
	for i := off; i < n; i += stride {
		out = append(out, int32(i))
	}
	return out
}

func checkIntGroups(t *testing.T, got []Int64Group, want map[int64]refGroup) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("groups = %d, want %d", len(got), len(want))
	}
	for _, gr := range got {
		w, ok := want[gr.Key]
		if !ok {
			t.Fatalf("unexpected group %d", gr.Key)
		}
		if gr.Rows != w.rows || gr.Count != w.count || gr.Sum != w.sum {
			t.Fatalf("group %d = %+v, want %+v", gr.Key, gr, w)
		}
		if w.count > 0 && gr.Avg != w.sum/float64(w.count) {
			t.Fatalf("group %d avg = %v, want %v", gr.Key, gr.Avg, w.sum/float64(w.count))
		}
	}
}

// TestTablesVsOracle folds identical fixtures through the OA table
// (Mod=0), the dense table (small + huge mod, incl. negative keys), and
// the dict table, comparing each against the map reference.
func TestTablesVsOracle(t *testing.T) {
	fix := htFixture(3000)
	keys := fix.Columns[0].Ints
	kn := fix.Columns[0].Nulls
	vals := fix.Columns[1].Floats
	vn := fix.Columns[1].Nulls
	strs := fix.Columns[2].Strings
	sn := fix.Columns[2].Nulls

	sels := map[string][]int32{"dense": nil, "stride3": stridedSel(3000, 3, 1)}
	for name, sel := range sels {
		n := 3000
		if sel != nil {
			n = len(sel)
		}
		// OA path (raw keys, negatives included).
		want, wng, hasNg := refFoldInt(keys, kn, vals, vn, true, 0, sel, n)
		_ = n
		g := NewGroupByInt64(0, 1, 0)
		b := htFixture(3000)
		b.Sel = sel
		if err := g.Add(b); err != nil {
			t.Fatalf("%s oa Add: %v", name, err)
		}
		if g.oa == nil {
			t.Fatalf("%s: Mod=0 must select the OA table", name)
		}
		checkIntGroups(t, g.SortedRows(), want)
		if g.HasNullGroup() != hasNg {
			t.Fatalf("%s: null group = %v, want %v", name, g.HasNullGroup(), hasNg)
		}
		if hasNg {
			ng := g.NullGroup()
			if ng.Rows != wng.rows || ng.Count != wng.count || ng.Sum != wng.sum {
				t.Fatalf("%s: null group = %+v, want %+v", name, ng, wng)
			}
		}

		// Dense path (mod 64 forces heavy collisions + negative keys).
		wantD, wngD, hasNgD := refFoldInt(keys, kn, vals, vn, true, 64, sel, len(selOrFull(sel, 3000)))
		gd := NewGroupByInt64(0, 1, 64)
		bd := htFixture(3000)
		bd.Sel = sel
		if err := gd.Add(bd); err != nil {
			t.Fatalf("%s dense Add: %v", name, err)
		}
		if gd.dense == nil {
			t.Fatalf("%s: mod=64 must select the dense table", name)
		}
		checkIntGroups(t, gd.SortedRows(), wantD)
		if gd.HasNullGroup() != hasNgD {
			t.Fatalf("%s dense: null group = %v, want %v", name, gd.HasNullGroup(), hasNgD)
		}
		if hasNgD {
			ng := gd.NullGroup()
			if ng.Rows != wngD.rows || ng.Count != wngD.count || ng.Sum != wngD.sum {
				t.Fatalf("%s dense: null group = %+v, want %+v", name, ng, wngD)
			}
		}

		// Dict path vs string reference.
		wantS := make(map[string]refGroup)
		var sng refGroup
		var hasSng bool
		one := func(r int) {
			if len(sn) > 0 && sn[r] {
				hasSng = true
				sng.rows++
				if !(len(vn) > 0 && vn[r]) {
					sng.count++
					sng.sum += vals[r]
				}
				return
			}
			gr := wantS[strs[r]]
			gr.rows++
			if !(len(vn) > 0 && vn[r]) {
				gr.count++
				gr.sum += vals[r]
			}
			wantS[strs[r]] = gr
		}
		if sel == nil {
			for r := 0; r < 3000; r++ {
				one(r)
			}
		} else {
			for _, s := range sel {
				one(int(s))
			}
		}
		gs := NewGroupByString(2, 1)
		bs := htFixture(3000)
		bs.Sel = sel
		if err := gs.Add(bs); err != nil {
			t.Fatalf("%s dict Add: %v", name, err)
		}
		rows := gs.SortedRows()
		if len(rows) != len(wantS) {
			t.Fatalf("%s dict: groups = %d, want %d", name, len(rows), len(wantS))
		}
		for _, gr := range rows {
			w, ok := wantS[gr.Key]
			if !ok {
				t.Fatalf("%s dict: unexpected group %q", name, gr.Key)
			}
			if gr.Rows != w.rows || gr.Count != w.count || gr.Sum != w.sum {
				t.Fatalf("%s dict: group %q = %+v, want %+v", name, gr.Key, gr, w)
			}
		}
		if gs.HasNullGroup() != hasSng {
			t.Fatalf("%s dict: null group = %v, want %v", name, gs.HasNullGroup(), hasSng)
		}
	}
}

func selOrFull(sel []int32, n int) []int32 {
	if sel == nil {
		out := make([]int32, n)
		for i := range out {
			out[i] = int32(i)
		}
		return out
	}
	return sel
}

// TestZeroValueAgg covers aggregate use without its constructor (the old
// map path built its table lazily; ensureTable preserves that).
func TestZeroValueAgg(t *testing.T) {
	var g GroupByInt64
	g.KeyCol, g.ValCol, g.Mod = 0, 1, 128
	if err := g.Add(htFixture(100)); err != nil {
		t.Fatalf("zero-value Add: %v", err)
	}
	if g.Len() == 0 {
		t.Fatal("zero-value agg folded nothing")
	}
	var s GroupByString
	s.KeyCol, s.ValCol = 2, 1
	if err := s.Add(htFixture(100)); err != nil {
		t.Fatalf("zero-value string Add: %v", err)
	}
	if s.Len() == 0 {
		t.Fatal("zero-value string agg folded nothing")
	}
}

// TestDenseMatchesOA cross-checks both int64 representations on positive
// keys (where dense biasing is identity) with count-only measures.
func TestDenseMatchesOA(t *testing.T) {
	b := htFixture(2000)
	for i := range b.Columns[0].Nulls {
		b.Columns[0].Nulls[i] = false // drop NULL keys: dense/OA agree trivially there
	}
	b.Columns[0].HasNulls = true
	oa := NewGroupByInt64(0, -1, 0)
	d := NewGroupByInt64(0, -1, 4096)
	if err := oa.Add(b); err != nil {
		t.Fatalf("oa Add: %v", err)
	}
	if err := d.Add(b); err != nil {
		t.Fatalf("dense Add: %v", err)
	}
	orows, drows := oa.SortedRows(), d.SortedRows()
	// Different key domains (raw vs mod-4096): compare per-group rows
	// against the reference instead of each other.
	want, _, _ := refFoldInt(b.Columns[0].Ints, b.Columns[0].Nulls, nil, nil, false, 0, nil, 2000)
	checkIntGroups(t, orows, want)
	wantD, _, _ := refFoldInt(b.Columns[0].Ints, b.Columns[0].Nulls, nil, nil, false, 4096, nil, 2000)
	checkIntGroups(t, drows, wantD)
}

// TestMergeEquivalence checks sequential Merge and the radix parallel
// merge agree with the single-shot fold, incl. Reset+reuse and the
// cross-representation (Mod-mismatch) fallback.
func TestMergeEquivalence(t *testing.T) {
	mk := func(mod int64) []*Batch {
		return []*Batch{htFixture(1500), htFixture(1500), htFixture(0)}
	}
	for _, mod := range []int64{0, 64, 100000} {
		bs := mk(mod)
		seq := NewGroupByInt64(0, 1, mod)
		for _, b := range bs {
			if err := seq.Add(b); err != nil {
				t.Fatalf("mod=%d seq Add: %v", mod, err)
			}
		}
		// Sequential two-way merge.
		a := NewGroupByInt64(0, 1, mod)
		if err := a.Add(bs[0]); err != nil {
			t.Fatal(err)
		}
		b2 := NewGroupByInt64(0, 1, mod)
		if err := b2.Add(bs[1]); err != nil {
			t.Fatal(err)
		}
		a.Merge(b2)
		compareIntAgg(t, mod, seq, a)
		// Parallel merge incl. -race coverage of the barrier paths.
		par, err := ParallelGroupByInt64(bs, 4, 0, 1, mod)
		if err != nil {
			t.Fatalf("mod=%d parallel: %v", mod, err)
		}
		compareIntAgg(t, mod, seq, par)
		// Merge of empty into non-empty is identity; Reset clears.
		empty := NewGroupByInt64(0, 1, mod)
		n := len(seq.SortedRows())
		seq.Merge(empty)
		if len(seq.SortedRows()) != n {
			t.Fatalf("mod=%d: merge with empty changed groups", mod)
		}
		seq.Reset()
		if seq.Len() != 0 || seq.HasNullGroup() {
			t.Fatalf("mod=%d: Reset must clear", mod)
		}
		// Reuse after Reset refolds identically.
		for _, b := range bs {
			if err := seq.Add(b); err != nil {
				t.Fatal(err)
			}
		}
		compareIntAgg(t, mod, a, seq)
	}
	// Cross-representation merge (different Mod) takes the generic path.
	x := NewGroupByInt64(0, 1, 64)
	y := NewGroupByInt64(0, 1, 0)
	if err := x.Add(htFixture(500)); err != nil {
		t.Fatal(err)
	}
	if err := y.Add(htFixture(500)); err != nil {
		t.Fatal(err)
	}
	x.Merge(y) // must not crash; groups union (domains differ, no exact oracle)
	if x.Len() == 0 {
		t.Fatal("cross-representation merge dropped everything")
	}

	// String parallel merge vs sequential (relative epsilon: fold order).
	sbs := []*Batch{htFixture(1200), htFixture(1200)}
	sseq := NewGroupByString(2, 1)
	for _, b := range sbs {
		if err := sseq.Add(b); err != nil {
			t.Fatal(err)
		}
	}
	spar, err := ParallelGroupByString(sbs, 4, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	sm, spm := rowsByString(sseq), rowsByString(spar)
	if len(sm) != len(spm) {
		t.Fatalf("string parallel groups = %d, want %d", len(spm), len(sm))
	}
	for k, w := range sm {
		p := spm[k]
		if p.Rows != w.Rows || p.Count != w.Count || !closeRel(p.Sum, w.Sum) {
			t.Fatalf("string group %q: parallel %+v != sequential %+v", k, p, w)
		}
	}
}

// TestParallelMergeGate forces the radix parallel merge (many groups ×
// workers) and checks exactness against sequential merge.
func TestParallelMergeGate(t *testing.T) {
	if wantParallelMerge(1000, 8) {
		t.Fatal("1K entries must stay sequential")
	}
	if wantParallelMerge(100000, 2) {
		t.Fatal("<4 workers must stay sequential")
	}
	if !wantParallelMerge(200000, 8) {
		t.Fatal("200K entries × 8 workers must fan out")
	}
	// 100K distinct raw keys × 8 locals → parallel OA merge.
	var locals []*int64HashAgg
	var tables []*int64HashAgg
	var total int
	for w := 0; w < 8; w++ {
		lt := newInt64HashAgg(1024)
		r := &lcg{s: uint64(w + 1)}
		for i := 0; i < 40000; i++ {
			lt.add(int64(r.next()%100000), float64(i)/10.0, true)
		}
		locals = append(locals, lt)
		tables = append(tables, lt)
		total += lt.used
	}
	t.Logf("total occupied = %d", total)
	// OA merge is sequential by design (see parthash.go): the workers
	// argument must not change the result — same fold order means EXACT
	// equality, sums included.
	seq := newInt64HashAgg(1024)
	mergeOA(seq, tables, 1)
	par := newInt64HashAgg(1024)
	mergeOA(par, tables, 8)
	if seq.used != par.used {
		t.Fatalf("OA merge groups = %d, want %d", par.used, seq.used)
	}
	for _, s := range seq.occupied {
		i := uint64(s)
		h := mixInt64(seq.keys[i])
		j, found := par.slotFor(h, seq.keys[i])
		if !found || par.rows[j] != seq.rows[i] || par.counts[j] != seq.counts[i] || par.sums[j] != seq.sums[i] {
			t.Fatalf("OA merge differs at key %d", seq.keys[i])
		}
	}
	_ = locals
	// Dense parallel merge over mod 100K.
	var dlocals []*denseIntAgg
	for w := 0; w < 8; w++ {
		d := newDenseIntAgg(100000)
		r := &lcg{s: uint64(w + 100)}
		for i := 0; i < 40000; i++ {
			k := int64(r.next() % 100000)
			d.addIdx(k, k, float64(i)/10.0, true)
		}
		dlocals = append(dlocals, d)
	}
	dseq := newDenseIntAgg(100000)
	mergeDense(dseq, dlocals, 1)
	dpar := newDenseIntAgg(100000)
	mergeDense(dpar, dlocals, 8)
	if dseq.used != dpar.used {
		t.Fatalf("parallel dense merge groups = %d, want %d", dpar.used, dseq.used)
	}
	for _, s := range dseq.occupied {
		idx := int64(s)
		if !dpar.present[idx] || dpar.rows[idx] != dseq.rows[idx] ||
			dpar.counts[idx] != dseq.counts[idx] || !closeRel(dpar.sums[idx], dseq.sums[idx]) {
			t.Fatalf("parallel dense merge differs at idx %d", idx)
		}
	}
}

func compareIntAgg(t *testing.T, mod int64, want, got *GroupByInt64) {
	t.Helper()
	wm, pm := rowsByInt(want), rowsByInt(got)
	if len(wm) != len(pm) {
		t.Fatalf("mod=%d: groups = %d, want %d", mod, len(pm), len(wm))
	}
	for k, w := range wm {
		p := pm[k]
		if p.Rows != w.Rows || p.Count != w.Count || !closeRel(p.Sum, w.Sum) {
			t.Fatalf("mod=%d group %d: got %+v, want %+v", mod, k, p, w)
		}
	}
	if want.HasNullGroup() != got.HasNullGroup() {
		t.Fatalf("mod=%d: null presence differs", mod)
	}
	if want.HasNullGroup() {
		wn, gn := want.NullGroup(), got.NullGroup()
		if wn.Rows != gn.Rows || wn.Count != gn.Count || !closeRel(wn.Sum, gn.Sum) {
			t.Fatalf("mod=%d: null group got %+v, want %+v", mod, wn, gn)
		}
	}
}

func rowsByInt(g *GroupByInt64) map[int64]Int64Group {
	m := make(map[int64]Int64Group)
	for _, gr := range g.SortedRows() {
		m[gr.Key] = gr
	}
	return m
}

func rowsByString(g *GroupByString) map[string]StringGroup {
	m := make(map[string]StringGroup)
	for _, gr := range g.SortedRows() {
		m[gr.Key] = gr
	}
	return m
}

// closeRel is the merge-order allowance: partitions sum in a different
// order than a sequential fold, so the last ulp may differ. PG parity
// uses floatEq (absolute 1e-6) instead — see postgres/vector_agg_test.go.
func closeRel(a, b float64) bool {
	if a == b {
		return true
	}
	d := a - b
	if d < 0 {
		d = -d
	}
	m := a
	if m < 0 {
		m = -m
	}
	if b < 0 {
		m += -b
	} else {
		m += b
	}
	return d <= 1e-9*m
}

func htMax(a, b int) int {
	if a > b {
		return a
	}
	return b
}
