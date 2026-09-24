package vector_test

import (
	"testing"

	"github.com/queryance/qpx/engine"
	"github.com/queryance/qpx/engine/vector"
)

var aggSchema = engine.Schema{Fields: []engine.Field{
	{Name: "id", Type: engine.Int64},
	{Name: "f", Type: engine.Float64},
	{Name: "t", Type: engine.String},
}}

// aggBatch builds a dense batch: id = base+1..base+n, f = float64(id)/10,
// t cycles a/b/c. n <= 0 yields an empty (but valid) batch.
func aggBatch(base, n int) *vector.Batch {
	b := vector.NewBatch(aggSchema, n)
	for i := 0; i < n; i++ {
		id := int64(base + i + 1)
		b.Columns[0].AppendInt(id)
		b.Columns[1].AppendFloat(float64(id) / 10.0)
		b.Columns[2].AppendString(string(rune('a' + (base+i)%3)))
		b.NumRows++
	}
	return b
}

func mustAdd(t *testing.T, add func(*vector.Batch) error, b *vector.Batch) {
	t.Helper()
	if err := add(b); err != nil {
		t.Fatalf("Add: %v", err)
	}
}

func TestGlobalAggDense(t *testing.T) {
	a := vector.NewGlobalFloatAgg(1)
	mustAdd(t, a.Add, aggBatch(0, 4)) // f = 0.1..0.4
	mustAdd(t, a.Add, aggBatch(4, 2)) // f = 0.5, 0.6
	r := a.Result()
	if r.Rows != 6 || r.Count != 6 {
		t.Fatalf("rows/count = %d/%d, want 6/6", r.Rows, r.Count)
	}
	if !closeEnough(r.Sum, 2.1) || !closeEnough(r.Avg, 0.35) {
		t.Fatalf("sum/avg = %v/%v, want 2.1/0.35", r.Sum, r.Avg)
	}
}

func TestGlobalAggNullsAndSel(t *testing.T) {
	b := aggBatch(0, 4)
	// NULL out f for id=2: rows still counts it, sum/avg skip it.
	b.Columns[1].Floats[1] = 0
	b.Columns[1].Nulls = make([]bool, 4)
	b.Columns[1].Nulls[1] = true
	b.Columns[1].HasNulls = true
	a := vector.NewGlobalFloatAgg(1)
	mustAdd(t, a.Add, b)
	r := a.Result()
	if r.Rows != 4 || r.Count != 3 {
		t.Fatalf("rows/count = %d/%d, want 4/3", r.Rows, r.Count)
	}
	if !closeEnough(r.Sum, 0.8) {
		t.Fatalf("sum = %v, want 0.8", r.Sum)
	}

	// Selection honored: keep id=1,3 only (rows 0,2).
	filt, err := vector.NewInt64Filter(0, vector.Gt, 0).Process(aggBatch(0, 4))
	if err != nil {
		t.Fatalf("filter: %v", err)
	}
	filt.Sel = []int32{0, 2}
	a2 := vector.NewGlobalFloatAgg(1)
	mustAdd(t, a2.Add, filt)
	if r := a2.Result(); r.Rows != 2 || r.Count != 2 || !closeEnough(r.Sum, 0.4) {
		t.Fatalf("selected result = %+v, want rows=2 count=2 sum=0.4", r)
	}

	// Empty batch is a no-op, empty agg averages zero.
	mustAdd(t, a2.Add, aggBatch(0, 0))
	if r := vector.NewGlobalFloatAgg(1).Result(); r.Avg != 0 || r.Rows != 0 {
		t.Fatalf("empty result = %+v, want zeros", r)
	}
}

func TestGlobalAggMerge(t *testing.T) {
	bs := []*vector.Batch{aggBatch(0, 100), aggBatch(100, 100), aggBatch(200, 50)}
	seq := vector.NewGlobalFloatAgg(1)
	for _, b := range bs {
		mustAdd(t, seq.Add, b)
	}
	par, err := vector.ParallelGlobalAgg(bs, 4, 1)
	if err != nil {
		t.Fatalf("ParallelGlobalAgg: %v", err)
	}
	if seq.Result() != par.Result() {
		t.Fatalf("parallel %+v != sequential %+v", par.Result(), seq.Result())
	}
	// workers=1 and workers > batches both collapse correctly.
	for _, w := range []int{1, 0, 99} {
		p, err := vector.ParallelGlobalAgg(bs, w, 1)
		if err != nil {
			t.Fatalf("workers=%d: %v", w, err)
		}
		if p.Result() != seq.Result() {
			t.Fatalf("workers=%d: %+v != %+v", w, p.Result(), seq.Result())
		}
	}
}

func TestGlobalAggRejects(t *testing.T) {
	a := vector.NewGlobalFloatAgg(1)
	if err := a.Add(nil); err == nil {
		t.Fatal("nil batch must fail")
	}
	if err := vector.NewGlobalFloatAgg(0).Add(aggBatch(0, 2)); err == nil {
		t.Fatal("int column as float measure must fail")
	}
	if err := vector.NewGlobalFloatAgg(9).Add(aggBatch(0, 2)); err == nil {
		t.Fatal("out-of-range column must fail")
	}
	if _, err := vector.ParallelGlobalAgg([]*vector.Batch{aggBatch(0, 2)}, 2, 9); err == nil {
		t.Fatal("parallel must propagate Add errors")
	}
}

func TestGroupByInt64Raw(t *testing.T) {
	// ids 1..6 with t cycling a/b/c above; group raw id%2 via Mod=2 is
	// covered separately — here group by id directly (6 groups).
	g := vector.NewGroupByInt64(0, 1, 0)
	mustAdd(t, g.Add, aggBatch(0, 6))
	rows := g.SortedRows()
	if len(rows) != 6 || g.Len() != 6 {
		t.Fatalf("groups = %d, want 6", len(rows))
	}
	if rows[0].Key != 1 || rows[0].Rows != 1 || rows[0].Count != 1 || rows[0].Sum != 0.1 {
		t.Fatalf("first group = %+v", rows[0])
	}
	if g.HasNullGroup() || g.NullGroup() != nil {
		t.Fatal("dense keys must not produce a NULL group")
	}
}

func TestGroupByInt64Mod(t *testing.T) {
	// ids 1..100, mod 4: groups 0..3 (id%4: 25 each).
	g := vector.NewGroupByInt64(0, 1, 4)
	mustAdd(t, g.Add, aggBatch(0, 100))
	rows := g.SortedRows()
	if len(rows) != 4 {
		t.Fatalf("groups = %d, want 4", len(rows))
	}
	var totalRows, totalCount int64
	for _, gr := range rows {
		if gr.Rows != 25 || gr.Count != 25 {
			t.Fatalf("group %+v: want 25/25 rows/count", gr)
		}
		// Avg must equal Sum/Count exactly (same division PG does).
		if gr.Avg != gr.Sum/float64(gr.Count) {
			t.Fatalf("group %d: avg %v != sum/count", gr.Key, gr.Avg)
		}
		totalRows += gr.Rows
		totalCount += gr.Count
	}
	if totalRows != 100 || totalCount != 100 {
		t.Fatalf("totals = %d/%d, want 100/100", totalRows, totalCount)
	}
	// Group 1 holds ids 1,5,...,97: sum = (1+5+...+97)/10 = 122.5.
	if !closeEnough(rows[1].Sum, 122.5) {
		t.Fatalf("group 1 sum = %v, want 122.5", rows[1].Sum)
	}
}

func TestGroupByInt64Nulls(t *testing.T) {
	b := aggBatch(0, 4) // ids 1..4
	// NULL key on id=1, NULL value on id=2.
	b.Columns[0].Ints[0] = 0
	b.Columns[0].Nulls = make([]bool, 4)
	b.Columns[0].Nulls[0] = true
	b.Columns[0].HasNulls = true
	b.Columns[1].Floats[1] = 0
	b.Columns[1].Nulls = make([]bool, 4)
	b.Columns[1].Nulls[1] = true
	b.Columns[1].HasNulls = true

	g := vector.NewGroupByInt64(0, 1, 0)
	mustAdd(t, g.Add, b)
	if !g.HasNullGroup() {
		t.Fatal("NULL key must form a group")
	}
	ng := g.NullGroup()
	if ng.Rows != 1 || ng.Count != 1 || ng.Sum != 0.1 {
		t.Fatalf("null group = %+v, want rows=1 count=1 sum=0.1", ng)
	}
	// id=2 group: row counted, NULL value skipped.
	var found bool
	for _, gr := range g.SortedRows() {
		if gr.Key == 2 {
			found = true
			if gr.Rows != 1 || gr.Count != 0 || gr.Sum != 0 || gr.Avg != 0 {
				t.Fatalf("null-value group = %+v", gr)
			}
		}
	}
	if !found {
		t.Fatal("group 2 missing")
	}
	if g.Len() != 4 { // ids 2,3,4 + NULL
		t.Fatalf("Len = %d, want 4", g.Len())
	}
}

func TestGroupByInt64CountOnlyAndSel(t *testing.T) {
	g := vector.NewGroupByInt64(0, -1, 2)
	b := aggBatch(0, 6)
	b.Sel = []int32{0, 1, 2, 3} // ids 1..4
	mustAdd(t, g.Add, b)
	rows := g.SortedRows()
	if len(rows) != 2 {
		t.Fatalf("groups = %d, want 2", len(rows))
	}
	for _, gr := range rows {
		if gr.Rows != 2 || gr.Count != 0 || gr.Sum != 0 || gr.Avg != 0 {
			t.Fatalf("count-only group = %+v", gr)
		}
	}
}

func TestGroupByInt64Merge(t *testing.T) {
	bs := []*vector.Batch{aggBatch(0, 100), aggBatch(100, 100), aggBatch(250, 40)}
	seq := vector.NewGroupByInt64(0, 1, 7)
	for _, b := range bs {
		mustAdd(t, seq.Add, b)
	}
	par, err := vector.ParallelGroupByInt64(bs, 3, 0, 1, 7)
	if err != nil {
		t.Fatalf("ParallelGroupByInt64: %v", err)
	}
	srows, prows := seq.SortedRows(), par.SortedRows()
	if len(srows) != len(prows) {
		t.Fatalf("group counts %d != %d", len(prows), len(srows))
	}
	for i := range srows {
		if !sameInt64Group(srows[i], prows[i]) {
			t.Fatalf("group %d: parallel %+v != sequential %+v", i, prows[i], srows[i])
		}
	}
	if (seq.NullGroup() == nil) != (par.NullGroup() == nil) {
		t.Fatal("NULL group presence differs after merge")
	}
	// Merge of empty into non-empty is identity.
	empty := vector.NewGroupByInt64(0, 1, 7)
	seq.Merge(empty)
	if len(seq.SortedRows()) != len(srows) {
		t.Fatal("merge with empty changed groups")
	}
	seq.Reset()
	if seq.Len() != 0 || seq.HasNullGroup() {
		t.Fatal("Reset must clear all groups")
	}
}

func TestGroupByString(t *testing.T) {
	// t cycles a/b/c over ids 1..9: 3 rows each.
	g := vector.NewGroupByString(2, 1)
	mustAdd(t, g.Add, aggBatch(0, 9))
	rows := g.SortedRows()
	if len(rows) != 3 {
		t.Fatalf("groups = %d, want 3", len(rows))
	}
	if rows[0].Key != "a" || rows[0].Rows != 3 || rows[0].Count != 3 {
		t.Fatalf("group a = %+v", rows[0])
	}
	// a holds ids 1,4,7: sum = 1.2.
	if !closeEnough(rows[0].Sum, 1.2) {
		t.Fatalf("group a sum = %v, want 1.2", rows[0].Sum)
	}

	// NULL key + NULL value.
	b := aggBatch(0, 3)
	b.Columns[2].Strings[0] = ""
	b.Columns[2].Nulls = make([]bool, 3)
	b.Columns[2].Nulls[0] = true
	b.Columns[2].HasNulls = true
	g2 := vector.NewGroupByString(2, 1)
	mustAdd(t, g2.Add, b)
	if !g2.HasNullGroup() || g2.Len() != 3 {
		t.Fatalf("want 3 groups incl NULL, got %d", g2.Len())
	}

	// Parallel equals sequential.
	bs := []*vector.Batch{aggBatch(0, 500), aggBatch(500, 500)}
	seq := vector.NewGroupByString(2, 1)
	for _, b := range bs {
		mustAdd(t, seq.Add, b)
	}
	par, err := vector.ParallelGroupByString(bs, 4, 2, 1)
	if err != nil {
		t.Fatalf("ParallelGroupByString: %v", err)
	}
	srows, prows := seq.SortedRows(), par.SortedRows()
	if len(srows) != len(prows) {
		t.Fatalf("group counts %d != %d", len(prows), len(srows))
	}
	for i := range srows {
		if !sameStringGroup(srows[i], prows[i]) {
			t.Fatalf("group %d: parallel %+v != sequential %+v", i, prows[i], srows[i])
		}
	}

	// Count-only string path.
	gc := vector.NewGroupByString(2, -1)
	mustAdd(t, gc.Add, aggBatch(0, 6))
	for _, gr := range gc.SortedRows() {
		if gr.Rows != 2 || gr.Count != 0 {
			t.Fatalf("count-only group = %+v", gr)
		}
	}
}

func TestParallelGroupByFilteredNulls(t *testing.T) {
	// The real filtered-agg shape: Filter-produced Sel plus NULL keys and
	// NULL values, merged across workers. Dense-only parallel tests miss
	// the Sel-indexed Add loops and the sharded null-key merge.
	mk := func() []*vector.Batch {
		bs := []*vector.Batch{aggBatch(0, 120), aggBatch(120, 120), aggBatch(240, 100)}
		f := vector.NewFloat64Filter(1, vector.Gt, 5.0)
		for _, b := range bs {
		n := b.NumRows
		b.Columns[0].Nulls = make([]bool, n)
		b.Columns[1].Nulls = make([]bool, n)
		b.Columns[2].Nulls = make([]bool, n)
		for r := 0; r < n; r++ {
			if r%17 == 0 {
				b.Columns[0].Nulls[r] = true
				b.Columns[0].HasNulls = true
			}
			if r%13 == 0 {
				b.Columns[1].Nulls[r] = true
				b.Columns[1].HasNulls = true
			}
			if r%19 == 0 {
				b.Columns[2].Nulls[r] = true
				b.Columns[2].HasNulls = true
			}
		}
		if _, err := f.Process(b); err != nil {
			t.Fatalf("filter: %v", err)
		}
		if b.Sel == nil {
			t.Fatalf("fixture must be Sel-filtered")
		}
	}
		return bs
	}

	// Int64 keys (mod 4 forces cross-worker group merges) with NULL keys.
	bs := mk()
	seq := vector.NewGroupByInt64(0, 1, 4)
	for _, b := range bs {
		mustAdd(t, seq.Add, b)
	}
	if !seq.HasNullGroup() {
		t.Fatal("fixture must retain a NULL int64 key after filtering")
	}
	par, err := vector.ParallelGroupByInt64(bs, 3, 0, 1, 4)
	if err != nil {
		t.Fatalf("ParallelGroupByInt64: %v", err)
	}
	srows, prows := seq.SortedRows(), par.SortedRows()
	if len(srows) != len(prows) {
		t.Fatalf("group counts %d != %d", len(prows), len(srows))
	}
	for i := range srows {
		if !sameInt64Group(srows[i], prows[i]) {
			t.Fatalf("group %d: parallel %+v != sequential %+v", i, prows[i], srows[i])
		}
	}
	if (seq.NullGroup() == nil) != (par.NullGroup() == nil) {
		t.Fatal("NULL group presence differs after merge")
	} else if seq.NullGroup() != nil && !sameInt64Group(*seq.NullGroup(), *par.NullGroup()) {
		t.Fatalf("NULL group: parallel %+v != sequential %+v", par.NullGroup(), seq.NullGroup())
	}

	// String keys with NULL keys over the same filtered batches.
	sseq := vector.NewGroupByString(2, 1)
	for _, b := range bs {
		mustAdd(t, sseq.Add, b)
	}
	if !sseq.HasNullGroup() {
		t.Fatal("fixture must retain a NULL string key after filtering")
	}
	spar, err := vector.ParallelGroupByString(bs, 3, 2, 1)
	if err != nil {
		t.Fatalf("ParallelGroupByString: %v", err)
	}
	ssrows, sprows := sseq.SortedRows(), spar.SortedRows()
	if len(ssrows) != len(sprows) {
		t.Fatalf("group counts %d != %d", len(sprows), len(ssrows))
	}
	for i := range ssrows {
		if !sameStringGroup(ssrows[i], sprows[i]) {
			t.Fatalf("group %d: parallel %+v != sequential %+v", i, sprows[i], ssrows[i])
		}
	}
	if (sseq.NullGroup() == nil) != (spar.NullGroup() == nil) {
		t.Fatal("string NULL group presence differs after merge")
	}
}

func TestAggRejects(t *testing.T) {	b := aggBatch(0, 2)
	if err := vector.NewGroupByInt64(2, 1, 0).Add(b); err == nil {
		t.Fatal("string key as int64 must fail")
	}
	if err := vector.NewGroupByInt64(0, 2, 0).Add(b); err == nil {
		t.Fatal("string value as float must fail")
	}
	if err := vector.NewGroupByString(0, 1).Add(b); err == nil {
		t.Fatal("int key as string must fail")
	}
	if err := vector.NewGroupByInt64(9, 1, 0).Add(b); err == nil {
		t.Fatal("out-of-range key must fail")
	}
	if err := vector.NewGroupByInt64(0, 1, 0).Add(nil); err == nil {
		t.Fatal("nil batch must fail")
	}
}

// closeEnough compares floats with a relative epsilon: parallel merges
// sum partitions in a different order, so the last ulp may differ from
// a sequential fold. Parity against PG uses absolute tolerances (see
// postgres/vector_agg_test.go); here we only assert merge order does not
// change the result beyond float noise.
func closeEnough(a, b float64) bool {
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

func sameInt64Group(a, b vector.Int64Group) bool {
	return a.Key == b.Key && a.Rows == b.Rows && a.Count == b.Count &&
		closeEnough(a.Sum, b.Sum) && closeEnough(a.Avg, b.Avg)
}

func sameStringGroup(a, b vector.StringGroup) bool {
	return a.Key == b.Key && a.Rows == b.Rows && a.Count == b.Count &&
		closeEnough(a.Sum, b.Sum) && closeEnough(a.Avg, b.Avg)
}
