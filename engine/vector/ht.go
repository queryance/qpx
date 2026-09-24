// DuckDB-inspired hash tables for group-by aggregation.
//
// Mapping to duckdb/duckdb@main (see package doc in agg.go for the QPX
// shape these replace):
//
//   - int64HashAgg: src/execution/aggregate_hashtable.cpp. Single
//     power-of-two table, slot = hash & (cap-1), contiguous key/payload
//     arrays, no Go map on the hot path. Deviation: DuckDB stores rows
//     in PartitionedTupleData and 8 B (salt+pointer) entries with an odd
//     salt-derived probe stride; we store keys inline in SoA arrays and
//     probe linearly (+1). Linear keeps the probe sequential (prefetch
//     friendly) and is optimal at our ≤67% load factor; the salt trick
//     matters most for wide rows with an extra key-compare indirection,
//     which we do not have.
//   - denseIntAgg: src/execution/perfect_aggregate_hashtable.cpp. When
//     the key domain is small and known (Mod path: index IS the reduced
//     key), there is no hash, no probe, no compare — Add is one indexed
//     payload update. DuckDB gates this on stats min/max and
//     total_bits ≤ 12; our Mod bound IS the range proof, so no stats
//     pass is needed.
//   - strDictAgg: dictionary encoding for string keys. The profile
//     blames string hashing (hash+compare ≈ 23.5% of string Add,
//     map+hash+equal ≈ 57%): hashing each row twice (load + store-back)
//     plus a 24 B groupState copy per row. The dict hashes each row once
//     (single map load to an id; stores happen only for unseen keys)
//     and folds payloads in place in SoA arrays — no value copy, no
//     store-back rehash.
//
// All tables keep per-row payload updates in place (rows/counts/sums
// arrays indexed by slot), which is the entire win over
// map[K]groupState: the old path paid TWO runtime probes per row (load
// + store-back with rehash and write barrier) plus a struct copy. Payload
// arrays hold no pointers, so their stores skip the write barrier too.
//
// These are worker-local structures: no locks, no atomics. Merging is
// single-threaded (or radix-partitioned, see parthash.go).
package vector

// Load and sizing constants, mirroring DuckDB's aggregate_hashtable:
// LOAD_FACTOR 1.5 (resize at 67% full), InitialCapacity 4096 slots.
const (
	htLoadNum    = 2 // grow when used*3 > cap*2, i.e. used/cap > 2/3
	htLoadDen    = 3
	htInitialCap = 1024 // slots; 128-group shapes never grow, 100K grows ×7
	// denseMaxGroups caps ONE dense table: 1M slots × ~33 B ≈ 33 MB.
	// Parallel folds hold W locals plus the output — see
	// wantDenseParallel in agg.go, which falls back to OA past the
	// 128 MB total budget so a near-cap Mod with many workers cannot OOM.
	denseMaxGroups = 1 << 20
)

// mixInt64 is a splitmix64 finalizer: bijective-ish avalanche so
// sequential ids (the bench shape) spread over all slot bits from a
// single cheap integer hash. One multiply alone would distribute
// sequential keys exactly, but collapses when keys share high bits
// (e.g. all multiples of 2^32); the full finalizer costs ~1ns/row.
func mixInt64(x int64) uint64 {
	z := uint64(x) + 0x9e3779b97f4a7c15
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// nextPow2 returns the smallest power of two >= n (n > 0).
func nextPow2(n int) int {
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// ---------------------------------------------------------------------------
// int64HashAgg: open-addressing linear-probing table for int64 keys.
// ---------------------------------------------------------------------------

// int64HashAgg maps int64 keys to (rows, count, sum) payloads held in
// parallel arrays. filled is 1 for occupied slots; hashes caches the
// full mix so repeat keys skip the key load on match (and mismatch
// probes compare hashes first); occupied lists live slots for O(groups)
// scan/merge without sweeping the capacity.
type int64HashAgg struct {
	mask     uint64
	used     int
	filled   []uint8
	hashes   []uint64
	keys     []int64
	rows     []int64
	counts   []int64
	sums     []float64
	occupied []int32
}

func newInt64HashAgg(hint int) *int64HashAgg {
	cap := nextPow2(max(hint*htLoadDen/htLoadNum, htInitialCap))
	return &int64HashAgg{
		mask:   uint64(cap - 1),
		filled: make([]uint8, cap),
		hashes: make([]uint64, cap),
		keys:   make([]int64, cap),
		rows:   make([]int64, cap),
		counts: make([]int64, cap),
		sums:   make([]float64, cap),
	}
}

// slotFor probes for h/k, returning the slot and whether it was already
// occupied. The caller inserts (claiming the slot) or folds into it.
func (t *int64HashAgg) slotFor(h uint64, k int64) (uint64, bool) {
	i := h & t.mask
	for {
		if t.filled[i] == 0 {
			return i, false
		}
		if t.hashes[i] == h && t.keys[i] == k {
			return i, true
		}
		i = (i + 1) & t.mask
	}
}

// add folds one row: single probe, in-place payload update. Insert zeroes
// the payload explicitly (slots are recycled by Reset, so stale data may
// linger — there is no sweep).
func (t *int64HashAgg) add(k int64, v float64, valid bool) {
	h := mixInt64(k)
	if t.used*htLoadDen >= len(t.filled)*htLoadNum {
		t.grow()
	}
	i, ok := t.slotFor(h, k)
	if !ok {
		t.filled[i] = 1
		t.hashes[i] = h
		t.keys[i] = k
		t.rows[i] = 1
		if valid {
			t.counts[i] = 1
			t.sums[i] = v
		} else {
			t.counts[i] = 0
			t.sums[i] = 0
		}
		t.occupied = append(t.occupied, int32(i))
		t.used++
		return
	}
	t.rows[i]++
	if valid {
		t.counts[i]++
		t.sums[i] += v
	}
}

// foldPayload merges an already-aggregated (rows, count, sum) triple for
// k — the merge path, where per-row Add semantics do not apply.
func (t *int64HashAgg) foldPayload(k int64, rows, count int64, sum float64) {
	h := mixInt64(k)
	if t.used*htLoadDen >= len(t.filled)*htLoadNum {
		t.grow()
	}
	i, ok := t.slotFor(h, k)
	if !ok {
		t.filled[i] = 1
		t.hashes[i] = h
		t.keys[i] = k
		t.rows[i] = rows
		t.counts[i] = count
		t.sums[i] = sum
		t.occupied = append(t.occupied, int32(i))
		t.used++
		return
	}
	t.rows[i] += rows
	t.counts[i] += count
	t.sums[i] += sum
}

// grow doubles the capacity and reinserts live slots using their cached
// hashes (no rehashing). Cost is amortized: each insert pays O(1)
// expected probes and each entry moves O(log n) times total.
func (t *int64HashAgg) grow() {
	oldFilled, oldHashes, oldKeys := t.filled, t.hashes, t.keys
	oldRows, oldCounts, oldSums := t.rows, t.counts, t.sums
	oldOcc := t.occupied
	cap := len(oldFilled) * 2
	t.mask = uint64(cap - 1)
	t.filled = make([]uint8, cap)
	t.hashes = make([]uint64, cap)
	t.keys = make([]int64, cap)
	t.rows = make([]int64, cap)
	t.counts = make([]int64, cap)
	t.sums = make([]float64, cap)
	t.occupied = make([]int32, 0, len(oldOcc))
	t.used = 0
	for _, s := range oldOcc {
		i := uint64(s)
		if oldFilled[i] == 0 {
			continue
		}
		j := oldHashes[i] & t.mask
		for t.filled[j] != 0 {
			j = (j + 1) & t.mask
		}
		t.filled[j] = 1
		t.hashes[j] = oldHashes[i]
		t.keys[j] = oldKeys[i]
		t.rows[j] = oldRows[i]
		t.counts[j] = oldCounts[i]
		t.sums[j] = oldSums[i]
		t.occupied = append(t.occupied, int32(j))
		t.used++
	}
}

// reserve ensures room for n additional distinct keys without growing.
func (t *int64HashAgg) reserve(n int) {
	for (t.used+n)*htLoadDen >= len(t.filled)*htLoadNum {
		t.grow()
	}
}

// reset clears occupancy without freeing backing arrays. Payloads are
// NOT zeroed (insert writes them); filled bytes are, over the capacity
// — O(cap), paid once per reuse, never per row.
func (t *int64HashAgg) reset() {
	for i := range t.filled {
		t.filled[i] = 0
	}
	t.occupied = t.occupied[:0]
	t.used = 0
}

// ---------------------------------------------------------------------------
// denseIntAgg: perfect-hash (dense array) table for small known domains.
// ---------------------------------------------------------------------------

// denseIntAgg is DuckDB's perfect_aggregate_hashtable ported to the one
// shape where QPX knows the domain upfront: the Mod path, where reduced
// keys lie in [0, mod) by construction. Index IS the key: no hash, no
// probe, no compare, no collision path. keys stores the first-seen
// reduced key per slot so negative raw keys (Go % keeps the sign, same
// as the old map path and PG's %) report their true group instead of a
// biased index.
type denseIntAgg struct {
	keys     []int64
	present  []bool
	rows     []int64
	counts   []int64
	sums     []float64
	occupied []int32
	used     int
}

func newDenseIntAgg(mod int64) *denseIntAgg {
	return &denseIntAgg{
		keys:    make([]int64, mod),
		present: make([]bool, mod),
		rows:    make([]int64, mod),
		counts:  make([]int64, mod),
		sums:    make([]float64, mod),
	}
}

// addIdx folds one row into slot idx (already reduced AND biased to
// [0,mod)); redKey is the unbiased reduced key reported by SortedRows.
func (t *denseIntAgg) addIdx(idx int64, redKey int64, v float64, valid bool) {
	if !t.present[idx] {
		t.present[idx] = true
		t.keys[idx] = redKey
		t.rows[idx] = 1
		if valid {
			t.counts[idx] = 1
			t.sums[idx] = v
		} else {
			// Explicit zero: slots are recycled by reset, so stale
			// payloads linger — a NULL-valued first touch must not
			// inherit the previous generation's count/sum.
			t.counts[idx] = 0
			t.sums[idx] = 0
		}
		t.occupied = append(t.occupied, int32(idx))
		t.used++
		return
	}
	t.rows[idx]++
	if valid {
		t.counts[idx]++
		t.sums[idx] += v
	}
}

// foldIdx merges a triple into slot idx (idx owned by disjoint merge
// partitions in parthash.go, or sequential otherwise).
func (t *denseIntAgg) foldIdx(idx int64, redKey int64, rows, count int64, sum float64) {
	if !t.present[idx] {
		t.present[idx] = true
		t.keys[idx] = redKey
		t.rows[idx] = rows
		t.counts[idx] = count
		t.sums[idx] = sum
		t.occupied = append(t.occupied, int32(idx))
		t.used++
		return
	}
	t.rows[idx] += rows
	t.counts[idx] += count
	t.sums[idx] += sum
}

func (t *denseIntAgg) reset() {
	for _, s := range t.occupied {
		t.present[s] = false
	}
	t.occupied = t.occupied[:0]
	t.used = 0
}

// ---------------------------------------------------------------------------
// strDictAgg: dictionary-encoded string aggregation.
// ---------------------------------------------------------------------------

// strDictAgg maps each distinct string to a dense id ONCE (single map
// load per row; stores only for unseen keys) and folds payloads into
// SoA arrays — the per-row store-back, rehash, and 24 B value copy of
// map[string]groupState disappear. Keys alias batch memory (no per-row
// key bytes), exactly as before. Dedup uses the runtime map itself, and
// the merge below is sequential, so no per-group hash is cached.
// strPayload is one group's measure. Array-of-struct (not three
// parallel slices): the per-row update touches rows+count+sum together,
// which fits one 24 B cache line instead of three.
type strPayload struct {
	rows  int64
	count int64
	sum   float64
}

type strDictAgg struct {
	ids  map[string]uint32
	keys []string
	pay  []strPayload
}

func newStrDictAgg(hint int) *strDictAgg {
	return &strDictAgg{ids: make(map[string]uint32, hint)}
}

// addCold inserts an unseen key (outlined cold path; the hot
// found-path lives inline in GroupByString.Add — see there).
func (t *strDictAgg) addCold(key string, v float64, valid bool) {
	id := uint32(len(t.keys))
	t.ids[key] = id
	t.keys = append(t.keys, key)
	p := strPayload{rows: 1}
	if valid {
		p.count = 1
		p.sum = v
	}
	t.pay = append(t.pay, p)
}

// foldPayload merges a triple for key, creating the id when absent.
func (t *strDictAgg) foldPayload(key string, rows, count int64, sum float64) {
	if id, ok := t.ids[key]; ok {
		p := &t.pay[id]
		p.rows += rows
		p.count += count
		p.sum += sum
		return
	}
	id := uint32(len(t.keys))
	t.ids[key] = id
	t.keys = append(t.keys, key)
	t.pay = append(t.pay, strPayload{rows: rows, count: count, sum: sum})
}

func (t *strDictAgg) reset() {
	for k := range t.ids {
		delete(t.ids, k)
	}
	t.keys = t.keys[:0]
	t.pay = t.pay[:0]
}
