// Parallel partitioned merge for group-by aggregation.
//
// Shape: workers fold batches into private local tables (no sharing, no
// locks — see agg.go), then the locals merge into one output. The merge
// itself is O(groups × workers) and runs once; at ≤1K groups it is noise
// (<10µs) and stays single-threaded. Past radixMergeMinTotal occupied
// entries with ≥4 workers, the DENSE fold fans out: merger p owns index
// range [p*mod/P,(p+1)*mod/P) of the SHARED output (index IS the key, so
// index ranges are hash partitions by construction) with no private
// tables and no concat pass at all — this is DuckDB's radix-partitioned
// merge, ported to the representation where partitions are free.
//
// Deliberate deviations from DuckDB, both measured (see README):
//   - The open-addressing and dict tables merge SEQUENTIALLY, always. A
//     private-table fan-out was built and benchmarked
//     (BenchmarkMergeOA8x100K): slicing occupied lists across mergers
//     does not partition KEYS, so the concat must re-fold every entry
//     and total work EXCEEDS the sequential fold (13ms vs ~6ms at
//     8×100K). True key-range partitioning would need an input scatter
//     pass (~1 extra scan of 1M rows) to eliminate a merge that costs
//     single-digit ms — net negative at our shapes.
//   - Input radix-partitioning (DuckDB's partitioned tuple data: every
//     thread owns disjoint keys end to end) is likewise rejected: it
//     trades a guaranteed full extra pass over all rows for a merge
//     that is sub-ms below 100K groups, and needs a cardinality
//     heuristic we cannot know upfront without a stats pass.
package vector

// radixMergeMinTotal gates the parallel merge: below this many occupied
// entries across all locals, goroutine + private-table overhead exceeds
// the fold it parallelizes. 128-group merges stay sequential (~µs);
// 100K-group × multi-worker merges fan out.
const radixMergeMinTotal = 65536

// radixMergeMaxParts bounds fan-out: enough to cover cores, few enough that
// per-merger chunks stay L2-friendly.
const radixMergeMaxParts = 16

// wantParallelMerge reports whether a parallel merge pays for
// totalOccupied entries over workers merger candidates.
func wantParallelMerge(totalOccupied, workers int) bool {
	if workers < 4 || totalOccupied < radixMergeMinTotal {
		return false
	}
	return true
}

// mergeParts clamps the merger count to [2, radixMergeMaxParts].
func mergeParts(workers int) int {
	if workers < 2 {
		return 2
	}
	if workers > radixMergeMaxParts {
		return radixMergeMaxParts
	}
	return workers
}

// barrier runs fn over parts goroutines and waits (same done-channel
// idiom as parallelRun in agg.go: channel receive happens-before the
// read, so per-part outputs need no mutex).
func barrier(parts int, fn func(p int)) {
	done := make(chan struct{}, parts)
	for p := 0; p < parts; p++ {
		go func(p int) {
			fn(p)
			done <- struct{}{}
		}(p)
	}
	for range parts {
		<-done
	}
}

// mergeDenseParallel folds locals into out (must be empty, same domain)
// with merger p owning a disjoint index range of the shared output:
// no private tables, no concat, no races by construction. Occupied lists
// come back in index order (part order), so SortedRows barely sorts.
func mergeDenseParallel(out *denseIntAgg, locals []*denseIntAgg, parts int) {
	mod := int64(len(out.present))
	occ := make([][]int32, parts)
	barrier(parts, func(p int) {
		lo := int64(p) * mod / int64(parts)
		hi := int64(p+1) * mod / int64(parts)
		var local []int32
		for _, l := range locals {
			for idx := lo; idx < hi; idx++ {
				if !l.present[idx] {
					continue
				}
				if !out.present[idx] {
					out.present[idx] = true
					out.keys[idx] = l.keys[idx]
					out.rows[idx] = l.rows[idx]
					out.counts[idx] = l.counts[idx]
					out.sums[idx] = l.sums[idx]
					local = append(local, int32(idx))
					continue
				}
				out.rows[idx] += l.rows[idx]
				out.counts[idx] += l.counts[idx]
				out.sums[idx] += l.sums[idx]
			}
		}
		occ[p] = local
	})
	for p := 0; p < parts; p++ {
		out.occupied = append(out.occupied, occ[p]...)
	}
	out.used = len(out.occupied)
}

// mergeDense folds locals into out, parallel past the gate.
func mergeDense(out *denseIntAgg, locals []*denseIntAgg, workers int) {
	var total int
	for _, l := range locals {
		total += l.used
	}
	if !wantParallelMerge(total, workers) {
		for _, l := range locals {
			for _, s := range l.occupied {
				idx := int64(s)
				out.foldIdx(idx, l.keys[idx], l.rows[idx], l.counts[idx], l.sums[idx])
			}
		}
		return
	}
	mergeDenseParallel(out, locals, mergeParts(workers))
}

// mergeOA folds locals into out, always sequentially (see package doc:
// fanning out costs more than it saves — the concat must re-fold).
// reserve hoists growth out of the fold. workers is accepted for API
// symmetry with mergeDense and ignored.
func mergeOA(out *int64HashAgg, locals []*int64HashAgg, _ int) {
	// Reserve the largest local, not the sum: workers fold the same key
	// domain, so distinct keys track max-used, and sizing for the sum
	// (e.g. 800K entries → 2M slots ≈ 96 MB) destroys probe locality
	// (measured 46ms vs ~8ms at 8×100K). Disjoint locals still grow
	// naturally during the fold — never worse than no reserve.
	mx := 0
	for _, l := range locals {
		if l.used > mx {
			mx = l.used
		}
	}
	out.reserve(mx)
	for _, l := range locals {
		for _, s := range l.occupied {
			i := uint64(s)
			out.foldPayload(l.keys[i], l.rows[i], l.counts[i], l.sums[i])
		}
	}
}

// mergeDict folds locals into out, always sequentially (see package
// doc). workers is accepted for API symmetry and ignored.
func mergeDict(out *strDictAgg, locals []*strDictAgg, _ int) {
	for _, l := range locals {
		for i, k := range l.keys {
			p := &l.pay[i]
			out.foldPayload(k, p.rows, p.count, p.sum)
		}
	}
}
