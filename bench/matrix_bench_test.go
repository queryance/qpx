package bench

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/queryance/qpx"
	"github.com/queryance/qpx/postgres"
)

// Matrix benchmarks: same SQL on QPX (Execute+drain), raw pgx
// (Values-decode+drain), and DuckDB (postgres_scan view + Scan-decode).
// M scan/filter/glow reuse the legacy Q1-Q3 benchmarks (see bench_test.go
// and duckdb_test.go) and are not duplicated here.
//
// Naming: BenchmarkMx{SYS}{QUERY}{S,L} for the size sweep, ...{M} suffix
// for M-only queries (AggM, GHighM, JoinM, SortM, LimitM), and
// BenchmarkMxQPXW{workers}{Scan,Filter} for the worker sweep (M only).
// L benches are slow: run with -benchtime=1x -count=2 (see matrix_doc.go).

// driveMatrix mirrors drive with an explicit scanned-row count so S/L
// report comparable scan_rows/s instead of the hardcoded M 1M.
func driveMatrix(b *testing.B, sql string, scanRows int64, fn func(context.Context, string) (int64, error)) {
	b.Helper()
	ensureMatrixDataset(b)
	if resolveDSN() == "" {
		b.Fatalf("bench: no working DSN (tried %v)", dsnCandidates)
	}
	// Warmup: shake out one-time costs (connection, plan cache) so the
	// timed iterations measure steady-state streaming.
	if _, err := fn(context.Background(), sql); err != nil {
		b.Fatalf("warmup: %v", err)
	}
	b.ResetTimer()
	var total int64
	for i := 0; i < b.N; i++ {
		n, err := fn(context.Background(), sql)
		if err != nil {
			b.Fatalf("iter %d: %v", i, err)
		}
		total += n
	}
	b.StopTimer()
	sec := b.Elapsed().Seconds()
	b.ReportMetric(float64(total)/sec, "rows/s")
	b.ReportMetric(float64(int64(b.N)*scanRows)/sec, "scan_rows/s")
	b.ReportMetric(float64(total)/float64(b.N), "rows/op")
}

// runQPXWorkers drains sql through qpx.Execute with an explicit worker
// count (default options otherwise), for the worker-scaling sweep.
func runQPXWorkers(ctx context.Context, sql string, workers int) (int64, error) {
	src := postgres.NewSource(postgres.Config{ConnString: resolveDSN(), SQL: sql})
	opts := qpx.DefaultExecuteOptions()
	opts.Workers = workers
	it, err := qpx.Execute(ctx, src, opts)
	if err != nil {
		return 0, err
	}
	defer func() { _ = it.Close() }()
	var rows int64
	for {
		c, ok, err := it.Next(ctx)
		if err != nil {
			return rows, err
		}
		if !ok {
			return rows, nil
		}
		rows += int64(c.NumRows())
	}
}

// mxOpenDuckDB reuses the existing DuckDB harness (openDuckDB maps M) and
// adds same-name views for S/J/L, so timed SQL stays byte-identical.
func mxOpenDuckDB(ctx context.Context) (*duckDB, error) {
	d, err := openDuckDB(ctx)
	if err != nil {
		return nil, err
	}
	dsn := resolveDSN()
	for _, tbl := range []string{mxTableS, mxTableJ, mxTableL} {
		scan := fmt.Sprintf("postgres_scan('%s', 'public', '%s')",
			strings.ReplaceAll(dsn, `'`, `''`), tbl)
		if _, err := d.conn.ExecContext(ctx,
			`CREATE OR REPLACE VIEW `+tbl+` AS SELECT * FROM `+scan); err != nil {
			d.close()
			return nil, fmt.Errorf("bench: duckdb view %s: %w", tbl, err)
		}
	}
	return d, nil
}

func driveMatrixDuckDB(b *testing.B, sql string, scanRows int64) {
	b.Helper()
	ensureMatrixDataset(b)
	ctx := context.Background()
	d, err := mxOpenDuckDB(ctx)
	if err != nil {
		b.Fatalf("duckdb setup: %v", err)
	}
	defer d.close()
	if _, err := d.runQuery(ctx, sql); err != nil {
		b.Fatalf("warmup: %v", err)
	}
	b.ResetTimer()
	var total int64
	for i := 0; i < b.N; i++ {
		n, err := d.runQuery(ctx, sql)
		if err != nil {
			b.Fatalf("iter %d: %v", i, err)
		}
		total += n
	}
	b.StopTimer()
	sec := b.Elapsed().Seconds()
	b.ReportMetric(float64(total)/sec, "rows/s")
	b.ReportMetric(float64(int64(b.N)*scanRows)/sec, "scan_rows/s")
	b.ReportMetric(float64(total)/float64(b.N), "rows/op")
}

// Size sweep, S (100K rows).

func BenchmarkMxQPXScanS(b *testing.B)   { driveMatrix(b, mxScanS, mxRowsS, runQPX) }
func BenchmarkMxPGXScanS(b *testing.B)   { driveMatrix(b, mxScanS, mxRowsS, runPGX) }
func BenchmarkMxDuckScanS(b *testing.B)  { driveMatrixDuckDB(b, mxScanS, mxRowsS) }
func BenchmarkMxQPXFilterS(b *testing.B) { driveMatrix(b, mxFilterS, mxRowsS, runQPX) }
func BenchmarkMxPGXFilterS(b *testing.B) { driveMatrix(b, mxFilterS, mxRowsS, runPGX) }
func BenchmarkMxDuckFilterS(b *testing.B) {
	driveMatrixDuckDB(b, mxFilterS, mxRowsS)
}
func BenchmarkMxQPXAggS(b *testing.B)   { driveMatrix(b, mxAggS, mxRowsS, runQPX) }
func BenchmarkMxPGXAggS(b *testing.B)   { driveMatrix(b, mxAggS, mxRowsS, runPGX) }
func BenchmarkMxDuckAggS(b *testing.B)  { driveMatrixDuckDB(b, mxAggS, mxRowsS) }
func BenchmarkMxQPXGlowS(b *testing.B)  { driveMatrix(b, mxGlowS, mxRowsS, runQPX) }
func BenchmarkMxPGXGlowS(b *testing.B)  { driveMatrix(b, mxGlowS, mxRowsS, runPGX) }
func BenchmarkMxDuckGlowS(b *testing.B) { driveMatrixDuckDB(b, mxGlowS, mxRowsS) }
func BenchmarkMxQPXGHighS(b *testing.B) { driveMatrix(b, mxGHighS, mxRowsS, runQPX) }
func BenchmarkMxPGXGHighS(b *testing.B) { driveMatrix(b, mxGHighS, mxRowsS, runPGX) }
func BenchmarkMxDuckGHighS(b *testing.B) {
	driveMatrixDuckDB(b, mxGHighS, mxRowsS)
}

// M-only queries (scan/filter/glow live in the legacy Q1-Q3 benchmarks).

func BenchmarkMxQPXAggM(b *testing.B)   { driveMatrix(b, mxAggQM, benchRows, runQPX) }
func BenchmarkMxPGXAggM(b *testing.B)   { driveMatrix(b, mxAggQM, benchRows, runPGX) }
func BenchmarkMxDuckAggM(b *testing.B)  { driveMatrixDuckDB(b, mxAggQM, benchRows) }
func BenchmarkMxQPXGHighM(b *testing.B) { driveMatrix(b, mxGHighQM, benchRows, runQPX) }
func BenchmarkMxPGXGHighM(b *testing.B) { driveMatrix(b, mxGHighQM, benchRows, runPGX) }
func BenchmarkMxDuckGHighM(b *testing.B) {
	driveMatrixDuckDB(b, mxGHighQM, benchRows)
}
func BenchmarkMxQPXJoinM(b *testing.B) { driveMatrix(b, mxJoinQM, mxRowsJ, runQPX) }
func BenchmarkMxPGXJoinM(b *testing.B) { driveMatrix(b, mxJoinQM, mxRowsJ, runPGX) }
func BenchmarkMxDuckJoinM(b *testing.B) {
	driveMatrixDuckDB(b, mxJoinQM, mxRowsJ)
}
func BenchmarkMxQPXSortM(b *testing.B)  { driveMatrix(b, mxSortQM, benchRows, runQPX) }
func BenchmarkMxPGXSortM(b *testing.B)  { driveMatrix(b, mxSortQM, benchRows, runPGX) }
func BenchmarkMxDuckSortM(b *testing.B) { driveMatrixDuckDB(b, mxSortQM, benchRows) }
func BenchmarkMxQPXLimitM(b *testing.B) { driveMatrix(b, mxLimitQM, 100, runQPX) }
func BenchmarkMxPGXLimitM(b *testing.B) { driveMatrix(b, mxLimitQM, 100, runPGX) }
func BenchmarkMxDuckLimitM(b *testing.B) {
	driveMatrixDuckDB(b, mxLimitQM, 100)
}

// Size sweep, L (10M rows). Slow: -benchtime=1x -count=2.

func BenchmarkMxQPXScanL(b *testing.B)   { driveMatrix(b, mxScanL, mxRowsL, runQPX) }
func BenchmarkMxPGXScanL(b *testing.B)   { driveMatrix(b, mxScanL, mxRowsL, runPGX) }
func BenchmarkMxDuckScanL(b *testing.B)  { driveMatrixDuckDB(b, mxScanL, mxRowsL) }
func BenchmarkMxQPXFilterL(b *testing.B) { driveMatrix(b, mxFilterL, mxRowsL, runQPX) }
func BenchmarkMxPGXFilterL(b *testing.B) { driveMatrix(b, mxFilterL, mxRowsL, runPGX) }
func BenchmarkMxDuckFilterL(b *testing.B) {
	driveMatrixDuckDB(b, mxFilterL, mxRowsL)
}
func BenchmarkMxQPXAggL(b *testing.B)   { driveMatrix(b, mxAggL, mxRowsL, runQPX) }
func BenchmarkMxPGXAggL(b *testing.B)   { driveMatrix(b, mxAggL, mxRowsL, runPGX) }
func BenchmarkMxDuckAggL(b *testing.B)  { driveMatrixDuckDB(b, mxAggL, mxRowsL) }
func BenchmarkMxQPXGlowL(b *testing.B)  { driveMatrix(b, mxGlowL, mxRowsL, runQPX) }
func BenchmarkMxPGXGlowL(b *testing.B)  { driveMatrix(b, mxGlowL, mxRowsL, runPGX) }
func BenchmarkMxDuckGlowL(b *testing.B) { driveMatrixDuckDB(b, mxGlowL, mxRowsL) }
func BenchmarkMxQPXGHighL(b *testing.B) { driveMatrix(b, mxGHighL, mxRowsL, runQPX) }
func BenchmarkMxPGXGHighL(b *testing.B) { driveMatrix(b, mxGHighL, mxRowsL, runPGX) }
func BenchmarkMxDuckGHighL(b *testing.B) {
	driveMatrixDuckDB(b, mxGHighL, mxRowsL)
}

// Worker sweep: ExecuteOptions.Workers 1/2/4/8/16 on M scan and
// filter+project (QPX only; pgx/DuckDB have no worker knob). Filter here
// is PG-side WHERE (sqlQ2, ~500k rows out): the vector head-to-head is
// BenchmarkVectorW*PGFilter; BenchmarkVectorW*EngineFilter is the
// engine-side variant (full 1M scan) and is not directly comparable.
func mxDriveWorkers(b *testing.B, sql string, workers int) {
	b.Helper()
	driveMatrix(b, sql, benchRows, func(ctx context.Context, s string) (int64, error) {
		return runQPXWorkers(ctx, s, workers)
	})
}

func BenchmarkMxQPXW1Scan(b *testing.B)   { mxDriveWorkers(b, sqlQ1, 1) }
func BenchmarkMxQPXW2Scan(b *testing.B)   { mxDriveWorkers(b, sqlQ1, 2) }
func BenchmarkMxQPXW4Scan(b *testing.B)   { mxDriveWorkers(b, sqlQ1, 4) }
func BenchmarkMxQPXW8Scan(b *testing.B)   { mxDriveWorkers(b, sqlQ1, 8) }
func BenchmarkMxQPXW16Scan(b *testing.B)  { mxDriveWorkers(b, sqlQ1, 16) }
func BenchmarkMxQPXW1Filter(b *testing.B) { mxDriveWorkers(b, sqlQ2, 1) }
func BenchmarkMxQPXW2Filter(b *testing.B) { mxDriveWorkers(b, sqlQ2, 2) }
func BenchmarkMxQPXW4Filter(b *testing.B) { mxDriveWorkers(b, sqlQ2, 4) }
func BenchmarkMxQPXW8Filter(b *testing.B) { mxDriveWorkers(b, sqlQ2, 8) }
func BenchmarkMxQPXW16Filter(b *testing.B) {
	mxDriveWorkers(b, sqlQ2, 16)
}
