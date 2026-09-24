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
