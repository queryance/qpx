package bench

import (
	"context"
	"testing"

	"github.com/queryance/qpx"
	"github.com/queryance/qpx/engine/vector"
	"github.com/queryance/qpx/postgres"
)

// Vector benchmarks: same SQL as the legacy Q1/Q2 benchmarks, drained
// through qpx.ExecuteVector. Run side by side with -benchmem to compare
// allocs/op:
//
//	go test ./bench/ -run XXX -bench 'QPXQ1FullScan|VectorQ1|QPXQ2FilterProjectOps|VectorQ2' -benchmem
//
// Q1 is a straight typed scan (no ops); Q2 is a full scan with the fused
// engine-side filter+project (f > 0.5, keep (id, f)), matching
// BenchmarkQPXQ2FilterProjectOps result-for-result.

func runVectorQ1(ctx context.Context, sql string) (int64, error) {
	src := postgres.NewVectorSource(postgres.Config{ConnString: resolveDSN(), SQL: sql})
	it, err := qpx.ExecuteVector(ctx, src, qpx.DefaultVectorOptions())
	if err != nil {
		return 0, err
	}
	defer func() { _ = it.Close() }()
	var rows int64
	for {
		b, ok, err := it.Next(ctx)
		if err != nil {
			return rows, err
		}
		if !ok {
			return rows, nil
		}
		rows += int64(b.Len())
	}
}

func runVectorQ2(ctx context.Context, sql string) (int64, error) {
	src := postgres.NewVectorSource(postgres.Config{ConnString: resolveDSN(), SQL: sql})
	opts := qpx.DefaultVectorOptions()
	opts.Ops = []vector.Operator{vector.NewFilterFloat64GTProject(1, 0.5, 0, 1)}
	it, err := qpx.ExecuteVector(ctx, src, opts)
	if err != nil {
		return 0, err
	}
	defer func() { _ = it.Close() }()
	var rows int64
	for {
		b, ok, err := it.Next(ctx)
		if err != nil {
			return rows, err
		}
		if !ok {
			return rows, nil
		}
		rows += int64(b.Len())
	}
}

func BenchmarkVectorQ1FullScan(b *testing.B) { drive(b, sqlQ1, runVectorQ1) }

func BenchmarkVectorQ2Fused(b *testing.B) {
	drive(b, sqlQ2FullScan, func(ctx context.Context, _ string) (int64, error) {
		return runVectorQ2(ctx, sqlQ2FullScan)
	})
}

// TestVectorQ2Parity guards the fused vector Q2 against the PG-side WHERE
// row count, mirroring TestQ2OpsParity for the legacy ops path.
func TestVectorQ2Parity(t *testing.T) {
	ensureDataset(t)
	if resolveDSN() == "" {
		t.Fatalf("bench: no working DSN (tried %v)", dsnCandidates)
	}
	got, err := runVectorQ2(context.Background(), sqlQ2FullScan)
	if err != nil {
		t.Fatalf("vector-q2: %v", err)
	}
	if got != 499990 {
		t.Fatalf("vector-q2: got %d rows, want 499990", got)
	}
}
