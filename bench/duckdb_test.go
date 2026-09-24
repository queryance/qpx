package bench

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"testing"

	_ "github.com/marcboeker/go-duckdb"
)

// DuckDB availability: github.com/marcboeker/go-duckdb v1.8.5 installs
// cleanly (module download + prebuilt static libs, ~11s build, CGO) and
// its bundled postgres extension reads the same qpx_bench database, so
// (c) is measured in-process below — no CLI fallback needed.
//
// Method: per benchmark, open one in-memory DuckDB, LOAD the postgres
// extension once, and expose the bench table under its own name:
//
//	CREATE VIEW qpx_bench_data AS SELECT * FROM postgres_scan('<dsn>', ...)
//
// so the timed SQL is byte-identical to sqlQ1/sqlQ2/sqlQ3. Each timed
// iteration runs the query and Scan-decodes every row into []any,
// discarding it — the same drain-and-discard semantics as runPGX.
// (Setup — open/LOAD/view — is hoisted out of the timed loop and noted
// as such; QPX/pgx reconnect per iteration at ~1ms cost, negligible
// against 50-300ms queries.)
type duckDB struct {
	db   *sql.DB
	conn *sql.Conn
}

func openDuckDB(ctx context.Context) (*duckDB, error) {
	dsn := resolveDSN()
	if dsn == "" {
		return nil, fmt.Errorf("bench: no working DSN")
	}
	db, err := sql.Open("duckdb", "")
	if err != nil {
		return nil, fmt.Errorf("bench: duckdb open: %w", err)
	}
	conn, err := db.Conn(ctx)
	if err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("bench: duckdb conn: %w", err)
	}
	if _, err := conn.ExecContext(ctx, `INSTALL postgres; LOAD postgres;`); err != nil {
		_ = conn.Close()
		_ = db.Close()
		return nil, fmt.Errorf("bench: duckdb postgres extension: %w", err)
	}
	scan := fmt.Sprintf("postgres_scan('%s', 'public', '%s')",
		strings.ReplaceAll(dsn, `'`, `''`), benchTable)
	if _, err := conn.ExecContext(ctx,
		`CREATE OR REPLACE VIEW `+benchTable+` AS SELECT * FROM `+scan); err != nil {
		_ = conn.Close()
		_ = db.Close()
		return nil, fmt.Errorf("bench: duckdb view: %w", err)
	}
	return &duckDB{db: db, conn: conn}, nil
}

func (d *duckDB) close() {
	_ = d.conn.Close()
	_ = d.db.Close()
}

// runDuckDBQuery runs sql and Scan-decodes/discards every row.
func (d *duckDB) runQuery(ctx context.Context, sql string) (int64, error) {
	rows, err := d.conn.QueryContext(ctx, sql)
	if err != nil {
		return 0, err
	}
	defer rows.Close()
	cols, err := rows.Columns()
	if err != nil {
		return 0, err
	}
	vals := make([]any, len(cols))
	ptrs := make([]any, len(cols))
	for i := range vals {
		ptrs[i] = &vals[i]
	}
	var n int64
	for rows.Next() {
		if err := rows.Scan(ptrs...); err != nil {
			return n, err
		}
		n++
	}
	return n, rows.Err()
}

func driveDuckDB(b *testing.B, sql string) {
	b.Helper()
	ensureDataset(b)
	ctx := context.Background()
	d, err := openDuckDB(ctx)
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
	b.ReportMetric(float64(int64(b.N)*benchRows)/sec, "scan_rows/s")
	b.ReportMetric(float64(total)/float64(b.N), "rows/op")
}

func BenchmarkDuckDBQ1FullScan(b *testing.B)      { driveDuckDB(b, sqlQ1) }
func BenchmarkDuckDBQ2FilterProject(b *testing.B) { driveDuckDB(b, sqlQ2) }
func BenchmarkDuckDBQ3GroupBy(b *testing.B)       { driveDuckDB(b, sqlQ3) }

// TestDuckDBSanity checks the DuckDB path returns the same result sizes
// as Postgres (Q1 1M rows, Q2 499,990, Q3 128 groups).
func TestDuckDBSanity(t *testing.T) {
	ensureDataset(t)
	ctx := context.Background()
	d, err := openDuckDB(ctx)
	if err != nil {
		t.Fatalf("duckdb setup: %v", err)
	}
	defer d.close()
	checks := map[string]struct {
		sql  string
		want int64
	}{
		"q1": {sqlQ1, benchRows},
		"q2": {sqlQ2, 499990},
		"q3": {sqlQ3, 128},
	}
	for name, c := range checks {
		got, err := d.runQuery(ctx, c.sql)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if got != c.want {
			t.Fatalf("%s: got %d rows, want %d", name, got, c.want)
		}
	}
}
