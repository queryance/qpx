package postgres

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/queryance/qpx/engine"
)

type scanner struct {
	ctx     context.Context
	conn    *pgx.Conn
	rows    pgx.Rows
	schema  engine.Schema
	size    int
	cur     engine.Chunk
	scanErr error
	closed  bool
}

// Schema returns the result schema. It is valid immediately after Open.
func (sc *scanner) Schema() engine.Schema {
	return sc.schema
}

// Next advances to the next chunk, packing up to size rows.
// Short rows are nil-filled (matching PackChunk) so chunks stay rectangular.
//
// The per-row Values slice is never retained — every cell is copied into
// its column before the next Next call — so no defensive copy of the row
// is needed even though pgx may reuse the slice.
func (sc *scanner) Next() bool {
	if sc.closed || sc.scanErr != nil {
		return false
	}
	ncols := len(sc.schema.Fields)
	cols := make([][]any, ncols)
	for i := range cols {
		cols[i] = make([]any, 0, sc.size)
	}
	count := 0
	for count < sc.size && sc.rows.Next() {
		vals, err := sc.rows.Values()
		if err != nil {
			sc.scanErr = fmt.Errorf("postgres: decode row: %w", err)
			return false
		}
		// Rectangular rows are the common case: copy cells straight
		// into their columns with no per-cell length check.
		if len(vals) == ncols {
			for i := 0; i < ncols; i++ {
				cols[i] = append(cols[i], vals[i])
			}
		} else {
			for i := 0; i < ncols; i++ {
				var v any
				if i < len(vals) {
					v = vals[i]
				}
				cols[i] = append(cols[i], v)
			}
		}
		count++
	}
	if err := sc.rows.Err(); err != nil {
		sc.scanErr = fmt.Errorf("postgres: rows: %w", err)
		return false
	}
	if count == 0 {
		return false
	}
	chunk, err := engine.NewChunk(sc.schema, cols)
	if err != nil {
		sc.scanErr = fmt.Errorf("postgres: invalid chunk: %w", err)
		return false
	}
	sc.cur = chunk
	return true
}

// Chunk returns the current chunk. It is valid until the next Next call.
func (sc *scanner) Chunk() engine.Chunk {
	return sc.cur
}

// Err reports any streaming failure, or nil at end of stream.
func (sc *scanner) Err() error {
	return sc.scanErr
}

// Close releases the rows and connection. It is idempotent.
// The connection is closed with a timeout derived from a
// cancellation-detached parent, so a hung close is bounded even when the
// run ctx is already cancelled (the scheduler cancels runCtx on error and
// teardown paths before/concurrent-with Close). Open always sets sc.ctx
// from the run ctx.
func (sc *scanner) Close() error {
	if sc.closed {
		return nil
	}
	sc.closed = true
	sc.rows.Close()
	ctx, cancel := context.WithTimeout(context.WithoutCancel(sc.ctx), closeTimeout)
	defer cancel()
	if err := sc.conn.Close(ctx); err != nil {
		return fmt.Errorf("postgres: close: %w", err)
	}
	return nil
}
