// Package postgres is the Postgres backend for qpx: it runs a single query
// and streams rows into columnar chunks. It is the only package allowed to
// import pgx; the engine never sees driver types.
//
// Layout: config.go (Config, chunk-size knobs), source.go (Source.Open),
// scanner.go (chunked row streaming), datatype.go (OID mapping),
// pack.go (row-major to column-major transpose).
package postgres

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/queryance/qpx/engine"
)

// closeTimeout bounds connection teardown so a hung close cannot block a
// Scheduler run forever.
const closeTimeout = 5 * time.Second

// Source runs the configured query and packs rows into chunks.
type Source struct {
	cfg Config
}

// Open runs the query; the Scanner streams its rows in chunks.
// The Scanner's Schema is valid immediately after Open, so the query runs
// exactly once per Open.
func (s *Source) Open(ctx context.Context) (engine.Scanner, error) {
	conn, err := pgx.Connect(ctx, s.cfg.ConnString)
	if err != nil {
		return nil, fmt.Errorf("postgres: connect: %w", err)
	}
	rows, err := conn.Query(ctx, s.cfg.SQL, s.cfg.Args...)
	if err != nil {
		// Detach from ctx: the caller may have cancelled it, which
		// must not fail connection teardown.
		closeCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), closeTimeout)
		defer cancel()
		_ = conn.Close(closeCtx)
		return nil, fmt.Errorf("postgres: query: %w", err)
	}
	return &scanner{
		ctx:    ctx,
		conn:   conn,
		rows:   rows,
		schema: describeSchema(rows.FieldDescriptions()),
		size:   s.chunkSize(),
	}, nil
}
