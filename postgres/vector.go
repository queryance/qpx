package postgres

import (
	"context"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/queryance/qpx/engine"
	"github.com/queryance/qpx/engine/vector"
)

// VectorSource runs the configured query and decodes rows straight from
// the pgx wire buffers into typed vector batches. It is the default scan
// path for queries that need no legacy row ops (func(row []any)); use
// Source when custom legacy operators must see []any cells.
//
// Decode bypasses pgx's Values()/Scan plan machinery entirely: RawValues
// hands back the per-row wire bytes and each column decodes with one
// big-endian load (int/float), one string copy (text), or one
// micros-to-time conversion (timestamptz). No interface boxing, no
// per-cell plan lookup, no time.Time heap churn — the column slices are
// appended directly. Exotic encodings (binary numeric, text timestamps,
// non-text-family binary strings) fall back to a shared pgtype.Map; only
// the benchmark-hot binary/text shapes take the zero-alloc fast path.
type VectorSource struct {
	cfg Config
}

// NewVectorSource builds a VectorSource from cfg. ChunkSize selects rows
// per batch (default vector.DefaultBatchSize = 4096).
func NewVectorSource(cfg Config) *VectorSource {
	return &VectorSource{cfg: cfg}
}

// SetChunkSize overrides the rows-per-batch hint (engine.ChunkSizer, so
// qpx.ExecuteVector applies VectorOptions.BatchSize). Non-positive values
// are ignored.
func (s *VectorSource) SetChunkSize(n int) {
	if n > 0 {
		s.cfg.ChunkSize = n
	}
}

func (s *VectorSource) batchSize() int {
	if s.cfg.ChunkSize <= 0 {
		return vector.DefaultBatchSize
	}
	return s.cfg.ChunkSize
}

// colMeta pins the decode recipe for one column: the engine type plus the
// Postgres OID and wire format observed at Open time.
type colMeta struct {
	typ    engine.DataType
	oid    uint32
	format int16
}

// OpenBatch runs the query and returns a Scanner over typed batches. The
// schema is valid immediately; the query runs exactly once per OpenBatch.
// Time-family columns are limited to timestamp/timestamptz/date —
// time/timetz/interval have no time.Time representation, so they are
// rejected here with an explicit error instead of silently mistyped.
// Serve those through the legacy Source.
func (s *VectorSource) OpenBatch(ctx context.Context) (vector.Scanner, error) {
	conn, err := pgx.Connect(ctx, s.cfg.ConnString)
	if err != nil {
		return nil, fmt.Errorf("postgres: connect: %w", err)
	}
	rows, err := conn.Query(ctx, s.cfg.SQL, s.cfg.Args...)
	if err != nil {
		closeCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), closeTimeout)
		defer cancel()
		_ = conn.Close(closeCtx)
		return nil, fmt.Errorf("postgres: query: %w", err)
	}
	fds := rows.FieldDescriptions()
	schema := describeSchema(fds)
	meta := make([]colMeta, len(fds))
	for i, fd := range fds {
		typ := MapDataType(fd.DataTypeOID)
		if typ == engine.Time {
			switch fd.DataTypeOID {
			case pgtype.TimestampOID, pgtype.TimestamptzOID, pgtype.DateOID:
			default:
				_ = conn.Close(context.WithoutCancel(ctx))
				rows.Close()
				return nil, fmt.Errorf("postgres: vector: column %q OID %d has no time.Time form (use legacy Source)", fd.Name, fd.DataTypeOID)
			}
		}
		meta[i] = colMeta{typ: typ, oid: fd.DataTypeOID, format: fd.Format}
	}
	return &vscanner{
		ctx:     ctx,
		conn:    conn,
		rows:    rows,
		schema:  schema,
		meta:    meta,
		size:    s.batchSize(),
		fallMap: pgtype.NewMap(),
	}, nil
}

type vscanner struct {
	ctx     context.Context
	conn    *pgx.Conn
	rows    pgx.Rows
	schema  engine.Schema
	meta    []colMeta
	size    int
	fallMap *pgtype.Map
	cur     *vector.Batch
	scanErr error
	closed  bool
}

// Schema returns the result schema. Valid immediately after OpenBatch.
func (sc *vscanner) Schema() engine.Schema {
	return sc.schema
}

// Next fills a fresh Batch with up to size rows. Ownership transfers to
// the caller: each Next allocates a new Batch (a few backing arrays per
// ~4K rows, ~0.002 allocs/row) so batches in flight through a scheduler
// worker pool are never aliased. No sync.Pool: at this amortization the
// pool's lifecycle complexity buys nothing measurable — string bytes
// dominate the profile, not backing arrays. Direct drivers that own the
// batch exclusively may Reset and refill it themselves.
func (sc *vscanner) Next() bool {
	if sc.closed || sc.scanErr != nil {
		return false
	}
	b := vector.NewBatch(sc.schema, sc.size)
	for b.NumRows < sc.size && sc.rows.Next() {
		raw := sc.rows.RawValues()
		if len(raw) == len(b.Columns) {
			for j := range b.Columns {
				if err := sc.appendCell(&b.Columns[j], &sc.meta[j], raw[j]); err != nil {
					sc.scanErr = err
