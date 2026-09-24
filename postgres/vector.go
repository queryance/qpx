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
					return false
				}
			}
		} else {
			// Ragged wire row (should not happen for table scans):
			// nil-fill short, ignore extras — same contract as the
			// legacy scanner, so both paths stay rectangular.
			for j := range b.Columns {
				var buf []byte
				if j < len(raw) {
					buf = raw[j]
				}
				if err := sc.appendCell(&b.Columns[j], &sc.meta[j], buf); err != nil {
					sc.scanErr = err
					return false
				}
			}
		}
		b.NumRows++
	}
	if err := sc.rows.Err(); err != nil {
		sc.scanErr = fmt.Errorf("postgres: vector rows: %w", err)
		return false
	}
	if b.NumRows == 0 {
		return false
	}
	sc.cur = b
	return true
}

// Batch returns the current batch, valid until the next Next call in
// direct-drive use; owned by the caller once handed to a scheduler.
func (sc *vscanner) Batch() *vector.Batch {
	return sc.cur
}

// Err reports any streaming failure, or nil at end of stream.
func (sc *vscanner) Err() error {
	return sc.scanErr
}

// Close releases the rows and connection. Idempotent; bounded by the same
// closeTimeout as the legacy scanner.
func (sc *vscanner) Close() error {
	if sc.closed {
		return nil
	}
	sc.closed = true
	sc.rows.Close()
	ctx, cancel := context.WithTimeout(context.WithoutCancel(sc.ctx), closeTimeout)
	defer cancel()
	if err := sc.conn.Close(ctx); err != nil {
		return fmt.Errorf("postgres: vector close: %w", err)
	}
	return nil
}

func (sc *vscanner) appendCell(c *vector.Column, m *colMeta, buf []byte) error {
	if buf == nil {
		c.AppendNull()
		return nil
	}
	var err error
	switch m.typ {
	case engine.Int64:
		var v int64
		v, err = decodeInt(buf, m.format)
		if err == nil {
			c.AppendInt(v)
		}
	case engine.Float64:
		var v float64
		v, err = decodeFloat(buf, m.format)
		if err == nil {
			c.AppendFloat(v)
		}
	case engine.Bool:
		var v bool
		v, err = decodeBool(buf, m.format)
		if err == nil {
			c.AppendBool(v)
		}
	case engine.String:
		var v string
		v, err = sc.decodeString(buf, m)
		if err == nil {
			c.AppendString(v)
		}
	case engine.Bytes:
		var v []byte
		v, err = decodeBytes(buf, m.format)
		if err == nil {
			c.AppendBytes(v)
		}
	case engine.Time:
		var v time.Time
		v, err = sc.decodeTime(buf, m)
		if err == nil {
			c.AppendTime(v)
		}
	case engine.Numeric:
		var v string
		v, err = sc.decodeNumeric(buf, m)
		if err == nil {
			c.AppendNumeric(v)
		}
	default:
		var v string
		v, err = sc.decodeString(buf, m)
		if err == nil {
			c.AppendString(v)
		}
	}
	if err != nil {
		return fmt.Errorf("postgres: vector: decode: %w", err)
	}
	return nil
}

// decodeInt handles binary int2/int4/int8 (width from payload length) and
// text integers.
func decodeInt(buf []byte, format int16) (int64, error) {
	if format == pgtype.BinaryFormatCode {
		switch len(buf) {
		case 2:
			return int64(int16(binary.BigEndian.Uint16(buf))), nil
		case 4:
			return int64(int32(binary.BigEndian.Uint32(buf))), nil
		case 8:
			return int64(binary.BigEndian.Uint64(buf)), nil
		default:
			return 0, fmt.Errorf("int: binary length %d", len(buf))
		}
	}
	v, err := strconv.ParseInt(string(buf), 10, 64)
	if err != nil {
		return 0, fmt.Errorf("int: %w", err)
	}
	return v, nil
}

// decodeFloat handles binary float4/float8 and text floats.
func decodeFloat(buf []byte, format int16) (float64, error) {
	if format == pgtype.BinaryFormatCode {
		switch len(buf) {
		case 4:
			return float64(math.Float32frombits(binary.BigEndian.Uint32(buf))), nil
		case 8:
			return math.Float64frombits(binary.BigEndian.Uint64(buf)), nil
		default:
			return 0, fmt.Errorf("float: binary length %d", len(buf))
		}
	}
	v, err := strconv.ParseFloat(string(buf), 64)
	if err != nil {
		return 0, fmt.Errorf("float: %w", err)
	}
	return v, nil
}

// decodeBool handles binary (1 byte) and text ('t'/'f', extended forms).
func decodeBool(buf []byte, format int16) (bool, error) {
	if format == pgtype.BinaryFormatCode {
		if len(buf) != 1 {
			return false, fmt.Errorf("bool: binary length %d", len(buf))
		}
		return buf[0] != 0, nil
	}
	switch string(buf) {
	case "t", "true", "1":
		return true, nil
	case "f", "false", "0":
		return false, nil
	default:
		return false, fmt.Errorf("bool: unrecognized %q", buf)
	}
}

// decodeString copies text bytes. Binary payloads of the text family
// (text/varchar/name/bpchar) are raw bytes too; any other binary OID
// goes through pgtype text coercion, else an explicit error.
func (sc *vscanner) decodeString(buf []byte, m *colMeta) (string, error) {
	if m.format == pgtype.TextFormatCode {
		return string(buf), nil
	}
	switch m.oid {
	case pgtype.TextOID, pgtype.VarcharOID, pgtype.NameOID, pgtype.BPCharOID:
		return string(buf), nil
	}
	var t pgtype.Text
	if err := sc.fallMap.Scan(m.oid, m.format, buf, &t); err != nil {
		return "", fmt.Errorf("string OID %d: %w (use legacy Source)", m.oid, err)
	}
	if !t.Valid {
		return "", fmt.Errorf("string OID %d: unexpected NULL decode", m.oid)
	}
	return t.String, nil
}

// decodeBytes clones binary payloads (RawValues is only valid until the
// next Next). Text bytea arrives hex-encoded (\x...); decode the prefix,
// else keep the raw copy.
func decodeBytes(buf []byte, format int16) ([]byte, error) {
	if format == pgtype.BinaryFormatCode {
		out := make([]byte, len(buf))
		copy(out, buf)
		return out, nil
	}
	if len(buf) >= 2 && buf[0] == '\\' && (buf[1] == 'x' || buf[1] == 'X') {
		out := make([]byte, hex.DecodedLen(len(buf)-2))
		if _, err := hex.Decode(out, buf[2:]); err != nil {
			return nil, fmt.Errorf("bytea hex: %w", err)
		}
		return out, nil
	}
	out := make([]byte, len(buf))
	copy(out, buf)
	return out, nil
}

// Decode timestamps. Binary timestamp/timestamptz is int64 micros since
// 2000-01-01, converted with pgx's exact formula so vector times are
// bit-identical to legacy Values() times. Binary date is int32 days since
// 2000-01-01. Anything else (text forms) goes through pgtype into
// time.Time. Infinities have no time.Time form and are an explicit error.
func (sc *vscanner) decodeTime(buf []byte, m *colMeta) (time.Time, error) {
	if m.format == pgtype.BinaryFormatCode {
		switch m.oid {
		case pgtype.DateOID:
			if len(buf) != 4 {
				return time.Time{}, fmt.Errorf("date: binary length %d", len(buf))
			}
			days := int64(int32(binary.BigEndian.Uint32(buf)))
			if days == 2147483647 || days == -2147483648 {
				return time.Time{}, fmt.Errorf("date: infinity unsupported in vector Time")
			}
			return time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC).AddDate(0, 0, int(days)), nil
		case pgtype.TimestampOID, pgtype.TimestamptzOID:
			if len(buf) != 8 {
				return time.Time{}, fmt.Errorf("timestamp: binary length %d", len(buf))
			}
			return decodeMicrosY2K(int64(binary.BigEndian.Uint64(buf)))
		}
	}
	var t time.Time
	if err := sc.fallMap.Scan(m.oid, m.format, buf, &t); err != nil {
		return time.Time{}, fmt.Errorf("time OID %d: %w (use legacy Source)", m.oid, err)
	}
	return t, nil
}

// microsecFromUnixEpochToY2K and the infinity sentinels mirror
// pgtype/timestamptz.go so the conversion below stays exact.
const microsecFromUnixEpochToY2K = 946_684_800 * 1_000_000

const (
	negInfinityMicros = -9223372036854775808
	posInfinityMicros = 9223372036854775807
)

func decodeMicrosY2K(micros int64) (time.Time, error) {
	switch micros {
	case negInfinityMicros, posInfinityMicros:
		return time.Time{}, fmt.Errorf("timestamp: infinity unsupported in vector Time")
	}
	// Identical arithmetic to pgx's binary timestamptz scan plan
	// (including the zero remainder term) so vector and legacy times
	// agree exactly, including pre-1970 negatives where Go division
	// truncates toward zero on both sides.
	sec := microsecFromUnixEpochToY2K/1_000_000 + micros/1_000_000
	nsec := (microsecFromUnixEpochToY2K%1_000_000)*1_000 + (micros%1_000_000)*1_000
	return time.Unix(sec, nsec), nil
}

// decodeNumeric keeps the text form: text numeric is copied, binary
// numeric coerced via pgtype.
func (sc *vscanner) decodeNumeric(buf []byte, m *colMeta) (string, error) {
	if m.format == pgtype.TextFormatCode {
		return string(buf), nil
	}
	var t pgtype.Text
	if err := sc.fallMap.Scan(m.oid, m.format, buf, &t); err != nil {
		return "", fmt.Errorf("numeric: %w (use legacy Source)", err)
	}
	if !t.Valid {
		return "", fmt.Errorf("numeric: unexpected NULL decode")
	}
	return t.String, nil
}
