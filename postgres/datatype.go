package postgres

import (
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/queryance/qpx/engine"
)

// MapDataType maps a Postgres type OID onto an engine DataType. Text-like
// and unknown types fall back to String; binary data maps to Bytes.
func MapDataType(oid uint32) engine.DataType {
	switch oid {
	case pgtype.Int2OID, pgtype.Int4OID, pgtype.Int8OID:
		return engine.Int64
	case pgtype.Float4OID, pgtype.Float8OID:
		return engine.Float64
	case pgtype.BoolOID:
		return engine.Bool
	case pgtype.ByteaOID:
		return engine.Bytes
	case pgtype.DateOID, pgtype.TimeOID, pgtype.TimetzOID,
		pgtype.TimestampOID, pgtype.TimestamptzOID, pgtype.IntervalOID:
		return engine.Time
	case pgtype.NumericOID:
		return engine.Numeric
	default:
		return engine.String
	}
}

func describeSchema(fds []pgconn.FieldDescription) engine.Schema {
	schema := engine.Schema{Fields: make([]engine.Field, len(fds))}
	for i, fd := range fds {
		schema.Fields[i] = engine.Field{Name: fd.Name, Type: MapDataType(fd.DataTypeOID)}
	}
	return schema
}
