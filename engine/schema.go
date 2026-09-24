package engine

// DataType is the set of column types the engine understands.
// Backends map their native types onto these; anything unknown falls
// back to String (or Bytes for binary data).
type DataType int

const (
	Int64 DataType = iota
	Float64
	Bool
	String
	Bytes
	Time
	Numeric
)

// String returns the human-readable name of the data type.
func (t DataType) String() string {
	switch t {
	case Int64:
		return "Int64"
	case Float64:
		return "Float64"
	case Bool:
		return "Bool"
	case String:
		return "String"
	case Bytes:
		return "Bytes"
	case Time:
		return "Time"
	case Numeric:
		return "Numeric"
	default:
		return "Unknown"
	}
}

// Field is a single named, typed column in a schema.
type Field struct {
	Name string
	Type DataType
}

// Schema is the ordered list of columns produced by a Source.
type Schema struct {
	Fields []Field
}
