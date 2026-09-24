package postgres

import (
	"encoding/binary"
	"math"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgtype"
)

// Pure decode tests: no database needed. Wire bytes are built by hand.

func beInt16(v int16) []byte {
	b := make([]byte, 2)
	binary.BigEndian.PutUint16(b, uint16(v))
	return b
}

func beInt32(v int32) []byte {
	b := make([]byte, 4)
	binary.BigEndian.PutUint32(b, uint32(v))
	return b
}

func beInt64(v int64) []byte {
	b := make([]byte, 8)
	binary.BigEndian.PutUint64(b, uint64(v))
	return b
}

func TestDecodeIntWidths(t *testing.T) {
	cases := []struct {
		buf  []byte
		want int64
	}{
		{beInt16(-32768), -32768},
		{beInt16(1234), 1234},
		{beInt32(-2147483648), -2147483648},
		{beInt32(100000), 100000},
		{beInt64(-9223372036854775807), -9223372036854775807},
		{beInt64(9876543210), 9876543210},
	}
	for _, tc := range cases {
		got, err := decodeInt(tc.buf, pgtype.BinaryFormatCode)
		if err != nil {
			t.Fatalf("decodeInt(%x): %v", tc.buf, err)
		}
		if got != tc.want {
			t.Errorf("decodeInt(%x) = %d, want %d", tc.buf, got, tc.want)
		}
	}
	if got, err := decodeInt([]byte("-42"), pgtype.TextFormatCode); err != nil || got != -42 {
		t.Errorf("text int = %d, %v; want -42", got, err)
	}
	if _, err := decodeInt([]byte{1, 2, 3}, pgtype.BinaryFormatCode); err == nil {
		t.Error("3-byte binary int must fail")
	}
}

func TestDecodeFloat(t *testing.T) {
	b4 := make([]byte, 4)
	binary.BigEndian.PutUint32(b4, math.Float32bits(1.5))
	if got, err := decodeFloat(b4, pgtype.BinaryFormatCode); err != nil || got != 1.5 {
		t.Errorf("float4 = %v, %v; want 1.5", got, err)
	}
	b8 := make([]byte, 8)
	binary.BigEndian.PutUint64(b8, math.Float64bits(-0.1))
	if got, err := decodeFloat(b8, pgtype.BinaryFormatCode); err != nil || got != -0.1 {
		t.Errorf("float8 = %v, %v; want -0.1", got, err)
	}
	if got, err := decodeFloat([]byte("0.5"), pgtype.TextFormatCode); err != nil || got != 0.5 {
		t.Errorf("text float = %v, %v; want 0.5", got, err)
	}
}

func TestDecodeBool(t *testing.T) {
	if v, _ := decodeBool([]byte{1}, pgtype.BinaryFormatCode); !v {
		t.Error("binary 1 must be true")
	}
	if v, _ := decodeBool([]byte{0}, pgtype.BinaryFormatCode); v {
		t.Error("binary 0 must be false")
	}
	for _, s := range []string{"t", "true", "1"} {
		if v, _ := decodeBool([]byte(s), pgtype.TextFormatCode); !v {
			t.Errorf("text %q must be true", s)
		}
	}
	for _, s := range []string{"f", "false", "0"} {
		if v, _ := decodeBool([]byte(s), pgtype.TextFormatCode); v {
			t.Errorf("text %q must be false", s)
		}
	}
	if _, err := decodeBool([]byte("yes"), pgtype.TextFormatCode); err == nil {
		t.Error("text 'yes' must fail (no silent coercion)")
	}
}

func TestDecodeMicrosY2K(t *testing.T) {
	// 2020-01-01 00:00:01 UTC = 631152001 seconds since unix epoch.
	want := time.Unix(631152001, 0)
	micros := int64(631152001-946_684_800) * 1_000_000
	got, err := decodeMicrosY2K(micros)
	if err != nil {
		t.Fatalf("decodeMicrosY2K: %v", err)
	}
	if !got.Equal(want) {
		t.Errorf("got %v, want %v", got, want)
	}
	// Pre-1970 negative micros: truncation-toward-zero must match pgx.
	// 1960-06-15 12:00:00 UTC = -301320000 unix seconds.
	wantNeg := time.Unix(-301320000, 123456000)
	microsNeg := int64(-301320000-946_684_800)*1_000_000 + 123456
	gotNeg, err := decodeMicrosY2K(microsNeg)
	if err != nil {
		t.Fatalf("decodeMicrosY2K negative: %v", err)
	}
	if !gotNeg.Equal(wantNeg) {
		t.Errorf("negative: got %v, want %v", gotNeg, wantNeg)
	}
	for _, inf := range []int64{posInfinityMicros, negInfinityMicros} {
		if _, err := decodeMicrosY2K(inf); err == nil {
			t.Errorf("infinity %d must fail explicitly", inf)
		}
	}
}

func TestDecodeBytesHex(t *testing.T) {
	got, err := decodeBytes([]byte(`\xdeadbeef`), pgtype.TextFormatCode)
	if err != nil {
		t.Fatalf("hex bytea: %v", err)
	}
	if len(got) != 4 || got[0] != 0xde || got[3] != 0xef {
		t.Errorf("hex decode wrong: %x", got)
	}
	raw := []byte{0, 1, 2, 255}
	got, err = decodeBytes(raw, pgtype.BinaryFormatCode)
	if err != nil {
		t.Fatalf("binary bytea: %v", err)
	}
	// Must be a copy: RawValues is only valid until the next Next.
	raw[0] = 9
	if got[0] != 0 {
		t.Error("binary bytea must be copied, not aliased")
	}
}
