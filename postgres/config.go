package postgres

import "github.com/queryance/qpx/engine"

// Config describes one query to stream.
type Config struct {
	ConnString string
	SQL        string
	Args       []any
	ChunkSize  int
}

// NewSource builds a Source from cfg.
func NewSource(cfg Config) *Source {
	return &Source{cfg: cfg}
}

// SetChunkSize overrides the rows-per-chunk hint. Non-positive values are ignored.
func (s *Source) SetChunkSize(n int) {
	if n > 0 {
		s.cfg.ChunkSize = n
	}
}

func (s *Source) chunkSize() int {
	if s.cfg.ChunkSize <= 0 {
		return engine.DefaultChunkSize
	}
	return s.cfg.ChunkSize
}
