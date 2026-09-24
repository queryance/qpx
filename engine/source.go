package engine

import "context"

// Source produces a stream of chunks. Implementations run a single
// underlying query or scan; parallelism happens downstream in the
// Scheduler's worker pool, not by fanning out the source.
// The result schema is exposed by the Scanner after Open, so opening the
// source runs the underlying query or scan exactly once.
type Source interface {
	// Open starts the underlying query or scan. The caller must Close
	// the returned Scanner.
	Open(ctx context.Context) (Scanner, error)
}

// Scanner streams chunks from an opened Source. Schema is valid immediately
// after Open. Next advances to the next chunk and reports whether one is
// available; Chunk returns it. When Next returns false, Err reports any
// failure (nil at end of stream).
type Scanner interface {
	Schema() Schema
	Next() bool
	Chunk() Chunk
	Err() error
	Close() error
}

// ChunkSizer is implemented by sources that accept a rows-per-chunk hint.
// The qpx package asserts this interface instead of an anonymous one.
type ChunkSizer interface {
	SetChunkSize(int)
}
