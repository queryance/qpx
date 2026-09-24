package postgres

import (
	"fmt"

	"github.com/queryance/qpx/engine"
)

// PackChunk transposes row-major rows into a column-major chunk. Rows are
// not retained; cell values are shared by reference. Short rows are
// nil-filled and extra values are ignored, so output is always rectangular
// and consistent with the scanner.
func PackChunk(schema engine.Schema, rows [][]any) (engine.Chunk, error) {
	ncols := len(schema.Fields)
	cols := make([][]any, ncols)
	for i := range cols {
		cols[i] = make([]any, len(rows))
	}
	// Branch per row, not per cell: the common case is a full-width row.
	for r, row := range rows {
		if len(row) >= ncols {
			for c := 0; c < ncols; c++ {
				cols[c][r] = row[c]
			}
			continue
		}
		for c := 0; c < len(row); c++ {
			cols[c][r] = row[c]
		}
		for c := len(row); c < ncols; c++ {
			cols[c][r] = nil
		}
	}
	chunk, err := engine.NewChunk(schema, cols)
	if err != nil {
		return engine.Chunk{}, fmt.Errorf("postgres: pack chunk: %w", err)
	}
	return chunk, nil
}
