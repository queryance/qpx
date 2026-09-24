// Command basic streams a Postgres query through qpx and prints chunk stats.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/queryance/qpx"
	"github.com/queryance/qpx/postgres"
)

func main() {
	dsn := os.Getenv("TEST_POSTGRES_DSN")
	if dsn == "" {
		dsn = os.Getenv("DATABASE_URL")
	}
	if dsn == "" {
		log.Fatal("set TEST_POSTGRES_DSN (or DATABASE_URL) to a Postgres DSN")
	}
	ctx := context.Background()
	src := postgres.NewSource(postgres.Config{
		ConnString: dsn,
		SQL:        `SELECT g AS id, 'user-' || g::text AS name FROM generate_series(1, 100000) g`,
		ChunkSize:  4096,
	})
	it, err := qpx.Execute(ctx, src, qpx.ExecuteOptions{Workers: 4, QueueSize: 16, ChunkSize: 4096})
	if err != nil {
		log.Fatalf("execute: %v", err)
	}
	defer func() {
		_ = it.Close()
	}()
	total := 0
	for {
		c, ok, err := it.Next(ctx)
		if err != nil {
			log.Fatalf("iterate: %v", err)
		}
		if !ok {
			break
		}
		total += c.NumRows()
	}
	fmt.Printf("streamed %d rows\n", total)
}
