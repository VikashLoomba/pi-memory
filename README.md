# pi-memory

MemTrust-inspired long-term memory extension for Pi.

It adds three cooperating memory layers:

- **Episodic Memory**: full turn summaries stored in a local vector database
- **Profile Memory**: durable facts stored in a SurrealDB relationship graph
- **Adaptive Forgetting**: stale memories decay and move to cold/archive storage

## Stack

- **Pi extension** in TypeScript
- **Embeddings**: OpenAI `text-embedding-3-large`
- **Episodic store**: `better-sqlite3` + `sqlite-vec`
- **Profile graph**: embedded SurrealDB via `surrealdb` + `@surrealdb/node`
- **Consolidation**: Pi model/provider path via `@mariozechner/pi-ai` `complete()` with a structured custom tool

## How it works

### Retrieval
On each user turn, the extension:

1. embeds the current prompt
2. recalls similar episodes from the vector store
3. recalls relevant profile facts from the graph
4. injects a compact `<MemoryContext>` block into the prompt before the model runs

### Write path
After each agent response, the extension:

1. serializes the completed turn into an episodic memory
2. embeds and stores it locally
3. every `N` unconsolidated episodes, runs consolidation
4. extracts durable facts and writes them into the profile graph
5. decays stale episodic/profile memories and archives them

## Storage layout

Project-local data is stored under:

```text
.pi/memory/
  episodic.sqlite
  profile.db
  schema-version.json
```

## Schema versioning and migrations

This package includes a tiny migration/versioning layer so future schema changes can be rolled out safely.

- Version state is stored in `.pi/memory/schema-version.json`
- Versions are tracked independently for the `episodic` and `profile` stores
- Each store applies migrations in ascending order on startup
- Downgrades are rejected if a database was initialized by a newer package version

To add a future migration:

- append a new migration object with the next integer version number
- do **not** rewrite old migrations in place
- keep migrations idempotent when possible

Current schema versions:

- `episodic`: `1`
- `profile`: `1`

## Requirements

You need an OpenAI API key for embeddings:

```bash
export OPENAI_API_KEY=...
```

Pi also needs a valid provider/model configuration for consolidation, because consolidation uses Pi's active model path.

## Commands

The extension registers these Pi commands:

- `/memory-status`
- `/memory-peek`
- `/memory-peek episodes 10`
- `/memory-peek profile 10`
- `/memory-context`
- `/memory-consolidate`
- `/memory-reset`

## Development

Install dependencies:

```bash
npm install
```

Type-check:

```bash
npm run check
```

Run tests:

```bash
npm test
```

## Test philosophy

Tests are behavior-driven, not implementation-driven.

They validate intended behavior such as:

- semantic episodic recall ordering
- persistence across reopen
- consolidation parsing behavior
- graph deduplication and recall
- archive behavior
- context-frame safety/escaping
- schema version manifest creation

The tests are allowed to fail if the implementation drifts from the intended design.

## Running the extension

From this project:

```bash
pi -e ./src/index.ts
```

Or place it in a project-local `.pi/extensions/` directory for Pi discovery.
