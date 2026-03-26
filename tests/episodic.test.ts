import test from "node:test";
import assert from "node:assert/strict";
import { createEpisodicStore } from "../src/episodic.js";
import { createTempDir, cleanupTempDir, makeEmbedding } from "./test-helpers.js";

test("episodic memory retrieves the most semantically similar memories first", () => {
	const dir = createTempDir("episodic-query");
	try {
		const store = createEpisodicStore(dir);
		store.init();

		store.insert("User prefers TypeScript", makeEmbedding(0), "pi");
		store.insert("Project uses SurrealDB", makeEmbedding(1), "pi");

		const results = store.query(makeEmbedding(0), 2);
		assert.equal(results.length, 2);
		assert.equal(results[0]?.content, "User prefers TypeScript");
		assert.equal(results[1]?.content, "Project uses SurrealDB");

		store.close();
	} finally {
		cleanupTempDir(dir);
	}
});

test("episodic memory persists across reopen and retains consolidation state", () => {
	const dir = createTempDir("episodic-persist");
	try {
		let store = createEpisodicStore(dir);
		store.init();

		const first = store.insert("Episode one", makeEmbedding(0), "pi");
		store.insert("Episode two", makeEmbedding(1), "pi");
		store.markConsolidated([first.id]);
		store.close();

		store = createEpisodicStore(dir);
		store.init();

		const stats = store.getStats();
		assert.equal(stats.activeCount, 2);
		assert.equal(stats.unconsolidatedCount, 1);
		assert.equal(store.listRecent(10).length, 2);

		store.close();
	} finally {
		cleanupTempDir(dir);
	}
});

test("episodic memory decay can archive stale memories out of active recall", () => {
	const dir = createTempDir("episodic-decay");
	let currentTime = 0;
	try {
		const store = createEpisodicStore(dir, { now: () => currentTime });
		store.init();

		const episode = store.insert("Old preference", makeEmbedding(0), "pi");
		currentTime = 72 * 3_600_000;

		const expired = store.decay(0.5);
		assert.deepEqual(expired, [episode.id]);

		store.archive(expired);
		const stats = store.getStats();
		assert.equal(stats.activeCount, 0);
		assert.equal(stats.coldCount, 1);
		assert.equal(store.query(makeEmbedding(0), 5).length, 0);

		store.close();
	} finally {
		cleanupTempDir(dir);
	}
});

test("episodic memory rejects embeddings that do not match the configured model dimensions", () => {
	const dir = createTempDir("episodic-dimensions");
	try {
		const store = createEpisodicStore(dir);
		store.init();

		assert.throws(
			() => store.insert("Bad embedding", new Float32Array(10), "pi"),
			/Expected embedding with 3072 dimensions/,
		);
		assert.throws(
			() => store.query(new Float32Array(10), 5),
			/Expected embedding with 3072 dimensions/,
		);

		store.close();
	} finally {
		cleanupTempDir(dir);
	}
});
