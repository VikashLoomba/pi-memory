import test from "node:test";
import assert from "node:assert/strict";
import { createProfileStore } from "../src/profile.js";
import { createTempDir, cleanupTempDir } from "./test-helpers.js";

test("profile memory stores durable facts as a deduplicated relationship graph", async () => {
	const dir = createTempDir("profile-dedup");
	const store = createProfileStore(dir);
	try {
		await store.init();
		await store.ingestFacts([
			{ source: "user", source_type: "user", relation: "prefers", target: "TypeScript", target_type: "language", confidence: 0.6 },
			{ source: "user", source_type: "user", relation: "PREFERS", target: "TypeScript", target_type: "language", confidence: 0.95 },
		]);

		const stats = await store.getStats();
		assert.equal(stats.entityCount, 2);
		assert.equal(stats.relationshipCount, 1);

		const facts = await store.queryByKeywords(["typescript"], 5);
		assert.equal(facts.length, 1);
		assert.equal(facts[0]?.source, "user");
		assert.equal(facts[0]?.relation, "PREFERS");
		assert.equal(facts[0]?.target, "TypeScript");
		assert.equal(facts[0]?.confidence, 0.95);
	} finally {
		await store.close();
		cleanupTempDir(dir);
	}
});

test("profile memory keyword recall prioritizes directly relevant facts and reinforcement increases their rank", async () => {
	const dir = createTempDir("profile-recall");
	const store = createProfileStore(dir);
	try {
		await store.init();
		await store.ingestFacts([
			{ source: "user", source_type: "user", relation: "USES", target: "TypeScript", target_type: "language", confidence: 0.8 },
			{ source: "project phoenix", source_type: "project", relation: "USES", target: "SurrealDB", target_type: "tool", confidence: 0.9 },
		]);

		const queryResults = await store.queryByKeywords(["typescript"], 5);
		assert.equal(queryResults[0]?.target, "TypeScript");

		const surrealFact = (await store.queryByKeywords(["surrealdb"], 5))[0];
		assert.ok(surrealFact, "expected a durable fact about SurrealDB");
		await store.reinforce(surrealFact.id);

		const top = await store.listTopRelationships(2);
		assert.equal(top[0]?.target, "SurrealDB");
	} finally {
		await store.close();
		cleanupTempDir(dir);
	}
});

test("profile memory archiving removes facts from active recall and increments archive stats", async () => {
	const dir = createTempDir("profile-archive");
	const store = createProfileStore(dir);
	try {
		await store.init();
		await store.ingestFacts([
			{ source: "user", relation: "LIKES", target: "Rust", confidence: 0.9 },
		]);

		const before = await store.listTopRelationships(10);
		assert.equal(before.length, 1);

		await store.archive([before[0]!.id]);

		const stats = await store.getStats();
		assert.equal(stats.relationshipCount, 0);
		assert.equal(stats.archivedRelationshipCount, 1);
		assert.equal((await store.queryByKeywords(["rust"], 5)).length, 0);
	} finally {
		await store.close();
		cleanupTempDir(dir);
	}
});

test("profile memory ignores malformed facts that are missing either endpoint", async () => {
	const dir = createTempDir("profile-invalid");
	const store = createProfileStore(dir);
	try {
		await store.init();
		const ingested = await store.ingestFacts([
			{ source: "", relation: "LIKES", target: "Go", confidence: 0.8 },
			{ source: "user", relation: "LIKES", target: "", confidence: 0.8 },
		]);

		assert.equal(ingested, 0);
		assert.deepEqual(await store.getStats(), {
			entityCount: 0,
			relationshipCount: 0,
			archivedRelationshipCount: 0,
		});
	} finally {
		await store.close();
		cleanupTempDir(dir);
	}
});
