import test from "node:test";
import assert from "node:assert/strict";
import { buildContextFrame } from "../src/context-frame.js";
import type { Episode } from "../src/episodic.js";
import type { Relationship } from "../src/profile.js";

function makeEpisode(overrides: Partial<Episode> = {}): Episode {
	return {
		id: 1,
		content: "User likes TypeScript & Rust <both>",
		sourceApp: "pi",
		timestamp: Date.now(),
		memoryStrength: 1,
		lastAccessed: Date.now(),
		consolidated: false,
		...overrides,
	};
}

function makeRelationship(overrides: Partial<Relationship> = {}): Relationship {
	return {
		id: "relates_to:1",
		sourceId: "entity:user",
		targetId: "entity:typescript",
		source: "user",
		target: "TypeScript",
		sourceType: "user",
		targetType: "language",
		relation: "PREFERS",
		confidence: 0.9,
		memoryStrength: 2,
		lastAccessed: new Date().toISOString(),
		createdAt: new Date().toISOString(),
		updatedAt: new Date().toISOString(),
		...overrides,
	};
}

test("context frame returns an empty frame when there is no recalled memory", () => {
	const result = buildContextFrame([], [], { maxTokens: 600 });
	assert.equal(result.frame, "");
	assert.equal(result.usedEpisodes, 0);
	assert.equal(result.usedFacts, 0);
});

test("context frame escapes XML and respects explicit caps on recalled items", () => {
	const episodes = [
		makeEpisode({ id: 1, content: "User likes <TypeScript> & Rust" }),
		makeEpisode({ id: 2, content: "This second episode should be dropped by the maxEpisodes cap" }),
	];
	const facts = [
		makeRelationship({ id: "1", target: "TypeScript & Rust" }),
		makeRelationship({ id: "2", target: "This fact should be dropped", relation: "USES" }),
	];

	const result = buildContextFrame(episodes, facts, { maxTokens: 1200, maxEpisodes: 1, maxFacts: 1 });
	assert.equal(result.usedEpisodes, 1);
	assert.equal(result.usedFacts, 1);
	assert.match(result.frame, /<MemoryContext>/);
	assert.match(result.frame, /&lt;TypeScript&gt;/);
	assert.match(result.frame, /TypeScript &amp; Rust/);
	assert.doesNotMatch(result.frame, /second episode should be dropped/);
	assert.doesNotMatch(result.frame, /This fact should be dropped/);
});

test("context frame truncates oversized memory payloads to stay compact", () => {
	const oversizedEpisode = makeEpisode({
		content: "x".repeat(900),
	});

	const result = buildContextFrame([oversizedEpisode], [], { maxTokens: 1200, maxEpisodes: 1 });
	assert.equal(result.usedEpisodes, 1);
	assert.ok(result.frame.includes("…"), "expected long episodic content to be truncated");
	assert.ok(result.estimatedTokens > 0);
});
