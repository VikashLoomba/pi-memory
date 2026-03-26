import test from "node:test";
import assert from "node:assert/strict";
import { extractFacts } from "../src/consolidation.js";

test("consolidation prefers structured tool output and deduplicates durable facts", async () => {
	const result = await extractFacts(
		["User says they prefer TypeScript over JavaScript."],
		{} as any,
		"test-key",
		{
			completeFn: (async () => ({
				role: "assistant",
				api: "test",
				provider: "test",
				model: "test",
				usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
				stopReason: "stop",
				content: [
					{
						type: "toolCall",
						id: "call-1",
						name: "emit_durable_facts",
						arguments: {
							facts: [
								{ source: " user ", relation: "prefers", target: "TypeScript", confidence: 0.7 },
								{ source: "user", relation: "PREFERS", target: "TypeScript", confidence: 0.95 },
							],
						},
					},
				],
				timestamp: 123,
			})) as any,
			now: () => 123,
		},
	);

	assert.equal(result.toolUsed, true);
	assert.equal(result.facts.length, 1);
	assert.deepEqual(result.facts[0], {
		source: "user",
		source_type: "concept",
		relation: "PREFERS",
		target: "TypeScript",
		target_type: "concept",
		confidence: 0.95,
	});
});

test("consolidation falls back to parsing JSON text when the model does not emit tool calls", async () => {
	const result = await extractFacts(
		["Project Phoenix uses SurrealDB for profile memory."],
		{} as any,
		"test-key",
		{
			completeFn: (async () => ({
				role: "assistant",
				api: "test",
				provider: "test",
				model: "test",
				usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
				stopReason: "stop",
				content: [
					{
						type: "text",
						text: JSON.stringify([
							{ source: "Project Phoenix", source_type: "project", relation: "uses", target: "SurrealDB", target_type: "tool", confidence: 0.88 },
						]),
					},
				],
				timestamp: 456,
			})) as any,
			now: () => 456,
		},
	);

	assert.equal(result.toolUsed, false);
	assert.deepEqual(result.facts, [
		{
			source: "Project Phoenix",
			source_type: "project",
			relation: "USES",
			target: "SurrealDB",
			target_type: "tool",
			confidence: 0.88,
		},
	]);
});

test("consolidation returns no durable facts when model output is malformed", async () => {
	const result = await extractFacts(
		["The user asked for a one-off shell command."],
		{} as any,
		"test-key",
		{
			completeFn: (async () => ({
				role: "assistant",
				api: "test",
				provider: "test",
				model: "test",
				usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 },
				stopReason: "stop",
				content: [{ type: "text", text: "not valid json" }],
				timestamp: 789,
			})) as any,
			now: () => 789,
		},
	);

	assert.equal(result.toolUsed, false);
	assert.deepEqual(result.facts, []);
});
