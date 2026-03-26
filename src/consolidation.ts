/**
 * Consolidation Engine
 *
 * Runs every N turns and distills recent episodic memories into durable facts.
 * The extraction is done with Pi's model provider stack via `complete()`.
 *
 * Per user requirement, the extraction is structured via a custom tool schema.
 */

import { complete, type AssistantMessage, type Message, type Model, type TextContent, type ToolCall } from "@mariozechner/pi-ai";
import { Type, type Static } from "@sinclair/typebox";
import type { ExtractedFact } from "./profile.js";

const DurableFactSchema = Type.Object({
	source: Type.String({ description: "Subject entity, e.g. user, Python, Project Alpha" }),
	source_type: Type.Optional(
		Type.String({ description: "Entity type such as user, person, project, organization, tool, concept" }),
	),
	relation: Type.String({ description: "Relationship verb in SCREAMING_SNAKE_CASE" }),
	target: Type.String({ description: "Object entity" }),
	target_type: Type.Optional(Type.String({ description: "Entity type of the target" })),
	confidence: Type.Number({ minimum: 0, maximum: 1 }),
});

const EmitDurableFactsTool = {
	name: "emit_durable_facts",
	description:
		"Emit the durable facts extracted from recent conversation episodes. Call this tool exactly once with the final structured facts.",
	parameters: Type.Object({
		facts: Type.Array(DurableFactSchema),
	}),
};

type EmitDurableFactsArgs = Static<typeof EmitDurableFactsTool.parameters>;

const EXTRACTION_SYSTEM_PROMPT = `You are a memory consolidation worker for an AI memory system.

Your task is to read recent conversation episodes and extract only durable, reusable facts suitable for long-term profile memory.

Extract facts about:
- user preferences
- project context and technology choices
- recurring workflows and behaviors
- important entities (people, projects, tools, organizations)
- stable relationship facts that future turns would benefit from

Do NOT extract:
- transient one-off requests
- implementation details that are obviously temporary
- low-confidence guesses
- sensitive facts unless they are clearly necessary and explicit in the conversation

You MUST call the emit_durable_facts tool exactly once.
Return an empty facts array if nothing durable should be remembered.`;

export interface ConsolidationResult {
	facts: ExtractedFact[];
	rawAssistant: AssistantMessage;
	toolUsed: boolean;
}

export interface ExtractFactsOptions {
	completeFn?: typeof complete;
	now?: () => number;
}

function isToolCall(content: AssistantMessage["content"][number]): content is ToolCall {
	return content.type === "toolCall";
}

function normalizeFact(fact: ExtractedFact): ExtractedFact | null {
	const source = fact.source?.trim();
	const target = fact.target?.trim();
	const relation = fact.relation
		?.trim()
		.toUpperCase()
		.replace(/[^A-Z0-9]+/g, "_")
		.replace(/^_+|_+$/g, "");
	if (!source || !target || !relation) return null;
	return {
		source,
		source_type: fact.source_type?.trim().toLowerCase() || "concept",
		relation,
		target,
		target_type: fact.target_type?.trim().toLowerCase() || "concept",
		confidence: Math.max(0, Math.min(1, fact.confidence ?? 0.8)),
	};
}

function dedupeFacts(facts: ExtractedFact[]): ExtractedFact[] {
	const byKey = new Map<string, ExtractedFact>();
	for (const fact of facts) {
		const key = `${fact.source}|${fact.relation}|${fact.target}`;
		const existing = byKey.get(key);
		if (!existing || (fact.confidence ?? 0) > (existing.confidence ?? 0)) {
			byKey.set(key, fact);
		}
	}
	return [...byKey.values()];
}

function parseFactsFromToolCalls(message: AssistantMessage): ExtractedFact[] {
	const toolCalls = message.content.filter(isToolCall).filter((call) => call.name === EmitDurableFactsTool.name);
	const facts: ExtractedFact[] = [];
	for (const call of toolCalls) {
		const args = call.arguments as Partial<EmitDurableFactsArgs>;
		if (!Array.isArray(args.facts)) continue;
		for (const rawFact of args.facts) {
			const normalized = normalizeFact(rawFact);
			if (normalized) facts.push(normalized);
		}
	}
	return dedupeFacts(facts);
}

function parseFactsFromTextFallback(message: AssistantMessage): ExtractedFact[] {
	const rawText = message.content
		.filter((content): content is TextContent => content.type === "text")
		.map((content) => content.text)
		.join("\n")
		.trim();

	if (!rawText) return [];

	try {
		const cleaned = rawText.replace(/```json\s*/g, "").replace(/```\s*/g, "").trim();
		const parsed = JSON.parse(cleaned);
		if (!Array.isArray(parsed)) return [];
		return dedupeFacts(parsed.map((fact) => normalizeFact(fact)).filter((fact): fact is ExtractedFact => Boolean(fact)));
	} catch {
		return [];
	}
}

export async function extractFacts(
	recentEpisodes: string[],
	model: Model<any>,
	apiKey: string,
	options: ExtractFactsOptions = {},
): Promise<ConsolidationResult> {
	const completeFn = options.completeFn ?? complete;
	const timestamp = options.now?.() ?? Date.now();

	const userMessage: Message = {
		role: "user",
		content: [
			{
				type: "text",
				text: recentEpisodes.map((episode, index) => `--- Episode ${index + 1} ---\n${episode}`).join("\n\n"),
			},
		],
		timestamp,
	};

	const assistant = await completeFn(
		model,
		{
			systemPrompt: EXTRACTION_SYSTEM_PROMPT,
			messages: [userMessage],
			tools: [EmitDurableFactsTool],
		},
		{ apiKey, temperature: 0 },
	);

	const factsFromTools = parseFactsFromToolCalls(assistant);
	if (factsFromTools.length > 0) {
		return { facts: factsFromTools, rawAssistant: assistant, toolUsed: true };
	}

	const fallbackFacts = parseFactsFromTextFallback(assistant);
	return { facts: fallbackFacts, rawAssistant: assistant, toolUsed: false };
}
