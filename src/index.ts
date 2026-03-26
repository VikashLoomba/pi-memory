/**
 * Pi Memory Extension
 *
 * MemTrust-inspired functional implementation for Pi:
 * - Episodic Memory: autobiographical vector memory (SQLite + sqlite-vec)
 * - Profile Memory: durable semantic graph (SurrealDB)
 * - Adaptive Forgetting: Ebbinghaus decay + cold storage/archive
 *
 * Retrieval is orchestrator-driven:
 * - every user turn runs recall in before_agent_start
 * - recalled memory is injected into the system prompt as a ContextFrame
 */

import type { AgentMessage } from "@mariozechner/pi-agent-core";
import type {
	AssistantMessage,
	Message,
	Model,
	TextContent,
	ToolCall,
	ToolResultMessage,
	UserMessage,
} from "@mariozechner/pi-ai";
import type { ExtensionAPI, ExtensionCommandContext, ExtensionContext } from "@mariozechner/pi-coding-agent";
import { Text } from "@mariozechner/pi-tui";
import { createEpisodicStore, type Episode, type EpisodicStore } from "./episodic.js";
import { embed } from "./embeddings.js";
import { createProfileStore, type ProfileStore } from "./profile.js";
import { extractFacts } from "./consolidation.js";
import { buildContextFrame, buildContextFrameFromRanked, MEMORY_INSTRUCTIONS } from "./context-frame.js";
import { fusionRank } from "./ranking.js";
import { maskPii, unmaskPii, deserializeMasks, type PiiMask } from "./pii.js";
import { createMemoryServer, type MemoryServer } from "./memory-server.js";
import { join } from "node:path";

const CONSOLIDATION_INTERVAL = 5;
const CONSOLIDATION_BATCH_SIZE = 12;
const RETRIEVAL_EPISODE_LIMIT = 5;
const RETRIEVAL_RELATION_LIMIT = 10;
const DECAY_THRESHOLD = 0.1;

interface RecallDebugInfo {
	query: string;
	usedEpisodes: number;
	usedFacts: number;
	estimatedTokens: number;
	keywords: string[];
	timestamp: number;
}

interface BashExecutionMessage {
	role: "bashExecution";
	command: string;
	output: string;
	exitCode: number;
	cancelled: boolean;
	truncated: boolean;
	fullOutputPath?: string;
	timestamp: number;
	excludeFromContext?: boolean;
}

function isUserMessage(message: AgentMessage): message is UserMessage {
	return typeof message === "object" && message !== null && "role" in message && message.role === "user";
}

function isAssistantMessage(message: AgentMessage): message is AssistantMessage {
	return typeof message === "object" && message !== null && "role" in message && message.role === "assistant";
}

function isToolResultMessage(message: AgentMessage): message is ToolResultMessage {
	return typeof message === "object" && message !== null && "role" in message && message.role === "toolResult";
}

function isBashExecutionMessage(message: AgentMessage): message is BashExecutionMessage {
	return typeof message === "object" && message !== null && "role" in message && message.role === "bashExecution";
}

function isTextContent(content: unknown): content is TextContent {
	return typeof content === "object" && content !== null && "type" in content && (content as { type?: string }).type === "text";
}

function isToolCallContent(content: unknown): content is ToolCall {
	return typeof content === "object" && content !== null && "type" in content && (content as { type?: string }).type === "toolCall";
}

function extractUserText(message: UserMessage): string {
	if (typeof message.content === "string") return message.content;
	return message.content.filter(isTextContent).map((part) => part.text).join("\n");
}

function extractAssistantText(message: AssistantMessage): string {
	return message.content.filter(isTextContent).map((part) => part.text).join("\n");
}

function extractKeywords(prompt: string): string[] {
	const stopWords = new Set([
		"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "to", "of", "in", "for", "on", "with",
		"at", "by", "from", "as", "and", "or", "if", "then", "than", "that", "this", "these", "those", "it", "its", "i",
		"me", "my", "you", "your", "we", "our", "they", "their", "please", "help", "want", "need", "make", "use", "get",
	]);
	return prompt
		.toLowerCase()
		.replace(/[^a-z0-9\s_-]/g, " ")
		.split(/\s+/)
		.map((token) => token.trim())
		.filter((token) => token.length > 2 && !stopWords.has(token))
		.slice(0, 10);
}

function approximatePromptBudget(ctx: ExtensionContext): number {
	const usage = ctx.getContextUsage();
	const contextWindow = (ctx.model as Model<any> | null | undefined)?.contextWindow ?? 200_000;
	if (!usage || usage.tokens === null) {
		return 1800;
	}
	const remaining = Math.max(0, contextWindow - usage.tokens);
	return Math.max(400, Math.min(3000, Math.floor(remaining * 0.12)));
}

function summarizeToolArguments(args: Record<string, unknown>): string {
	try {
		const json = JSON.stringify(args);
		return json.length > 300 ? `${json.slice(0, 299)}…` : json;
	} catch {
		return "{}";
	}
}

function serializeTurn(messages: AgentMessage[]): string | null {
	const parts: string[] = [];

	for (const message of messages) {
		if (isUserMessage(message)) {
			const text = extractUserText(message).trim();
			if (text) parts.push(`[User] ${text}`);
			continue;
		}

		if (isAssistantMessage(message)) {
			const text = extractAssistantText(message).trim();
			if (text) parts.push(`[Assistant] ${text}`);
			for (const content of message.content) {
				if (isToolCallContent(content)) {
					parts.push(`[ToolCall ${content.name}] ${summarizeToolArguments(content.arguments)}`);
				}
			}
			continue;
		}

		if (isToolResultMessage(message)) {
			const text = message.content.filter(isTextContent).map((part) => part.text).join("\n").trim();
			if (text) {
				const truncated = text.length > 700 ? `${text.slice(0, 699)}…` : text;
				parts.push(`[ToolResult ${message.toolName}] ${truncated}`);
			}
			continue;
		}

		if (isBashExecutionMessage(message)) {
			const output = message.output?.trim() || "";
			const truncated = output.length > 700 ? `${output.slice(0, 699)}…` : output;
			parts.push(`[Bash ${message.command}] exit=${message.exitCode}${truncated ? ` output=${truncated}` : ""}`);
		}
	}

	if (parts.length === 0) return null;
	return parts.join("\n");
}

async function getOpenAiApiKey(ctx: ExtensionContext): Promise<string | undefined> {
	if (process.env.OPENAI_API_KEY) return process.env.OPENAI_API_KEY;
	const modelRegistry = ctx.modelRegistry as { getApiKeyForProvider?: (provider: string) => Promise<string | undefined> };
	if (typeof modelRegistry.getApiKeyForProvider === "function") {
		return modelRegistry.getApiKeyForProvider("openai");
	}
	return undefined;
}

function sendMemoryMessage(pi: ExtensionAPI, content: string) {
	pi.sendMessage(
		{
			customType: "pi-memory",
			content,
			display: true,
		},
		{ triggerTurn: false },
	);
}

export default function memoryExtension(pi: ExtensionAPI): void {
	let episodic: EpisodicStore | null = null;
	let profile: ProfileStore | null = null;
	let memoryServer: MemoryServer | null = null;
	let lastInjectedFrame = "";
	let lastRecall: RecallDebugInfo | null = null;
	let consolidationInFlight = false;
	let dataDir = "";

	pi.registerMessageRenderer("pi-memory", (message, _options, theme) => {
		const content =
			typeof message.content === "string"
				? message.content
				: message.content.filter(isTextContent).map((part) => part.text).join("\n");
		return new Text(theme.fg("accent", content), 0, 0);
	});

	pi.registerCommand("memory-status", {
		description: "Show memory system stats",
		handler: async (_args, ctx) => {
			await ctx.waitForIdle();
			if (!episodic || !profile) {
				sendMemoryMessage(pi, "Memory is not initialized.");
				return;
			}
			const episodeStats = episodic.getStats();
			const profileStats = await profile.getStats();
			const recallSummary = lastRecall
				? `Last recall: ${lastRecall.usedEpisodes} episodes, ${lastRecall.usedFacts} facts, ~${lastRecall.estimatedTokens} tokens for query \"${lastRecall.query}\"`
				: "Last recall: none yet";
			sendMemoryMessage(
				pi,
				[
					`Memory directory: ${dataDir || "(not initialized)"}`,
					`Episodes: active=${episodeStats.activeCount}, cold=${episodeStats.coldCount}, unconsolidated=${episodeStats.unconsolidatedCount}`,
					`Profile graph: entities=${profileStats.entityCount}, relationships=${profileStats.relationshipCount}, archived=${profileStats.archivedRelationshipCount}`,
					recallSummary,
				].join("\n"),
			);
		},
	});

	pi.registerCommand("memory-peek", {
		description: "Inspect recent episodic memories and top profile facts",
		handler: async (args, ctx) => {
			await ctx.waitForIdle();
			if (!episodic || !profile) {
				sendMemoryMessage(pi, "Memory is not initialized.");
				return;
			}
			const [modeArg, countArg] = args.trim().split(/\s+/, 2);
			const mode = modeArg || "all";
			const count = Math.max(1, Math.min(20, Number.parseInt(countArg || "5", 10) || 5));
			const sections: string[] = [];

			if (mode === "all" || mode === "episodes") {
				const episodes = episodic.listRecent(count);
				sections.push(
					["Recent episodes:", ...episodes.map((episode) => `- #${episode.id} ${episode.content.slice(0, 240).replace(/\n/g, " ")}`)].join("\n"),
				);
			}

			if (mode === "all" || mode === "profile") {
				const facts = await profile.listTopRelationships(count);
				sections.push(
					["Top profile facts:", ...facts.map((fact) => `- ${fact.source} ${fact.relation} ${fact.target} (strength=${fact.memoryStrength.toFixed(2)}, confidence=${fact.confidence.toFixed(2)})`)].join("\n"),
				);
			}

			sendMemoryMessage(pi, sections.join("\n\n") || "No memory available yet.");
		},
	});

	pi.registerCommand("memory-context", {
		description: "Show the last injected memory context frame",
		handler: async (_args, ctx) => {
			await ctx.waitForIdle();
			sendMemoryMessage(pi, lastInjectedFrame || "No context frame has been injected yet.");
		},
	});

	pi.registerCommand("memory-consolidate", {
		description: "Force a consolidation pass now",
		handler: async (_args, ctx) => {
			await ctx.waitForIdle();
			await runConsolidation(ctx, true);
		},
	});

	pi.registerCommand("memory-reset", {
		description: "Clear episodic memory, profile memory, and archives",
		handler: async (_args, ctx) => {
			await ctx.waitForIdle();
			if (!episodic || !profile) {
				sendMemoryMessage(pi, "Memory is not initialized.");
				return;
			}
			if (ctx.hasUI) {
				const ok = await ctx.ui.confirm("Reset memory?", "This will permanently delete active and archived memories for this project.");
				if (!ok) return;
			}
			await profile.clear();
			episodic.clear();
			lastInjectedFrame = "";
			lastRecall = null;
			sendMemoryMessage(pi, "Memory reset complete.");
		},
	});

	pi.registerCommand("memory-server", {
		description: "Show memory server status and connection info",
		handler: async (_args, ctx) => {
			await ctx.waitForIdle();
			if (!memoryServer || !memoryServer.port) {
				sendMemoryMessage(pi, "Memory server is not running.");
				return;
			}
			sendMemoryMessage(
				pi,
				[
					`Memory server: http://127.0.0.1:${memoryServer.port}`,
					`Token: ${memoryServer.token.slice(0, 8)}...`,
					`Endpoints:`,
					`  GET /api/memory/episodes?q=<query>&limit=5`,
					`  GET /api/memory/profile?keywords=<kw1>,<kw2>&limit=10`,
					`  GET /api/memory/stats`,
					`Auth: Bearer <token>`,
				].join("\n"),
			);
		},
	});

	pi.on("session_start", async (_event, ctx) => {
		try {
			if (memoryServer) await memoryServer.stop();
			if (episodic) episodic.close();
			if (profile) await profile.close();

			dataDir = join(ctx.cwd, ".pi", "memory");
			episodic = createEpisodicStore(dataDir);
			episodic.init();
			profile = createProfileStore(dataDir);
			await profile.init();
			lastInjectedFrame = "";
			lastRecall = null;
			consolidationInFlight = false;

			// Start memory server for cross-application access
			const openAiKey = await getOpenAiApiKey(ctx);
			memoryServer = createMemoryServer({
				episodic,
				profile,
				dataDir,
				embeddingOptions: openAiKey ? { apiKey: openAiKey } : undefined,
			});
			try {
				const { port } = await memoryServer.start();
				console.log(`[pi-memory] server listening on 127.0.0.1:${port}`);
			} catch (error) {
				console.error("[pi-memory] server start failed", error);
			}

			if (ctx.hasUI) {
				ctx.ui.setStatus("pi-memory", "🧠 memory ready");
				setTimeout(() => ctx.ui.setStatus("pi-memory", undefined), 2000);
			}
		} catch (error) {
			console.error("[pi-memory] session_start failed", error);
			if (ctx.hasUI) ctx.ui.notify(`Memory init failed: ${(error as Error).message}`, "error");
		}
	});

	pi.on("before_agent_start", async (event, ctx) => {
		if (!episodic || !profile || !event.prompt?.trim()) return;

		try {
			const openAiKey = await getOpenAiApiKey(ctx);
			if (!openAiKey) return;

			const queryEmbedding = await embed(event.prompt, { apiKey: openAiKey });
			const keywords = extractKeywords(event.prompt);
			const [episodesResult, profileResult] = await Promise.allSettled([
				Promise.resolve(episodic.query(queryEmbedding, RETRIEVAL_EPISODE_LIMIT)),
				profile.queryByKeywords(keywords, RETRIEVAL_RELATION_LIMIT),
			]);

			const episodes = episodesResult.status === "fulfilled" ? episodesResult.value : [];
			const relationships = profileResult.status === "fulfilled" ? profileResult.value : [];

			for (const episode of episodes) episodic.reinforce(episode.id);
			for (const relationship of relationships) await profile.reinforce(relationship.id);

			const ranked = fusionRank(episodes, relationships, keywords);
			const frameResult = buildContextFrameFromRanked(ranked, {
				maxTokens: approximatePromptBudget(ctx),
			});

			lastInjectedFrame = frameResult.frame;
			lastRecall = {
				query: event.prompt,
				usedEpisodes: frameResult.usedEpisodes,
				usedFacts: frameResult.usedFacts,
				estimatedTokens: frameResult.estimatedTokens,
				keywords,
				timestamp: Date.now(),
			};

			if (!frameResult.frame) return;
			return {
				systemPrompt: `${event.systemPrompt}\n\n${MEMORY_INSTRUCTIONS}\n\n${frameResult.frame}`,
			};
		} catch (error) {
			console.error("[pi-memory] recall failed", error);
			return;
		}
	});

	pi.on("agent_end", async (event, ctx) => {
		if (!episodic || !profile) return;

		try {
			const openAiKey = await getOpenAiApiKey(ctx);
			const episodeText = serializeTurn(event.messages);
			if (openAiKey && episodeText) {
				const { masked, masks } = maskPii(episodeText);
				const piiMasksJson = masks.length > 0 ? JSON.stringify(masks) : null;
				const episodeEmbedding = await embed(masked, { apiKey: openAiKey });
				episodic.insert(masked, episodeEmbedding, "pi", piiMasksJson);
			}

			if (episodic.countUnconsolidated() >= CONSOLIDATION_INTERVAL && !consolidationInFlight) {
				void runConsolidation(ctx, false);
			}
		} catch (error) {
			console.error("[pi-memory] agent_end failed", error);
		}
	});

	pi.on("session_shutdown", async () => {
		try {
			if (memoryServer) await memoryServer.stop();
			episodic?.close();
			await profile?.close();
		} catch (error) {
			console.error("[pi-memory] session_shutdown failed", error);
		}
		memoryServer = null;
		episodic = null;
		profile = null;
	});

	async function runConsolidation(ctx: ExtensionContext | ExtensionCommandContext, forced: boolean): Promise<void> {
		if (!episodic || !profile || consolidationInFlight) return;
		const unconsolidatedCount = episodic.countUnconsolidated();
		if (!forced && unconsolidatedCount < CONSOLIDATION_INTERVAL) return;
		consolidationInFlight = true;

		try {
			if (ctx.hasUI) ctx.ui.setStatus("pi-memory", "🧠 consolidating memory…");
			const batch = episodic.getUnconsolidated(CONSOLIDATION_BATCH_SIZE);
			if (batch.length === 0) return;
			if (!ctx.model) return;
			const apiKey = await ctx.modelRegistry.getApiKey(ctx.model);
			if (!apiKey) return;

			const result = await extractFacts(
				batch.map((episode) => episode.content),
				ctx.model,
				apiKey,
			);

			const ingested = result.facts.length > 0 ? await profile.ingestFacts(result.facts) : 0;
			episodic.markConsolidated(batch.map((episode) => episode.id));

			const decayedEpisodeIds = episodic.decay(DECAY_THRESHOLD);
			if (decayedEpisodeIds.length > 0) episodic.archive(decayedEpisodeIds);

			const decayedRelationshipIds = await profile.decay(DECAY_THRESHOLD);
			if (decayedRelationshipIds.length > 0) await profile.archive(decayedRelationshipIds);

			const summary = [
				`Consolidation complete${forced ? " (forced)" : ""}.`,
				`Episodes processed: ${batch.length}`,
				`Durable facts ingested: ${ingested}`,
				`Extraction mode: ${result.toolUsed ? "tool-call" : "text-fallback"}`,
				`Archived episodes: ${decayedEpisodeIds.length}`,
				`Archived profile facts: ${decayedRelationshipIds.length}`,
			].join("\n");

			if (forced) {
				sendMemoryMessage(pi, summary);
			} else if (ctx.hasUI) {
				ctx.ui.notify(`Memory consolidated: ${ingested} facts, ${decayedEpisodeIds.length + decayedRelationshipIds.length} archived`, "info");
			}
		} catch (error) {
			console.error("[pi-memory] consolidation failed", error);
			if (ctx.hasUI) ctx.ui.notify(`Memory consolidation failed: ${(error as Error).message}`, "error");
		} finally {
			consolidationInFlight = false;
			if (ctx.hasUI) ctx.ui.setStatus("pi-memory", undefined);
		}
	}
}
