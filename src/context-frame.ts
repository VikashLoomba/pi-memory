/**
 * Context Frame Builder
 *
 * MemTrust retrieval returns a packaged ContextFrame that is injected before the
 * model reasons. This file builds a compact XML frame with explicit budgeting.
 */

import type { Episode } from "./episodic.js";
import type { Relationship } from "./profile.js";

export interface ContextFrameOptions {
	maxTokens: number;
	maxEpisodes?: number;
	maxFacts?: number;
}

export interface ContextFrameResult {
	frame: string;
	usedEpisodes: number;
	usedFacts: number;
	estimatedTokens: number;
}

function escapeXml(value: string): string {
	return value
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/\"/g, "&quot;")
		.replace(/'/g, "&apos;");
}

function approximateTokens(value: string): number {
	return Math.ceil(value.length / 4);
}

function truncate(value: string, maxChars: number): string {
	if (value.length <= maxChars) return value;
	return `${value.slice(0, maxChars - 1)}…`;
}

function formatAge(timestamp: number): string {
	const ageMs = Date.now() - timestamp;
	const minutes = Math.floor(ageMs / 60_000);
	const hours = Math.floor(ageMs / 3_600_000);
	const days = Math.floor(ageMs / 86_400_000);
	if (minutes < 1) return "just-now";
	if (minutes < 60) return `${minutes}m`;
	if (hours < 24) return `${hours}h`;
	return `${days}d`;
}

export function buildContextFrame(
	episodes: Episode[],
	relationships: Relationship[],
	options: ContextFrameOptions,
): ContextFrameResult {
	const maxTokens = Math.max(256, options.maxTokens);
	const maxEpisodes = options.maxEpisodes ?? 5;
	const maxFacts = options.maxFacts ?? 12;

	let frame = `<MemoryContext>\n`;
	let usedEpisodes = 0;
	let usedFacts = 0;

	const episodeBudget = Math.floor(maxTokens * 0.55);
	const factBudget = Math.floor(maxTokens * 0.45);

	let episodeSection = "";
	let factSection = "";

	for (const episode of episodes.slice(0, maxEpisodes)) {
		const item =
			`  <RecentEpisode source="${escapeXml(episode.sourceApp)}" age="${formatAge(episode.timestamp)}">` +
			`${escapeXml(truncate(episode.content.trim(), 700))}</RecentEpisode>\n`;
		if (approximateTokens(episodeSection + item) > episodeBudget) break;
		episodeSection += item;
		usedEpisodes++;
	}

	for (const relationship of relationships.slice(0, maxFacts)) {
		const item =
			`  <ProfileFact confidence="${relationship.confidence.toFixed(2)}">` +
			`${escapeXml(relationship.source)} ${escapeXml(relationship.relation)} ${escapeXml(relationship.target)}` +
			`</ProfileFact>\n`;
		if (approximateTokens(factSection + item) > factBudget) break;
		factSection += item;
		usedFacts++;
	}

	if (episodeSection) {
		frame += episodeSection;
	}
	if (factSection) {
		frame += factSection;
	}
	frame += `</MemoryContext>`;

	return {
		frame: usedEpisodes === 0 && usedFacts === 0 ? "" : frame,
		usedEpisodes,
		usedFacts,
		estimatedTokens: approximateTokens(frame),
	};
}

export const MEMORY_INSTRUCTIONS = `
You may receive a <MemoryContext> block containing relevant long-term memory.
- <RecentEpisode> items are episodic memories recalled by semantic similarity.
- <ProfileFact> items are durable facts distilled into profile memory.
Use these memories naturally when relevant.
Do not mention the memory system itself unless the user asks about it.
If memory conflicts with the current user message, prefer the current user message.
`.trim();
