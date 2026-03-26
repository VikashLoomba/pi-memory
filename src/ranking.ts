/**
 * Fusion Ranking
 *
 * MemTrust retrieval combines multiple retrieval paths into a unified ranking.
 * This module implements weighted fusion scoring across episodic (vector) and
 * profile (keyword/graph) results with temporal recency and memory strength.
 */

import type { EpisodeWithDistance } from "./episodic.js";
import type { Relationship } from "./profile.js";

export interface ScoredMemoryItem {
	type: "episode" | "fact";
	item: EpisodeWithDistance | Relationship;
	fusionScore: number;
}

export interface FusionWeights {
	semantic: number;
	keyword: number;
	recency: number;
	strength: number;
}

const DEFAULT_WEIGHTS: FusionWeights = {
	semantic: 0.4,
	keyword: 0.3,
	recency: 0.2,
	strength: 0.1,
};

const RECENCY_HALF_LIFE_HOURS = 168; // 1 week
const MAX_MEMORY_STRENGTH = 20;

function isEpisode(item: ScoredMemoryItem): item is ScoredMemoryItem & { item: EpisodeWithDistance } {
	return item.type === "episode";
}

function computeRecencyScore(timestampMs: number): number {
	const hoursSince = Math.max(0, (Date.now() - timestampMs) / 3_600_000);
	return Math.exp(-hoursSince / RECENCY_HALF_LIFE_HOURS);
}

export function fusionRank(
	episodes: EpisodeWithDistance[],
	relationships: Relationship[],
	keywords: string[],
	weights: FusionWeights = DEFAULT_WEIGHTS,
): ScoredMemoryItem[] {
	const items: ScoredMemoryItem[] = [];
	const normalizedKeywords = keywords.map((k) => k.toLowerCase());

	for (const episode of episodes) {
		// Semantic: convert distance to 0-1 similarity
		const semanticScore = 1 / (1 + episode.distance);

		// Keyword: check if episode content matches any keywords
		const contentLower = episode.content.toLowerCase();
		let keywordScore = 0;
		for (const kw of normalizedKeywords) {
			if (contentLower.includes(kw)) keywordScore += 1;
		}
		keywordScore = normalizedKeywords.length > 0 ? keywordScore / normalizedKeywords.length : 0;

		const recencyScore = computeRecencyScore(episode.timestamp);
		const strengthScore = Math.min(episode.memoryStrength, MAX_MEMORY_STRENGTH) / MAX_MEMORY_STRENGTH;

		const fusionScore =
			weights.semantic * semanticScore +
			weights.keyword * keywordScore +
			weights.recency * recencyScore +
			weights.strength * strengthScore;

		items.push({ type: "episode", item: episode, fusionScore });
	}

	for (const rel of relationships) {
		// Semantic: profile facts have no vector distance, use 0
		const semanticScore = 0;

		// Keyword: check if relationship fields match keywords
		const haystacks = [rel.source.toLowerCase(), rel.target.toLowerCase(), rel.relation.toLowerCase()];
		let keywordScore = 0;
		for (const kw of normalizedKeywords) {
			if (haystacks.some((h) => h.includes(kw))) keywordScore += 1;
		}
		keywordScore = normalizedKeywords.length > 0 ? keywordScore / normalizedKeywords.length : 0;

		const lastAccessedMs = new Date(rel.lastAccessed).getTime();
		const recencyScore = computeRecencyScore(lastAccessedMs);
		const strengthScore = Math.min(rel.memoryStrength, MAX_MEMORY_STRENGTH) / MAX_MEMORY_STRENGTH;

		const fusionScore =
			weights.semantic * semanticScore +
			weights.keyword * keywordScore +
			weights.recency * recencyScore +
			weights.strength * strengthScore;

		items.push({ type: "fact", item: rel, fusionScore });
	}

	items.sort((a, b) => b.fusionScore - a.fusionScore);
	return items;
}

export { isEpisode };
