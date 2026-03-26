/**
 * Profile Memory Layer
 *
 * MemTrust functional mapping:
 * - durable facts are distilled from episodic episodes into a graph
 * - nodes represent entities, edges represent typed relationships
 * - successful retrieval reinforces memory strength
 * - low-retention relationships are archived/pruned
 */

import { createNodeEngines } from "@surrealdb/node";
import { RecordId, Surreal, Table, createRemoteEngines } from "surrealdb";
import { join } from "node:path";
import { mkdirSync } from "node:fs";
import { applyAsyncMigrations } from "./migrations.js";

export interface ExtractedFact {
	source: string;
	source_type?: string;
	relation: string;
	target: string;
	target_type?: string;
	confidence?: number;
}

type TemporalValue = string | Date;

interface EntityRecord {
	id: string;
	type: string;
	name: string;
	normalized_name: string;
	created_at: TemporalValue;
	updated_at: TemporalValue;
}

interface RelationRecord {
	id: string;
	in: EntityRecord | string;
	out: EntityRecord | string;
	relation: string;
	confidence: number;
	memory_strength: number;
	last_accessed: TemporalValue;
	created_at: TemporalValue;
	updated_at: TemporalValue;
}

export interface Relationship {
	id: string;
	sourceId: string;
	targetId: string;
	source: string;
	target: string;
	sourceType: string;
	targetType: string;
	relation: string;
	confidence: number;
	memoryStrength: number;
	lastAccessed: string;
	createdAt: string;
	updatedAt: string;
}

export interface ProfileStats {
	entityCount: number;
	relationshipCount: number;
	archivedRelationshipCount: number;
}

export interface ProfileStore {
	init(): Promise<void>;
	ingestFacts(facts: ExtractedFact[]): Promise<number>;
	queryByKeywords(keywords: string[], limit?: number): Promise<Relationship[]>;
	reinforce(relationshipId: string): Promise<void>;
	decay(threshold?: number): Promise<string[]>;
	archive(ids: string[]): Promise<void>;
	listTopRelationships(limit?: number): Promise<Relationship[]>;
	getStats(): Promise<ProfileStats>;
	clear(): Promise<void>;
	close(): Promise<void>;
}

export interface ProfileStoreOptions {
	now?: () => Date;
}

function slugify(input: string): string {
	const slug = input
		.normalize("NFKD")
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, "-")
		.replace(/^-+|-+$/g, "")
		.slice(0, 80);
	return slug || "unknown";
}

function normalizeEntityName(input: string): string {
	return input.trim().replace(/\s+/g, " ");
}

function normalizeRelation(input: string): string {
	return input
		.trim()
		.toUpperCase()
		.replace(/[^A-Z0-9]+/g, "_")
		.replace(/^_+|_+$/g, "")
		.slice(0, 64) || "RELATED_TO";
}

function entityRecord(type: string, name: string): RecordId<"entity", string> {
	return new RecordId("entity", `${slugify(type)}-${slugify(name)}`);
}

function toIso(value: TemporalValue): string {
	return value instanceof Date ? value.toISOString() : value;
}

function asRelationship(record: RelationRecord): Relationship {
	const sourceRecord = typeof record.in === "string" ? { id: record.in, type: "concept", name: record.in, normalized_name: record.in, created_at: record.created_at, updated_at: record.updated_at } : record.in;
	const targetRecord = typeof record.out === "string" ? { id: record.out, type: "concept", name: record.out, normalized_name: record.out, created_at: record.created_at, updated_at: record.updated_at } : record.out;
	return {
		id: record.id,
		sourceId: sourceRecord.id,
		targetId: targetRecord.id,
		source: sourceRecord.name,
		target: targetRecord.name,
		sourceType: sourceRecord.type,
		targetType: targetRecord.type,
		relation: record.relation,
		confidence: record.confidence,
		memoryStrength: record.memory_strength,
		lastAccessed: toIso(record.last_accessed),
		createdAt: toIso(record.created_at),
		updatedAt: toIso(record.updated_at),
	};
}

export function createProfileStore(dataDir: string, options: ProfileStoreOptions = {}): ProfileStore {
	mkdirSync(dataDir, { recursive: true });
	const dbPath = join(dataDir, "profile.db");
	const db = new Surreal({
		engines: {
			...createRemoteEngines(),
			...createNodeEngines(),
		},
	});
	const now = options.now ?? (() => new Date());

	async function fetchAllRelationships(limit = 250): Promise<Relationship[]> {
		const result = await db.query<[RelationRecord[]]>(`
			SELECT * FROM relates_to
			LIMIT ${Math.max(1, Math.min(limit, 1000))}
			FETCH in, out
		`);
		return (result[0] ?? []).map(asRelationship);
	}

	async function upsertEntity(type: string, name: string): Promise<RecordId<"entity", string>> {
		const normalizedName = normalizeEntityName(name);
		const recordId = entityRecord(type, normalizedName);
		const currentTime = now();
		await db
			.upsert(recordId)
			.content({
				type: type.trim() || "concept",
				name: normalizedName,
				normalized_name: normalizedName.toLowerCase(),
				created_at: currentTime,
				updated_at: currentTime,
			});
		return recordId;
	}

	async function findExistingRelationship(
		source: RecordId<"entity", string>,
		target: RecordId<"entity", string>,
		relation: string,
	): Promise<Relationship | null> {
		const result = await db.query<[RelationRecord[]]>(
			`SELECT * FROM relates_to WHERE in = $source AND out = $target AND relation = $relation LIMIT 1 FETCH in, out`,
			{ source, target, relation },
		);
		const row = result[0]?.[0];
		return row ? asRelationship(row) : null;
	}

	async function getRelationshipById(id: string): Promise<Relationship | null> {
		const result = await db.query<[RelationRecord[]]>(`SELECT * FROM $id FETCH in, out`, { id });
		const row = result[0]?.[0];
		return row ? asRelationship(row) : null;
	}

	async function upsertRelationship(
		source: RecordId<"entity", string>,
		target: RecordId<"entity", string>,
		relation: string,
		confidence = 0.8,
	): Promise<void> {
		const existing = await findExistingRelationship(source, target, relation);
		if (existing) {
			const mergedConfidence = Math.max(existing.confidence, confidence);
			const nextStrength = Math.min(existing.memoryStrength + 0.3, 20);
			await db.query(
				`UPDATE $id SET confidence = $confidence, memory_strength = $memoryStrength, last_accessed = time::now(), updated_at = time::now()`,
				{ id: existing.id, confidence: mergedConfidence, memoryStrength: nextStrength },
			);
			return;
		}

		const currentTime = now();
		await db.relate(source, new Table("relates_to"), target, {
			relation,
			confidence,
			memory_strength: 1.0,
			last_accessed: currentTime,
			created_at: currentTime,
			updated_at: currentTime,
		});
	}

	return {
		async init() {
			await db.connect(`surrealkv://${dbPath}`);
			await db.use({ namespace: "pi_memory", database: "profile" });

			await applyAsyncMigrations(dataDir, "profile", [
				{
					version: 1,
					description: "Initial profile memory schema",
					up: async () => {
						await db.query(`
							DEFINE TABLE IF NOT EXISTS entity SCHEMAFULL;
							DEFINE FIELD IF NOT EXISTS type ON entity TYPE string;
							DEFINE FIELD IF NOT EXISTS name ON entity TYPE string;
							DEFINE FIELD IF NOT EXISTS normalized_name ON entity TYPE string;
							DEFINE FIELD IF NOT EXISTS created_at ON entity TYPE datetime VALUE $value OR time::now();
							DEFINE FIELD IF NOT EXISTS updated_at ON entity TYPE datetime VALUE $value OR time::now();
							DEFINE INDEX IF NOT EXISTS idx_entity_normalized_name ON entity COLUMNS normalized_name;

							DEFINE TABLE IF NOT EXISTS relates_to SCHEMAFULL TYPE RELATION IN entity OUT entity;
							DEFINE FIELD IF NOT EXISTS relation ON relates_to TYPE string;
							DEFINE FIELD IF NOT EXISTS confidence ON relates_to TYPE float;
							DEFINE FIELD IF NOT EXISTS memory_strength ON relates_to TYPE float;
							DEFINE FIELD IF NOT EXISTS last_accessed ON relates_to TYPE datetime;
							DEFINE FIELD IF NOT EXISTS created_at ON relates_to TYPE datetime;
							DEFINE FIELD IF NOT EXISTS updated_at ON relates_to TYPE datetime;

							DEFINE TABLE IF NOT EXISTS relation_archive SCHEMALESS;
						`);
					},
				},
			]);
		},

		async ingestFacts(facts: ExtractedFact[]): Promise<number> {
			let ingested = 0;
			for (const fact of facts) {
				const source = normalizeEntityName(fact.source || "");
				const target = normalizeEntityName(fact.target || "");
				if (!source || !target) continue;

				const sourceType = (fact.source_type?.trim() || "concept").toLowerCase();
				const targetType = (fact.target_type?.trim() || "concept").toLowerCase();
				const relation = normalizeRelation(fact.relation || "RELATED_TO");
				const confidence = Math.max(0, Math.min(1, fact.confidence ?? 0.8));

				const sourceId = await upsertEntity(sourceType, source);
				const targetId = await upsertEntity(targetType, target);
				await upsertRelationship(sourceId, targetId, relation, confidence);
				ingested++;
			}
			return ingested;
		},

		async queryByKeywords(keywords: string[], limit = 12): Promise<Relationship[]> {
			const normalized = keywords.map((k) => k.trim().toLowerCase()).filter(Boolean);
			const all = await fetchAllRelationships(500);
			if (all.length === 0) return [];

			if (normalized.length === 0) {
				return all
					.sort((a, b) => b.memoryStrength - a.memoryStrength || b.confidence - a.confidence)
					.slice(0, limit);
			}

			const seedMatches = new Map<string, number>();
			for (const rel of all) {
				let score = 0;
				const haystacks = [rel.source.toLowerCase(), rel.target.toLowerCase(), rel.relation.toLowerCase()];
				for (const keyword of normalized) {
					if (haystacks.some((value) => value.includes(keyword))) score += 2;
				}
				if (score > 0) seedMatches.set(rel.id, score);
			}

			const expandedEntityIds = new Set<string>();
			for (const rel of all) {
				if (seedMatches.has(rel.id)) {
					expandedEntityIds.add(rel.sourceId);
					expandedEntityIds.add(rel.targetId);
				}
			}

			const scored = all
				.map((rel) => {
					const seedScore = seedMatches.get(rel.id) ?? 0;
					const hopScore =
						seedScore > 0
							? 1
							: expandedEntityIds.has(rel.sourceId) || expandedEntityIds.has(rel.targetId)
								? 0.75
								: 0;
					const strengthScore = rel.memoryStrength * 0.5;
					const confidenceScore = rel.confidence * 0.5;
					const totalScore = seedScore + hopScore + strengthScore + confidenceScore;
					return { rel, totalScore };
				})
				.filter((entry) => entry.totalScore > 0)
				.sort((a, b) => b.totalScore - a.totalScore)
				.slice(0, limit)
				.map((entry) => entry.rel);

			return scored.length > 0
				? scored
				: all
						.sort((a, b) => b.memoryStrength - a.memoryStrength || b.confidence - a.confidence)
						.slice(0, limit);
		},

		async reinforce(relationshipId: string) {
			const existing = await getRelationshipById(relationshipId);
			if (!existing) return;
			await db.query(
				`UPDATE $id SET memory_strength = $memoryStrength, last_accessed = time::now(), updated_at = time::now()`,
				{ id: relationshipId, memoryStrength: Math.min(existing.memoryStrength + 0.5, 20) },
			);
		},

		async decay(threshold = 0.1): Promise<string[]> {
			const all = await fetchAllRelationships(1000);
			const currentTime = now().getTime();
			const expired: string[] = [];

			for (const rel of all) {
				const lastAccessed = new Date(rel.lastAccessed).getTime();
				const hoursSinceAccess = Math.max(0, (currentTime - lastAccessed) / 3_600_000);
				const retention = Math.exp(-hoursSinceAccess / Math.max(0.1, rel.memoryStrength));
				if (retention < threshold) expired.push(rel.id);
			}

			return expired;
		},

		async archive(ids: string[]) {
			if (ids.length === 0) return;
			for (const id of ids) {
				const result = await db.query<[RelationRecord[]]>(`SELECT * FROM $id FETCH in, out`, { id });
				const relation = result[0]?.[0];
				if (!relation) continue;
				const normalized = asRelationship(relation);
				await db.insert(new Table("relation_archive"), {
					relationship_id: normalized.id,
					source: normalized.source,
					source_id: normalized.sourceId,
					source_type: normalized.sourceType,
					target: normalized.target,
					target_id: normalized.targetId,
					target_type: normalized.targetType,
					relation: normalized.relation,
					confidence: normalized.confidence,
					memory_strength: normalized.memoryStrength,
					last_accessed: normalized.lastAccessed,
					created_at: normalized.createdAt,
					updated_at: normalized.updatedAt,
					archived_at: now().toISOString(),
				});
				await db.query(`DELETE $id`, { id });
			}
		},

		async listTopRelationships(limit = 20): Promise<Relationship[]> {
			const all = await fetchAllRelationships(Math.max(limit, 50));
			return all
				.sort((a, b) => b.memoryStrength - a.memoryStrength || b.confidence - a.confidence)
				.slice(0, limit);
		},

		async getStats(): Promise<ProfileStats> {
			const entityCount = (await db.query<[Array<{ count: number }>]>(`SELECT count() AS count FROM entity GROUP ALL`))[0]?.[0]?.count ?? 0;
			const relationshipCount = (await db.query<[Array<{ count: number }>]>(`SELECT count() AS count FROM relates_to GROUP ALL`))[0]?.[0]?.count ?? 0;
			const archivedRelationshipCount =
				(await db.query<[Array<{ count: number }>]>(`SELECT count() AS count FROM relation_archive GROUP ALL`))[0]?.[0]?.count ?? 0;
			return { entityCount, relationshipCount, archivedRelationshipCount };
		},

		async clear() {
			await db.query(`DELETE relation_archive`);
			await db.query(`DELETE relates_to`);
			await db.query(`DELETE entity`);
		},

		async close() {
			await db.close();
		},
	};
}
