/**
 * Episodic Memory Layer
 *
 * MemTrust functional mapping:
 * - "Every user query, tool execution result, and agent response is serialized into a discrete Episode"
 * - Episodes are stored in a vector database for semantic recall
 * - Successful retrievals reinforce memory strength
 * - Low-retention memories move to cold storage
 */

import Database from "better-sqlite3";
import * as sqliteVec from "sqlite-vec";
import { join } from "node:path";
import { mkdirSync } from "node:fs";
import { applySyncMigrations } from "./migrations.js";

export const EMBEDDING_DIMENSIONS = 3072; // OpenAI text-embedding-3-large

export interface Episode {
	id: number;
	content: string;
	sourceApp: string;
	timestamp: number;
	memoryStrength: number;
	lastAccessed: number;
	consolidated: boolean;
}

export interface EpisodicStats {
	activeCount: number;
	coldCount: number;
	unconsolidatedCount: number;
}

export interface EpisodicStore {
	init(): void;
	insert(content: string, embedding: Float32Array, sourceApp: string): Episode;
	query(embedding: Float32Array, limit?: number): Episode[];
	reinforce(id: number): void;
	getUnconsolidated(limit?: number): Episode[];
	countUnconsolidated(): number;
	markConsolidated(ids: number[]): void;
	decay(threshold?: number): number[];
	archive(ids: number[]): void;
	listRecent(limit?: number): Episode[];
	getStats(): EpisodicStats;
	clear(): void;
	close(): void;
}

export interface EpisodicStoreOptions {
	now?: () => number;
}

function assertEmbeddingDimensions(embedding: Float32Array): void {
	if (embedding.length !== EMBEDDING_DIMENSIONS) {
		throw new Error(
			`Expected embedding with ${EMBEDDING_DIMENSIONS} dimensions, received ${embedding.length}.`,
		);
	}
}

function embeddingToJson(embedding: Float32Array): string {
	assertEmbeddingDimensions(embedding);
	return `[${Array.from(embedding).join(",")}]`;
}

function mapEpisodeRow(row: {
	id: number;
	content: string;
	source_app: string;
	timestamp: number;
	memory_strength: number;
	last_accessed: number;
	consolidated: number;
}): Episode {
	return {
		id: row.id,
		content: row.content,
		sourceApp: row.source_app,
		timestamp: row.timestamp,
		memoryStrength: row.memory_strength,
		lastAccessed: row.last_accessed,
		consolidated: row.consolidated === 1,
	};
}

export function createEpisodicStore(dataDir: string, options: EpisodicStoreOptions = {}): EpisodicStore {
	mkdirSync(dataDir, { recursive: true });
	const dbPath = join(dataDir, "episodic.sqlite");
	const db = new Database(dbPath);
	const now = options.now ?? (() => Date.now());

	db.pragma("journal_mode = WAL");
	db.pragma("foreign_keys = ON");
	sqliteVec.load(db);

	const initializeSchema = () => {
		applySyncMigrations(dataDir, "episodic", [
			{
				version: 1,
				description: "Initial episodic memory schema",
				up: () => {
					db.exec(`
						CREATE TABLE IF NOT EXISTS episodes (
							id INTEGER PRIMARY KEY AUTOINCREMENT,
							content TEXT NOT NULL,
							source_app TEXT NOT NULL,
							timestamp INTEGER NOT NULL,
							memory_strength REAL NOT NULL DEFAULT 1.0,
							last_accessed INTEGER NOT NULL,
							consolidated INTEGER NOT NULL DEFAULT 0 CHECK (consolidated IN (0, 1))
						);

						CREATE VIRTUAL TABLE IF NOT EXISTS episodes_vec USING vec0(
							embedding float[${EMBEDDING_DIMENSIONS}]
						);

						CREATE TABLE IF NOT EXISTS episodes_cold (
							id INTEGER PRIMARY KEY,
							content TEXT NOT NULL,
							source_app TEXT NOT NULL,
							timestamp INTEGER NOT NULL,
							memory_strength REAL NOT NULL,
							last_accessed INTEGER NOT NULL,
							consolidated INTEGER NOT NULL,
							archived_at INTEGER NOT NULL
						);

						CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp DESC);
						CREATE INDEX IF NOT EXISTS idx_episodes_consolidated ON episodes(consolidated, timestamp ASC);
						CREATE INDEX IF NOT EXISTS idx_episodes_last_accessed ON episodes(last_accessed ASC);
					`);
				},
			},
		]);
	};

	initializeSchema();

	const insertMetaStmt = db.prepare(`
		INSERT INTO episodes (content, source_app, timestamp, memory_strength, last_accessed, consolidated)
		VALUES (?, ?, ?, 1.0, ?, 0)
	`);
	const insertVecStmt = db.prepare(`
		INSERT INTO episodes_vec (rowid, embedding)
		VALUES (last_insert_rowid(), ?)
	`);
	const reinforceStmt = db.prepare(`
		UPDATE episodes
		SET memory_strength = memory_strength + 0.5,
		    last_accessed = ?
		WHERE id = ?
	`);
	const markConsolidatedStmt = db.prepare(`UPDATE episodes SET consolidated = 1 WHERE id = ?`);
	const archiveStmt = db.prepare(`
		INSERT INTO episodes_cold (
			id, content, source_app, timestamp, memory_strength, last_accessed, consolidated, archived_at
		)
		SELECT id, content, source_app, timestamp, memory_strength, last_accessed, consolidated, ?
		FROM episodes
		WHERE id = ?
	`);
	const deleteVecStmt = db.prepare(`DELETE FROM episodes_vec WHERE rowid = ?`);
	const deleteMetaStmt = db.prepare(`DELETE FROM episodes WHERE id = ?`);

	return {
		init() {
			initializeSchema();
		},

		insert(content: string, embedding: Float32Array, sourceApp: string): Episode {
			const timestamp = now();
			const embeddingJson = embeddingToJson(embedding);

			const insertTxn = db.transaction(() => {
				const info = insertMetaStmt.run(content, sourceApp, timestamp, timestamp);
				insertVecStmt.run(embeddingJson);
				const id = Number(info.lastInsertRowid);
				const row = db
					.prepare(`SELECT * FROM episodes WHERE id = ?`)
					.get(id) as Parameters<typeof mapEpisodeRow>[0] | undefined;
				if (!row) {
					throw new Error(`Failed to read back inserted episode ${id}`);
				}
				return mapEpisodeRow(row);
			});

			return insertTxn();
		},

		query(embedding: Float32Array, limit = 5): Episode[] {
			const rows = db
				.prepare(`
					SELECT
						e.id,
						e.content,
						e.source_app,
						e.timestamp,
						e.memory_strength,
						e.last_accessed,
						e.consolidated,
						v.distance
					FROM episodes_vec v
					JOIN episodes e ON e.id = v.rowid
					WHERE v.embedding MATCH ?
					  AND k = ?
					ORDER BY v.distance ASC, e.timestamp DESC
				`)
				.all(embeddingToJson(embedding), limit) as (Parameters<typeof mapEpisodeRow>[0] & { distance: number })[];

			return rows.map(mapEpisodeRow);
		},

		reinforce(id: number) {
			reinforceStmt.run(now(), id);
		},

		getUnconsolidated(limit = 10): Episode[] {
			const rows = db
				.prepare(`
					SELECT * FROM episodes
					WHERE consolidated = 0
					ORDER BY timestamp ASC
					LIMIT ?
				`)
				.all(limit) as Parameters<typeof mapEpisodeRow>[0][];

			return rows.map(mapEpisodeRow);
		},

		countUnconsolidated(): number {
			const row = db.prepare(`SELECT COUNT(*) AS count FROM episodes WHERE consolidated = 0`).get() as { count: number };
			return row.count;
		},

		markConsolidated(ids: number[]) {
			if (ids.length === 0) return;
			const txn = db.transaction((episodeIds: number[]) => {
				for (const id of episodeIds) {
					markConsolidatedStmt.run(id);
				}
			});
			txn(ids);
		},

		decay(threshold = 0.1): number[] {
			const currentTime = now();
			const rows = db.prepare(`SELECT id, memory_strength, last_accessed FROM episodes`).all() as {
				id: number;
				memory_strength: number;
				last_accessed: number;
			}[];

			const expired: number[] = [];
			for (const row of rows) {
				const hoursSinceAccess = Math.max(0, (currentTime - row.last_accessed) / 3_600_000);
				const retention = Math.exp(-hoursSinceAccess / Math.max(0.1, row.memory_strength));
				if (retention < threshold) expired.push(row.id);
			}
			return expired;
		},

		archive(ids: number[]) {
			if (ids.length === 0) return;
			const archivedAt = now();
			const txn = db.transaction((episodeIds: number[]) => {
				for (const id of episodeIds) {
					archiveStmt.run(archivedAt, id);
					deleteVecStmt.run(id);
					deleteMetaStmt.run(id);
				}
			});
			txn(ids);
		},

		listRecent(limit = 10): Episode[] {
			const rows = db
				.prepare(`SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?`)
				.all(limit) as Parameters<typeof mapEpisodeRow>[0][];
			return rows.map(mapEpisodeRow);
		},

		getStats(): EpisodicStats {
			const activeCount = (db.prepare(`SELECT COUNT(*) AS count FROM episodes`).get() as { count: number }).count;
			const coldCount = (db.prepare(`SELECT COUNT(*) AS count FROM episodes_cold`).get() as { count: number }).count;
			const unconsolidatedCount = this.countUnconsolidated();
			return { activeCount, coldCount, unconsolidatedCount };
		},

		clear() {
			const txn = db.transaction(() => {
				db.prepare(`DELETE FROM episodes_vec`).run();
				db.prepare(`DELETE FROM episodes`).run();
				db.prepare(`DELETE FROM episodes_cold`).run();
			});
			txn();
		},

		close() {
			db.close();
		},
	};
}
