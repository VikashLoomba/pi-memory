/**
 * Tiny schema migration/versioning layer.
 *
 * Versions are tracked in a project-local sidecar file under the memory data
 * directory so both embedded stores can evolve independently:
 *
 *   .pi/memory/schema-version.json
 *
 * Future schema changes should append a new migration with an incremented
 * version number rather than editing older migrations in place.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

export interface SchemaVersionState {
	stores: Record<string, number>;
	updatedAt: string;
}

export interface SyncMigration {
	version: number;
	description: string;
	up(): void;
}

export interface AsyncMigration {
	version: number;
	description: string;
	up(): Promise<void>;
}

const VERSION_FILE = "schema-version.json";

function versionFilePath(dataDir: string): string {
	return join(dataDir, VERSION_FILE);
}

function emptyState(): SchemaVersionState {
	return {
		stores: {},
		updatedAt: new Date().toISOString(),
	};
}

function validateMigrations<T extends { version: number }>(migrations: readonly T[]): void {
	let previous = 0;
	for (const migration of migrations) {
		if (!Number.isInteger(migration.version) || migration.version <= 0) {
			throw new Error(`Invalid migration version: ${migration.version}`);
		}
		if (migration.version <= previous) {
			throw new Error(`Migration versions must be strictly increasing. Found ${migration.version} after ${previous}.`);
		}
		previous = migration.version;
	}
}

function readState(dataDir: string): SchemaVersionState {
	mkdirSync(dataDir, { recursive: true });
	const path = versionFilePath(dataDir);
	if (!existsSync(path)) return emptyState();

	const raw = readFileSync(path, "utf8").trim();
	if (!raw) return emptyState();

	let parsed: unknown;
	try {
		parsed = JSON.parse(raw);
	} catch (error) {
		throw new Error(`Invalid schema version file at ${path}: ${(error as Error).message}`);
	}

	if (typeof parsed !== "object" || parsed === null) {
		throw new Error(`Invalid schema version file at ${path}: expected an object.`);
	}

	const stores = typeof (parsed as { stores?: unknown }).stores === "object" && (parsed as { stores?: unknown }).stores !== null
		? Object.fromEntries(
			Object.entries((parsed as { stores: Record<string, unknown> }).stores).map(([key, value]) => {
				if (!Number.isInteger(value) || (value as number) < 0) {
					throw new Error(`Invalid schema version for store \"${key}\" in ${path}`);
				}
				return [key, value as number];
			}),
		)
		: {};

	return {
		stores,
		updatedAt:
			typeof (parsed as { updatedAt?: unknown }).updatedAt === "string"
				? (parsed as { updatedAt: string }).updatedAt
				: new Date().toISOString(),
	};
}

function writeState(dataDir: string, state: SchemaVersionState): void {
	mkdirSync(dataDir, { recursive: true });
	writeFileSync(versionFilePath(dataDir), `${JSON.stringify(state, null, 2)}\n`, "utf8");
}

function latestVersion<T extends { version: number }>(migrations: readonly T[]): number {
	return migrations[migrations.length - 1]?.version ?? 0;
}

export function readSchemaVersions(dataDir: string): SchemaVersionState {
	return readState(dataDir);
}

export function applySyncMigrations(
	dataDir: string,
	store: string,
	migrations: readonly SyncMigration[],
): number {
	validateMigrations(migrations);
	const state = readState(dataDir);
	const currentVersion = state.stores[store] ?? 0;
	const targetVersion = latestVersion(migrations);

	if (currentVersion > targetVersion) {
		throw new Error(
			`Database schema version for \"${store}\" (${currentVersion}) is newer than this package supports (${targetVersion}). Downgrade is not supported.`,
		);
	}

	for (const migration of migrations) {
		if (migration.version <= currentVersion) continue;
		migration.up();
		state.stores[store] = migration.version;
		state.updatedAt = new Date().toISOString();
		writeState(dataDir, state);
	}

	if (!(store in state.stores)) {
		state.stores[store] = currentVersion;
		state.updatedAt = new Date().toISOString();
		writeState(dataDir, state);
	}

	return state.stores[store] ?? 0;
}

export async function applyAsyncMigrations(
	dataDir: string,
	store: string,
	migrations: readonly AsyncMigration[],
): Promise<number> {
	validateMigrations(migrations);
	const state = readState(dataDir);
	const currentVersion = state.stores[store] ?? 0;
	const targetVersion = latestVersion(migrations);

	if (currentVersion > targetVersion) {
		throw new Error(
			`Database schema version for \"${store}\" (${currentVersion}) is newer than this package supports (${targetVersion}). Downgrade is not supported.`,
		);
	}

	for (const migration of migrations) {
		if (migration.version <= currentVersion) continue;
		await migration.up();
		state.stores[store] = migration.version;
		state.updatedAt = new Date().toISOString();
		writeState(dataDir, state);
	}

	if (!(store in state.stores)) {
		state.stores[store] = currentVersion;
		state.updatedAt = new Date().toISOString();
		writeState(dataDir, state);
	}

	return state.stores[store] ?? 0;
}
