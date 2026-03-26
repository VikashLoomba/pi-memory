import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createTempDir, cleanupTempDir } from "./test-helpers.js";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function runMemoryScript(script: string, memoryDir: string): string {
	const result = spawnSync(process.execPath, ["--import", "tsx", "-e", script], {
		cwd: projectRoot,
		encoding: "utf8",
		env: { ...process.env, MEMORY_DIR: memoryDir },
	});

	if (result.status !== 0) {
		throw new Error(`Subprocess failed\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`);
	}

	return result.stdout.trim();
}

test("schema version manifest is written and survives a process restart without losing persisted data", () => {
	const dir = createTempDir("migrations");
	try {
		runMemoryScript(
			`
				import { createEpisodicStore } from './src/episodic.js';
				import { createProfileStore } from './src/profile.js';
				const memoryDir = process.env.MEMORY_DIR;
				if (!memoryDir) throw new Error('MEMORY_DIR is required');
				const episodic = createEpisodicStore(memoryDir);
				episodic.init();
				const embedding = new Float32Array(3072);
				embedding[0] = 1;
				episodic.insert('Persistent episode', embedding, 'pi');
				const profile = createProfileStore(memoryDir);
				await profile.init();
				await profile.ingestFacts([{ source: 'user', relation: 'USES', target: 'TypeScript', confidence: 0.9 }]);
				await profile.close();
				episodic.close();
				console.log('seeded');
				process.exit(0);
			`,
			dir,
		);

		const raw = runMemoryScript(
			`
				import { readFileSync } from 'node:fs';
				import { join } from 'node:path';
				import { createEpisodicStore } from './src/episodic.js';
				import { createProfileStore } from './src/profile.js';
				const memoryDir = process.env.MEMORY_DIR;
				if (!memoryDir) throw new Error('MEMORY_DIR is required');
				const episodic = createEpisodicStore(memoryDir);
				episodic.init();
				const profile = createProfileStore(memoryDir);
				await profile.init();
				const result = {
					episodes: episodic.listRecent(10).length,
					relationships: (await profile.listTopRelationships(10)).length,
					versions: JSON.parse(readFileSync(join(memoryDir, 'schema-version.json'), 'utf8')).stores,
				};
				await profile.close();
				episodic.close();
				console.log(JSON.stringify(result));
				process.exit(0);
			`,
			dir,
		);

		const parsed = JSON.parse(raw) as {
			episodes: number;
			relationships: number;
			versions: Record<string, number>;
		};

		assert.equal(parsed.episodes, 1);
		assert.equal(parsed.relationships, 1);
		assert.equal(parsed.versions.episodic, 2);
		assert.equal(parsed.versions.profile, 1);
	} finally {
		cleanupTempDir(dir);
	}
});
