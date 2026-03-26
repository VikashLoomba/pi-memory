import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createEpisodicStore } from "../src/episodic.js";
import { createProfileStore } from "../src/profile.js";
import { createTempDir, cleanupTempDir, makeEmbedding } from "./test-helpers.js";

test("store initialization writes schema versions for both databases and does not destroy existing data", async () => {
	const dir = createTempDir("migrations");
	let episodic = createEpisodicStore(dir);
	const profile = createProfileStore(dir);
	try {
		episodic.init();
		episodic.insert("Persistent episode", makeEmbedding(0), "pi");
		await profile.init();
		await profile.ingestFacts([{ source: "user", relation: "USES", target: "TypeScript", confidence: 0.9 }]);
		await profile.close();
		episodic.close();

		episodic = createEpisodicStore(dir);
		episodic.init();
		const reopenedProfile = createProfileStore(dir);
		await reopenedProfile.init();

		assert.equal(episodic.listRecent(10).length, 1);
		assert.equal((await reopenedProfile.listTopRelationships(10)).length, 1);

		const versions = JSON.parse(readFileSync(join(dir, "schema-version.json"), "utf8")) as {
			stores: Record<string, number>;
		};
		assert.equal(versions.stores.episodic, 1);
		assert.equal(versions.stores.profile, 1);

		await reopenedProfile.close();
		episodic.close();
	} finally {
		cleanupTempDir(dir);
	}
});
