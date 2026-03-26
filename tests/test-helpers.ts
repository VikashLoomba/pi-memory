import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

export function createTempDir(prefix: string): string {
	return mkdtempSync(join(tmpdir(), `pi-memory-${prefix}-`));
}

export function cleanupTempDir(path: string): void {
	rmSync(path, { recursive: true, force: true });
}

export function makeEmbedding(activeIndex = 0, magnitude = 1): Float32Array {
	const vector = new Float32Array(3072);
	vector[activeIndex] = magnitude;
	return vector;
}
