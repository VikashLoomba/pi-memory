/**
 * Embedding Service
 *
 * Uses OpenAI's official TypeScript SDK with text-embedding-3-large.
 */

import OpenAI from "openai";

const MODEL = "text-embedding-3-large";
const DIMENSIONS = 3072;

const clients = new Map<string, OpenAI>();

function resolveApiKey(explicitApiKey?: string): string {
	const apiKey = explicitApiKey ?? process.env.OPENAI_API_KEY;
	if (!apiKey) {
		throw new Error("No OpenAI API key available for embeddings. Set OPENAI_API_KEY or configure the OpenAI provider in pi.");
	}
	return apiKey;
}

function getClient(explicitApiKey?: string): OpenAI {
	const apiKey = resolveApiKey(explicitApiKey);
	const existing = clients.get(apiKey);
	if (existing) return existing;
	const client = new OpenAI({ apiKey });
	clients.set(apiKey, client);
	return client;
}

export interface EmbeddingOptions {
	apiKey?: string;
}

export async function embed(text: string, options: EmbeddingOptions = {}): Promise<Float32Array> {
	const response = await getClient(options.apiKey).embeddings.create({
		model: MODEL,
		input: text,
		dimensions: DIMENSIONS,
	});
	return new Float32Array(response.data[0].embedding);
}

export async function embedBatch(texts: string[], options: EmbeddingOptions = {}): Promise<Float32Array[]> {
	if (texts.length === 0) return [];
	const response = await getClient(options.apiKey).embeddings.create({
		model: MODEL,
		input: texts,
		dimensions: DIMENSIONS,
	});
	return response.data
		.sort((a, b) => a.index - b.index)
		.map((item) => new Float32Array(item.embedding));
}

export { DIMENSIONS, MODEL as EMBEDDING_MODEL };
