/**
 * Embedding Service
 *
 * MemTrust uses local bge-m3 within TEE. This module provides an EmbeddingProvider
 * interface with OpenAI (default) and optional local model support.
 */

import OpenAI from "openai";

export interface EmbeddingProvider {
	embed(text: string): Promise<Float32Array>;
	embedBatch(texts: string[]): Promise<Float32Array[]>;
	readonly dimensions: number;
	readonly model: string;
}

// --- OpenAI Provider ---

const OPENAI_MODEL = "text-embedding-3-large";
const OPENAI_DIMENSIONS = 3072;
const clients = new Map<string, OpenAI>();

function getClient(apiKey: string): OpenAI {
	const existing = clients.get(apiKey);
	if (existing) return existing;
	const client = new OpenAI({ apiKey });
	clients.set(apiKey, client);
	return client;
}

export function createOpenAIProvider(apiKey: string): EmbeddingProvider {
	const client = getClient(apiKey);
	return {
		dimensions: OPENAI_DIMENSIONS,
		model: OPENAI_MODEL,
		async embed(text: string): Promise<Float32Array> {
			const response = await client.embeddings.create({
				model: OPENAI_MODEL,
				input: text,
				dimensions: OPENAI_DIMENSIONS,
			});
			return new Float32Array(response.data[0].embedding);
		},
		async embedBatch(texts: string[]): Promise<Float32Array[]> {
			if (texts.length === 0) return [];
			const response = await client.embeddings.create({
				model: OPENAI_MODEL,
				input: texts,
				dimensions: OPENAI_DIMENSIONS,
			});
			return response.data
				.sort((a, b) => a.index - b.index)
				.map((item) => new Float32Array(item.embedding));
		},
	};
}

// --- Local Provider (requires @xenova/transformers) ---

export function createLocalProvider(modelName?: string): EmbeddingProvider {
	// Dynamic import to avoid requiring @xenova/transformers unless used
	const model = modelName ?? process.env.LOCAL_EMBEDDING_MODEL ?? "Xenova/bge-small-en-v1.5";
	const LOCAL_DIMENSIONS = 384; // bge-small-en-v1.5 default
	let pipeline: any = null;

	async function getPipeline() {
		if (pipeline) return pipeline;
		try {
			// @ts-ignore — optional dependency, only loaded when EMBEDDING_PROVIDER=local
			const transformers = await import("@xenova/transformers");
			pipeline = await transformers.pipeline("feature-extraction", model);
			return pipeline;
		} catch (error) {
			throw new Error(
				`Failed to load local embedding model "${model}". ` +
				`Install @xenova/transformers: npm install @xenova/transformers\n` +
				`Original error: ${(error as Error).message}`,
			);
		}
	}

	return {
		dimensions: LOCAL_DIMENSIONS,
		model,
		async embed(text: string): Promise<Float32Array> {
			const pipe = await getPipeline();
			const output = await pipe(text, { pooling: "mean", normalize: true });
			return new Float32Array(output.data);
		},
		async embedBatch(texts: string[]): Promise<Float32Array[]> {
			if (texts.length === 0) return [];
			const results: Float32Array[] = [];
			for (const text of texts) {
				results.push(await this.embed(text));
			}
			return results;
		},
	};
}

// --- Convenience functions (backwards-compatible API) ---

export interface EmbeddingOptions {
	apiKey?: string;
}

function resolveApiKey(explicitApiKey?: string): string {
	const apiKey = explicitApiKey ?? process.env.OPENAI_API_KEY;
	if (!apiKey) {
		throw new Error("No OpenAI API key available for embeddings. Set OPENAI_API_KEY or configure the OpenAI provider in pi.");
	}
	return apiKey;
}

export async function embed(text: string, options: EmbeddingOptions = {}): Promise<Float32Array> {
	const provider = createOpenAIProvider(resolveApiKey(options.apiKey));
	return provider.embed(text);
}

export async function embedBatch(texts: string[], options: EmbeddingOptions = {}): Promise<Float32Array[]> {
	return createOpenAIProvider(resolveApiKey(options.apiKey)).embedBatch(texts);
}

// For backwards compatibility
const DIMENSIONS = OPENAI_DIMENSIONS;
const MODEL = OPENAI_MODEL;
export { DIMENSIONS, MODEL as EMBEDDING_MODEL };

export function resolveProvider(apiKey?: string): EmbeddingProvider {
	const providerType = process.env.EMBEDDING_PROVIDER ?? "openai";
	if (providerType === "local") {
		return createLocalProvider();
	}
	const key = apiKey ?? resolveApiKey();
	return createOpenAIProvider(key);
}
