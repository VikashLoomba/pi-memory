/**
 * Memory Server
 *
 * MemTrust defines an OAuth-like "Context from MemTrust" protocol for
 * cross-application memory access. This is a pragmatic localhost HTTP API
 * that other local agents or tools can use to query the memory store.
 */

import { createServer, type Server, type IncomingMessage, type ServerResponse } from "node:http";
import { writeFileSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { randomBytes } from "node:crypto";
import type { EpisodicStore } from "./episodic.js";
import type { ProfileStore } from "./profile.js";
import { embed, type EmbeddingOptions } from "./embeddings.js";

export interface MemoryServerOptions {
	episodic: EpisodicStore;
	profile: ProfileStore;
	dataDir: string;
	embeddingOptions?: EmbeddingOptions;
}

export interface MemoryServer {
	start(): Promise<{ port: number; token: string }>;
	stop(): Promise<void>;
	readonly port: number | null;
	readonly token: string;
}

function parseUrl(req: IncomingMessage): URL {
	return new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
}

function jsonResponse(res: ServerResponse, status: number, data: unknown) {
	res.writeHead(status, { "Content-Type": "application/json" });
	res.end(JSON.stringify(data));
}

export function createMemoryServer(options: MemoryServerOptions): MemoryServer {
	const { episodic, profile, dataDir, embeddingOptions } = options;
	const token = randomBytes(32).toString("hex");
	const serverJsonPath = join(dataDir, "server.json");
	let server: Server | null = null;
	let serverPort: number | null = null;

	function authenticate(req: IncomingMessage): boolean {
		const authHeader = req.headers.authorization;
		if (!authHeader) return false;
		const parts = authHeader.split(" ");
		return parts[0] === "Bearer" && parts[1] === token;
	}

	async function handleRequest(req: IncomingMessage, res: ServerResponse) {
		// CORS for local use
		res.setHeader("Access-Control-Allow-Origin", "http://localhost");
		res.setHeader("Access-Control-Allow-Headers", "Authorization, Content-Type");
		if (req.method === "OPTIONS") {
			res.writeHead(204);
			res.end();
			return;
		}

		if (!authenticate(req)) {
			jsonResponse(res, 401, { error: "Unauthorized" });
			return;
		}

		const url = parseUrl(req);
		const path = url.pathname;

		try {
			if (path === "/api/memory/episodes" && req.method === "GET") {
				const q = url.searchParams.get("q");
				const limit = Math.min(20, Math.max(1, Number(url.searchParams.get("limit") ?? "5")));

				if (q && embeddingOptions?.apiKey) {
					const queryEmbedding = await embed(q, embeddingOptions);
					const episodes = episodic.query(queryEmbedding, limit);
					jsonResponse(res, 200, { episodes });
				} else {
					const episodes = episodic.listRecent(limit);
					jsonResponse(res, 200, { episodes });
				}
				return;
			}

			if (path === "/api/memory/profile" && req.method === "GET") {
				const keywords = (url.searchParams.get("keywords") ?? "").split(",").filter(Boolean);
				const limit = Math.min(20, Math.max(1, Number(url.searchParams.get("limit") ?? "10")));
				const facts = await profile.queryByKeywords(keywords, limit);
				jsonResponse(res, 200, { facts });
				return;
			}

			if (path === "/api/memory/stats" && req.method === "GET") {
				const episodicStats = episodic.getStats();
				const profileStats = await profile.getStats();
				jsonResponse(res, 200, { episodic: episodicStats, profile: profileStats });
				return;
			}

			jsonResponse(res, 404, { error: "Not found" });
		} catch (error) {
			jsonResponse(res, 500, { error: (error as Error).message });
		}
	}

	return {
		get port() { return serverPort; },
		get token() { return token; },

		start(): Promise<{ port: number; token: string }> {
			return new Promise((resolve, reject) => {
				server = createServer((req, res) => {
					handleRequest(req, res).catch((error) => {
						jsonResponse(res, 500, { error: (error as Error).message });
					});
				});

				server.listen(0, "127.0.0.1", () => {
					const addr = server!.address();
					if (typeof addr === "object" && addr) {
						serverPort = addr.port;
						// Write discovery file
						writeFileSync(serverJsonPath, JSON.stringify({
							port: serverPort,
							token,
							pid: process.pid,
						}));
						resolve({ port: serverPort, token });
					} else {
						reject(new Error("Failed to bind server"));
					}
				});

				server.on("error", reject);
			});
		},

		async stop(): Promise<void> {
			if (server) {
				return new Promise((resolve) => {
					server!.close(() => {
						server = null;
						serverPort = null;
						try { unlinkSync(serverJsonPath); } catch { /* ignore */ }
						resolve();
					});
				});
			}
		},
	};
}
