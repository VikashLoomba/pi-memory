/**
 * Crypto-Shredding Support
 *
 * MemTrust uses per-memory-unit Data Unit Keys (DUK) with AES-256-GCM.
 * Destroying the DUK makes the encrypted data unrecoverable.
 */

import { randomBytes, createCipheriv, createDecipheriv } from "node:crypto";

const ALGORITHM = "aes-256-gcm";
const IV_LENGTH = 12;
const TAG_LENGTH = 16;

export interface EncryptedPayload {
	ciphertext: string; // base64
	iv: string; // base64
	tag: string; // base64
}

export function generateDuk(): string {
	return randomBytes(32).toString("hex");
}

export function encrypt(plaintext: string, dukHex: string): EncryptedPayload {
	const key = Buffer.from(dukHex, "hex");
	const iv = randomBytes(IV_LENGTH);
	const cipher = createCipheriv(ALGORITHM, key, iv, { authTagLength: TAG_LENGTH });
	const encrypted = Buffer.concat([cipher.update(plaintext, "utf8"), cipher.final()]);
	const tag = cipher.getAuthTag();

	return {
		ciphertext: encrypted.toString("base64"),
		iv: iv.toString("base64"),
		tag: tag.toString("base64"),
	};
}

export function decrypt(payload: EncryptedPayload, dukHex: string): string {
	const key = Buffer.from(dukHex, "hex");
	const iv = Buffer.from(payload.iv, "base64");
	const tag = Buffer.from(payload.tag, "base64");
	const ciphertext = Buffer.from(payload.ciphertext, "base64");

	const decipher = createDecipheriv(ALGORITHM, key, iv, { authTagLength: TAG_LENGTH });
	decipher.setAuthTag(tag);
	const decrypted = Buffer.concat([decipher.update(ciphertext), decipher.final()]);
	return decrypted.toString("utf8");
}

export function serializeEncrypted(payload: EncryptedPayload): string {
	return JSON.stringify(payload);
}

export function deserializeEncrypted(json: string): EncryptedPayload | null {
	try {
		const parsed = JSON.parse(json);
		if (parsed && typeof parsed.ciphertext === "string" && typeof parsed.iv === "string" && typeof parsed.tag === "string") {
			return parsed as EncryptedPayload;
		}
		return null;
	} catch {
		return null;
	}
}
