/**
 * PII Detection and Masking
 *
 * MemTrust uses BERT + spaCy NER + pattern matching for PII sanitization.
 * This is a pragmatic regex-based implementation for local use.
 */

export interface PiiMask {
	original: string;
	token: string;
	type: string;
}

export interface MaskResult {
	masked: string;
	masks: PiiMask[];
}

interface PiiPattern {
	type: string;
	regex: RegExp;
}

const PII_PATTERNS: PiiPattern[] = [
	{ type: "EMAIL", regex: /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g },
	{ type: "SSN", regex: /\b\d{3}-\d{2}-\d{4}\b/g },
	{ type: "CREDIT_CARD", regex: /\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b/g },
	{ type: "OPENAI_KEY", regex: /\bsk-[a-zA-Z0-9]{20,}\b/g },
	{ type: "GITHUB_TOKEN", regex: /\bghp_[a-zA-Z0-9]{36}\b/g },
	{ type: "AWS_KEY", regex: /\bAKIA[0-9A-Z]{16}\b/g },
	{ type: "PHONE", regex: /\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b/g },
	{ type: "IP_ADDRESS", regex: /\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/g },
];

export function maskPii(text: string): MaskResult {
	const masks: PiiMask[] = [];
	const counters = new Map<string, number>();
	let masked = text;

	for (const pattern of PII_PATTERNS) {
		// Reset regex state
		pattern.regex.lastIndex = 0;

		// Collect all matches first, then replace to avoid shifting indices
		const matches: string[] = [];
		let match: RegExpExecArray | null;
		while ((match = pattern.regex.exec(masked)) !== null) {
			if (!matches.includes(match[0])) {
				matches.push(match[0]);
			}
		}

		for (const original of matches) {
			const count = (counters.get(pattern.type) ?? 0) + 1;
			counters.set(pattern.type, count);
			const token = `[${pattern.type}_${count}]`;
			masks.push({ original, token, type: pattern.type });
			// Replace all occurrences of this exact match
			masked = masked.split(original).join(token);
		}
	}

	return { masked, masks };
}

export function unmaskPii(text: string, masks: PiiMask[]): string {
	let unmasked = text;
	for (const mask of masks) {
		unmasked = unmasked.split(mask.token).join(mask.original);
	}
	return unmasked;
}

export function serializeMasks(masks: PiiMask[]): string {
	return JSON.stringify(masks);
}

export function deserializeMasks(json: string): PiiMask[] {
	try {
		const parsed = JSON.parse(json);
		if (!Array.isArray(parsed)) return [];
		return parsed;
	} catch {
		return [];
	}
}
