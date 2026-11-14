/** Inputs used to rank legal model/provider routes before any upstream calls are made. */
export interface CandidateRank {
  providerId: string;
  modelId: string;
  ratingMilli: number;
}

/** Higher quality first, then stable provider/model names for deterministic ties. */
export function compareCandidateRanks(a: CandidateRank, b: CandidateRank): number {
  return b.ratingMilli - a.ratingMilli || a.providerId.localeCompare(b.providerId) || a.modelId.localeCompare(b.modelId);
}

/** One sequential provider attempt. A refusal can mark its provider permanently exhausted. */
export interface CandidateAttempt<T> {
  ok: boolean;
  value?: T;
  retireProvider?: boolean;
}

/**
 * Walk an already-ranked list exactly once, skipping only providers confirmed exhausted.
 * Transient refusal classification belongs to the caller, which has the provider response body.
 */
export async function failoverCandidates<C, T>(
  candidates: readonly C[],
  retiredProviders: Set<string>,
  providerOf: (candidate: C) => string,
  attempt: (candidate: C) => Promise<CandidateAttempt<T>>,
): Promise<{ candidate: C; value: T } | null> {
  for (const candidate of candidates) {
    const provider = providerOf(candidate);
    if (retiredProviders.has(provider)) continue;
    const result = await attempt(candidate);
    if (result.ok && result.value !== undefined) return { candidate, value: result.value };
    if (result.retireProvider) retiredProviders.add(provider);
  }
  return null;
}

/** Refusal classification: scheduled failures/concurrency move on; confirmed depletion retires. */
export function providerIsExhausted(status: number, body: unknown): boolean {
  let message = '';
  if (body !== null && typeof body === 'object') {
    const root = body as Record<string, unknown>;
    const error = root.error !== null && typeof root.error === 'object' ? root.error as Record<string, unknown> : {};
    if (typeof error.message === 'string') message = error.message;
  }
  if (/\brefused this call\b|\bconcurrent_requests\b/i.test(message)) return false;
  if (/(?:budget\s+is\s+spent|budget\s+spent|provider\s+is\s+out|\bis\s+out\b)/i.test(message)) return true;
  return status === 402 && message === '';
}
