import type { Recipe } from '../types.ts';

/**
 * Ollama Cloud (https://ollama.com/v1) — OpenAI-compatible hosted endpoint.
 * Model IDs are Ollama library tags with size suffixes, passed through
 * verbatim: `deepseek-v4-flash:0731`, `deepseek-v4-pro:0813`.
 *
 * Wire facts (verified 2026-09-09):
 * - Base URL: https://ollama.com/v1
 * - Auth: Bearer OLLAMA_API_KEY (Doppler secret in this household)
 * - Chat: POST /v1/chat/completions
 * - DeepSeek V4 models think by default; chain-of-thought arrives in a
 *   `reasoning` field (NOT `reasoning_content`) with `content` populated on
 *   terminal turns when the output budget survives. With a tight max_tokens
 *   the CoT burns the entire budget and content comes back EMPTY with
 *   finish_reason "length" — the same failure class the commandcode shim
 *   exists to fix.
 * - Top-level `reasoning_effort` accepts none|low|medium|high|max; "none"
 *   is the ONLY off switch for thinking (extra_body.thinking is ignored).
 *   Injecting "none" for flash extraction workloads makes it answer
 *   directly (verified live: content "PONG" with reasoning_effort none).
 * - No embedding models on the cloud catalog (19 chat models, 0 embed) —
 *   this recipe is chat-only; local ollama stays for embeddings.
 *
 * Auth: OLLAMA_API_KEY — subscribe at https://ollama.com
 */

/**
 * Transport shim, modeled on the commandcode recipe's compat fetch:
 *
 * 1. REQUEST: inject `reasoning_effort: "none"` for
 *    `deepseek-v4-flash:0731` (prefixed id) unless already set. Prevents the
 *    CoT-burns-output-budget failure class on extraction phases (propose_takes
 *    etc.) — flash answers directly. Pro keeps CoT for think/skillopt where
 *    reasoning is the point.
 *
 * 2. RESPONSE: promote `reasoning` → `content` when content is empty on a
 *    terminal text turn (no tool calls), mirroring the commandcode shim.
 *    Ollama Cloud uses `reasoning`, not `reasoning_content`.
 *
 * Fail-open: any error returns the original response. Non-streaming JSON
 * chat completions only.
 */
// Cast through `unknown` because TS's `typeof fetch` includes a `preconnect`
// member the arrow function does not implement.
export const ollamaCloudCompatFetch = (async (
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> => {
  let reqInit = init;
  try {
    const rawBody = typeof init?.body === 'string' ? init.body : null;
    if (rawBody) {
      const b = JSON.parse(rawBody) as { model?: string; reasoning_effort?: string };
      if (b.model === 'deepseek-v4-flash:0731' && b.reasoning_effort === undefined) {
        b.reasoning_effort = 'none';
        const headers = new Headers(init?.headers);
        headers.delete('content-length');
        reqInit = { ...init, headers, body: JSON.stringify(b) };
      }
    }
  } catch {
    // Malformed body — leave the request untouched, fail open.
  }
  const res = await fetch(input as any, reqInit as any);
  try {
    if (!res.ok) return res;
    const ctype = res.headers.get('content-type') ?? '';
    if (!ctype.includes('application/json')) return res;
    const json = await res.clone().json();
    const choices = Array.isArray(json?.choices) ? json.choices : [];
    let modified = false;
    for (const choice of choices) {
      const msg = choice?.message;
      if (!msg) continue;
      // A tool-call turn legitimately carries content:null — the answer is the
      // tool call, not text. Only promote on a terminal text turn.
      const hasToolCalls = Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0;
      const content = msg.content;
      const reasoning = msg.reasoning;
      const contentEmpty = content == null || (typeof content === 'string' && content.trim() === '');
      if (!hasToolCalls && contentEmpty && typeof reasoning === 'string' && reasoning.trim() !== '') {
        msg.content = reasoning;
        modified = true;
      }
    }
    if (!modified) return res;
    const headers = new Headers(res.headers);
    headers.delete('content-length');
    headers.delete('content-encoding');
    return new Response(JSON.stringify(json), {
      status: res.status,
      statusText: res.statusText,
      headers,
    });
  } catch {
    return res;
  }
}) as unknown as typeof fetch;

export const ollamaCloud: Recipe = {
  id: 'ollama-cloud',
  name: 'Ollama Cloud',
  tier: 'openai-compat',
  implementation: 'openai-compatible',
  base_url_default: 'https://ollama.com/v1',
  auth_env: {
    required: ['OLLAMA_API_KEY'],
    setup_url: 'https://ollama.com',
  },
  touchpoints: {
    chat: {
      models: [
        'deepseek-v4-flash:0731',
        'deepseek-v4-pro:0813',
        'qwen3.5:397b',
        'glm-5.3',
        'glm-5.3-flash',
        'kimi-k2.7-code',
        'minimax-m3',
        'nemotron-3-nano:30b',
      ],
      supports_tools: true,
      supports_subagent_loop: true,
      supports_prompt_cache: false,
      max_context_tokens: 1_000_000,
      // Pricing (verified 2026-09-09 at ollama.com/pricing, off-peak per-1M):
      // flash $0.22 in / $0.66 out; pro $0.66 in / $1.98 out. Peak
      // (12:00-18:00 UTC Mon-Fri) doubles both.
      cost_per_1m_input_usd: 0.22,
      cost_per_1m_output_usd: 0.66,
      price_last_verified: '2026-09-09',
    },
  },
  compat: { fetch: ollamaCloudCompatFetch },
  setup_hint:
    'Subscribe at https://ollama.com, then `export OLLAMA_API_KEY=...`. Use `ollama-cloud:<model-tag>` strings (e.g. ollama-cloud:deepseek-v4-flash:0731).',
};
