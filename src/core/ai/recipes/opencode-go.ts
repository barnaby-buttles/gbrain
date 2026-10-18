import type { Recipe } from '../types.ts';

import { randomUUID } from 'node:crypto';

/**
 * OpenCode Go rejects any request without `x-opencode-session` (HTTP 400
 * MissingSessionID); the header also carries the relay's routing/cache
 * affinity. Hermes treats it as mandatory on every OpenCode transport
 * (hermes#105841) — this recipe must do the same. One id per process: every
 * request a gbrain run makes stays routable to the same relay shard.
 */
const OPENCODE_SESSION_HEADER = 'x-opencode-session';
const OPENCODE_SESSION_ID = `gbrain-${randomUUID().replace(/-/g, '').slice(0, 16)}`;

/**
 * Models that must answer directly rather than spending their whole output
 * budget on chain-of-thought when asked to extract structured data (the
 * failure the shim below fixes). The v4.1 generation inherits v4-flash's
 * behavior on this wire, so both ids carry the `reasoning_effort: 'none'`
 * injection.
 */
const DIRECT_ANSWER_MODELS: ReadonlySet<string> = new Set(['deepseek-v4-flash', 'deepseek-v4.1-flash']);

/**
 * Reasoning models (qwen3.5-plus, deepseek-v4-pro) return chain-of-thought
 * in a separate `reasoning_content` field, leaving `content` empty on
 * pure-reasoning turns. The AI SDK's openai-compatible adapter reads only
 * `content`, so the model appears to answer with nothing. This transport
 * shim promotes `reasoning_content` into `content` when `content` is empty,
 * before the adapter parses the body. Fail-open: any error returns the
 * original response.
 *
 * Non-streaming JSON chat completions only.
 */
// Cast through `unknown` because TS's `typeof fetch` includes a `preconnect`
// member the arrow function does not implement.
export const reasoningContentCompatFetch = (async (
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> => {
  // deepseek-v4-flash burns its entire output budget on chain-of-thought
  // when asked to extract structured data (observed: finish_reason="length",
  // content="", 2048/2048 reasoning tokens) — the JSON array never appears,
  // and propose_takes re-fails the same pages every cycle. The Go endpoint
  // accepts OpenAI-style `reasoning_effort`; 'none' makes flash answer
  // directly (verified 2026-08-14: reasoning 2048→0 tokens, content 0→2.7KB
  // JSON, finish_reason="stop"). Inject only for flash — pro keeps CoT for
  // think/skillopt where reasoning is the point.
  let reqInit = init;
  try {
    const headers = new Headers(init?.headers);
    let body = typeof init?.body === 'string' ? init.body : null;
    let changed = false;
    // The relay refuses a request without the session header before any model
    // work happens, so this is applied to every call, not just flash.
    if (!headers.has(OPENCODE_SESSION_HEADER)) {
      headers.set(OPENCODE_SESSION_HEADER, OPENCODE_SESSION_ID);
      changed = true;
    }
    if (body) {
      const b = JSON.parse(body) as { model?: string; reasoning_effort?: string };
      if (b.model !== undefined && DIRECT_ANSWER_MODELS.has(b.model) && b.reasoning_effort === undefined) {
        b.reasoning_effort = 'none';
        body = JSON.stringify(b);
        headers.delete('content-length');
        changed = true;
      }
    }
    if (changed) reqInit = { ...init, headers, ...(body === null ? {} : { body }) };
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
      const reasoning = msg.reasoning_content;
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

/**
 * OpenCode Go — low-cost subscription ($10/mo) providing curated open models
 * via an OpenAI-compatible /v1/chat/completions endpoint.
 *
 * Full model catalog: https://opencode.ai/zen/go/v1/models
 * API key: OPENCODE_GO_API_KEY — subscribe at https://opencode.ai/auth
 * Endpoint docs: https://opencode.ai/docs/go/
 */
export const opencodeGo: Recipe = {
  id: 'opencode-go',
  name: 'OpenCode Go',
  tier: 'openai-compat',
  implementation: 'openai-compatible',
  base_url_default: 'https://opencode.ai/zen/go/v1',
  auth_env: {
    required: ['OPENCODE_GO_API_KEY'],
    setup_url: 'https://opencode.ai/auth',
  },
  touchpoints: {
    chat: {
      models: [
        'deepseek-v4-flash',
        'deepseek-v4.1-flash',
        'deepseek-v4-pro',
        'glm-5',
        'glm-5.1',
        'glm-5.2',
        'grok-4.5',
        'kimi-k2.5',
        'kimi-k2.6',
        'kimi-k2.7-code',
        'kimi-k3',
        'minimax-m2.5',
        'minimax-m2.7',
        'minimax-m3',
        'mimo-v2-omni',
        'mimo-v2-pro',
        'mimo-v2.5',
        'mimo-v2.5-pro',
        'qwen3.5-plus',
        'qwen3.6-plus',
        'qwen3.7-max',
        'qwen3.7-plus',
      ],
      supports_tools: true,
      supports_subagent_loop: true,
      // DeepSeek caches the request prefix server-side and bills hits at the
      // cache-read rate; nothing is sent client-side (same doctrine as
      // openrouter.ts, which already returns true for `deepseek/`). Verified on
      // this wire 2026-10-12: an identical ~1.2k-token prefix reported
      // cached_tokens=1152 on the second call, 0 on the first. The other Go
      // families are unverified, so they stay false rather than promising
      // caching the upstream may not do.
      supports_prompt_cache: (modelId: string) => modelId.trim().toLowerCase().startsWith('deepseek'),
      max_context_tokens: 200_000,
      // Pricing: deepseek-v4-flash baseline (cheapest; other models vary)
      cost_per_1m_input_usd: 0.14,
      cost_per_1m_output_usd: 0.28,
      price_last_verified: '2026-07-17',
    },
  },
  compat: { fetch: reasoningContentCompatFetch },
  setup_hint:
    'Subscribe at https://opencode.ai/auth ($5 first month, $10/mo after), then `export OPENCODE_GO_API_KEY=...`. Use `opencode-go:<model-id>` strings (e.g. opencode-go:deepseek-v4-pro).',
};
