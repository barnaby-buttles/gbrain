import type { Recipe } from '../types.ts';

/**
 * CommandCode (commandcode.ai / Langbase "Provider API") — OpenAI-compatible
 * provider endpoint. Model IDs are vendor-prefixed and pass through verbatim:
 * `deepseek/deepseek-v4-flash`, `deepseek/deepseek-v4-pro`,
 * `deepseek/deepseek-v4-flash-vision-exp` — all 1M context.
 *
 * Wire facts (verified 2026-08-23):
 * - Base URL: https://api.commandcode.ai/provider/v1
 * - Catalog: GET /provider/v1/models is PUBLIC (no auth), 58 models.
 * - Chat: POST /provider/v1/chat/completions, Bearer auth.
 * - DeepSeek models return chain-of-thought in a `reasoning` field (NOT
 *   `reasoning_content`) with `content` populated on terminal turns — the
 *   openai-compatible adapter reads `content`, so extraction works without a
 *   shim in the normal case.
 * - `reasoning_effort` is REJECTED (400: expected low|medium|high|xhigh|max).
 *   `enable_thinking: false` IS honored — verified: kills CoT, content
 *   returns directly. This replaces the opencode-go recipe's
 *   `reasoning_effort: 'none'` injection for flash extraction workloads.
 * - Cloudflare blocks Python urllib TLS fingerprint (403 error code: 1010)
 *   but NOT httpx/curl — hindsight's httpx client is fine.
 *
 * Auth: COMMANDCODE_API_KEY — subscribe at https://commandcode.ai
 */

/**
 * Transport shim, modeled on the opencode-go recipe's
 * `reasoningContentCompatFetch`:
 *
 * 1. REQUEST: inject `enable_thinking: false` for `deepseek/deepseek-v4-flash`
 *    (prefixed id) unless already set. Prevents the CoT-burns-output-budget
 *    failure class on extraction phases (propose_takes etc.) — flash answers
 *    directly. Pro keeps CoT for think/skillopt where reasoning is the point.
 *    `reasoning_effort` must NOT be used — commandcode rejects it (400).
 *
 * 2. RESPONSE: promote `reasoning` → `content` when content is empty on a
 *    terminal text turn (no tool calls), mirroring the deepseek recipe's
 *    fail-open shim. commandcode uses `reasoning`, not `reasoning_content`.
 *
 * Fail-open: any error returns the original response. Non-streaming JSON
 * chat completions only.
 */
// Cast through `unknown` because TS's `typeof fetch` includes a `preconnect`
// member the arrow function does not implement.
export const commandcodeCompatFetch = (async (
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<Response> => {
  let reqInit = init;
  try {
    const rawBody = typeof init?.body === 'string' ? init.body : null;
    if (rawBody) {
      const b = JSON.parse(rawBody) as { model?: string; enable_thinking?: boolean };
      if (b.model === 'deepseek/deepseek-v4-flash' && b.enable_thinking === undefined) {
        b.enable_thinking = false;
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

export const commandcode: Recipe = {
  id: 'commandcode',
  name: 'CommandCode',
  tier: 'openai-compat',
  implementation: 'openai-compatible',
  base_url_default: 'https://api.commandcode.ai/provider/v1',
  auth_env: {
    required: ['COMMANDCODE_API_KEY'],
    setup_url: 'https://commandcode.ai',
  },
  touchpoints: {
    chat: {
      models: [
        'deepseek/deepseek-v4-flash',
        'deepseek/deepseek-v4-pro',
        'deepseek/deepseek-v4-flash-vision-exp',
        'google/gemini-3.5-flash',
        'Qwen/Qwen3.7-plus',
        'moonshotai/Kimi-K3',
      ],
      supports_tools: true,
      supports_subagent_loop: true,
      supports_prompt_cache: false,
      max_context_tokens: 1_000_000,
      // Pricing: deepseek-v4-flash baseline (cheapest; other models vary)
      cost_per_1m_input_usd: 0.14,
      cost_per_1m_output_usd: 0.28,
      price_last_verified: '2026-08-23',
    },
  },
  compat: { fetch: commandcodeCompatFetch },
  setup_hint:
    'Subscribe at https://commandcode.ai, then `export COMMANDCODE_API_KEY=...`. Use `commandcode:<model-id>` strings with the vendor prefix (e.g. commandcode:deepseek/deepseek-v4-flash).',
};
