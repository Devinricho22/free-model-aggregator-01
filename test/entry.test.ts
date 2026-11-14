import test from 'node:test';
import assert from 'node:assert/strict';
import { buildCatalog, type Probe, type RatingsFile } from '../lib/providers.ts';
import { anthropicEventsFromOpenAI, anthropicToOpenAI, geminiToOpenAI, openAIRequestInputTokens, openAIStreamWithFinalUsage, openAIToAnthropicRequest, openAIToGemini } from '../lib/wire.ts';
import { failoverCandidates, providerIsExhausted } from '../lib/failover.ts';

test('catalog keeps every provider and trusts rating free/context truth', () => {
  const probe: Probe = { provider: { id: 'p', name: 'P', wire: 'openai', auth: 'bearer', base: '/p/p', key_env: 'P_KEY', key: 'secret' }, ok: true, status: 200,
    models: [{ id: 'free', display_name: 'Free', free: null, context_length: 100, tools: null, vision: null, json: null }, { id: 'paid', display_name: 'Paid', free: true, context_length: 100, tools: null, vision: null, json: null }], error: null };
  const ratings: RatingsFile = { schema: 'ratings', best: [], ratings: [
    { provider: 'p', model: 'free', rating_milli: 900, free: true, context_length: 500, tools: true, vision: false, json: true },
    { provider: 'p', model: 'paid', rating_milli: 999, free: false, context_length: 100, tools: false, vision: false, json: false },
  ] };
  assert.deepEqual(buildCatalog('http://env', [probe], ratings), { providers: [{ id: 'p', base_url: '/p/p', key_env: 'P_KEY', wire: 'openai', models: [
    { id: 'free', free: true, context_length: 500 }, { id: 'paid', free: false, context_length: 100 },
  ] }] });
});

test('Anthropic request translation preserves system, tools, JSON mode, max tokens and image input', () => {
  const result = anthropicToOpenAI({ system: 'system rule', max_tokens: 77, output_config: { format: { type: 'json_schema' } },
    tools: [{ name: 'lookup', input_schema: { type: 'object' } }], messages: [{ role: 'user', content: [
      { type: 'text', text: 'inspect' }, { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'AA==' } },
    ] }] }, 'chosen');
  assert.equal(result.model, 'chosen'); assert.equal(result.max_tokens, 77);
  assert.deepEqual(result.response_format, { type: 'json_object' });
  assert.deepEqual(result.messages?.[0], { role: 'system', content: 'system rule' });
  assert.deepEqual(result.tools?.[0], { type: 'function', function: { name: 'lookup', description: '', parameters: { type: 'object' } } });
  const user = (result.messages as any[])[1]; assert.match(user.content[1].image_url.url, /^data:image\/png;base64,/);
});

test('OpenAI request maps system, image, JSON mode and token cap to Anthropic and Gemini', () => {
  const req = { model: 'ignored', max_tokens: 55, stop: ['\n\n###'], response_format: { type: 'json_object' }, messages: [
    { role: 'system', content: 'policy' }, { role: 'user', content: [{ type: 'text', text: 'hi' }, { type: 'image_url', image_url: { url: 'data:image/jpeg;base64,AA==' } }] },
  ] };
  const anthropic = openAIToAnthropicRequest(req, 'm'); assert.equal(anthropic.max_tokens, 55); assert.deepEqual(anthropic.output_config, { format: { type: 'json' } });
  assert.equal(anthropic.system?.[0].text, 'policy'); assert.equal(anthropic.messages?.[0].content[1].source.media_type, 'image/jpeg');
  const gemini = openAIToGemini(req); assert.deepEqual(gemini.generationConfig, { maxOutputTokens: 55, stopSequences: ['\n\n###'], responseMimeType: 'application/json' });
  assert.equal(gemini.systemInstruction.parts[0].text, 'policy'); assert.equal(gemini.contents[0].parts[1].inlineData.mimeType, 'image/jpeg');
});

test('Gemini response carries text, finish reason and usage into OpenAI shape', () => {
  const result = geminiToOpenAI({ candidates: [{ content: { parts: [{ text: 'ok' }] }, finishReason: 'MAX_TOKENS' }], usageMetadata: { promptTokenCount: 4, candidatesTokenCount: 3, totalTokenCount: 7 } }, 'm');
  assert.equal(result.choices?.[0].message.content, 'ok'); assert.equal(result.choices?.[0].finish_reason, 'length');
  assert.deepEqual(result.usage, { prompt_tokens: 4, completion_tokens: 3, total_tokens: 7 });
});

test('OpenAI request carries Anthropic system and stop controls', () => {
  const result = openAIToAnthropicRequest({ max_tokens: 20, stop: ['END'], tools: [{ type: 'function', function: { name: 'f', parameters: { type: 'object' } } }], messages: [{ role: 'assistant', content: 'ready' }] }, 'm');
  assert.equal(result.max_tokens, 20); assert.deepEqual(result.system, undefined); assert.equal(result.messages?.[0].content[0].text, 'ready'); assert.equal(result.tools?.[0].name, 'f');
});

test('translated Anthropic SSE has ordered message and block completion events with usage', async () => {
  const lines = [
    { id: 'c', choices: [{ delta: { content: 'hello' }, finish_reason: null }] },
    { id: 'c', choices: [{ delta: {}, finish_reason: 'stop' }] },
    { id: 'c', choices: [], usage: { prompt_tokens: 5, completion_tokens: 1 } },
  ].map((x) => `data: ${JSON.stringify(x)}\n\n`).join('') + 'data: [DONE]\n\n';
  const stream = new ReadableStream<Uint8Array>({ start(c) { c.enqueue(new TextEncoder().encode(lines)); c.close(); } });
  const events: string[] = [];
  for await (const event of anthropicEventsFromOpenAI(stream, 'client-model')) events.push(event);
  const names = events.map((e) => /^event: ([^\n]+)/.exec(e)?.[1]);
  assert.deepEqual(names, ['message_start', 'content_block_start', 'content_block_delta', 'content_block_stop', 'message_delta', 'message_stop']);
  assert.match(events[4], /"output_tokens":1/);
});

test('OpenAI stream synthesizes exact final usage before DONE when upstream omits it', async () => {
  const source = [
    { id: 's', object: 'chat.completion.chunk', created: 3, model: 'm', choices: [{ delta: { content: 'hello' }, finish_reason: null }] },
    { id: 's', object: 'chat.completion.chunk', created: 3, model: 'm', choices: [{ delta: {}, finish_reason: 'stop' }] },
  ].map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join('') + 'data: [DONE]\n\n';
  const stream = new ReadableStream<Uint8Array>({ start(c) { c.enqueue(new TextEncoder().encode(source)); c.close(); } });
  const out: string[] = [];
  for await (const event of openAIStreamWithFinalUsage(stream, { messages: [{ role: 'user', content: 'abcd' }] })) out.push(event);
  const usageFrame = JSON.parse(out.at(-2)!.split('data: ')[1]);
  assert.deepEqual(usageFrame.usage, { prompt_tokens: 2, completion_tokens: 2, total_tokens: 4 });
  assert.equal(out.at(-1), 'data: [DONE]\n\n');
  assert.equal(openAIRequestInputTokens({ messages: [{ role: 'user', content: 'abcd' }] }), 2);
});

test('OpenAI stream preserves upstream final usage when present', async () => {
  const source = `data: ${JSON.stringify({ id: 's', choices: [], usage: { prompt_tokens: 8, completion_tokens: 6, total_tokens: 14 } })}\n\n` + 'data: [DONE]\n\n';
  const stream = new ReadableStream<Uint8Array>({ start(c) { c.enqueue(new TextEncoder().encode(source)); c.close(); } });
  const out: string[] = [];
  for await (const event of openAIStreamWithFinalUsage(stream, { messages: [{ role: 'user', content: 'fallback' }] })) out.push(event);
  assert.deepEqual(JSON.parse(out.at(-2)!.split('data: ')[1]).usage, { prompt_tokens: 8, completion_tokens: 6, total_tokens: 14 });
});

test('Anthropic streaming synthesizes output usage when its OpenAI upstream omits usage', async () => {
  const source = `data: ${JSON.stringify({ choices: [{ delta: { content: 'hello' }, finish_reason: null }] })}\n\n` +
    `data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: 'stop' }] })}\n\n` + 'data: [DONE]\n\n';
  const stream = new ReadableStream<Uint8Array>({ start(c) { c.enqueue(new TextEncoder().encode(source)); c.close(); } });
  const events: string[] = [];
  for await (const event of anthropicEventsFromOpenAI(stream, 'm', 7)) events.push(event);
  assert.match(events.find((event) => event.startsWith('event: message_delta'))!, /"output_tokens":2/);
});

test('transient 429 attempts another eligible model and only explicit exhaustion retires provider', async () => {
  const candidates = [{ provider: 'p', model: 'a' }, { provider: 'p', model: 'b' }, { provider: 'q', model: 'c' }];
  const retired = new Set<string>();
  const tried: string[] = [];
  const result = await failoverCandidates(candidates, retired, (c) => c.provider, async (candidate) => {
    tried.push(candidate.model);
    return candidate.model === 'a' ? { ok: false } : { ok: true, value: candidate.model };
  });
  assert.deepEqual(tried, ['a', 'b']); assert.equal(result?.value, 'b'); assert.equal(retired.size, 0);
  assert.equal(providerIsExhausted(429, { error: { code: 'rate_limit', message: 'provider refused this call' } }), false);
  assert.equal(providerIsExhausted(402, { error: { code: 'rate_limit', message: 'provider refused this call' } }), false);
  assert.equal(providerIsExhausted(429, { error: { code: 'rate_limit', message: "provider's free-tier budget is spent" } }), true);
  assert.equal(providerIsExhausted(429, { error: { code: 'rate_limit', message: 'provider is out' } }), true);
});
