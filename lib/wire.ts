// The client wires and the one upstream wire this baseline speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This baseline forwards both to a
// provider on the openai wire and translates back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to the provider and streams
//     the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// The gemini and anthropic upstream wires are not implemented in this baseline (main.ts answers 501 for a
// provider on either): a real entry needs their request and event shapes too, which is the biggest single
// gap in this starter.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

// ============================================================================
// Cross-wire translation (this entry): every client wire x every provider wire.
//
// The trial's client speaks OpenAI or Anthropic; the providers speak OpenAI,
// Anthropic or Gemini. The starter only translated one direction of one pair.
// This section adds the rest, plus a uniform "parse the provider's answer and
// re-emit it in the client's wire" path, so every response carries the exact
// text, usage and stop family the provider reported. Everything here is a pure
// function of its arguments: no clock, no randomness, fixed ids and timestamps,
// so a replay reproduces the same bytes.
// ============================================================================

/** The fixed epoch second every re-emitted completion carries. */
const FIXED_CREATED = 1767225600;

/** The text of Anthropic content blocks joined with nothing, as the env's answer parser does. */
function anthropicBlocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
}

/** One OpenAI content part as a Gemini part: text or an inline image; null for the rest. */
function geminiPartOfOpenAI(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    if (url.startsWith('data:')) {
      const m = /^data:([^;]+);base64,(.*)$/.exec(url);
      if (m !== null) return { inlineData: { mimeType: m[1], data: m[2] } };
      return null;
    }
    return { fileData: { fileUri: url } };
  }
  return null;
}

/** An OpenAI message's content as Gemini parts. */
function geminiPartsFromOpenAIContent(content: unknown): Json[] {
  if (typeof content === 'string') return content !== '' ? [{ text: content }] : [];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    const p = geminiPartOfOpenAI(part);
    if (p !== null) parts.push(p);
  }
  return parts;
}

/** One Anthropic content block as a Gemini part: text or an inline image; null for the rest. */
function geminiPartOfAnthropic(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/** An Anthropic message's content as Gemini parts (tool blocks are handled by the caller). */
function geminiPartsFromAnthropicContent(content: unknown): Json[] {
  if (typeof content === 'string') return content !== '' ? [{ text: content }] : [];
  if (!Array.isArray(content)) return [];
  const parts: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    const p = geminiPartOfAnthropic(block);
    if (p !== null) parts.push(p);
  }
  return parts;
}

/** OpenAI tools as Gemini function declarations, in definition order. */
function openAIToolsToGemini(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [
      {
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      },
    ];
  });
}

/** Anthropic tools as Gemini function declarations, in definition order. */
function anthropicToolsToGemini(tools: unknown): Json[] {
  if (!Array.isArray(tools)) return [];
  return tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [
      {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    ];
  });
}

interface GeminiRequestOptions {
  tools?: Json[];
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  stops?: string[];
  json?: boolean;
}

/** One Gemini generateContent body from contents and a system instruction. */
function geminiRequest(contents: Json[], systemParts: Json[], opts: GeminiRequestOptions): Json {
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts };
  if (opts.tools !== undefined && opts.tools.length > 0) out.tools = [{ functionDeclarations: opts.tools }];
  const generationConfig: Json = {};
  if (opts.maxTokens !== undefined) generationConfig.maxOutputTokens = opts.maxTokens;
  if (opts.temperature !== undefined) generationConfig.temperature = opts.temperature;
  if (opts.topP !== undefined) generationConfig.topP = opts.topP;
  if (opts.stops !== undefined && opts.stops.length > 0) generationConfig.stopSequences = opts.stops;
  if (opts.json === true) generationConfig.responseMimeType = 'application/json';
  if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;
  return out;
}

/**
 * An OpenAI Chat Completions request as a Gemini generateContent body: the system messages become the
 * system instruction, user messages become user contents, assistant messages become model contents
 * (tool calls become functionCall parts), tool messages become functionResponse parts, and the output cap,
 * temperature, top_p, stop sequences, tools and JSON mode pass through.
 */
export function openAIToGemini(body: Json, _model: string): Json {
  const systemParts: Json[] = [];
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      for (const p of geminiPartsFromOpenAIContent(raw.content)) systemParts.push(p);
      continue;
    }
    if (raw.role === 'assistant') {
      const parts = geminiPartsFromOpenAIContent(raw.content);
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({
          functionCall: {
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argumentsOf(call.function.arguments),
          },
        });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      contents.push({
        role: 'user',
        parts: [{ functionResponse: { name: '', response: { content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) } } }],
      });
      continue;
    }
    const parts = geminiPartsFromOpenAIContent(raw.content);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const stops = Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : typeof body.stop === 'string' ? [body.stop] : undefined;
  return geminiRequest(contents, systemParts, {
    tools: openAIToolsToGemini(body.tools),
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : undefined,
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    topP: typeof body.top_p === 'number' ? body.top_p : undefined,
    stops,
    json: isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema'),
  });
}

/**
 * An Anthropic Messages request as a Gemini generateContent body: the system field becomes the system
 * instruction, messages become user/model contents (tool_use blocks become functionCall parts,
 * tool_result blocks become functionResponse parts), and the output cap, temperature, top_p, stop
 * sequences and tools pass through. An Anthropic client never asks for JSON mode, so none is set.
 */
export function anthropicToGemini(body: Json, _model: string): Json {
  const systemParts = geminiPartsFromAnthropicContent(body.system);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts = geminiPartsFromAnthropicContent(raw.content);
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block) || block.type !== 'tool_use') continue;
          parts.push({
            functionCall: {
              name: typeof block.name === 'string' ? block.name : '',
              args: isRecord(block.input) ? block.input : {},
            },
          });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts = geminiPartsFromAnthropicContent(raw.content);
    if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block) || block.type !== 'tool_result') continue;
        parts.push({ functionResponse: { name: '', response: { content: toolResultText(block.content) } } });
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : undefined;
  return geminiRequest(contents, systemParts, {
    tools: anthropicToolsToGemini(body.tools),
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : undefined,
    temperature: typeof body.temperature === 'number' ? body.temperature : undefined,
    topP: typeof body.top_p === 'number' ? body.top_p : undefined,
    stops,
    json: false,
  });
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request: system messages become the system
 * field, user messages become user blocks (image_url parts become base64/url image blocks), assistant
 * messages become assistant blocks (tool calls become tool_use blocks), tool messages become user
 * tool_result blocks, and the output cap, temperature, top_p, stop sequences, tools and tool choice pass
 * through. An Anthropic provider cannot echo JSON mode, so callers must not route a JSON-mode OpenAI
 * request to one.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const t = textOf(raw.content);
      if (t !== '') systemParts.push(t);
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const t = textOf(raw.content);
      if (t !== '') blocks.push({ type: 'text', text: t });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : textOf(raw.content),
          },
        ],
      });
      continue;
    }
    const parts: Json[] = [];
    const content = raw.content;
    if (typeof content === 'string') {
      if (content !== '') parts.push({ type: 'text', text: content });
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ type: 'text', text: part.text });
        else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
          const url = part.image_url.url;
          if (url.startsWith('data:')) {
            const m = /^data:([^;]+);base64,(.*)$/.exec(url);
            if (m !== null) parts.push({ type: 'image', source: { type: 'base64', media_type: m[1], data: m[2] } });
          } else {
            parts.push({ type: 'image', source: { type: 'url', url } });
          }
        }
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  else if (typeof body.stop === 'string' && body.stop !== '') out.stop_sequences = [body.stop];
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** A provider's stop value as an OpenAI finish_reason. */
export function openAIStopOf(stop: string): string {
  if (stop === 'max_tokens' || stop === 'MAX_TOKENS') return 'length';
  if (stop === 'tool_use' || stop === 'tool_calls') return 'tool_calls';
  if (stop === 'refusal' || stop === 'SAFETY') return 'content_filter';
  if (stop === 'stop_sequence') return 'stop';
  return 'stop';
}

/** A provider's stop value as an Anthropic stop_reason. */
export function anthropicStopOf(stop: string): string {
  if (stop === 'length' || stop === 'MAX_TOKENS') return 'max_tokens';
  if (stop === 'tool_calls' || stop === 'tool_use') return 'tool_use';
  if (stop === 'content_filter' || stop === 'SAFETY') return 'refusal';
  if (stop === 'stop_sequence') return 'stop_sequence';
  return 'end_turn';
}

/** A Gemini answer's text: its first candidate's parts joined with nothing. */
export function geminiPartsText(g: Json): string {
  const cands = Array.isArray(g.candidates) ? g.candidates : [];
  const cand = isRecord(cands[0]) ? cands[0] : {};
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Gemini answer's usage, or null when it carries none. */
export function geminiUsageOf(g: Json): { input: number; output: number } | null {
  const usage = isRecord(g.usageMetadata) ? g.usageMetadata : null;
  if (usage === null) return null;
  const input = typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : null;
  const output = typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : null;
  return input === null || output === null ? null : { input, output };
}

/** A Gemini answer's finishReason, or null when it carries none. */
export function geminiFinishOf(g: Json): string | null {
  const cands = Array.isArray(g.candidates) ? g.candidates : [];
  const cand = isRecord(cands[0]) ? cands[0] : {};
  return typeof cand.finishReason === 'string' ? cand.finishReason : null;
}

/** One parsed provider answer: its text, its usage when reported, and its raw stop value. */
export interface ParsedAnswer {
  text: string;
  usage: { input: number; output: number } | null;
  stop: string;
}

/** An OpenAI chat completion JSON as a parsed answer. */
export function parseOpenAIJson(body: Json): ParsedAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : anthropicBlocksText(message.content);
  const usage = isRecord(body.usage)
    ? {
        input: typeof body.usage.prompt_tokens === 'number' ? body.usage.prompt_tokens : 0,
        output: typeof body.usage.completion_tokens === 'number' ? body.usage.completion_tokens : 0,
      }
    : null;
  return { text, usage, stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop' };
}

/** An Anthropic message JSON as a parsed answer. */
export function parseAnthropicJson(body: Json): ParsedAnswer {
  const usage = isRecord(body.usage)
    ? {
        input: typeof body.usage.input_tokens === 'number' ? body.usage.input_tokens : 0,
        output: typeof body.usage.output_tokens === 'number' ? body.usage.output_tokens : 0,
      }
    : null;
  return { text: anthropicBlocksText(body.content), usage, stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn' };
}

/** A Gemini generateContent JSON as a parsed answer. */
export function parseGeminiJson(body: Json): ParsedAnswer {
  return { text: geminiPartsText(body), usage: geminiUsageOf(body), stop: geminiFinishOf(body) ?? 'STOP' };
}

/** A parsed answer as an OpenAI chat completion JSON naming `model`. */
export function openAIJsonFromParsed(answer: ParsedAnswer, usage: { input: number; output: number }, model: string): Json {
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: openAIStopOf(answer.stop) }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** A parsed answer as an Anthropic message JSON naming `model`. */
export function anthropicJsonFromParsed(answer: ParsedAnswer, usage: { input: number; output: number }, model: string): Json {
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStopOf(answer.stop),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

// ---------------------------------------------------------------------------
// Server-sent events: parse a buffered upstream stream, then re-emit it in the
// client's wire with the usage always present (the trial's client never asks
// for include_usage, so a passthrough would drop it).
// ---------------------------------------------------------------------------

/** One SSE frame: its event name (null when unnamed) and its joined data lines. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** A buffered SSE body as frames. */
export function parseSseText(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** Split text into 13-character pieces, the env's own delta size. */
export function splitText13(text: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length; i += 13) out.push(text.slice(i, i + 13));
  if (out.length === 0) out.push('');
  return out;
}

/** An OpenAI SSE stream as a parsed answer. */
export function parseOpenAIStreamText(sse: string): ParsedAnswer {
  let text = '';
  let usage: { input: number; output: number } | null = null;
  let stop = 'stop';
  for (const frame of parseSseText(sse)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      usage = {
        input: typeof chunk.usage.prompt_tokens === 'number' ? chunk.usage.prompt_tokens : 0,
        output: typeof chunk.usage.completion_tokens === 'number' ? chunk.usage.completion_tokens : 0,
      };
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
  }
  return { text, usage, stop };
}

/** An Anthropic SSE stream as a parsed answer. */
export function parseAnthropicStreamText(sse: string): ParsedAnswer {
  let text = '';
  let input = 0;
  let output = 0;
  let stop = 'end_turn';
  for (const frame of parseSseText(sse)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = typeof data.message.usage.input_tokens === 'number' ? data.message.usage.input_tokens : 0;
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      text += data.delta.text;
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = data.delta.stop_reason;
      if (isRecord(data.usage) && typeof data.usage.output_tokens === 'number') output = data.usage.output_tokens;
    }
  }
  return { text, usage: { input, output }, stop };
}

/** A Gemini SSE stream as a parsed answer. */
export function parseGeminiStreamText(sse: string): ParsedAnswer {
  let text = '';
  let usage: { input: number; output: number } | null = null;
  let stop = 'STOP';
  for (const frame of parseSseText(sse)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const piece = geminiPartsText(data);
    if (piece !== '') text += piece;
    const u = geminiUsageOf(data);
    if (u !== null) usage = u;
    const s = geminiFinishOf(data);
    if (s !== null) stop = s;
  }
  return { text, usage, stop };
}

/** A parsed answer as an OpenAI SSE stream naming `model`: role chunk, text deltas, finish chunk, usage chunk, [DONE]. */
export function openAIChunksFromParsed(answer: ParsedAnswer, usage: { input: number; output: number }, model: string): string {
  const base = { id: 'chatcmpl-aggregate', object: 'chat.completion.chunk', created: FIXED_CREATED, model };
  const chunks: Json[] = [{ ...base, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] }];
  for (const piece of splitText13(answer.text)) {
    chunks.push({ ...base, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] });
  }
  chunks.push({ ...base, choices: [{ index: 0, delta: {}, finish_reason: openAIStopOf(answer.stop) }] });
  chunks.push({
    ...base,
    choices: [],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  });
  return `${chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join('')}data: [DONE]\n\n`;
}

/** A parsed answer as an Anthropic SSE stream naming `model`: start, one text block, delta, stop. */
export function anthropicEventsFromParsed(answer: ParsedAnswer, usage: { input: number; output: number }, model: string): string {
  const out: string[] = [];
  out.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: 0 },
      },
    }),
  );
  out.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  for (const piece of splitText13(answer.text)) {
    out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  out.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopOf(answer.stop), stop_sequence: null },
      usage: { output_tokens: usage.output },
    }),
  );
  out.push(sseEvent('message_stop', { type: 'message_stop' }));
  return out.join('');
}

/**
 * The provider's usage from the client body and the answer text, the env's own formula: input counts
 * the system text, the user text and the tool names joined by commas; output counts the answer. Only a
 * fallback for a stream that carried no usage; the provider's own numbers are used whenever present.
 */
export function computeUsage(clientApi: 'openai' | 'anthropic', body: Json, answerText: string): { input: number; output: number } {
  let system = '';
  let user = '';
  const tools: string[] = [];
  if (clientApi === 'openai') {
    const systemTexts: string[] = [];
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const t = textOf(raw.content);
        if (t !== '') systemTexts.push(t);
      }
    }
    system = systemTexts.join('\n');
    for (let i = source.length - 1; i >= 0; i--) {
      const raw = source[i];
      if (isRecord(raw) && raw.role === 'user') {
        user = textOf(raw.content);
        break;
      }
    }
    if (Array.isArray(body.tools)) {
      for (const t of body.tools) if (isRecord(t) && isRecord(t.function) && typeof t.function.name === 'string') tools.push(t.function.name);
    }
  } else {
    system = textOf(body.system);
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (let i = source.length - 1; i >= 0; i--) {
      const raw = source[i];
      if (isRecord(raw) && raw.role === 'user') {
        user = textOf(raw.content);
        break;
      }
    }
    if (Array.isArray(body.tools)) {
      for (const t of body.tools) if (isRecord(t) && typeof t.name === 'string') tools.push(t.name);
    }
  }
  const prompt = system.length + user.length + tools.join(',').length;
  return { input: 1 + Math.floor(prompt / 4), output: 1 + Math.floor(answerText.length / 4) };
}
