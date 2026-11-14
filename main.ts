// free-model-aggregator-01: the best free AI model aggregator.
//
// One server, two client wires (OpenAI Chat Completions, Anthropic Messages), ten simulated free-tier
// providers on three upstream wires (openai, anthropic, gemini). Read the brief at $MAKEFUN_PROMPT_FILE,
// probe every provider's catalog route, build one catalog, register it, serve the client's twenty
// requests, poll /work and exit 0 when done. `node main.ts --list` prints the catalog and exits.
//
// Routing (quality 60%): every free model that meets the request's requirements is ranked by the quality
// table's rating_milli, best first; ties go to the provider that has served the fewest calls so far, so
// load spreads instead of draining one provider. A provider that answers that its free-tier budget is
// spent, or that it is out (outage), is dead for the rest of the trial. A call that is refused (fail_call),
// rate-limited or a 5xx is transient: the request moves to the next candidate, and a provider that failed
// on this request is retried only once every other candidate has been tried. Exactly one successful
// upstream call is made per request, and the answer is re-emitted in the client's wire with the exact
// text, usage and stop family the provider reported.
//
// Wires: all six client-x-provider combinations are translated (the starter only did two). A JSON-mode
// OpenAI request is never routed to an anthropic-wire provider, which cannot echo JSON mode. Streams are
// buffered, parsed and re-emitted, so the usage is always present even though the client never asks for
// include_usage (the upstream OpenAI call sets it).
//
// Determinism: no clock, no randomness; every decision follows from observed provider answers, every
// response carries fixed ids and timestamps, so a replay reproduces the same bytes.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  ratingsProbe,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type ProviderDirectoryEntry,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromParsed,
  anthropicJsonFromParsed,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  computeUsage,
  openAIChunksFromParsed,
  openAIError,
  openAIJsonFromParsed,
  openAIModelList,
  openAIToAnthropicRequest,
  openAIToGemini,
  parseAnthropicJson,
  parseAnthropicStreamText,
  parseGeminiJson,
  parseGeminiStreamText,
  parseOpenAIJson,
  parseOpenAIStreamText,
  type ParsedAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream provider call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Upstream calls made so far per provider, used to spread load across equal ratings. */
const providerCalls = new Map<string, number>();
/** Providers proven dead this trial: budget spent, outage, or a configuration error. */
const providerDead = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider (one retry) and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(
    brief.directory.providers.map(async (provider) => {
      let probe = await probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS);
      if (!probe.ok) {
        log(`probe ${provider.id} failed (${probe.error ?? 'unknown'}); retrying once`);
        probe = await probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS);
      }
      if (!probe.ok) {
        log(`probe ${provider.id} failed twice; routing it from the quality table`);
        return ratingsProbe(provider, brief.ratings);
      }
      return probe;
    }),
  );
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  provider: ProviderDirectoryEntry;
  model: ProbedModel;
  rating: number;
  key: string;
}

/** Whether this OpenAI client body asks for JSON mode (an anthropic-wire provider cannot echo it). */
function jsonModeRequired(clientApi: 'openai' | 'anthropic', body: Json): boolean {
  return (
    clientApi === 'openai' &&
    isRecord(body.response_format) &&
    (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')
  );
}

/**
 * Every free model on a live provider that meets the request's requirements. Recomputed per attempt, so
 * a provider that died mid-request drops out at once.
 */
function allCandidates(requires: Requirement, clientApi: 'openai' | 'anthropic', body: Json): Candidate[] {
  const out: Candidate[] = [];
  const jsonRequired = jsonModeRequired(clientApi, body);
  for (const probe of probes) {
    if (!probe.ok || providerDead.has(probe.provider.id)) continue;
    if (jsonRequired && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ provider: probe.provider, model, rating: row?.rating_milli ?? 0, key: `${probe.provider.id}/${model.id}` });
    }
  }
  return out;
}

/** Best rating first; ties to the provider with the fewest calls so far, then by id for determinism. */
function sortCandidates(list: Candidate[]): Candidate[] {
  return [...list].sort(
    (a, b) =>
      b.rating - a.rating ||
      (providerCalls.get(a.provider.id) ?? 0) - (providerCalls.get(b.provider.id) ?? 0) ||
      (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface ServedAnswer {
  streamText: string | null;
  json: Json | null;
}

type CallFailure = { ok: false; kind: 'dead' | 'transient'; message: string };
type CallOutcome = { ok: true; answer: ServedAnswer } | CallFailure;

/** A failed provider answer classified by its message: budget and outage are permanent, the rest is not. */
function classifyError(status: number, text: string): CallFailure {
  const message = text.slice(0, 300);
  if (/free-tier budget is spent/i.test(message) || / is out\b/i.test(message)) {
    return { ok: false, kind: 'dead', message: `HTTP ${status}: ${message}` };
  }
  if (status === 400 || status === 401 || status === 404 || status === 405) {
    return { ok: false, kind: 'dead', message: `HTTP ${status}: ${message}` };
  }
  return { ok: false, kind: 'transient', message: `HTTP ${status}: ${message}` };
}

/** One provider answer, parsed and re-emitted in the client's wire. */
function translateAnswer(
  wire: 'openai' | 'anthropic' | 'gemini',
  clientApi: 'openai' | 'anthropic',
  stream: boolean,
  rawText: string,
  clientBody: Json,
  model: string,
): ServedAnswer {
  const requestedModel = typeof clientBody.model === 'string' && clientBody.model !== '' ? clientBody.model : model;
  if (stream) {
    const parsed: ParsedAnswer =
      wire === 'openai' ? parseOpenAIStreamText(rawText) : wire === 'anthropic' ? parseAnthropicStreamText(rawText) : parseGeminiStreamText(rawText);
    const usage = parsed.usage ?? computeUsage(clientApi, clientBody, parsed.text);
    if (clientApi === 'openai') return { streamText: openAIChunksFromParsed(parsed, usage, model), json: null };
    return { streamText: anthropicEventsFromParsed(parsed, usage, requestedModel), json: null };
  }
  let upstreamJson: unknown;
  try {
    upstreamJson = JSON.parse(rawText);
  } catch {
    throw new Error('the provider answered a body that is not JSON');
  }
  if (!isRecord(upstreamJson)) throw new Error('the provider answered a JSON value that is not one object');
  const parsed: ParsedAnswer =
    wire === 'openai' ? parseOpenAIJson(upstreamJson) : wire === 'anthropic' ? parseAnthropicJson(upstreamJson) : parseGeminiJson(upstreamJson);
  const usage = parsed.usage ?? computeUsage(clientApi, clientBody, parsed.text);
  if (clientApi === 'openai') return { streamText: null, json: openAIJsonFromParsed(parsed, usage, model) };
  return { streamText: null, json: anthropicJsonFromParsed(parsed, usage, requestedModel) };
}

/** One attempt on one candidate: build the upstream request in the provider's wire and translate its answer. */
async function callCandidate(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<CallOutcome> {
  const provider = candidate.provider;
  const model = candidate.model.id;
  const stream = body.stream === true;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientApi === 'anthropic' ? anthropicToOpenAI(body, model) : { ...body, model };
    if (stream) upstreamBody.stream_options = { include_usage: true };
    headers.accept = stream ? 'text/event-stream' : 'application/json';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    headers.accept = stream ? 'text/event-stream' : 'application/json';
    upstreamBody = clientApi === 'anthropic' ? { ...body, model } : openAIToAnthropicRequest(body, model);
  } else {
    url = stream
      ? `${base}/v1beta/models/${encodeURIComponent(model)}:streamGenerateContent?alt=sse`
      : `${base}/v1beta/models/${encodeURIComponent(model)}:generateContent`;
    headers.accept = stream ? 'text/event-stream' : 'application/json';
    upstreamBody = clientApi === 'anthropic' ? anthropicToGemini(body, model) : openAIToGemini(body, model);
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch (err) {
    return { ok: false, kind: 'transient', message: `unreachable: ${(err as Error).message}` };
  }
  const rawText = await res.text();
  if (!res.ok) return classifyError(res.status, rawText);
  try {
    return { ok: true, answer: translateAnswer(provider.wire, clientApi, stream, rawText, body, model) };
  } catch (err) {
    return { ok: false, kind: 'transient', message: `bad upstream answer: ${(err as Error).message}` };
  }
}

/**
 * Serve one client request: try the ranked candidates, best first, until one answers. Two passes: the
 * first avoids a provider that already failed on this request; the second retries it (its next call may
 * be past its refused call). Returns null when no candidate serves the request.
 */
async function serveRequest(clientApi: 'openai' | 'anthropic', body: Json, requires: Requirement): Promise<ServedAnswer | null> {
  for (let pass = 0; pass < 2; pass++) {
    const tried = new Set<string>();
    const failedProviders = new Set<string>();
    for (;;) {
      const list = sortCandidates(allCandidates(requires, clientApi, body)).filter(
        (c) => !tried.has(c.key) && (pass === 1 || !failedProviders.has(c.provider.id)),
      );
      if (list.length === 0) break;
      const candidate = list[0];
      tried.add(candidate.key);
      providerCalls.set(candidate.provider.id, (providerCalls.get(candidate.provider.id) ?? 0) + 1);
      const outcome = await callCandidate(candidate, clientApi, body);
      if (outcome.ok) {
        log(`served by ${candidate.provider.id}/${candidate.model.id} (rating ${candidate.rating})`);
        return outcome.answer;
      }
      log(`${candidate.provider.id}/${candidate.model.id} failed: ${outcome.message}`);
      if (outcome.kind === 'dead') providerDead.add(candidate.provider.id);
      failedProviders.add(candidate.provider.id);
    }
  }
  return null;
}

/** Serve one client request on either wire and write the answer. */
async function serve(clientApi: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const answer = await serveRequest(clientApi, body, requirementsFor(body));
  if (answer === null) {
    return sendError(res, clientApi, 503, 'no_free_model', 'no free model on the directory serves this request');
  }
  if (answer.streamText !== null) {
    const text = answer.streamText;
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', 'content-length': Buffer.byteLength(text) });
    res.end(text);
    return;
  }
  sendJson(res, 200, answer.json);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
