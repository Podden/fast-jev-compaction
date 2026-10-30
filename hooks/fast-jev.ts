import type {
  On,
  PluginOptions,
  Register,
  SessionMessage,
  ToolResultSummary,
  ToolUseSummary,
  TurnCompleteInput,
} from 'claude-code';

import { compact, reductionRatio, resolveOptions } from '../src/compact.js';
import {
  buildCondenseRequest,
  CONDENSE_SYSTEM,
  condensePrompt,
  parseCondenseResponse,
} from '../src/condense.js';
import { buildJevRequest, DEFAULT_MODEL, parseJevResponse } from '../src/request.js';
import type {
  CompactOptions,
  CompactResult,
  EpisodeCondenser,
  JevAsker,
  Message,
  ToolResult,
  ToolUse,
} from '../src/types.js';

const HOOK_DEFAULTS = {
  compactAtPercent: 60,
  minReductionRatio: 0.25,
  model: DEFAULT_MODEL,
  condenseModel: 'qwen3.8:27b',
};

export type HookFetchInit = {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
};

export type HookFetchResponse = {
  status: number;
  ok: boolean;
  text: string;
};

/** The shape of `$.http.fetch`, so the hook can be driven without an engine. */
export type HookFetch = (url: string, init?: HookFetchInit) => Promise<HookFetchResponse>;

export type HookConfig = CompactOptions & {
  apiKey?: string;
  /** A System One endpoint other than TypeSafe's, e.g. a self-hosted decider; needs no key. */
  baseUrl?: string;
  compactAtPercent: number;
  minReductionRatio: number;
  model: string;
  /** OpenAI-compatible chat completions URL; set, kept episodes get a fact note. */
  condenseUrl?: string;
  condenseModel: string;
  /** A Claude model (`haiku` or a full id) that writes the fact note through the session's own client; wins over `condenseUrl`. */
  condenseClaudeModel?: string;
};

/** The shape of `$.model.complete`. */
export type ModelComplete = (request: {
  model: string;
  prompt: string;
  system?: string;
  maxTokens?: number;
}) => Promise<string>;

function optionNumber(options: PluginOptions, key: string, fallback: number): number {
  const value = options[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function optionString(options: PluginOptions, key: string): string | undefined {
  const value = options[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

/** Reads the plugin's `userConfig` values; anything missing takes the defaults. */
export function resolveHookConfig(options: PluginOptions): HookConfig {
  const numbers: Partial<Omit<CompactOptions, 'goal'>> = {};
  for (const key of [
    'keepThreshold',
    'preserveRecentMessages',
    'maxStateTokens',
    'maxRequestTokens',
    'truncateHeadChars',
    'maxConcurrentRequests',
    'condenseMinChars',
    'maxConcurrentCondense',
  ] as const) {
    const value = options[key];
    if (typeof value === 'number' && Number.isFinite(value)) numbers[key] = value;
  }
  const config: HookConfig = {
    ...numbers,
    compactAtPercent: optionNumber(options, 'compactAtPercent', HOOK_DEFAULTS.compactAtPercent),
    minReductionRatio: optionNumber(
      options,
      'minReductionRatio',
      HOOK_DEFAULTS.minReductionRatio,
    ),
    model: optionString(options, 'model') ?? HOOK_DEFAULTS.model,
    condenseModel: optionString(options, 'condenseModel') ?? HOOK_DEFAULTS.condenseModel,
  };
  const condenseUrl = optionString(options, 'condenseUrl');
  if (condenseUrl) config.condenseUrl = condenseUrl;
  const condenseClaudeModel = optionString(options, 'condenseClaudeModel');
  if (condenseClaudeModel) config.condenseClaudeModel = condenseClaudeModel;
  const apiKey = optionString(options, 'apiKey');
  if (apiKey) config.apiKey = apiKey;
  const baseUrl = optionString(options, 'baseUrl');
  if (baseUrl) config.baseUrl = baseUrl;
  const goal = optionString(options, 'goal');
  if (goal) config.goal = goal;
  return config;
}

/** A `JevAsker` over the engine's `$.http.fetch`. */
export function jevAsker(
  fetchFn: HookFetch,
  apiKey: string | undefined,
  model: string,
  baseUrl?: string,
): JevAsker {
  return {
    async ask(state, questions) {
      const request = buildJevRequest({ apiKey, model, baseUrl }, state, questions);
      const response = await fetchFn(request.url, {
        method: request.method,
        headers: request.headers,
        body: request.body,
      });
      return parseJevResponse(response.status, response.ok, response.text);
    },
  };
}

function toolUseSummary(tool: ToolUse): ToolUseSummary {
  const summary: ToolUseSummary = {
    tool_use_id: tool.tool_use_id,
    tool: tool.tool,
    input: tool.input,
  };
  if (tool.text !== undefined) summary.text = tool.text;
  if (tool.isError) summary.isError = true;
  return summary;
}

function toolResultSummary(result: ToolResult): ToolResultSummary {
  return {
    tool_use_id: result.tool_use_id,
    text: result.text,
    isError: result.isError ?? false,
  };
}

/**
 * Maps the library's output back onto session messages. Whatever came back
 * unchanged (a message, a tool use, a tool result) is the engine's own object,
 * handle included; anything rebuilt is a fresh message without a handle, so the
 * engine takes the edited content instead of its original.
 */
export function toSessionMessages(
  input: readonly SessionMessage[],
  output: readonly Message[],
): SessionMessage[] {
  const messages = new Map<Message, SessionMessage>();
  const uses = new Map<ToolUse, ToolUseSummary>();
  const results = new Map<ToolResult, ToolResultSummary>();
  for (const message of input) {
    messages.set(message, message);
    for (const tool of message.toolUses) uses.set(tool, tool);
    for (const result of message.toolResults ?? []) results.set(result, result);
  }
  return output.map((message) => {
    const own = messages.get(message);
    if (own) return own;
    const rebuilt: SessionMessage = {
      role: message.role,
      text: message.text,
      toolUses: message.toolUses.map((tool) => uses.get(tool) ?? toolUseSummary(tool)),
    };
    if (message.toolResults && message.toolResults.length > 0) {
      rebuilt.toolResults = message.toolResults.map(
        (result) => results.get(result) ?? toolResultSummary(result),
      );
    }
    return rebuilt;
  });
}

/** An `EpisodeCondenser` over the engine's `$.http.fetch`. */
export function llmCondenser(fetchFn: HookFetch, url: string, model: string): EpisodeCondenser {
  return {
    async condense(input) {
      const request = buildCondenseRequest({ url, model }, input);
      const response = await fetchFn(request.url, request);
      return parseCondenseResponse(response.status, response.ok, response.text);
    },
  };
}

/**
 * An `EpisodeCondenser` over `$.model.complete`: the session's own client and
 * credentials, so no key and no `$.http.fetch` time limit.
 */
export function claudeCondenser(complete: ModelComplete, model: string): EpisodeCondenser {
  return {
    async condense(input) {
      const note = (
        await complete({ model, system: CONDENSE_SYSTEM, prompt: condensePrompt(input), maxTokens: 1000 })
      ).trim();
      if (!note) throw new Error(`condense reply from ${model} has no text`);
      return note;
    },
  };
}

/**
 * Drops the engine's `handle` so each kept message is written as a fresh
 * record after the compact boundary. With handles, kept records keep their
 * parentUuid and message.id from before the boundary, and `--resume` walks
 * back into the full history or repeats them (#89, #129, #145;
 * anthropics/claude-code#95328). The engine's hidden bookkeeping for those
 * messages (reasoning, images) is lost; their text and tool pairs stay.
 */
export function withoutHandles(messages: readonly SessionMessage[]): SessionMessage[] {
  return messages.map(({ handle: _handle, ...rest }) => rest);
}

export type SessionCompaction = {
  result: CompactResult;
  messages: SessionMessage[];
};

/** Runs the library over a session transcript; throws when TypeSafe's key is missing or Jev fails. */
export async function compactSession(
  messages: readonly SessionMessage[],
  config: HookConfig,
  fetchFn: HookFetch,
  archive?: CompactOptions['archive'],
  complete?: ModelComplete,
): Promise<SessionCompaction> {
  if (!config.apiKey && !config.baseUrl) throw new Error('TYPESAFE_API_KEY is not configured');
  const condenser =
    config.condenseClaudeModel && complete
      ? claudeCondenser(complete, config.condenseClaudeModel)
      : config.condenseUrl
        ? llmCondenser(fetchFn, config.condenseUrl, config.condenseModel)
        : undefined;
  const result = await compact(
    messages,
    jevAsker(fetchFn, config.apiKey, config.model, config.baseUrl),
    { ...config, ...(condenser ? { condenser, archive } : {}) },
  );
  return { result, messages: toSessionMessages(messages, result.messages) };
}

function percent(ratio: number): string {
  return `${Math.round(ratio * 100)}%`;
}

function seconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)} s`;
}

export function summarize(result: CompactResult): string {
  const { stats } = result;
  const parts = [
    `${stats.replaced} of ${stats.replaced + stats.kept} episodes replaced (${stats.callsRemoved} calls)`,
    stats.condensed > 0 || stats.condenseFailed > 0
      ? `${stats.condensed} condensed${
          stats.condenseFailed > 0 ? `, ${stats.condenseFailed} condense failed` : ''
        } in ${seconds(stats.condenseMs)}`
      : '',
    stats.skipped > 0 ? `${stats.skipped} too small` : '',
  ].filter(Boolean);
  return `${percent(reductionRatio(result))} reduction; ${parts.join(', ')}; state up to ~${
    stats.stateTokens
  } tokens in ${stats.requests} request(s), ${seconds(stats.deciderMs)}; total ${seconds(stats.ms)}`;
}

const UI_LOG_MAX_CHARS = 4096;

export function decisionLog(result: CompactResult): string {
  return result.decisions
    .map((d) => {
      const condense =
        d.condenseMs === undefined
          ? ''
          : `/condense=${d.condenseError === undefined ? 'ok' : 'failed'}:${seconds(d.condenseMs).replace(' ', '')}`;
      return `${d.id}:${d.calls}calls/${d.chars}ch:${d.action}/keep=${d.keep.toFixed(2)}${condense}`;
    })
    .join(' ');
}

/** One line naming each distinct condenser failure and how often it happened; none without failures. */
export function condenseErrorLines(result: CompactResult): string[] {
  const counts = new Map<string, number>();
  for (const { condenseError } of result.decisions) {
    if (condenseError !== undefined) counts.set(condenseError, (counts.get(condenseError) ?? 0) + 1);
  }
  if (counts.size === 0) return [];
  const listed = [...counts].map(([error, count]) => `${count}× ${error.slice(0, 300)}`).join(' | ');
  return [`condense errors: ${listed}`.slice(0, UI_LOG_MAX_CHARS)];
}

export function decisionLogLines(
  result: CompactResult,
  maxChars: number = UI_LOG_MAX_CHARS,
): string[] {
  const entries = decisionLog(result).split(' ').filter(Boolean);
  if (entries.length === 0) return ['decisions: (none)'];
  const chunks: string[] = [];
  let current = '';
  for (const entry of entries) {
    const next = current ? `${current} ${entry}` : entry;
    if (current && next.length > maxChars - 24) {
      chunks.push(current);
      current = entry;
    } else current = next;
  }
  chunks.push(current);
  return chunks.map((chunk, index) =>
    chunks.length === 1
      ? `decisions: ${chunk}`
      : `decisions (${index + 1}/${chunks.length}): ${chunk}`,
  );
}

async function getApiKey(
  $: {
    env: { get: (name: string) => Promise<string | undefined> };
    settings: { read: () => Promise<Readonly<Record<string, unknown>>> };
  },
  config: HookConfig,
): Promise<string | undefined> {
  if (config.apiKey) return config.apiKey;
  const fromEnv = await $.env.get('TYPESAFE_API_KEY');
  if (fromEnv) return fromEnv;
  const settings = await $.settings.read();
  const env = settings['env'];
  if (env && typeof env === 'object') {
    const value = (env as Record<string, unknown>)['TYPESAFE_API_KEY'];
    if (typeof value === 'string' && value) return value;
  }
  return undefined;
}

function notify(
  $: {
    ui: {
      log: (text: string) => void;
      toast: (text: string, options?: { timeoutMs?: number }) => void;
    };
  },
  text: string,
): void {
  $.ui.log(text);
  $.ui.toast(text, { timeoutMs: 15_000 });
}

export const register: Register = (on: On, options: PluginOptions) => {
  const configured = resolveHookConfig(options);
  let compacting = false;

  on('session.compact', async ($, event, next) => {
    // Subagent and fork transcripts are short-lived; leave them to the engine.
    if (event.agentId) return next(event);
    const tag = `[${event.trigger}]`;
    // A speculative compaction would cost a full Jev run per turn (#107).
    if (event.trigger === 'precompute') {
      $.ui.log(`${tag} skipped: fast-jev-compaction does not precompute`);
      return { skip: 'fast-jev-compaction does not precompute' };
    }
    $.ui.log(`${tag} start: ${event.messages.length} messages`);
    const fallBack = async (reason: string) => {
      notify($, `${tag} fallback to built-in summary (${reason})`);
      const started = Date.now();
      try {
        const out = await next(event);
        $.ui.log(
          `${tag} built-in summary ${
            out.skip !== undefined ? `skipped (${out.skip})` : `done: ${out.messages?.length ?? 0} messages`
          } in ${seconds(Date.now() - started)}`,
        );
        return out;
      } catch (error) {
        $.ui.log(
          `${tag} built-in summary failed in ${seconds(Date.now() - started)} (${
            error instanceof Error ? error.message : String(error)
          })`,
        );
        throw error;
      }
    };
    try {
      const config = { ...configured, apiKey: await getApiKey($, configured) };
      const folder = `.claude/fast-jev/${new Date().toISOString().replace(/[:.]/g, '-')}`;
      const { result, messages } = await compactSession(
        event.messages,
        config,
        async (url, init) => {
          const response = await $.http.fetch(url, init);
          return { status: response.status, ok: response.ok, text: response.text };
        },
        async (episode, text) => {
          // Full tool outputs stay on this machine; keep them out of git.
          await $.fs.write('.claude/fast-jev/.gitignore', '*\n');
          const path = `${folder}/${episode.id}.txt`;
          await $.fs.write(path, text);
          return path;
        },
        (request) => $.model.complete(request),
      );
      for (const line of decisionLogLines(result)) $.ui.log(line);
      for (const line of condenseErrorLines(result)) $.ui.log(line);
      if (reductionRatio(result) < config.minReductionRatio) {
        return fallBack(`below ${percent(config.minReductionRatio)} minimum: ${summarize(result)}`);
      }
      notify(
        $,
        `${tag} kept ${messages.length}/${event.messages.length} messages, no summary (${summarize(result)})`,
      );
      return { messages: withoutHandles(messages) };
    } catch (error) {
      return fallBack(error instanceof Error ? error.message : String(error));
    }
  });

  on('turn.complete', async ($, event: TurnCompleteInput, next) => {
    if (event.agentId || event.reason !== 'answer' || compacting) return next(event);
    // Set before the first await, so a second turn.complete cannot slip past (#107).
    compacting = true;
    const skipped = (error: unknown) =>
      $.ui.log(`auto-compact skipped (${error instanceof Error ? error.message : String(error)})`);
    try {
      const { context } = await $.session.usage();
      if ((context.percent ?? 0) < configured.compactAtPercent) {
        compacting = false;
        return next(event);
      }
      $.ui.log(
        `auto-compact requested at ${context.percent}% (${context.tokens ?? '?'} tokens, compactAtPercent ${configured.compactAtPercent})`,
      );
      // Inside this dispatch the turn still counts as in flight and compact()
      // is rejected; run it once the dispatch has resolved (#149).
      $.clock.after(0, () => {
        $.session
          .compact()
          .then((out) => {
            if (out.skip !== undefined) $.ui.log(`auto-compact skipped (${out.skip})`);
          })
          .catch(skipped)
          .finally(() => {
            compacting = false;
          });
      });
    } catch (error) {
      skipped(error);
      compacting = false;
    }
    return next(event);
  });
};

export { resolveOptions };
