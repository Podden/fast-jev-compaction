import { noulAnswer } from './request.js';
import {
  collectEpisodes,
  collectToolCalls,
  condensedNote,
  episodeEntry,
  episodeText,
  groundedShare,
  estimateTokens,
  goalFromMessages,
  replacementNote,
  STATE_CONTEXT,
} from './state.js';
import type {
  CompactOptions,
  CompactResult,
  Episode,
  EpisodeDecision,
  EpisodeEntry,
  EpisodeState,
  CondenseInput,
  JevAsker,
  JevQuestions,
  Message,
  ResolvedCompactOptions,
} from './types.js';

export const DEFAULT_OPTIONS: ResolvedCompactOptions = {
  goal: '',
  keepThreshold: 0.5,
  preserveRecentMessages: 6,
  maxStateTokens: 25_000,
  maxRequestTokens: 30_000,
  truncateHeadChars: 300,
  maxConcurrentRequests: 0,
  condenseMinChars: 4000,
  maxConcurrentCondense: 4,
  condenseMinGrounded: 0.3,
};

/** Tokens the request envelope (`model`, key names) adds around state and questions. */
const REQUEST_OVERHEAD_TOKENS = 20;

/** Episodes holding less tool input and output than this are not worth a question. */
export const MIN_EPISODE_CHARS = 1000;

/** Characters of each call's output shown to Jev, tried in turn until an episode fits. */
const STATE_HEAD_CHARS = [240, 60, 0] as const;

function finite(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export function resolveOptions(options: CompactOptions = {}): ResolvedCompactOptions {
  return {
    goal: options.goal ?? DEFAULT_OPTIONS.goal,
    keepThreshold: finite(options.keepThreshold, DEFAULT_OPTIONS.keepThreshold),
    preserveRecentMessages: Math.max(
      0,
      Math.floor(
        finite(options.preserveRecentMessages, DEFAULT_OPTIONS.preserveRecentMessages),
      ),
    ),
    maxStateTokens: Math.max(1, finite(options.maxStateTokens, DEFAULT_OPTIONS.maxStateTokens)),
    maxRequestTokens: Math.max(
      1,
      finite(options.maxRequestTokens, DEFAULT_OPTIONS.maxRequestTokens),
    ),
    truncateHeadChars: Math.max(
      0,
      Math.floor(finite(options.truncateHeadChars, DEFAULT_OPTIONS.truncateHeadChars)),
    ),
    maxConcurrentRequests: Math.max(
      0,
      Math.floor(finite(options.maxConcurrentRequests, DEFAULT_OPTIONS.maxConcurrentRequests)),
    ),
    condenseMinChars: Math.max(0, finite(options.condenseMinChars, DEFAULT_OPTIONS.condenseMinChars)),
    condenseMinGrounded: finite(options.condenseMinGrounded, DEFAULT_OPTIONS.condenseMinGrounded),
    maxConcurrentCondense: Math.max(
      0,
      Math.floor(finite(options.maxConcurrentCondense, DEFAULT_OPTIONS.maxConcurrentCondense)),
    ),
  };
}

/**
 * The one `noul` question asked about an episode: does the text after it
 * report what its calls showed? Asked one episode per request; with many
 * episodes in one state decider-4b mixes them up.
 */
export function questionFor(episode: Pick<Episode, 'id'>): JevQuestions {
  return {
    [`outcome_${episode.id}`]: {
      type: 'noul',
      instructions: `Does \`after\` of episode ${episode.id} report the result of its tool calls?`,
      criteria: {
        true: 'yes: `after` reports what the calls showed (a finding, status, cause, count, or that a file was written)',
        false: 'no: `after` only says what comes next or that the analysis is done, without the result itself',
      },
    },
  };
}

export interface SizedEpisode {
  episode: Episode;
  entry: EpisodeEntry;
  /** Estimated tokens of the entry in the state. */
  entryTokens: number;
}

/**
 * The episode with as much of each output as still fits one request alone;
 * undefined when not even the bare call lines do.
 */
export function sizeEpisode(
  messages: readonly Message[],
  episode: Episode,
  baseTokens: number,
  options: Pick<ResolvedCompactOptions, 'maxStateTokens' | 'maxRequestTokens'>,
): SizedEpisode | undefined {
  const questionTokens = estimateTokens(JSON.stringify(questionFor(episode)));
  for (const head of STATE_HEAD_CHARS) {
    const entry = episodeEntry(messages, episode, head);
    const entryTokens = estimateTokens(JSON.stringify(entry)) + 1;
    if (
      baseTokens + entryTokens <= options.maxStateTokens &&
      baseTokens + entryTokens + questionTokens + REQUEST_OVERHEAD_TOKENS <= options.maxRequestTokens
    ) {
      return { episode, entry, entryTokens };
    }
  }
  return undefined;
}

/**
 * Maps `items` through `fn` with at most `limit` calls in flight (0: all at
 * once), keeping the order. A server that queues requests (decider) would
 * otherwise count the wait for earlier batches against each later request.
 */
async function mapLimited<T, R>(
  items: readonly T[],
  limit: number,
  fn: (item: T) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const index = next++;
      results[index] = await fn(items[index]!);
    }
  };
  const workers = Math.min(limit || items.length, items.length);
  await Promise.all(Array.from({ length: workers }, worker));
  return results;
}

/**
 * Rebuilds the conversation: the calls of every replaced episode disappear
 * together with their results, and the episode's first message gets the
 * replacement note (or its entry in `customNotes`, by episode id). Texts stay
 * verbatim; messages that lose all content are removed; untouched messages
 * are returned as the same objects.
 */
export function applyEpisodes(
  messages: readonly Message[],
  replaced: readonly Episode[],
  errorChars: number,
  customNotes: ReadonlyMap<string, string> = new Map(),
): Message[] {
  const removed = new Set(replaced.flatMap((episode) => episode.calls.map((call) => call.tool_use_id)));
  const notes = new Map(
    replaced.map((episode) => [
      episode.start,
      customNotes.get(episode.id) ?? replacementNote(episode, errorChars),
    ]),
  );
  const kept: Message[] = [];
  messages.forEach((message, index) => {
    const note = notes.get(index);
    const toolUses = message.toolUses.filter((tool) => !removed.has(tool.tool_use_id));
    const toolResults = (message.toolResults ?? []).filter(
      (result) => !removed.has(result.tool_use_id),
    );
    if (
      !note &&
      toolUses.length === message.toolUses.length &&
      toolResults.length === (message.toolResults ?? []).length
    ) {
      kept.push(message);
      return;
    }
    const text = note ? (message.text.trim().length > 0 ? `${message.text}\n\n${note}` : note) : message.text;
    if (text.trim().length === 0 && toolUses.length === 0 && toolResults.length === 0) return;
    const rebuilt: Message = { role: message.role, text, toolUses };
    if (toolResults.length > 0) rebuilt.toolResults = toolResults;
    kept.push(rebuilt);
  });
  return kept;
}

/** Characters of text, tool input and tool output a message holds. */
export function messageChars(message: Message): number {
  let total = message.text.length;
  for (const tool of message.toolUses) {
    try {
      total += JSON.stringify(tool.input).length;
    } catch {
      total += 20;
    }
  }
  for (const result of message.toolResults ?? []) total += result.text.length;
  return total;
}

export function reductionRatio(result: Pick<CompactResult, 'stats'>): number {
  const { charsBefore, charsAfter } = result.stats;
  return charsBefore === 0 ? 0 : (charsBefore - charsAfter) / charsBefore;
}

/**
 * Compacts a transcript by episodes: for every search run outside the pinned
 * first and newest messages, Jev is asked, one request per episode, whether
 * the text the assistant wrote right after it reports what the calls showed.
 * Episodes whose keep probability (1 − that answer) falls below
 * `keepThreshold` are replaced by one line per call. With a `condenser`, kept
 * episodes of at least `condenseMinChars` are replaced by its fact note
 * instead; a failed condenser call keeps the episode. Throws when Jev fails;
 * the caller decides whether to fall back.
 */
export async function compact(
  messages: readonly Message[],
  asker: JevAsker,
  options: CompactOptions = {},
): Promise<CompactResult> {
  const started = Date.now();
  const resolved = resolveOptions(options);
  const calls = collectToolCalls(messages, resolved.preserveRecentMessages);
  const episodes = collectEpisodes(messages, calls, resolved.preserveRecentMessages);
  const charsBefore = messages.reduce((sum, message) => sum + messageChars(message), 0);

  const goal = resolved.goal || goalFromMessages(messages);
  const baseTokens = estimateTokens(JSON.stringify({ context: STATE_CONTEXT, goal, episodes: [] }));
  const sized: SizedEpisode[] = [];
  for (const episode of episodes) {
    if (episode.chars < MIN_EPISODE_CHARS) continue;
    const item = sizeEpisode(messages, episode, baseTokens, resolved);
    if (item) sized.push(item);
  }
  const deciderStarted = Date.now();
  const answered = await mapLimited(sized, resolved.maxConcurrentRequests, async (item) => {
    const state: EpisodeState = { context: STATE_CONTEXT, goal, episodes: [item.entry] };
    const { answers } = await asker.ask(state, questionFor(item.episode));
    return 1 - noulAnswer(answers, `outcome_${item.episode.id}`);
  });
  const deciderMs = Date.now() - deciderStarted;

  const decisions: EpisodeDecision[] = sized.map(({ episode }, index) => {
    const keep = answered[index]!;
    return {
      id: episode.id,
      calls: episode.calls.length,
      chars: episode.chars,
      keep,
      action: keep >= resolved.keepThreshold ? 'keep' : 'replace',
    };
  });
  const condenseStarted = Date.now();
  const notes = new Map<string, string>();
  let condenseFailed = 0;
  const { condenser, archive } = options;
  if (condenser) {
    const chosen = decisions.filter((d) => d.action === 'keep' && d.chars >= resolved.condenseMinChars);
    await mapLimited(chosen, resolved.maxConcurrentCondense, async (decision) => {
      const episode = episodes.find((e) => e.id === decision.id)!;
      const input = condenseInput(messages, episode, goal);
      const callStarted = Date.now();
      let facts: string;
      try {
        facts = await condenser.condense(input);
      } catch (error) {
        decision.condenseMs = Date.now() - callStarted;
        decision.condenseError = error instanceof Error ? error.message : String(error);
        condenseFailed += 1;
        return;
      }
      decision.condenseMs = Date.now() - callStarted;
      // A note in the wrong shape or with identifiers the outputs never held is
      // made up (seen: a whole invented shell session); keep the episode then.
      facts = facts
        .split('\n')
        .filter((line) => line.startsWith('- '))
        .join('\n');
      const grounded = facts ? groundedShare(facts, condenseSource(input)) : 0;
      if (!facts || grounded < resolved.condenseMinGrounded) {
        decision.condenseError = facts
          ? `note rejected: grounded ${grounded.toFixed(2)} < ${resolved.condenseMinGrounded}`
          : 'note rejected: no "- " fact lines';
        condenseFailed += 1;
        return;
      }
      const archivedAt = archive
        ? await archive(episode, episodeText(episode)).catch(() => undefined)
        : undefined;
      notes.set(episode.id, condensedNote(episode, facts, resolved.truncateHeadChars, archivedAt));
      decision.action = 'condense';
    });
  }
  const condenseMs = condenser ? Date.now() - condenseStarted : 0;

  const replacedIds = new Set(decisions.filter((d) => d.action !== 'keep').map((d) => d.id));
  const replaced = episodes.filter((episode) => replacedIds.has(episode.id));
  const kept = applyEpisodes(messages, replaced, resolved.truncateHeadChars, notes);
  return {
    messages: kept,
    decisions,
    stats: {
      messagesBefore: messages.length,
      messagesAfter: kept.length,
      charsBefore,
      charsAfter: kept.reduce((sum, message) => sum + messageChars(message), 0),
      episodes: episodes.length,
      replaced: replaced.length - notes.size,
      kept: decisions.length - replaced.length + notes.size,
      condensed: notes.size,
      condenseFailed,
      condenseMs,
      deciderMs,
      skipped: episodes.length - decisions.length,
      callsRemoved: replaced.reduce((sum, episode) => sum + episode.calls.length, 0),
      stateTokens: Math.max(0, ...sized.map((item) => baseTokens + item.entryTokens)),
      requests: sized.length,
      ms: Date.now() - started,
    },
  };
}

/** The episode as a condenser sees it: the texts around it and every call with its full output. */
export function condenseInput(
  messages: readonly Message[],
  episode: Episode,
  goal: string,
): CondenseInput {
  const input: CondenseInput = {
    goal,
    after: messages[episode.end]?.text ?? '',
    calls: episode.calls.map((call) => ({
      tool: call.tool,
      input: call.input,
      isError: call.isError,
      output: call.result,
    })),
  };
  const before = messages[episode.start]?.text ?? '';
  if (before.trim()) input.before = before;
  return input;
}

/** Everything a condenser may quote from: the texts around the episode and its calls. */
function condenseSource(input: CondenseInput): string {
  return [
    input.before ?? '',
    input.after,
    ...input.calls.map((call) => `${JSON.stringify(call.input)}\n${call.output}`),
  ].join('\n');
}
