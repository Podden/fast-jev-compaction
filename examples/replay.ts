/**
 * Replays real Claude Code sessions and measures, per compaction strategy,
 * how much of the history goes and how many later-used facts from tool
 * outputs stay. Up to CUTS points per session (one session per project) are
 * compacted; the history before a point is the newest messages that fit
 * WINDOW_CHARS (roughly what a context holds), plus the session's first one.
 *
 *   npx tsx examples/replay.ts <out-dir> [sessions=15]
 *
 * Env: SYSTEM_ONE_BASE_URL (decider), CONDENSE_URL, CONDENSE_MODEL, PROJECTS_DIR.
 *
 * A fact is a distinctive token (≥ 8 chars with a digit, `_`, `/` or `\`) from
 * a tool output before the boundary that the agent writes again (text or tool
 * input) within the next FUTURE_MESSAGES messages. Following #99, tokens the agent or the
 * user wrote before the boundary don't count (they stay in context as text),
 * and neither do tokens a later tool output brings back before the agent uses
 * them, nor tokens in the pinned newest messages.
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  collectToolCalls,
  compact,
  isPinned,
  JevClient,
  LlmCondenser,
  messageChars,
  type CompactOptions,
  type JevAsker,
  type Message,
} from '../src/index.js';

const outDir = process.argv[2] ?? 'replay-out';
const sessionCount = Number(process.argv[3] ?? 15);
const projectsDir = process.env.PROJECTS_DIR ?? join(process.env.USERPROFILE ?? process.env.HOME ?? '', '.claude', 'projects');
const deciderUrl = process.env.SYSTEM_ONE_BASE_URL ?? 'http://192.168.107.103:5810/v1/systemone';
const condenseUrl = process.env.CONDENSE_URL ?? 'http://192.168.107.103:11500/v1/chat/completions';
const condenseModel = process.env.CONDENSE_MODEL ?? 'qwen3.8:27b';
const FUTURE_MESSAGES = 150;
const WINDOW_CHARS = 500_000;
const CUTS = 3;
const PRESERVE = 6;

const BASE: CompactOptions = {
  preserveRecentMessages: PRESERVE,
  keepThreshold: 0.6,
  maxStateTokens: 24_000,
  maxConcurrentRequests: 1,
};

type Entry = Record<string, any>;

function blockText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .filter((b) => b?.type === 'text' && typeof b.text === 'string')
    .map((b) => b.text)
    .join('\n');
}

/** All main-thread messages of a session, compactions ignored. */
export function parseSession(file: string): Message[] {
  const messages: Message[] = [];
  for (const line of readFileSync(file, 'utf8').split('\n')) {
    if (!line) continue;
    let e: Entry;
    try {
      e = JSON.parse(line);
    } catch {
      continue;
    }
    if ((e.type !== 'user' && e.type !== 'assistant') || e.isSidechain || e.isCompactSummary) continue;
    const content = e.message?.content;
    const m: Message = { role: e.type, text: '', toolUses: [] };
    if (typeof content === 'string') m.text = content;
    else if (Array.isArray(content)) {
      for (const b of content) {
        if (b?.type === 'text') m.text += b.text;
        else if (b?.type === 'tool_use') m.toolUses.push({ tool_use_id: b.id, tool: b.name, input: b.input ?? {} });
        else if (b?.type === 'tool_result') {
          (m.toolResults ??= []).push({ tool_use_id: b.tool_use_id, text: blockText(b.content), isError: !!b.is_error });
        }
      }
    }
    if (m.text || m.toolUses.length || m.toolResults) messages.push(m);
  }
  return messages;
}

/**
 * Compaction points: the history before each (first message plus the newest
 * that fit WINDOW_CHARS, starting at a user prompt) and the messages after.
 */
export function cutPoints(all: Message[]): { at: number; prefix: Message[]; future: Message[] }[] {
  const candidates: number[] = [];
  for (let at = 200; at + FUTURE_MESSAGES <= all.length; at += 100) candidates.push(at);
  const step = Math.max(1, Math.ceil(candidates.length / CUTS));
  return candidates
    .filter((_, i) => i % step === step - 1 || candidates.length <= CUTS)
    .slice(0, CUTS)
    .map((at) => {
      let start = at;
      let size = 0;
      while (start > 1 && size + messageChars(all[start - 1]!) <= WINDOW_CHARS) size += messageChars(all[--start]!);
      while (start < at && !(all[start]!.role === 'user' && !all[start]!.toolResults)) start++;
      return { at, prefix: [all[0]!, ...all.slice(start, at)], future: all.slice(at, at + FUTURE_MESSAGES) };
    })
    .filter(({ prefix }) => prefix.length > 50);
}

const TOKEN = /[A-Za-z0-9_~.][A-Za-z0-9_.\/\\:#@+=-]{6,}[A-Za-z0-9_]/g;

function tokens(text: string): Set<string> {
  const out = new Set<string>();
  for (const [t] of text.matchAll(TOKEN)) if (/[\d_\/\\]/.test(t) && !/^https?:\/\/$/.test(t)) out.add(t);
  return out;
}

function inputText(m: Message): string {
  return m.toolUses.map((t) => JSON.stringify(t.input)).join('\n');
}

export function facts(prefix: Message[], future: Message[]): Set<string> {
  const wrote = new Set<string>();
  const pinnedResults = new Set<string>();
  const candidates = new Set<string>();
  prefix.forEach((m, i) => {
    for (const t of tokens(`${m.text}\n${inputText(m)}`)) wrote.add(t);
    for (const r of m.toolResults ?? []) {
      const target = isPinned(i, prefix.length, PRESERVE) ? pinnedResults : candidates;
      for (const t of tokens(r.text)) target.add(t);
    }
  });
  const used = new Set<string>();
  const recarried = new Set<string>();
  for (const m of future) {
    for (const r of m.toolResults ?? []) for (const t of tokens(r.text)) if (!used.has(t)) recarried.add(t);
    if (m.role === 'assistant') {
      for (const t of tokens(`${m.text}\n${inputText(m)}`)) if (!recarried.has(t)) used.add(t);
    }
  }
  return new Set([...candidates].filter((t) => used.has(t) && !wrote.has(t) && !pinnedResults.has(t)));
}

export function serialize(messages: readonly Message[]): string {
  return messages
    .map((m) => [m.text, inputText(m), ...(m.toolResults ?? []).map((r) => r.text)].join('\n'))
    .join('\n');
}

export function chars(messages: readonly Message[]): number {
  return messages.reduce((sum, m) => sum + messageChars(m), 0);
}

/** Every unpinned tool output cut to `cap` characters, 70 % head and 30 % tail. */
function headTail(messages: readonly Message[], cap: number): Message[] {
  return messages.map((m, i) => {
    if (!m.toolResults || isPinned(i, messages.length, PRESERVE)) return m;
    return {
      ...m,
      toolResults: m.toolResults.map((r) =>
        r.text.length <= cap
          ? r
          : { ...r, text: `${r.text.slice(0, Math.floor(cap * 0.7))}\n[…]\n${r.text.slice(r.text.length - Math.ceil(cap * 0.3))}` },
      ),
    };
  });
}

/** head+tail with the per-output cap that lands closest to `targetChars`. */
export function headTailMatched(messages: readonly Message[], targetChars: number): Message[] {
  let lo = 0;
  let hi = 2_000_000;
  while (hi - lo > 20) {
    const mid = Math.floor((lo + hi) / 2);
    if (chars(headTail(messages, mid)) > targetChars) hi = mid;
    else lo = mid;
  }
  return headTail(messages, lo);
}

function cached(asker: JevAsker): JevAsker {
  const memo = new Map<string, ReturnType<JevAsker['ask']>>();
  return {
    ask(state, questions) {
      const key = JSON.stringify([state, questions]);
      if (!memo.has(key)) memo.set(key, asker.ask(state, questions));
      return memo.get(key)!;
    },
  };
}

const always = (noul: number): JevAsker => ({
  async ask(_state, questions) {
    return { answers: Object.fromEntries(Object.keys(questions).map((k) => [k, { type: 'noul' as const, noul }])) };
  },
});

export function pickSessions(): { project: string; file: string }[] {
  const rows: { project: string; file: string; toolUses: number }[] = [];
  for (const project of readdirSync(projectsDir)) {
    const dir = join(projectsDir, project);
    if (!statSync(dir).isDirectory()) continue;
    for (const name of readdirSync(dir)) {
      if (!name.endsWith('.jsonl')) continue;
      const file = join(dir, name);
      const size = statSync(file).size;
      if (size < 300_000 || size > 40_000_000) continue;
      const text = readFileSync(file, 'utf8');
      rows.push({ project, file, toolUses: text.split('"type":"tool_use"').length - 1 });
    }
  }
  rows.sort((a, b) => b.toolUses - a.toolUses);
  const seen = new Set<string>();
  return rows.filter((r) => !seen.has(r.project) && seen.add(r.project)).slice(0, sessionCount);
}

type Row = Record<string, string | number>;

async function main() {
  mkdirSync(outDir, { recursive: true });
  const decider = cached(new JevClient({ baseUrl: deciderUrl }));
  const condenser = new LlmCondenser({ url: condenseUrl, model: condenseModel });
  // A rerun into the same folder resumes: finished points are skipped.
  const resultsFile = join(outDir, 'results.json');
  const previous = existsSync(resultsFile) ? JSON.parse(readFileSync(resultsFile, 'utf8')) : {};
  const rows: Row[] = previous.rows ?? [];
  const samples: { session: string; facts: string; outputChars: number; grounded: number }[] =
    previous.samples ?? [];
  const done = new Set(rows.map((r) => r.session));
  for (const { project, file } of pickSessions()) for (const { at, prefix, future } of cutPoints(parseSession(file))) {
    const session = `${project.replace(/^C--Users-Daniel-Code-/i, '')}/${file.slice(-42, -34)}@${at}`;
    if (done.has(session)) continue;
    const wanted = facts(prefix, future);
    if (wanted.size === 0) continue;
    const before = chars(prefix);
    const kept = (messages: readonly Message[], extra = '') => {
      const text = serialize(messages) + extra;
      return [...wanted].filter((t) => text.includes(t)).length;
    };
    const row = (strategy: string, messages: readonly Message[], ms: number, extra: Partial<Row> = {}) => {
      rows.push({
        session,
        strategy,
        charsBefore: before,
        reduction: +(1 - chars(messages) / before).toFixed(3),
        facts: wanted.size,
        retained: kept(messages),
        ms,
        ...extra,
      });
    };

    const dropAll = await compact(prefix, always(1), BASE);
    row('drop-all', dropAll.messages, dropAll.stats.ms);

    const episodes = await compact(prefix, decider, BASE);
    row('episodes', episodes.messages, episodes.stats.ms, {
      replaced: episodes.stats.replaced,
      kept: episodes.stats.kept,
    });

    const matched = headTailMatched(prefix, chars(episodes.messages));
    row('head+tail', matched, 0);

    const archives = new Map<string, string>();
    const archiveDir = join(outDir, session.replace(/[/@]/g, '_'));
    mkdirSync(archiveDir, { recursive: true });
    const combo = await compact(prefix, decider, {
      condenseMinChars: Number(process.env.CONDENSE_MIN_CHARS ?? 4000),
      maxConcurrentCondense: Number(process.env.CONDENSE_CONCURRENCY ?? 4),
      ...BASE,
      condenser: {
        async condense(input) {
          const note = await condenser.condense(input);
          const source = tokens(input.calls.map((c) => `${JSON.stringify(c.input)}\n${c.output}`).join('\n') + input.after + (input.before ?? ''));
          const own = [...tokens(note)];
          samples.push({
            session,
            facts: note,
            outputChars: input.calls.reduce((s, c) => s + c.output.length, 0),
            grounded: own.length === 0 ? 1 : own.filter((t) => source.has(t)).length / own.length,
          });
          return note;
        },
      },
      archive: async (episode, text) => {
        const path = join(archiveDir, `${episode.id}.txt`);
        writeFileSync(path, text);
        archives.set(path, text);
        return path;
      },
    });
    row(process.env.COMBO_LABEL ?? 'episodes+condense', combo.messages, combo.stats.condenseMs, {
      condensed: combo.stats.condensed,
      condenseFailed: combo.stats.condenseFailed,
      retainedWithArchive: kept(combo.messages, [...archives.values()].join('\n')),
    });
    writeFileSync(join(outDir, 'results.json'), JSON.stringify({ rows, samples }, null, 2));
    console.log(rows.slice(-4).map((r) => `${r.session} ${r.strategy}: -${Math.round(+r.reduction * 100)}% facts ${r.retained}/${r.facts}`).join('\n'));
  }
  writeFileSync(join(outDir, 'results.json'), JSON.stringify({ rows, samples }, null, 2));
}

// Run only as a script, so the helpers can be imported for follow-up analyses.
if (process.argv[1]?.replace(/\\/g, '/').endsWith('examples/replay.ts')) await main();
