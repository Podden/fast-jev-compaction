import { describe, expect, it } from 'vitest';
import {
  compactSession,
  decisionLog,
  decisionLogLines,
  register,
  resolveHookConfig,
  summarize,
  toSessionMessages,
} from '../hooks/fast-jev.ts';
import { applyEpisodes, collectEpisodes, collectToolCalls, replacementNote, type Message } from '../src/index.js';

type SessionMessage = Message & { handle?: string };

function message(role: Message['role'], text: string, extra: Partial<SessionMessage> = {}): SessionMessage {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): SessionMessage {
  return message('assistant', '', {
    toolUses: [{ tool_use_id: id, tool, input, text }],
    handle: `h-${id}`,
  });
}

function result(id: string, text: string, isError = false): SessionMessage {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }], handle: `r-${id}` });
}

const fileA = 'export const a = 1;\n'.repeat(50);
const failure = `FAIL b.test.ts: expected 2 to be 3\n${'    at test (b.test.ts:3:5)\n'.repeat(40)}`;

function transcript(): SessionMessage[] {
  return [
    message('user', 'Fix the failing test.', { handle: 'h-0' }),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    message('assistant', 'a.ts is fine.', { handle: 'h-3' }),
    call('tool-2', 'Bash', { command: 'npm test' }, failure),
    result('tool-2', failure, true),
    message('assistant', 'b.test.ts expects 3; fixing now.', { handle: 'h-6' }),
    message('user', 'go ahead', { handle: 'h-7' }),
  ];
}

function jevFetch(answer: (name: string) => number, bodies: string[] = []) {
  return async (_url: string, init?: { body?: string }) => {
    bodies.push(init?.body ?? '');
    const { questions } = JSON.parse(init?.body ?? '{}') as { questions: Record<string, unknown> };
    const answers = Object.fromEntries(
      Object.keys(questions).map((key) => [key, { type: 'noul', noul: answer(key) }]),
    );
    return { status: 200, ok: true, text: JSON.stringify({ answers }) };
  };
}

describe('hook config', () => {
  it('reads userConfig values and falls back to defaults', () => {
    expect(resolveHookConfig({})).toEqual({
      compactAtPercent: 60,
      minReductionRatio: 0.25,
      model: 'jev-latest',
      condenseModel: 'qwen3.8:27b',
    });
    expect(
      resolveHookConfig({ apiKey: 'k', keepThreshold: 0.3, maxStateTokens: 1000, model: 'jev-x', goal: 'g', compactAtPercent: 'no' }),
    ).toEqual({
      apiKey: 'k',
      keepThreshold: 0.3,
      maxStateTokens: 1000,
      model: 'jev-x',
      goal: 'g',
      compactAtPercent: 60,
      minReductionRatio: 0.25,
      condenseModel: 'qwen3.8:27b',
    });
  });
});

describe('session message mapping', () => {
  it('returns the engine objects for untouched messages and handle-less copies for rebuilt ones', () => {
    const messages = transcript();
    const [e1] = collectEpisodes(messages, collectToolCalls(messages, 1), 1);
    const out = toSessionMessages(messages, applyEpisodes(messages, [e1!], 300));
    expect(out).toHaveLength(messages.length - 1);
    expect(out[0]).toBe(messages[0]);
    expect(out[1]).toEqual({ role: 'assistant', text: replacementNote(e1!, 300), toolUses: [] });
    expect(out[1]?.handle).toBeUndefined();
    out.slice(2).forEach((m, i) => expect(m).toBe(messages[i + 3]));
  });
});

describe('compactSession', () => {
  it('runs the library over the engine fetch and reports the outcome', async () => {
    const bodies: string[] = [];
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k', model: 'jev-x' };
    const { result: output, messages } = await compactSession(
      transcript(),
      config,
      jevFetch((name) => (name === 'outcome_e1' ? 0.9 : 0.1), bodies),
    );
    expect(bodies).toHaveLength(2);
    expect(JSON.parse(bodies[0]!).model).toBe('jev-x');
    expect(output.decisions.map((d) => d.action)).toEqual(['replace', 'keep']);
    expect(messages.map((m) => m.handle)).toEqual(['h-0', undefined, 'h-3', 'h-tool-2', 'r-tool-2', 'h-6', 'h-7']);
    expect(summarize(output)).toMatch(
      /^\d+% reduction; 1 of 2 episodes replaced \(1 calls\); state up to ~\d+ tokens in 2 request\(s\)$/,
    );
    const [e1, e2] = output.decisions;
    expect(decisionLog(output)).toBe(
      `e1:1calls/${e1!.chars}ch:replace/keep=0.10 e2:1calls/${e2!.chars}ch:keep/keep=0.90`,
    );
    expect(decisionLogLines(output)).toEqual([`decisions: ${decisionLog(output)}`]);
  });

  it('splits a long decision log into ui.log lines under the host limit', async () => {
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k' };
    const { result: output } = await compactSession(transcript(), config, jevFetch(() => 0.9));
    const [first, second] = decisionLog(output).split(' ');
    const lines = decisionLogLines(output, 60);
    expect(lines).toEqual([`decisions (1/2): ${first}`, `decisions (2/2): ${second}`]);
    expect(lines.every((line) => line.length <= 60)).toBe(true);
    expect(decisionLogLines({ ...output, decisions: [] })).toEqual(['decisions: (none)']);
  });

  it('cuts output heads without leaving a lone surrogate in the request (#128)', async () => {
    const bodies: string[] = [];
    const emoji = `${'x'.repeat(239)}🔥${'y'.repeat(1200)}`;
    const messages = [
      message('user', 'go'),
      call('tool-1', 'Bash', { command: 'cat' }, emoji),
      result('tool-1', emoji),
      message('assistant', 'done'),
      message('user', 'next'),
    ];
    const config = { ...resolveHookConfig({ preserveRecentMessages: 1 }), apiKey: 'k' };
    await compactSession(messages, config, jevFetch(() => 0.9, bodies));
    expect(bodies).toHaveLength(1);
    expect(bodies[0]).not.toMatch(/\\ud[89a-f][0-9a-f]{2}/i);
  });

  it('condenses kept episodes over the engine fetch and archives their outputs', async () => {
    const urls: string[] = [];
    const prompts: string[] = [];
    const decider = jevFetch(() => 0);
    const fetchFn = async (url: string, init?: { body?: string }) => {
      urls.push(url);
      if (!url.includes('chat/completions')) return decider(url, init);
      const body = JSON.parse(init?.body ?? '{}') as { model: string; messages: { content: string }[] };
      expect(body.model).toBe('qwen3.8:27b');
      prompts.push(body.messages[1]!.content);
      return { status: 200, ok: true, text: JSON.stringify({ choices: [{ message: { content: '- FAIL b.test.ts: expected 2 to be 3' } }] }) };
    };
    const config = {
      ...resolveHookConfig({ preserveRecentMessages: 1, condenseUrl: 'http://llm.local/v1/chat/completions', condenseMinChars: 0 }),
      apiKey: 'k',
    };
    const files: string[] = [];
    const { result: output, messages } = await compactSession(transcript(), config, fetchFn, async (episode) => {
      files.push(episode.id);
      return `.claude/fast-jev/x/${episode.id}.txt`;
    });
    expect(output.stats.condensed).toBe(2);
    expect(files).toEqual(['e1', 'e2']);
    expect(urls.filter((u) => u.includes('chat/completions'))).toHaveLength(2);
    expect(prompts[1]).toContain('FAIL b.test.ts');
    expect(messages.some((m) => m.text.includes('Full outputs: .claude/fast-jev/x/e2.txt'))).toBe(true);
    expect(summarize(output)).toContain('2 condensed in');
  });

  it('throws on a missing key and on failed requests so the hook falls back', async () => {
    const config = resolveHookConfig({ preserveRecentMessages: 1 });
    await expect(compactSession(transcript(), config, jevFetch(() => 0))).rejects.toThrow(/TYPESAFE_API_KEY/);
    const selfHosted = resolveHookConfig({ preserveRecentMessages: 1, baseUrl: 'http://decider.local/v1/systemone' });
    await expect(compactSession(transcript(), selfHosted, jevFetch(() => 0.9))).resolves.toBeDefined();
    await expect(
      compactSession(transcript(), { ...config, apiKey: 'k' }, async () => ({ status: 500, ok: false, text: 'x' })),
    ).rejects.toThrow(/500/);
  });
});

type Handler = (...args: any[]) => Promise<unknown>;

/** Registers the hook against a fake engine and returns its handlers and recorded effects. */
function engine(percent = 80) {
  const handlers: Record<string, Handler> = {};
  register(((name: string, handler: Handler) => {
    handlers[name] = handler;
  }) as never, { preserveRecentMessages: 1, baseUrl: 'http://decider.local/v1/systemone' });
  const deferred: (() => void)[] = [];
  const effects = { compacts: 0, logs: [] as string[] };
  let compactResult: Promise<unknown> = Promise.resolve({});
  const $ = {
    http: { fetch: async (url: string, init: { body?: string }) => jevFetch(() => 0.9)(url, init) },
    env: { get: async () => undefined },
    settings: { read: async () => ({}) },
    ui: { log: (text: string) => effects.logs.push(text), toast: () => {} },
    session: {
      usage: async () => ({ context: { percent } }),
      compact: () => {
        effects.compacts += 1;
        return compactResult;
      },
    },
    clock: { after: (_ms: number, fn: () => void) => deferred.push(fn) },
  };
  const next = async (event: unknown) => ({ passed: event });
  return {
    $,
    next,
    handlers,
    deferred,
    effects,
    failCompact: (error: Error) => {
      compactResult = Promise.reject(error);
    },
  };
}

describe('register', () => {
  it('returns kept messages without engine handles so --resume stays compacted (#89)', async () => {
    const { $, next, handlers } = engine();
    const out = (await handlers['session.compact']!($, { trigger: 'manual', messages: transcript() }, next)) as {
      messages: SessionMessage[];
    };
    expect(out.messages.length).toBeGreaterThan(0);
    expect(out.messages.every((m) => m.handle === undefined)).toBe(true);
    expect(out.messages[0]?.text).toBe('Fix the failing test.');
  });

  it('leaves subagent compactions to the engine and skips precompute (#107)', async () => {
    const { $, next, handlers } = engine();
    const sub = { trigger: 'auto', agentId: 'a1', messages: transcript() };
    expect(await handlers['session.compact']!($, sub, next)).toEqual({ passed: sub });
    expect(await handlers['session.compact']!($, { trigger: 'precompute', messages: transcript() }, next)).toEqual({
      skip: expect.any(String),
    });
  });

  it('defers auto-compaction past the turn and guards against a second request (#149)', async () => {
    const { $, next, handlers, deferred, effects } = engine();
    await handlers['turn.complete']!($, { reason: 'answer' }, next);
    await handlers['turn.complete']!($, { reason: 'answer' }, next);
    expect(effects.compacts).toBe(0);
    expect(deferred).toHaveLength(1);
    deferred[0]!();
    expect(effects.compacts).toBe(1);
    await new Promise((resolve) => setTimeout(resolve, 0));
    await handlers['turn.complete']!($, { reason: 'answer' }, next);
    expect(deferred).toHaveLength(2);
  });

  it('ignores subagent and non-answer turns and releases the guard after a failure', async () => {
    const { $, next, handlers, deferred, effects, failCompact } = engine();
    await handlers['turn.complete']!($, { reason: 'answer', agentId: 'a1' }, next);
    await handlers['turn.complete']!($, { reason: 'aborted' }, next);
    expect(deferred).toHaveLength(0);
    failCompact(new Error('a turn is in flight'));
    await handlers['turn.complete']!($, { reason: 'answer' }, next);
    deferred[0]!();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(effects.logs).toContain('auto-compact skipped (a turn is in flight)');
    await handlers['turn.complete']!($, { reason: 'answer' }, next);
    expect(deferred).toHaveLength(2);
  });

  it('does nothing below compactAtPercent', async () => {
    const { $, next, handlers, deferred } = engine(10);
    await handlers['turn.complete']!($, { reason: 'answer' }, next);
    await handlers['turn.complete']!($, { reason: 'answer' }, next);
    expect(deferred).toHaveLength(0);
  });
});
