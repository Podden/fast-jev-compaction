import { describe, expect, it } from 'vitest';
import {
  applyEpisodes,
  buildJevRequest,
  collectEpisodes,
  collectToolCalls,
  compact,
  compactMessages,
  episodeEntry,
  estimateTokens,
  groundedShare,
  JevClient,
  parseJevResponse,
  reductionRatio,
  replacementNote,
  resolveOptions,
  sizeEpisode,
  type CondenseInput,
  type EpisodeState,
  type JevAsker,
  type JevQuestions,
  type Message,
} from '../src/index.js';

function message(role: Message['role'], text: string, extra: Partial<Message> = {}): Message {
  return { role, text, toolUses: [], ...extra };
}

function call(id: string, tool: string, input: Record<string, unknown>, text: string): Message {
  return message('assistant', '', { toolUses: [{ tool_use_id: id, tool, input, text }] });
}

function result(id: string, text: string, isError = false): Message {
  return message('user', '', { toolResults: [{ tool_use_id: id, text, isError }] });
}

const fileA = 'export const a = 1;\n'.repeat(50);
const fileB = 'export const b = 2;\n'.repeat(50);

function transcript(): Message[] {
  return [
    message('user', 'Never edit anything under src/generated. Fix the failing test.'),
    call('tool-1', 'Read', { file_path: 'src/a.ts' }, fileA),
    result('tool-1', fileA),
    message('assistant', 'a.ts looks fine; checking b.ts'),
    call('tool-2', 'Read', { file_path: 'src/b.ts' }, fileB),
    result('tool-2', fileB),
    call('tool-3', 'Bash', { command: 'npm test' }, 'FAIL b.test.ts'),
    result('tool-3', 'FAIL b.test.ts: expected 2 to be 3', true),
    message('assistant', 'The failure is in b.test.ts; fixing now.'),
    message('user', 'go ahead'),
  ];
}

type Seen = { state: unknown; questions: string[] };

function fakeJev(answer: (name: string) => number, seen: Seen[] = []): JevAsker {
  return {
    async ask(state, questions: JevQuestions) {
      seen.push({ state, questions: Object.keys(questions) });
      return {
        answers: Object.fromEntries(
          Object.keys(questions).map((key) => [key, { type: 'noul' as const, noul: answer(key) }]),
        ),
      };
    },
  };
}

describe('options', () => {
  it('fills in defaults and ignores non-finite values', () => {
    expect(resolveOptions()).toMatchObject({
      keepThreshold: 0.5,
      preserveRecentMessages: 6,
      maxStateTokens: 25_000,
      maxRequestTokens: 30_000,
      truncateHeadChars: 300,
    });
    expect(resolveOptions({
      keepThreshold: Number.NaN,
      preserveRecentMessages: 2.7,
      truncateHeadChars: -1.2,
    })).toMatchObject({
      keepThreshold: 0.5,
      preserveRecentMessages: 2,
      truncateHeadChars: 0,
    });
  });
});

describe('token estimate', () => {
  it('charges words, digits and symbols separately and never undercounts JSON badly', () => {
    expect(estimateTokens('')).toBe(0);
    expect(estimateTokens('hello world')).toBe(2);
    expect(estimateTokens('internationalization')).toBe(4);
    expect(estimateTokens('12345678')).toBe(4);
    const json = JSON.stringify({ file_path: '/Users/x/src/a.ts', old_string: 'a = 1;', n: 42 });
    expect(estimateTokens(json)).toBeGreaterThanOrEqual(Math.ceil(json.length / 3));
  });
});

describe('tool call collection', () => {
  it('pairs each tool call with its result and pins recent ones', () => {
    const calls = collectToolCalls(transcript(), 3);
    expect(calls.map((c) => [c.id, c.tool, c.callIndex, c.resultIndex, c.pinned])).toEqual([
      ['t1', 'Read', 1, 2, false],
      ['t2', 'Read', 4, 5, false],
      ['t3', 'Bash', 6, 7, true],
    ]);
    expect(calls[2]?.isError).toBe(true);
    expect(calls[0]?.resultChars).toBe(fileA.length);
  });

  it('ignores calls without a result', () => {
    expect(collectToolCalls([message('user', 'hi'), call('x', 'Read', {}, '')], 0)).toHaveLength(0);
  });
});

describe('episodes', () => {
  it('groups the calls between two texts and keeps only episodes an assistant text closes', () => {
    const messages = transcript();
    const episodes = collectEpisodes(messages, collectToolCalls(messages, 1), 1);
    expect(episodes.map((e) => [e.id, e.start, e.end, e.calls.map((c) => c.id)])).toEqual([
      ['e1', 1, 3, ['t1']],
      ['e2', 4, 8, ['t2', 't3']],
    ]);
    expect(episodes[0]?.chars).toBe(JSON.stringify({ file_path: 'src/a.ts' }).length + fileA.length);
  });

  it('leaves out episodes reaching into the pinned messages and runs cut short by a prompt', () => {
    const messages = transcript();
    expect(collectEpisodes(messages, collectToolCalls(messages, 3), 3).map((e) => e.id)).toEqual(['e1']);
    const interrupted = [
      message('user', 'look around'),
      call('x', 'Read', { file_path: 'a' }, fileA),
      result('x', fileA),
      message('user', 'stop, do something else'),
      message('assistant', 'ok'),
    ];
    expect(collectEpisodes(interrupted, collectToolCalls(interrupted, 0), 0)).toEqual([]);
  });

  it('shows Jev each call with the head of its output, and the texts around it', () => {
    const messages = transcript();
    const [, e2] = collectEpisodes(messages, collectToolCalls(messages, 1), 1);
    const entry = episodeEntry(messages, e2!, 20);
    expect(entry).toEqual({
      id: 'e2',
      calls: [
        `Read file_path=src/b.ts → ok, ${fileB.length} chars: export const b = 2;`,
        'Bash command=npm test → error, 34 chars: FAIL b.test.ts: expe',
      ],
      after: 'The failure is in b.test.ts; fixing now.',
    });
    expect(episodeEntry(messages, e2!, 0).calls[0]).toBe(`Read file_path=src/b.ts → ok, ${fileB.length} chars`);
  });

  it('replaces an episode by one line per call, failed calls with the head of their output', () => {
    const messages = transcript();
    const [, e2] = collectEpisodes(messages, collectToolCalls(messages, 1), 1);
    expect(replacementNote(e2!, 10)).toBe(
      [
        '[fast-jev-compaction replaced 2 tool call(s) here; their outcome is in the next message. Re-run a tool if the details are needed.',
        `- Read file_path=src/b.ts → ok, ${fileB.length} chars`,
        '- Bash command=npm test → error: FAIL b.tes]',
      ].join('\n'),
    );
    expect(replacementNote(e2!, 0)).toMatch(/- Bash command=npm test → error\]$/);
  });

  it('removes the calls of replaced episodes and keeps every text verbatim', () => {
    const messages = transcript();
    messages[4]!.text = 'Now b.ts.';
    const [, e2] = collectEpisodes(messages, collectToolCalls(messages, 1), 1);
    const kept = applyEpisodes(messages, [e2!], 300);
    expect(kept.map((m) => m.text.split('\n')[0])).toEqual([
      'Never edit anything under src/generated. Fix the failing test.',
      '',
      '',
      'a.ts looks fine; checking b.ts',
      'Now b.ts.',
      'The failure is in b.test.ts; fixing now.',
      'go ahead',
    ]);
    expect(kept.slice(0, 4)).toEqual(messages.slice(0, 4));
    expect(kept[0]).toBe(messages[0]);
    expect(kept[4]).toEqual({ role: 'assistant', text: `Now b.ts.\n\n${replacementNote(e2!, 300)}`, toolUses: [] });
    expect(kept[5]).toBe(messages[8]);
    expect(kept.flatMap((m) => m.toolResults ?? []).map((r) => r.tool_use_id)).toEqual(['tool-1']);
  });

  it('gives an episode as much output as fits one request, or nothing when it never fits', () => {
    const messages = transcript();
    const [, e2] = collectEpisodes(messages, collectToolCalls(messages, 1), 1);
    const roomy = sizeEpisode(messages, e2!, 100, { maxStateTokens: 25_000, maxRequestTokens: 30_000 });
    expect(roomy?.entry.calls[0]).toContain('export const b = 2;');
    const bare = estimateTokens(JSON.stringify(episodeEntry(messages, e2!, 0))) + 1;
    const tight = sizeEpisode(messages, e2!, 100, { maxStateTokens: 100 + bare, maxRequestTokens: 30_000 });
    expect(tight?.entry.calls[0]).toBe(`Read file_path=src/b.ts → ok, ${fileB.length} chars`);
    expect(sizeEpisode(messages, e2!, 100, { maxStateTokens: 100 + bare - 1, maxRequestTokens: 30_000 })).toBeUndefined();
  });
});

describe('compact', () => {
  it('asks one question per episode, one episode per request, and replaces what the next text reports', async () => {
    const seen: Seen[] = [];
    const messages = transcript();
    const output = await compact(
      messages,
      fakeJev((name) => (name === 'outcome_e1' ? 0.9 : 0.2), seen),
      { preserveRecentMessages: 1, goal: 'fix the test' },
    );

    expect(seen.map((r) => r.questions)).toEqual([['outcome_e1'], ['outcome_e2']]);
    expect(seen.map((r) => (r.state as EpisodeState).episodes.map((e) => e.id))).toEqual([['e1'], ['e2']]);
    expect((seen[0]!.state as EpisodeState).goal).toBe('fix the test');
    expect(output.decisions.map((d) => [d.id, d.action, d.keep.toFixed(1)])).toEqual([
      ['e1', 'replace', '0.1'],
      ['e2', 'keep', '0.8'],
    ]);
    expect(output.stats).toMatchObject({ episodes: 2, replaced: 1, kept: 1, skipped: 0, callsRemoved: 1, requests: 2 });
    expect(output.messages.map((m) => m.text.split('\n')[0])).toEqual([
      'Never edit anything under src/generated. Fix the failing test.',
      '[fast-jev-compaction replaced 1 tool call(s) here; their outcome is in the next message. Re-run a tool if the details are needed.',
      'a.ts looks fine; checking b.ts',
      '',
      '',
      '',
      '',
      'The failure is in b.test.ts; fixing now.',
      'go ahead',
    ]);
    expect(reductionRatio(output)).toBeGreaterThan(0);
  });

  it('keeps at most maxConcurrentRequests requests in flight', async () => {
    const run = async (maxConcurrentRequests: number) => {
      let inFlight = 0;
      let peak = 0;
      const inner = fakeJev(() => 0.1);
      const asker: JevAsker = {
        async ask(state, questions) {
          peak = Math.max(peak, ++inFlight);
          await new Promise((resolve) => setTimeout(resolve, 5));
          inFlight--;
          return inner.ask(state, questions);
        },
      };
      const output = await compact(transcript(), asker, { preserveRecentMessages: 1, maxConcurrentRequests });
      return { peak, output };
    };
    const serial = await run(1);
    expect(serial.peak).toBe(1);
    expect(serial.output.decisions.map((d) => d.action)).toEqual(['keep', 'keep']);
    expect((await run(0)).peak).toBe(2);
  });

  it('leaves small episodes and conversations without tool calls alone, without asking', async () => {
    const seen: Seen[] = [];
    const small = transcript().map((m) => ({
      ...m,
      toolResults: m.toolResults?.map((r) => ({ ...r, text: 'short' })),
    }));
    const output = await compact(small, fakeJev(() => 1, seen), { preserveRecentMessages: 1 });
    expect(seen).toHaveLength(0);
    expect(output.stats).toMatchObject({ episodes: 2, skipped: 2, requests: 0, stateTokens: 0 });
    expect(output.messages).toEqual(small);

    const chat = [message('user', 'hello'), message('assistant', 'hi')];
    expect((await compact(chat, fakeJev(() => 0, seen))).messages).toEqual(chat);
    expect(seen).toHaveLength(0);
  });

  it('rejects malformed answers', async () => {
    const broken: JevAsker = {
      ask: async () => ({ answers: { outcome_e9: { noul: 0.5 } } }),
    };
    await expect(compact(transcript(), broken, { preserveRecentMessages: 1 })).rejects.toThrow(
      /Invalid Jev answer/,
    );
  });
});

describe('HTTP client', () => {
  it('builds a System One request', () => {
    const request = buildJevRequest({ apiKey: 'k' }, { a: 1 }, {
      q: { type: 'noul', instructions: 'x' },
    });
    expect(request.url).toBe('https://api.typesafe.ai/v1/systemone');
    expect(request.headers.authorization).toBe('Bearer k');
    expect(JSON.parse(request.body)).toEqual({
      model: 'jev-latest',
      state: { a: 1 },
      questions: { q: { type: 'noul', instructions: 'x' } },
    });
  });

  it('rejects failed and malformed responses', () => {
    expect(() => parseJevResponse(500, false, 'boom')).toThrow(/500/);
    expect(() => parseJevResponse(200, true, 'not json')).toThrow(/malformed/);
    expect(() => parseJevResponse(200, true, '{}')).toThrow(/missing answers/);
    expect(parseJevResponse(200, true, '{"answers":{}}')).toEqual({ answers: {} });
  });

  it('asks over fetch and refuses to run without a key', async () => {
    const bodies: string[] = [];
    const client = new JevClient({
      apiKey: 'k',
      model: 'jev-test',
      fetch: (async (_url: string | URL | Request, init?: RequestInit) => {
        bodies.push(String(init?.body));
        return new Response(JSON.stringify({ answers: { q: { noul: 0.4 } } }), { status: 200 });
      }) as typeof fetch,
    });
    const response = await client.ask('state', { q: { type: 'noul', instructions: 'x' } });
    expect(response.answers.q).toEqual({ noul: 0.4 });
    expect(JSON.parse(bodies[0]!).model).toBe('jev-test');

    const custom: string[] = [];
    const selfHosted = new JevClient({
      apiKey: '',
      baseUrl: 'http://decider.local/v1/systemone',
      fetch: (async (url: string | URL | Request, init?: RequestInit) => {
        custom.push(String(url));
        expect(new Headers(init?.headers).has('authorization')).toBe(false);
        return new Response(JSON.stringify({ answers: {} }), { status: 200 });
      }) as typeof fetch,
    });
    await selfHosted.ask('s', {});
    expect(custom).toEqual(['http://decider.local/v1/systemone']);

    const keyless = new JevClient({ apiKey: '' });
    await expect(keyless.ask('s', {})).rejects.toThrow(/TYPESAFE_API_KEY/);
    await expect(
      compactMessages(transcript(), { apiKey: '', preserveRecentMessages: 1 }),
    ).rejects.toThrow(/TYPESAFE_API_KEY/);
  });
});

describe('condenser stage', () => {
  it('replaces kept episodes above condenseMinChars by the fact note and points to the archive', async () => {
    const inputs: CondenseInput[] = [];
    const archived: string[] = [];
    const output = await compact(transcript(), fakeJev(() => 0), {
      preserveRecentMessages: 1,
      condenseMinChars: 1050,
      condenser: {
        async condense(input) {
          inputs.push(input);
          return '- FAIL b.test.ts: expected 2 to be 3';
        },
      },
      archive: async (episode, text) => {
        archived.push(text);
        return `.claude/fast-jev/${episode.id}.txt`;
      },
    });
    expect(output.decisions.map((d) => d.action)).toEqual(['keep', 'condense']);
    expect(inputs).toHaveLength(1);
    expect(inputs[0]!.calls.map((c) => c.output)).toEqual([fileB, 'FAIL b.test.ts: expected 2 to be 3']);
    expect(inputs[0]!.after).toBe('The failure is in b.test.ts; fixing now.');
    expect(archived[0]).toContain(fileB);
    const note = output.messages.find((m) => m.text.includes('condensed 2 tool call(s)'));
    expect(note?.text).toContain('Full outputs: .claude/fast-jev/e2.txt');
    expect(note?.text).toContain('- FAIL b.test.ts: expected 2 to be 3');
    expect(output.messages.some((m) => m.toolUses.some((t) => t.tool_use_id === 'tool-1'))).toBe(true);
    expect(output.messages.some((m) => m.toolUses.some((t) => t.tool_use_id === 'tool-2'))).toBe(false);
    expect(output.stats).toMatchObject({ replaced: 0, kept: 2, condensed: 1, condenseFailed: 0 });
  });

  it('keeps an episode verbatim when the condenser fails', async () => {
    const output = await compact(transcript(), fakeJev(() => 0), {
      preserveRecentMessages: 1,
      condenseMinChars: 0,
      condenser: { condense: async () => Promise.reject(new Error('down')) },
    });
    expect(output.decisions.map((d) => d.action)).toEqual(['keep', 'keep']);
    expect(output.messages).toHaveLength(transcript().length);
    expect(output.stats).toMatchObject({ condensed: 0, condenseFailed: 2 });
  });
});

describe('condenser guard', () => {
  it('rejects notes that are not fact lines or name identifiers the outputs never held', async () => {
    const notes = [
      "I'll start by exploring the project.\n<call n=\"1\">",
      '- the failure is in src/parser_v2.ts:118 and tests/parser_v2.test.ts:40',
    ];
    const output = await compact(transcript(), fakeJev(() => 0), {
      preserveRecentMessages: 1,
      condenseMinChars: 0,
      maxConcurrentCondense: 1,
      condenser: { condense: async () => notes.shift()! },
    });
    expect(output.decisions.map((d) => d.action)).toEqual(['keep', 'keep']);
    expect(output.stats).toMatchObject({ condensed: 0, condenseFailed: 2 });
  });

  it('measures the share of identifiers found in the source', () => {
    expect(groundedShare('- read src/a.ts and src/b_9.ts', 'src/a.ts …')).toBe(0.5);
    expect(groundedShare('- nothing distinctive', '')).toBe(1);
  });
});
