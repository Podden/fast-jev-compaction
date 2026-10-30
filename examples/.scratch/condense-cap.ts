// Replays the hook on a real session cut right before a logged plugin run, with
// the host's 30 s per-fetch cap emulated, then once without a cap.
// npx tsx examples/.scratch/condense-cap.ts <session.jsonl> <outcome-line-substring>
import { readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  compactSession,
  condenseErrorLines,
  decisionLog,
  resolveHookConfig,
  summarize,
  type HookFetch,
} from '../../hooks/fast-jev.js';
import { parseSession } from './replay-lib.js';

const [file, marker] = process.argv.slice(2);
const lines = readFileSync(file!, 'utf8').split('\n');
const cut = lines.findIndex((line) => line.includes(marker!));
if (cut < 0) throw new Error('marker not found');
const prefixFile = join(tmpdir(), 'condense-cap-prefix.jsonl');
writeFileSync(prefixFile, lines.slice(0, cut).join('\n'));
const messages = parseSession(prefixFile);

const config = {
  ...resolveHookConfig({
    baseUrl: 'http://192.168.107.103:5810/v1/systemone',
    maxStateTokens: 24000,
    keepThreshold: 0.6,
    maxConcurrentRequests: 1,
    condenseUrl: 'http://192.168.107.103:11500/v1/chat/completions',
    condenseMinChars: 12000,
    maxConcurrentCondense: Number(process.env.CONCURRENCY ?? 12),
  }),
};

const usage: string[] = [];
function capped(capMs: number): HookFetch {
  return async (url, init) => {
    let body = init?.body;
    if (process.env.MAX_TOKENS && url.includes('chat/completions') && body) {
      body = JSON.stringify({ ...JSON.parse(body), max_tokens: Number(process.env.MAX_TOKENS) });
    }
    try {
      const started = Date.now();
      const response = await fetch(url, { ...init, body, signal: AbortSignal.timeout(capMs) });
      const text = await response.text();
      if (url.includes('chat/completions')) {
        const parsed = JSON.parse(text) as {
          usage?: { prompt_tokens: number; completion_tokens: number };
          choices?: { finish_reason: string; message: { content: string } }[];
        };
        const u = parsed.usage;
        if (process.env.DUMP) {
          const choice = parsed.choices?.[0];
          const content = choice?.message.content ?? '';
          const lines = content.split('\n');
          console.log(
            `-- finish=${choice?.finish_reason} lines=${lines.length} distinct=${new Set(lines).size}\n${content.slice(0, 500)}\n[…]\n${content.slice(-300)}`,
          );
        }
        usage.push(`${u?.prompt_tokens}in/${u?.completion_tokens}out/${((Date.now() - started) / 1000).toFixed(1)}s`);
      }
      return { status: response.status, ok: response.ok, text };
    } catch (error) {
      if ((error as Error).name === 'TimeoutError') {
        throw new Error(`$.http.fetch(${url}) aborted: no complete answer within ${capMs}ms`);
      }
      throw error;
    }
  };
}

console.log(`${messages.length} messages, concurrency ${config.maxConcurrentCondense}`);
for (const capMs of (process.env.CAPS ?? '30000,600000').split(',').map(Number)) {
  const { result } = await compactSession(messages, config, capped(capMs));
  console.log(`\n== cap ${capMs / 1000} s: ${summarize(result)}`);
  for (const line of condenseErrorLines(result)) console.log(line);
  if (usage.length) console.log('usage:', usage.splice(0).join(' '));
  console.log(
    decisionLog(result)
      .split(' ')
      .filter((entry) => entry.includes('condense='))
      .join('\n'),
  );
}
