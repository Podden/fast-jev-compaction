import { readFileSync, readdirSync } from 'node:fs';
import { compact, JevClient, type JevAsker, type Message } from '../../src/index.js';
const files = ['src', 'hooks', 'tests'].flatMap((d) => readdirSync(d).filter((f) => f.endsWith('.ts')).map((f) => `${d}/${f}`));
const msgs: Message[] = [{ role: 'user', text: 'Add support for self-hosted System One endpoints and keep all tests green.', toolUses: [] }];
let n = 0;
for (let round = 0; round < 8; round++) for (const f of files) {
  const id = `toolu_${++n}`; const text = readFileSync(f, 'utf8').slice(0, 3000);
  msgs.push({ role: 'assistant', text: `Round ${round}: checking ${f}.`, toolUses: [{ tool_use_id: id, tool: 'Write', input: { file_path: f, content: text } }] });
  msgs.push({ role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: id, text: 'File written.' }] });
}
msgs.push({ role: 'user', text: 'Now update the README.', toolUses: [] });
const client = new JevClient({ baseUrl: 'http://192.168.107.103:5810/v1/systemone' });
for (const c of [1, 0]) {
  const waits: number[] = [];
  const asker: JevAsker = { async ask(s, q) { const t = Date.now(); try { return await client.ask(s, q); } finally { waits.push(Date.now() - t); } } };
  const t = Date.now();
  const st = (await compact(msgs, asker, { maxStateTokens: 12000, maxQuestionsPerRequest: 64, maxConcurrentRequests: c })).stats;
  console.log(`concurrency ${c}: calls ${n}, requests ${st.requests}, total ${Date.now() - t} ms, slowest single request ${Math.max(...waits)} ms`);
}
