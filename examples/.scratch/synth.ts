import { readFileSync, readdirSync } from 'node:fs';
import { compact, JevClient, type Message } from '../../src/index.js';
const files = ['src', 'hooks', 'tests'].flatMap((d) => readdirSync(d).filter((f) => f.endsWith('.ts')).map((f) => `${d}/${f}`));
const msgs: Message[] = [{ role: 'user', text: 'Add support for self-hosted System One endpoints to the compaction library and keep all tests green.', toolUses: [] }];
let n = 0;
for (let round = 0; round < 4; round++) for (const f of files) {
  const id = `toolu_${++n}`; const text = readFileSync(f, 'utf8').slice(0, 3000);
  msgs.push({ role: 'assistant', text: `Round ${round}: checking ${f} for the endpoint handling and related tests.`, toolUses: [{ tool_use_id: id, tool: 'Write', input: { file_path: f, content: text } }] });
  msgs.push({ role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: id, text: 'File written.' }] });
}
msgs.push({ role: 'user', text: 'Now update the README with the new option.', toolUses: [] });
console.log(`messages ${msgs.length}, calls ${n}`);
const asker = new JevClient({ baseUrl: 'http://192.168.107.103:5810/v1/systemone' });
for (const [st, rq, mq] of [[12000, 30000, 64], [25000, 30000, 64]] as const) {
  const t = Date.now();
  try {
    const s = (await compact(msgs, asker, { maxStateTokens: st, maxRequestTokens: rq, maxQuestionsPerRequest: mq })).stats;
    console.log(`state ${st}/${rq}/q${mq}: ${Date.now() - t} ms, req ${s.requests}, stage ${s.stateStage}, stateTok ${s.stateTokens}, kept ${s.kept} res- ${s.resultsDropped} call- ${s.callsDropped}, reduction ${((s.charsBefore - s.charsAfter) / s.charsBefore * 100).toFixed(0)}%`);
  } catch (e) { console.log(`state ${st}/${rq}/q${mq}: FAIL after ${Date.now() - t} ms: ${(e as Error).message.slice(0, 150)}`); }
}
