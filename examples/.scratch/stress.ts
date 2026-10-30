import { readFileSync } from 'node:fs';
import { compact, JevClient, type Message } from '../../src/index.js';
const lines = readFileSync(process.argv[2]!, 'utf8').split('\n').filter(Boolean);
const msgs: Message[] = [];
for (const l of lines) {
  let e: any; try { e = JSON.parse(l); } catch { continue; }
  if ((e.type !== 'user' && e.type !== 'assistant') || e.isSidechain) continue;
  const c = e.message?.content; const m: Message = { role: e.type, text: '', toolUses: [] };
  if (typeof c === 'string') m.text = c;
  else for (const b of c ?? []) {
    if (b.type === 'text') m.text += b.text;
    if (b.type === 'tool_use') m.toolUses.push({ tool_use_id: b.id, tool: b.name, input: b.input ?? {} });
    if (b.type === 'tool_result') (m.toolResults ??= []).push({ tool_use_id: b.tool_use_id, text: typeof b.content === 'string' ? b.content : JSON.stringify(b.content), isError: !!b.is_error });
  }
  if (m.text || m.toolUses.length || m.toolResults) msgs.push(m);
}
const slice = msgs.slice(0, Number(process.argv[3] ?? 300));
const calls = slice.reduce((n, m) => n + m.toolUses.length, 0);
console.log(`messages ${slice.length}, calls ${calls}`);
const asker = new JevClient({ baseUrl: 'http://192.168.107.103:5810/v1/systemone' });
for (const [st, rq, mq] of [[3000, 6000, 64], [6000, 9000, 64], [12000, 16000, 64], [6000, 9000, 32]] as const) {
  const t = Date.now();
  try {
    const r = await compact(slice, asker, { maxStateTokens: st, maxRequestTokens: rq, maxQuestionsPerRequest: mq });
    const s = r.stats;
    console.log(`state ${st}/${rq}/q${mq}: ${Date.now() - t} ms, req ${s.requests}, stage ${s.stateStage}, stateTok ${s.stateTokens}, kept ${s.kept} res- ${s.resultsDropped} call- ${s.callsDropped}, reduction ${((s.charsBefore - s.charsAfter) / s.charsBefore * 100).toFixed(0)}%`);
  } catch (e) { console.log(`state ${st}/${rq}/q${mq}: FAIL after ${Date.now() - t} ms: ${(e as Error).message.slice(0, 150)}`); }
}
