import { pickSessions, parseSession, tokens, inputText } from './replay-lib.js';
import { isPinned } from '../../src/index.js';
for (const { project, file } of pickSessions().slice(0, 6)) {
  const p = parseSession(file); if (!p) { console.log(project, 'no parse'); continue; }
  const { prefix, future } = p;
  const wrote = new Set<string>(), cand = new Set<string>();
  prefix.forEach((m, i) => { for (const t of tokens(`${m.text}\n${inputText(m)}`)) wrote.add(t);
    if (!isPinned(i, prefix.length, 6)) for (const r of m.toolResults ?? []) for (const t of tokens(r.text)) cand.add(t); });
  const usedAll = new Set<string>(); const recar = new Set<string>(); const usedStrict = new Set<string>();
  for (const m of future) { for (const r of m.toolResults ?? []) for (const t of tokens(r.text)) if (!usedStrict.has(t)) recar.add(t);
    if (m.role === 'assistant') for (const t of tokens(`${m.text}\n${inputText(m)}`)) { usedAll.add(t); if (!recar.has(t)) usedStrict.add(t); } }
  const c = [...cand];
  console.log(project.slice(21), 'prefix', prefix.length, 'future', future.length, 'futureAsst', future.filter(m=>m.role==='assistant').length,
    '| cand', c.length, 'usedAny', c.filter(t => usedAll.has(t)).length, 'usedNotWrote', c.filter(t => usedAll.has(t) && !wrote.has(t)).length,
    'strict', c.filter(t => usedStrict.has(t) && !wrote.has(t)).length);
}
