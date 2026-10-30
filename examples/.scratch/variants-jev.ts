import { compact, type ToolCall } from '../../src/index.js';
import { messages } from './msgs.js';
const URL = 'https://openrouter.ai/api/alpha/decisions';
let state: unknown; let calls: { id: string; tool: string; chars: number }[] = [];
await compact(messages, { async ask(s, q) { state = s; calls = Object.keys(q).filter(k => k.startsWith('call_')).map(k => ({ id: k.slice(5), tool: '', chars: 0 })); return { answers: Object.fromEntries(Object.keys(q).map(k => [k, { noul: 1 }])) }; } }, { preserveRecentMessages: 2 });
const hist = (state as any).history as any[];
const meta = new Map<string, { tool: string; result: string }>();
for (const h of hist) for (const c of h.tool_calls ?? []) if (typeof c === 'object') meta.set(c.id, c);
type V = (id: string, tool: string, res: string) => Record<string, any>;
const variants: Record<string, V> = {
  A_current: (id, tool, res) => ({
    [`call_${id}`]: { type: 'noul', instructions: `Tool call ${id} (${tool}) should stay in the history: knowing this call was made, with its input, still matters for what the assistant does next` },
    [`result_${id}`]: { type: 'noul', instructions: `The full output of tool call ${id} (${tool}, ${res}) should stay in the history verbatim: the assistant still needs its contents and re-running the tool would not do` } }),
  B_criteria: (id, tool, res) => ({
    [`call_${id}`]: { type: 'noul', instructions: `Tool call ${id} (${tool}) should stay in the history: knowing this call was made, with its input, still matters for what the assistant does next`,
      criteria: { true: 'the call is part of the current work: it changed files the task builds on, or it records a constraint, decision, or finding still relevant', false: 'the call is stale: exploration that led nowhere, a superseded check, or unrelated to the goal' } },
    [`result_${id}`]: { type: 'noul', instructions: `The full output of tool call ${id} (${tool}, ${res}) should stay in the history verbatim: the assistant still needs its contents and re-running the tool would not do`,
      criteria: { true: 'the exact output is still needed for the next steps and cannot simply be re-read or re-run', false: 'a short note of the outcome is enough, or the output is stale, superseded, or can be re-run' } } }),
  C_question: (id, tool, res) => ({
    [`call_${id}`]: { type: 'noul', instructions: `Does the assistant still need to know that tool call \`${id}\` (${tool}) was made, with its input, for the current goal?`,
      criteria: { true: 'yes: it changed files the task builds on, or it records a constraint, decision, or finding still relevant', false: 'no: exploration that led nowhere, a superseded check, or unrelated to the goal' } },
    [`result_${id}`]: { type: 'noul', instructions: `Does the assistant still need the full verbatim output of tool call \`${id}\` (${tool}, ${res}) for the current goal?`,
      criteria: { true: 'yes: the exact output is needed for the next steps and re-running the tool would not do', false: 'no: a short note of the outcome is enough, or the output is stale or can be re-run' } } }),
};
for (const [name, v] of Object.entries(variants)) {
  const q = Object.assign({}, ...[...meta.entries()].map(([id, c]) => v(id, c.tool, c.result)));
  const t = Date.now();
  const r = await fetch(URL, { method: 'POST', headers: { 'content-type': 'application/json', authorization: 'Bearer ' + process.env.ORK }, body: JSON.stringify({ model: 'typesafe/jev-1.13', state, questions: q }) });
  const a = (await r.json()).answers;
  console.log(name, Date.now() - t, 'ms');
  for (const [id, c] of meta) console.log(`  ${id} ${c.tool.padEnd(5)} call=${a[`call_${id}`].noul.toFixed(2)} result=${a[`result_${id}`].noul.toFixed(2)}`);
}
console.log('goal:', (state as any).goal);
