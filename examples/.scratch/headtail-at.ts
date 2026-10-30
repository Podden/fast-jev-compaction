// head+tail retention at the reduction another strategy reached, per replay point.
import { readFileSync } from 'node:fs';

const [resultsFile, strategy] = process.argv.slice(2);
process.argv.splice(2); // replay.ts reads its own argv on import
const { chars, cutPoints, facts, headTailMatched, parseSession, pickSessions, serialize } = await import('../replay.js');
const rows = JSON.parse(readFileSync(resultsFile!, 'utf8')).rows as Record<string, any>[];
let kept = 0;
let total = 0;
for (const { project, file } of pickSessions()) for (const { at, prefix, future } of cutPoints(parseSession(file))) {
  const session = `${project.replace(/^C--Users-Daniel-Code-/i, '')}/${file.slice(-42, -34)}@${at}`;
  const row = rows.find((r) => r.session === session && r.strategy === strategy);
  if (!row) continue;
  const wanted = facts(prefix, future);
  const text = serialize(headTailMatched(prefix, chars(prefix) * (1 - row.reduction)));
  kept += [...wanted].filter((t) => text.includes(t)).length;
  total += wanted.size;
}
console.log(`head+tail at ${strategy} size: ${kept}/${total} = ${(kept / total).toFixed(3)}`);
