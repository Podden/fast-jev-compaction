import { pickSessions, parseSession, cutPoints, facts } from './replay-lib.js';
let total = 0, cuts = 0;
for (const { project, file } of pickSessions()) for (const c of cutPoints(parseSession(file))) {
  const n = facts(c.prefix, c.future).size; total += n; cuts++;
  console.log(project.slice(21, 45), c.at, 'prefix', c.prefix.length, 'facts', n);
}
console.log('cuts', cuts, 'facts', total);
