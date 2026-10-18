# fast-jev-compaction

Claude Code plugin that replaces the compaction summary with Jev decisions:
every search run (the tool calls between two assistant texts) is checked in one
fast request, and a run whose outcome the next text already reports shrinks to
one line per call. Everything else stays verbatim. Also usable as an npm
library.

## What and why

Most context compaction asks an LLM to summarize old turns. A summary is
lossy: a file path, exact error, constraint, or command can disappear even when
it matters later. This library never rewrites text. Most of a coding session is
the path to a finding: greps, reads and commands whose outputs the assistant
has already condensed into a sentence such as "the 13 rules are hidden because
plugin X is missing". Once that sentence exists, the path can go. User and
assistant text stays verbatim and in order, and failed calls stay listed with
the head of their error, so the pitfalls on the way are not repeated.

The repository is both an npm package (`src/`) and a Claude Code plugin
(`hooks/`, `.claude-plugin/`) that uses the package to replace Claude Code's
built-in compaction summary with the original messages.

## How it works

1. Every `tool_use` is paired with its `tool_result` by `tool_use_id`.
2. The conversation is split into **episodes**: the tool calls between one
   text and the next. An episode is a candidate when an assistant text closes
   it (not a user prompt cutting it short), none of its messages is pinned (the
   first message and the newest `preserveRecentMessages`), and its calls hold
   at least 1000 characters of input and output.
3. Each candidate gets its own request. The state holds that episode only: the
   text it started with (`before`), one line per call (tool, input, ok or
   error, size and the first 240 characters of the output), the text that
   closes it (`after`), and the `goal` (the last three prompts by default).
   One `noul` question asks whether `after` reports the result of the calls.
   An episode too big for `maxStateTokens` / `maxRequestTokens` is shown with
   60-character output heads, then with none, and left alone if it still does
   not fit. Tokens are estimated without a tokenizer (a word per six letters,
   half a token per digit, ~one per other symbol), calibrated to land a little
   above the counts Jev reports.
4. The keep probability is 1 minus that answer. Below `keepThreshold` the
   episode is replaced: its calls and results disappear, and its first message
   gets a note with one line per call
   (`- Grep pattern=carp path=docs → ok, 312 chars`; a failed call shows the
   first `truncateHeadChars` characters of its error instead of the size).
5. The message list is rebuilt: every text stays as it was, a message that
   loses all its content is removed, untouched messages are returned as the
   same objects, and no result is ever left without its call.

One episode per request is deliberate. With two dozen episodes in one state,
decider-4b mixed them up and its scores no longer told stated findings from
unstated ones; asked about one episode at a time, it separated them cleanly.

Jev failures, malformed answers or a missing key throw; the caller (or the
Claude Code hook) decides what to fall back to.

## Install and usage

```sh
npm install fast-jev-compaction
export TYPESAFE_API_KEY=...
```

```ts
import { compactMessages, reductionRatio, type Message } from 'fast-jev-compaction';

const transcript: Message[] = [
  { role: 'user', text: 'Fix the failing test. Never edit src/generated.', toolUses: [] },
  {
    role: 'assistant',
    text: '',
    toolUses: [{ tool_use_id: 'toolu_1', tool: 'Read', input: { file_path: 'src/a.ts' } }],
  },
  { role: 'user', text: '', toolUses: [], toolResults: [{ tool_use_id: 'toolu_1', text: '…file…' }] },
  // …
];

const result = await compactMessages(transcript, { preserveRecentMessages: 4 });
console.log(result.messages, result.decisions, result.stats);
if (reductionRatio(result) < 0.25) {
  // not worth it: keep the original transcript, or summarize instead
}
```

`Message` is a subset of Claude Code's `SessionMessage`, so a session transcript
can be passed in as is.

To bring your own transport, implement `JevAsker` (one `ask(state, questions)`
method) and call `compact(messages, asker, options)`; `buildJevRequest` and
`parseJevResponse` give you the HTTP request body and response validation.
The building blocks (`collectToolCalls`, `collectEpisodes`, `episodeEntry`,
`questionFor`, `sizeEpisode`, `applyEpisodes`, `replacementNote`) are exported
too.

`apiKey` defaults to `process.env.TYPESAFE_API_KEY`. Never commit the key or
put it in a source file.

### Other System One models (self-hosted decider)

Any server that speaks TypeSafe's `POST /v1/systemone` wire format can stand in
for Jev, e.g. [decider](https://huggingface.co/Mapika/decider-4b) served with
`decider.serve`. Point `baseUrl` at it; no key is sent when `apiKey` is empty.

```ts
await compactMessages(transcript, {
  baseUrl: 'http://192.168.107.103:5810/v1/systemone',
  maxStateTokens: 24_000,
  keepThreshold: 0.6,
  maxConcurrentRequests: 1,
});
```

decider-4b-v2.1 reads states of at least 31k tokens in full: sentences planted
at both ends of a 70k-character state were found (0.97–0.98) and absent ones
rejected (0.03–0.15). One episode is usually 1–3k tokens. On a real
281k-character session with 24 candidate episodes, all requests together took
5.7 s and saved 65 %. Requests to one server run one after another, so send
them one at a time (`maxConcurrentRequests: 1`): the Claude Code host aborts
each fetch after 30 s, and a queued request spends its wait inside that limit.
`npm run demo` honours `SYSTEM_ONE_BASE_URL`.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `apiKey` | `TYPESAFE_API_KEY` | TypeSafe API key (`compactMessages`/`JevClient`); not needed with a custom `baseUrl` |
| `model` | `jev-latest` | Jev model name (ignored by decider) |
| `baseUrl` | `https://api.typesafe.ai/v1/systemone` | System One endpoint; any server with the same wire API works |
| `fetch` | native `fetch` | Injectable fetch implementation for tests |
| `goal` | last 3 user prompts | Ongoing task description included in the state |
| `keepThreshold` | `0.5` | Minimum keep probability (1 − P(the next text reports the outcome)) for an episode to stay call by call |
| `preserveRecentMessages` | `6` | Newest messages never touched (the first is always kept) |
| `maxStateTokens` | `25000` | Estimated token ceiling for the state of one request (one episode) |
| `maxRequestTokens` | `30000` | Estimated ceiling for state plus question of one request |
| `truncateHeadChars` | `300` | Characters of a failed call's output kept in the replacement note |
| `maxConcurrentRequests` | `0` | Requests in flight at once; `1` for decider; `0` for no limit |
| `condenser` | none | `EpisodeCondenser` (e.g. `LlmCondenser`) that turns kept episodes into a fact note |
| `condenseMinChars` | `4000` | Smallest kept episode (tool input and output chars) handed to the condenser |
| `maxConcurrentCondense` | `4` | Condenser calls in flight at once; `0` for no limit |
| `condenseMinGrounded` | `0.3` | Smallest share of a note's identifiers found in the episode; below, or without `- ` fact lines, the note is dropped and the episode kept |
| `archive` | none | `(episode, text) => Promise<path>`: stores a condensed episode's full outputs; the note points there |

`result.stats` reports message and character counts before and after, how
many episodes were found, replaced, kept and skipped, the calls removed, the
largest state in estimated tokens, and the number of requests.

### Condensing kept episodes (optional)

An episode Jev keeps holds outputs whose result the assistant never wrote
down, so it stays verbatim and expensive. With a `condenser`, every kept
episode of at least `condenseMinChars` goes to a fast LLM instead, which
writes the facts from the outputs (paths, errors, ids, counts, verbatim) as a
note; with `archive`, the full outputs are stored and the note names the file.
A failed condenser call keeps the episode. Episodes Jev replaces never reach
the condenser: their outcome is already in the text after them.

```ts
import { compact, JevClient, LlmCondenser } from 'fast-jev-compaction';

await compact(transcript, new JevClient({ baseUrl: 'http://192.168.107.103:5810/v1/systemone' }), {
  maxConcurrentRequests: 1,
  condenser: new LlmCondenser({
    url: 'http://192.168.107.103:11500/v1/chat/completions',
    model: 'qwen3.8:27b',
  }),
});
```

`examples/replay.ts` replays local Claude Code sessions and compares drop-all,
episodes, head+tail at the same size, and episodes plus condensing by
reduction and by how many later-used facts from tool outputs survive.

## Limitations

- Only tool calls and results are removed; texts are never removed or
  shortened in the output (they are only abridged in the state Jev sees).
- An episode is kept or replaced as a whole. A run Jev keeps stays complete,
  including outputs nobody needs any more.
- A finding the assistant never wrote down ("analysis done, writing the page
  now") keeps its episode. That is the point, but such runs stay expensive.
- Token sizes are estimates from character counts, not a tokenizer.
- A probability is not a proof that an outcome is stated. The assistant can
  always re-run a tool.

## Claude Code plugin

The repository root is a Claude Code function-hook plugin: `hooks/fast-jev.ts`
is a thin adapter that feeds `session.compact` transcripts through `src/` and
falls back to Claude Code's built-in summary on errors or insufficient
reduction. See [`hooks/README.md`](hooks/README.md) for configuration and the
Claude Code 2.1.274 type reference.

### Install in Claude Code

Function hooks are an early-access Claude Code feature (2.1.274+), so the
opt-in flag must be set wherever Claude Code runs, e.g. in `~/.claude/settings.json`:

```json
{ "env": { "CLAUDE_CODE_ENABLE_FUNCTION_HOOKS": "1", "TYPESAFE_API_KEY": "<your key>" } }
```

Then add this repository as a plugin marketplace and install the plugin,
either from the shell or as slash commands inside a session:

```sh
claude plugin marketplace add tamaratran/fast-jev-compaction
claude plugin install fast-jev-compaction@fast-jev-compaction
```

The install prompts for the plugin options (API key, thresholds, `truncateHeadChars`,
…); leave them at their defaults to use `TYPESAFE_API_KEY` from the environment.
Restart Claude Code or run `/reload-plugins`. From then on `/compact` (and
auto-compaction) goes through Jev: the toast reads
`fast-jev-compaction: kept N/M messages, no summary (…)` when the pruned history
replaced the built-in summary, or `fallback to built-in summary (…)` when Jev
could not remove enough (short sessions, or when it fails).

To run from a checkout without installing: `CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1 claude --plugin-dir .`
from the repository root. No publishing step is required; the marketplace is
just the repo's `.claude-plugin/marketplace.json`.

## Development

```sh
npm install
npm run typecheck        # library + hook
npm test
npm run build
npm run validate:plugin  # claude plugin validate
TYPESAFE_API_KEY="$(cat ~/.typesafe_key)" npm run demo
```

The unit tests use a fake Jev and never contact TypeSafe. The demo is the live
network check.

## Animated demo (macOS)

`demo/JevDemo` is a small native SwiftUI app that plays a scripted, dramatized
version of the compaction flow inside a Claude Code-style terminal: the tool
calls of a canned transcript are scored, results and calls Jev lets go turn red
and collapse away, and the rest stays verbatim. It never calls the API; it
exists to be screen recorded.

```sh
demo/JevDemo/build.sh   # builds demo/JevDemo/build/JevDemo.app and launches it
```

Press space in the app to replay from the start.
