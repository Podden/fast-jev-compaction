export type Role = 'user' | 'assistant';

/**
 * A tool_use block of an assistant message. `text` and `isError` mirror the
 * outcome once the transcript holds it (Claude Code attaches them).
 */
export interface ToolUse {
  tool_use_id: string;
  tool: string;
  input: Record<string, unknown>;
  text?: string;
  isError?: boolean;
}

/** A tool_result block of a user message. */
export interface ToolResult {
  tool_use_id: string;
  text: string;
  isError?: boolean;
}

/**
 * One transcript message. The shape is a subset of Claude Code's
 * `SessionMessage`, so a session transcript can be passed in as is.
 */
export interface Message {
  role: Role;
  text: string;
  toolUses: ToolUse[];
  toolResults?: ToolResult[];
}

/** A tool call paired with its result by `tool_use_id`. */
export interface ToolCall {
  /** Short id used in the Jev state and question names (`t1`, `t2`, ...). */
  id: string;
  tool_use_id: string;
  tool: string;
  input: Record<string, unknown>;
  /** Index of the message holding the tool_use block. */
  callIndex: number;
  /** Index of the message holding the tool_result block. */
  resultIndex: number;
  /** The tool_result text. */
  result: string;
  resultChars: number;
  isError: boolean;
  /** In the first or the newest preserved messages; never a candidate. */
  pinned: boolean;
}

/**
 * A search run: the tool calls the assistant made between two texts, e.g.
 * greps, reads and commands that led to a finding stated in the text after.
 */
export interface Episode {
  /** Short id used in the Jev state and question names (`e1`, `e2`, ...). */
  id: string;
  /** Index of the message holding the episode's first tool call. */
  start: number;
  /** Index of the message whose text closes the episode (`after`). */
  end: number;
  calls: ToolCall[];
  /** Characters of tool input and output the episode holds. */
  chars: number;
}

/** One episode as Jev sees it. */
export interface EpisodeEntry {
  id: string;
  /** The text the episode started with, if its first message had one. */
  before?: string;
  /** One line per call: tool, input, outcome, and the head of the output. */
  calls: string[];
  /** The text that closes the episode; where the finding should be stated. */
  after: string;
}

/** The state of one Jev request: the episodes asked about in it. */
export interface EpisodeState {
  context: string;
  goal: string;
  episodes: EpisodeEntry[];
}

export interface EpisodeDecision {
  id: string;
  calls: number;
  chars: number;
  /** 1 − Jev's probability that the text after the episode reports its outcome. */
  keep: number;
  /** `condense`: kept by Jev, then replaced by the condenser's fact note. */
  action: 'keep' | 'replace' | 'condense';
  /** Wall time of the condenser call; set when the episode went to the condenser. */
  condenseMs?: number;
  /** Why the condenser's note was not used: the call's error, or the guard's verdict. */
  condenseError?: string;
}

/** One call of an episode handed to a condenser, with its full output. */
export interface CondenseCall {
  tool: string;
  input: Record<string, unknown>;
  isError: boolean;
  output: string;
}

/** What a condenser gets: an episode whose outcome the text after does not report. */
export interface CondenseInput {
  goal: string;
  before?: string;
  after: string;
  calls: CondenseCall[];
}

/**
 * Writes the facts of an episode as a short note (e.g. a fast LLM). A
 * rejection keeps the episode as it is.
 */
export interface EpisodeCondenser {
  condense(input: CondenseInput): Promise<string>;
}

export interface CompactOptions {
  /** Ongoing task description; defaults to the last few user prompts. */
  goal?: string;
  /** Minimum keep probability for an episode to stay call by call. Default 0.5. */
  keepThreshold?: number;
  /** Newest messages never touched (the first message is always kept). Default 6. */
  preserveRecentMessages?: number;
  /** Estimated token ceiling for the state of one request (one episode). Default 25000. */
  maxStateTokens?: number;
  /** Estimated token ceiling for state plus question of one request. Default 30000. */
  maxRequestTokens?: number;
  /** Characters of a failed call's output kept in the replacement note. Default 300. */
  truncateHeadChars?: number;
  /** Requests in flight at once; 1 for a single-GPU server like decider. 0 for no limit. Default 0. */
  maxConcurrentRequests?: number;
  /** Condenses episodes Jev keeps into a fact note; unset leaves them verbatim. */
  condenser?: EpisodeCondenser;
  /** Smallest kept episode (tool input and output chars) worth condensing. Default 4000. */
  condenseMinChars?: number;
  /** Condenser calls in flight at once; 0 for no limit. Default 4. */
  maxConcurrentCondense?: number;
  /**
   * Smallest share of a note's identifiers (paths, ids, numbers) that must
   * occur in the episode; a note below keeps the episode. Default 0.3.
   */
  condenseMinGrounded?: number;
  /**
   * Stores a condensed episode's full calls and outputs and resolves where
   * (a file path), so the note can point there. Optional; a rejection drops
   * the pointer, not the note.
   */
  archive?: (episode: Episode, text: string) => Promise<string>;
}

export interface ResolvedCompactOptions {
  goal: string;
  keepThreshold: number;
  preserveRecentMessages: number;
  maxStateTokens: number;
  maxRequestTokens: number;
  truncateHeadChars: number;
  maxConcurrentRequests: number;
  condenseMinChars: number;
  maxConcurrentCondense: number;
  condenseMinGrounded: number;
}

export interface CompactResult {
  /** The compacted transcript; untouched messages are the input objects. */
  messages: Message[];
  /** One per episode Jev was asked about. */
  decisions: EpisodeDecision[];
  stats: {
    messagesBefore: number;
    messagesAfter: number;
    charsBefore: number;
    charsAfter: number;
    /** Closed episodes outside the pinned messages, asked about or not. */
    episodes: number;
    replaced: number;
    kept: number;
    /** Kept episodes replaced by a condenser note (counted in `kept` too). */
    condensed: number;
    /** Condenser calls that failed; their episodes stayed verbatim. */
    condenseFailed: number;
    /** Wall time of the condenser stage. */
    condenseMs: number;
    /** Wall time of the Jev requests. */
    deciderMs: number;
    /** Episodes left alone without asking: too small, or too big for one request. */
    skipped: number;
    callsRemoved: number;
    /** Largest state of any request. */
    stateTokens: number;
    requests: number;
    ms: number;
  };
}

/** The `state` of a Jev request: a string or any JSON-serialisable object. */
export type JevState = string | object;

export interface NoulQuestion {
  type: 'noul';
  instructions: string;
  criteria?: {
    true?: string;
    false?: string;
  };
}

export interface ChoiceQuestion {
  type: 'choice';
  instructions: string;
  criteria: Record<string, string | null>;
}

export interface ScoreQuestion {
  type: 'score';
  instructions: string;
  criteria: string[];
}

export type JevQuestion = NoulQuestion | ChoiceQuestion | ScoreQuestion;
export type JevQuestions = Record<string, JevQuestion>;

export interface NoulAnswer {
  type?: 'noul';
  noul: number;
}

export interface ChoiceAnswer {
  type?: 'choice';
  choice: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export interface ScoreAnswer {
  type?: 'score';
  score: number;
  confidence: number;
  probabilities: Record<string, number>;
}

export type JevAnswer = NoulAnswer | ChoiceAnswer | ScoreAnswer;

export interface JevResponse {
  model?: string;
  answers: Record<string, JevAnswer>;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
  };
  [key: string]: unknown;
}

/** Anything that can answer Jev questions: `JevClient`, or a host-provided adapter. */
export interface JevAsker {
  ask(state: JevState, questions: JevQuestions): Promise<JevResponse>;
}
