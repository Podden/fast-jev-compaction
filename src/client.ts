import { buildCondenseRequest, parseCondenseResponse, type CondenseRequestParams } from './condense.js';
import { buildJevRequest, parseJevResponse } from './request.js';
import type {
  CondenseInput,
  EpisodeCondenser,
  JevAsker,
  JevQuestions,
  JevResponse,
  JevState,
} from './types.js';

export interface JevClientOptions {
  /** Defaults to `process.env.TYPESAFE_API_KEY`. */
  apiKey?: string;
  /** Defaults to `jev-latest`. */
  model?: string;
  /** Defaults to the TypeSafe System One endpoint; a custom one needs no key. */
  baseUrl?: string;
  /** Defaults to the global `fetch`. */
  fetch?: typeof fetch;
}

/** Asks Jev over HTTP with the global `fetch` (or an injected one). */
export class JevClient implements JevAsker {
  private readonly apiKey: string;
  private readonly model: string | undefined;
  private readonly baseUrl: string | undefined;
  private readonly fetcher: typeof fetch;

  constructor(options: JevClientOptions = {}) {
    this.apiKey = options.apiKey ?? process.env.TYPESAFE_API_KEY ?? '';
    this.model = options.model;
    this.baseUrl = options.baseUrl;
    this.fetcher = options.fetch ?? fetch;
  }

  async ask(state: JevState, questions: JevQuestions): Promise<JevResponse> {
    if (!this.apiKey && !this.baseUrl) throw new Error('TYPESAFE_API_KEY is not configured');
    const request = buildJevRequest(
      { apiKey: this.apiKey, model: this.model, baseUrl: this.baseUrl },
      state,
      questions,
    );
    const response = await this.fetcher(request.url, {
      method: request.method,
      headers: request.headers,
      body: request.body,
    });
    return parseJevResponse(response.status, response.ok, await response.text());
  }
}

/** An `EpisodeCondenser` over an OpenAI-compatible endpoint with `fetch`. */
export class LlmCondenser implements EpisodeCondenser {
  constructor(
    private readonly params: CondenseRequestParams,
    private readonly fetcher: typeof fetch = fetch,
  ) {}

  async condense(input: CondenseInput): Promise<string> {
    const request = buildCondenseRequest(this.params, input);
    const response = await this.fetcher(request.url, request);
    return parseCondenseResponse(response.status, response.ok, await response.text());
  }
}
