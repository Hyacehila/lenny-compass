import type {
  ChatMessage,
  Conversation,
  DocumentChunkLocator,
  HealthStatus,
  IndexStatus,
  RagMode,
  SourceWindowResponse,
  SseEvent,
} from "./types";

const API_ROOT = "/api/v1";

type DoneEvent = Extract<SseEvent, { type: "done" }>;
type ErrorEvent = Extract<SseEvent, { type: "error" }>;
type TerminalEvent = DoneEvent | ErrorEvent;

type StoredChatMessage = ChatMessage & {
  response_payload?: Pick<
    ChatMessage,
    "advice" | "user_premises" | "clarification" | "verification" | "insufficient_reason"
  >;
};

const SSE_EVENT_TYPES = new Set<SseEvent["type"]>([
  "phase",
  "retrieval_summary",
  "evidence",
  "answer_delta",
  "done",
  "error",
]);

export class AnswerStreamError extends Error {
  readonly code: string;
  readonly retryable: boolean;

  constructor(code: string, message: string, retryable: boolean, options?: ErrorOptions) {
    super(message, options);
    this.name = "AnswerStreamError";
    this.code = code;
    this.retryable = retryable;
  }
}

async function jsonRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

function normalizeStoredMessage(row: StoredChatMessage): ChatMessage {
  const { response_payload: payload, ...message } = row;
  return {
    ...message,
    advice: message.advice ?? payload?.advice,
    user_premises: message.user_premises ?? payload?.user_premises,
    clarification: message.clarification ?? payload?.clarification,
    verification: message.verification ?? payload?.verification,
    insufficient_reason: message.insufficient_reason ?? payload?.insufficient_reason,
  };
}

export const api = {
  health: () => jsonRequest<HealthStatus>("/health"),
  conversations: () => jsonRequest<Conversation[]>("/conversations"),
  createConversation: () =>
    jsonRequest<Conversation>("/conversations", {
      method: "POST",
      body: JSON.stringify({ title: "新对话" }),
    }),
  renameConversation: (id: string, title: string) =>
    jsonRequest<Conversation>(`/conversations/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    }),
  deleteConversation: (id: string) =>
    jsonRequest<void>(`/conversations/${encodeURIComponent(id)}`, { method: "DELETE" }),
  messages: async (id: string, signal?: AbortSignal) => {
    const rows = await jsonRequest<StoredChatMessage[]>(
      `/conversations/${encodeURIComponent(id)}/messages`,
      { signal },
    );
    return rows.map(normalizeStoredMessage);
  },
  indexStatus: () => jsonRequest<IndexStatus>("/index/status"),
  readSourceWindow: (locator: DocumentChunkLocator, window = 2, signal?: AbortSignal) =>
    jsonRequest<SourceWindowResponse>("/facts/read-source-window", {
      method: "POST",
      body: JSON.stringify({ locator, window }),
      signal,
    }),
};

async function responseError(response: Response): Promise<AnswerStreamError> {
  const body = await response.text();
  try {
    const parsed = JSON.parse(body) as {
      error?: { code?: unknown; message?: unknown; retryable?: unknown };
      detail?: unknown;
    };
    if (parsed.error && typeof parsed.error.message === "string") {
      return new AnswerStreamError(
        typeof parsed.error.code === "string" ? parsed.error.code : "ANSWER_REQUEST_FAILED",
        parsed.error.message,
        typeof parsed.error.retryable === "boolean" ? parsed.error.retryable : response.status >= 500,
      );
    }
    if (typeof parsed.detail === "string") {
      return new AnswerStreamError("ANSWER_REQUEST_FAILED", parsed.detail, response.status >= 500);
    }
  } catch {
    // Fall through to the plain-text response below.
  }
  return new AnswerStreamError(
    "ANSWER_REQUEST_FAILED",
    body || `${response.status} ${response.statusText}`,
    response.status >= 500,
  );
}

function parseEvent(payload: string): SseEvent {
  let candidate: unknown;
  try {
    candidate = JSON.parse(payload);
  } catch (cause) {
    throw new AnswerStreamError(
      "ANSWER_STREAM_INVALID_EVENT",
      "Answer stream contained invalid JSON",
      true,
      { cause },
    );
  }
  if (
    !isRecord(candidate)
    || !("type" in candidate)
    || typeof candidate.type !== "string"
    || !SSE_EVENT_TYPES.has(candidate.type as SseEvent["type"])
  ) {
    throw new AnswerStreamError(
      "ANSWER_STREAM_INVALID_EVENT",
      "Answer stream contained an unknown event",
      true,
    );
  }

  if (candidate.type === "error" && !isErrorEvent(candidate)) {
    throw new AnswerStreamError(
      "ANSWER_STREAM_INVALID_EVENT",
      "Answer stream contained an invalid error event",
      true,
    );
  }
  if (candidate.type === "done" && !isDoneEvent(candidate)) {
    throw new AnswerStreamError(
      "ANSWER_STREAM_INVALID_EVENT",
      "Answer stream contained an invalid done event",
      true,
    );
  }
  return candidate as SseEvent;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function isAnswerClaim(value: unknown): boolean {
  if (!isRecord(value) || !Array.isArray(value.evidence_ids)) return false;
  const evidenceIds = value.evidence_ids;
  return isNonEmptyString(value.claim_id)
    && isNonEmptyString(value.text)
    && (value.role === undefined || ["core", "supporting", "advice_rationale"].includes(String(value.role)))
    && evidenceIds.length > 0
    && evidenceIds.every(isNonEmptyString)
    && new Set(evidenceIds).size === evidenceIds.length;
}

function isAdviceItem(value: unknown): boolean {
  if (!isRecord(value) || !Array.isArray(value.rationale_claim_ids)) return false;
  return isNonEmptyString(value.advice_id)
    && isNonEmptyString(value.text)
    && (value.basis === "evidence_synthesis" || value.basis === "general_pm_guidance")
    && value.rationale_claim_ids.every(isNonEmptyString);
}

function isErrorEvent(candidate: Record<string, unknown>): candidate is ErrorEvent {
  if (!isRecord(candidate.error)) return false;
  return isNonEmptyString(candidate.error.code)
    && isNonEmptyString(candidate.error.message)
    && typeof candidate.error.retryable === "boolean";
}

function isDoneEvent(candidate: Record<string, unknown>): candidate is DoneEvent {
  if (!isRecord(candidate.response)) return false;
  const response = candidate.response;
  if (
    !isNonEmptyString(response.request_id)
    || (response.mode !== "direct" && response.mode !== "agentic")
    || !isNonEmptyString(response.answer)
    || !Array.isArray(response.evidence)
    || !Array.isArray(response.claims)
    || !response.claims.every(isAnswerClaim)
  ) {
    return false;
  }
  const contractVersion = response.contract_version ?? "v1";
  if (contractVersion === "v1") {
    return response.answer_kind === "insufficient_evidence"
      ? response.evidence.length === 0 && response.claims.length === 0
      : response.answer_kind === "grounded"
        && response.evidence.length > 0
        && response.claims.length > 0;
  }
  if (contractVersion !== "v2" || !Array.isArray(response.advice)) return false;
  if (!response.advice.every(isAdviceItem)) return false;
  if (response.response_kind === "grounded_answer") {
    return response.answer_kind === "grounded"
      && response.evidence.length > 0
      && response.claims.length > 0
      && isRecord(response.verification);
  }
  if (response.response_kind === "general_guidance") {
    return response.answer_kind === null
      && response.evidence.length === 0
      && response.claims.length === 0
      && response.advice.length > 0;
  }
  if (response.response_kind === "clarification") {
    return response.answer_kind === null
      && response.evidence.length === 0
      && response.claims.length === 0
      && isRecord(response.clarification)
      && isNonEmptyString(response.clarification.prompt);
  }
  return response.response_kind === "insufficient_evidence"
    && response.answer_kind === "insufficient_evidence"
    && response.evidence.length === 0
    && response.claims.length === 0;
}

function payloadFromFrame(frame: string): string {
  return frame
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart())
    .join("\n");
}

export async function streamAnswer(
  conversationId: string,
  question: string,
  mode: RagMode,
  idempotencyKey: string,
  onEvent: (event: SseEvent) => void,
  signal?: AbortSignal,
  coaching?: {
    coachingId: string;
    replyToMessageId: string;
    skipClarification?: boolean;
  },
): Promise<DoneEvent["response"]> {
  const response = await fetch(
    `${API_ROOT}/conversations/${encodeURIComponent(conversationId)}/ask`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({
        question,
        mode,
        idempotency_key: idempotencyKey,
        ...(coaching ? {
          coaching_id: coaching.coachingId,
          reply_to_message_id: coaching.replyToMessageId,
          skip_clarification: Boolean(coaching.skipClarification),
        } : {}),
      }),
      signal,
    },
  );
  if (!response.ok) throw await responseError(response);
  if (!response.body) {
    throw new AnswerStreamError(
      "ANSWER_STREAM_INCOMPLETE",
      "Answer stream returned no response body",
      true,
    );
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let terminal: TerminalEvent | null = null;
  let reachedEof = false;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop() ?? "";
      if (done && buffer.trim()) {
        frames.push(buffer);
        buffer = "";
      }
      for (const frame of frames) {
        const payload = payloadFromFrame(frame);
        if (!payload) continue;
        const event = parseEvent(payload);
        if (terminal) {
          const duplicateTerminal = event.type === "done" || event.type === "error";
          throw new AnswerStreamError(
            duplicateTerminal
              ? "ANSWER_STREAM_MULTIPLE_TERMINALS"
              : "ANSWER_STREAM_EVENT_AFTER_TERMINAL",
            duplicateTerminal
              ? "Answer stream emitted more than one terminal event"
              : "Answer stream emitted an event after completion",
            true,
          );
        }
        if (event.type === "done" || event.type === "error") terminal = event;
        onEvent(event);
      }
      if (done) {
        reachedEof = true;
        break;
      }
    }
  } finally {
    if (!reachedEof) {
      try {
        await reader.cancel();
      } catch {
        // An aborted transport may already have cancelled the reader.
      }
    }
    reader.releaseLock();
  }

  if (!terminal) {
    throw new AnswerStreamError(
      "ANSWER_STREAM_INCOMPLETE",
      "Answer stream ended without a done event",
      true,
    );
  }
  if (terminal.type === "error") {
    throw new AnswerStreamError(
      terminal.error.code,
      terminal.error.message,
      terminal.error.retryable,
    );
  }
  return terminal.response;
}
