import { applyTraceEvent, emptyProcessTrace } from "./citations";
import type { AnswerRunError, ChatMessage, RagMode, SseEvent } from "./types";

export type ConversationLoadStatus = "idle" | "loading" | "ready" | "error";

export interface ConversationMessageEntry {
  messages: ChatMessage[];
  loadStatus: ConversationLoadStatus;
  loadError?: string;
  loadRequestId?: string;
  loadRevision: number;
  revision: number;
  activeRunId?: string;
  phase: string;
}

export interface ConversationMessageState {
  byConversation: Record<string, ConversationMessageEntry>;
}

export const initialConversationMessageState: ConversationMessageState = {
  byConversation: {},
};

export type ConversationMessageAction =
  | { type: "ensure_conversation"; conversationId: string; ready?: boolean }
  | { type: "remove_conversation"; conversationId: string }
  | { type: "load_started"; conversationId: string; requestId: string }
  | { type: "load_cancelled"; conversationId: string; requestId: string }
  | { type: "load_succeeded"; conversationId: string; requestId: string; messages: ChatMessage[] }
  | { type: "load_failed"; conversationId: string; requestId: string; error: string }
  | {
      type: "run_started";
      conversationId: string;
      runId: string;
      userMessage: ChatMessage;
      assistantMessage: ChatMessage;
    }
  | {
      type: "stream_event";
      conversationId: string;
      runId: string;
      event: SseEvent;
      phaseText?: string;
    }
  | { type: "run_stopped"; conversationId: string; runId: string }
  | { type: "run_failed"; conversationId: string; runId: string; error: AnswerRunError };

function emptyEntry(loadStatus: ConversationLoadStatus = "idle"): ConversationMessageEntry {
  return {
    messages: [],
    loadStatus,
    loadRevision: 0,
    revision: 0,
    phase: "",
  };
}

function replaceEntry(
  state: ConversationMessageState,
  conversationId: string,
  entry: ConversationMessageEntry,
): ConversationMessageState {
  return {
    ...state,
    byConversation: {
      ...state.byConversation,
      [conversationId]: entry,
    },
  };
}

function updateRunMessage(
  messages: ChatMessage[],
  runId: string,
  update: (message: ChatMessage) => ChatMessage,
): ChatMessage[] {
  return messages.map((message) =>
    message.role === "assistant" && message.run_id === runId ? update(message) : message,
  );
}

export function conversationMessageReducer(
  state: ConversationMessageState,
  action: ConversationMessageAction,
): ConversationMessageState {
  if (action.type === "ensure_conversation") {
    if (state.byConversation[action.conversationId]) return state;
    return replaceEntry(
      state,
      action.conversationId,
      emptyEntry(action.ready ? "ready" : "idle"),
    );
  }

  if (action.type === "remove_conversation") {
    if (!state.byConversation[action.conversationId]) return state;
    const byConversation = { ...state.byConversation };
    delete byConversation[action.conversationId];
    return { ...state, byConversation };
  }

  if (action.type === "load_started") {
    const current = state.byConversation[action.conversationId] ?? emptyEntry();
    if (current.loadStatus === "ready" || current.loadStatus === "loading" || current.activeRunId) {
      return state;
    }
    return replaceEntry(state, action.conversationId, {
      ...current,
      loadStatus: "loading",
      loadError: undefined,
      loadRequestId: action.requestId,
      loadRevision: current.revision,
    });
  }

  if (action.type === "load_cancelled") {
    const current = state.byConversation[action.conversationId];
    if (!current || current.loadRequestId !== action.requestId) return state;
    return replaceEntry(state, action.conversationId, {
      ...current,
      loadStatus: "idle",
      loadRequestId: undefined,
    });
  }

  if (action.type === "load_succeeded") {
    const current = state.byConversation[action.conversationId];
    if (
      !current
      || current.loadRequestId !== action.requestId
      || current.loadRevision !== current.revision
    ) {
      return state;
    }
    return replaceEntry(state, action.conversationId, {
      ...current,
      messages: action.messages,
      loadStatus: "ready",
      loadError: undefined,
      loadRequestId: undefined,
      activeRunId: undefined,
      phase: "",
    });
  }

  if (action.type === "load_failed") {
    const current = state.byConversation[action.conversationId];
    if (
      !current
      || current.loadRequestId !== action.requestId
      || current.loadRevision !== current.revision
    ) {
      return state;
    }
    return replaceEntry(state, action.conversationId, {
      ...current,
      loadStatus: "error",
      loadError: action.error,
      loadRequestId: undefined,
    });
  }

  if (action.type === "run_started") {
    const current = state.byConversation[action.conversationId] ?? emptyEntry("ready");
    if (current.activeRunId) return state;
    return replaceEntry(state, action.conversationId, {
      ...current,
      messages: [...current.messages, action.userMessage, action.assistantMessage],
      loadStatus: "ready",
      loadError: undefined,
      loadRequestId: undefined,
      revision: current.revision + 1,
      activeRunId: action.runId,
      phase: "正在准备问题",
    });
  }

  if (action.type === "stream_event") {
    const current = state.byConversation[action.conversationId];
    if (!current || current.activeRunId !== action.runId) return state;

    const event = action.event;
    const messages = updateRunMessage(current.messages, action.runId, (message) => {
      const processTrace = applyTraceEvent(message.process_trace ?? emptyProcessTrace(), event);
      if (event.type === "answer_delta") {
        return { ...message, content: message.content + event.delta, process_trace: processTrace };
      }
      if (event.type === "evidence") {
        const citations = message.citations ?? [];
        const nextCitations = citations.some((citation) => citation.evidence_id === event.evidence.evidence_id)
          ? citations
          : [...citations, event.evidence];
        return { ...message, citations: nextCitations, process_trace: processTrace };
      }
      if (event.type === "done") {
        return {
          ...message,
          content: message.content || event.response.answer,
          citations: event.response.evidence,
          answer_kind: event.response.answer_kind,
          contract_version: event.response.contract_version,
          response_kind: event.response.response_kind,
          claims: event.response.claims,
          advice: event.response.advice,
          user_premises: event.response.user_premises,
          clarification: event.response.clarification,
          verification: event.response.verification,
          insufficient_reason: event.response.insufficient_reason,
          process_trace: processTrace,
          request_id: event.response.request_id,
          assistant_message_id: event.response.assistant_message_id,
          run_status: "completed",
          run_error: undefined,
        };
      }
      return { ...message, process_trace: processTrace };
    });

    return replaceEntry(state, action.conversationId, {
      ...current,
      messages,
      activeRunId: event.type === "done" ? undefined : current.activeRunId,
      phase: event.type === "done"
        ? ""
        : event.type === "phase"
          ? action.phaseText ?? event.message ?? event.phase
          : current.phase,
    });
  }

  if (action.type === "run_stopped") {
    const current = state.byConversation[action.conversationId];
    if (!current || !current.messages.some((message) => message.run_id === action.runId)) return state;
    return replaceEntry(state, action.conversationId, {
      ...current,
      messages: updateRunMessage(current.messages, action.runId, (message) => ({
        ...message,
        run_status: "stopped",
        run_error: undefined,
      })),
      activeRunId: current.activeRunId === action.runId ? undefined : current.activeRunId,
      phase: current.activeRunId === action.runId ? "" : current.phase,
    });
  }

  const current = state.byConversation[action.conversationId];
  if (!current || !current.messages.some((message) => message.run_id === action.runId)) return state;
  return replaceEntry(state, action.conversationId, {
    ...current,
    messages: updateRunMessage(current.messages, action.runId, (message) => ({
      ...message,
      run_status: "failed",
      run_error: action.error,
    })),
    activeRunId: current.activeRunId === action.runId ? undefined : current.activeRunId,
    phase: current.activeRunId === action.runId ? "" : current.phase,
  });
}

export function createOptimisticMessages(
  conversationId: string,
  runId: string,
  question: string,
  mode: RagMode,
  createdAt: string,
): { userMessage: ChatMessage; assistantMessage: ChatMessage } {
  return {
    userMessage: {
      id: `${runId}-user`,
      conversation_id: conversationId,
      role: "user",
      content: question,
      mode,
      created_at: createdAt,
    },
    assistantMessage: {
      id: `${runId}-assistant`,
      conversation_id: conversationId,
      role: "assistant",
      content: "",
      mode,
      created_at: createdAt,
      citations: [],
      process_trace: emptyProcessTrace(),
      run_id: runId,
      run_status: "streaming",
      retry_question: question,
    },
  };
}
