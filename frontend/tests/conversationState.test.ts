import { describe, expect, it } from "vitest";
import {
  conversationMessageReducer,
  createOptimisticMessages,
  initialConversationMessageState,
} from "../src/conversationState";

function startRun(conversationId: string, runId: string, question = "Question") {
  const messages = createOptimisticMessages(
    conversationId,
    runId,
    question,
    "direct",
    "2026-08-29T00:00:00.000Z",
  );
  return {
    type: "run_started" as const,
    conversationId,
    runId,
    ...messages,
  };
}

describe("conversation message reducer", () => {
  it("does not let a stale load overwrite optimistic run messages", () => {
    let state = conversationMessageReducer(initialConversationMessageState, {
      type: "load_started",
      conversationId: "conversation-a",
      requestId: "load-a",
    });
    state = conversationMessageReducer(state, startRun("conversation-a", "run-a"));
    const afterStaleLoad = conversationMessageReducer(state, {
      type: "load_succeeded",
      conversationId: "conversation-a",
      requestId: "load-a",
      messages: [],
    });

    expect(afterStaleLoad.byConversation["conversation-a"].messages).toHaveLength(2);
    expect(afterStaleLoad.byConversation["conversation-a"].activeRunId).toBe("run-a");
  });

  it("returns an aborted conversation load to an idle, retryable state", () => {
    let state = conversationMessageReducer(initialConversationMessageState, {
      type: "load_started",
      conversationId: "conversation-a",
      requestId: "load-a",
    });
    state = conversationMessageReducer(state, {
      type: "load_cancelled",
      conversationId: "conversation-a",
      requestId: "load-a",
    });

    expect(state.byConversation["conversation-a"].loadStatus).toBe("idle");
    expect(state.byConversation["conversation-a"].loadRequestId).toBeUndefined();
  });

  it("isolates stream events by conversation and run id", () => {
    let state = conversationMessageReducer(initialConversationMessageState, startRun("a", "run-a"));
    state = conversationMessageReducer(state, startRun("b", "run-b"));
    state = conversationMessageReducer(state, {
      type: "stream_event",
      conversationId: "a",
      runId: "run-a",
      event: { type: "answer_delta", delta: "Visible" },
    });
    const unchanged = conversationMessageReducer(state, {
      type: "stream_event",
      conversationId: "b",
      runId: "wrong-run",
      event: { type: "answer_delta", delta: "Leaked" },
    });

    expect(unchanged.byConversation.a.messages.at(-1)?.content).toBe("Visible");
    expect(unchanged.byConversation.b.messages.at(-1)?.content).toBe("");
  });

  it("keeps an explicit stopped state even when no answer text arrived", () => {
    let state = conversationMessageReducer(initialConversationMessageState, startRun("a", "run-a"));
    state = conversationMessageReducer(state, {
      type: "run_stopped",
      conversationId: "a",
      runId: "run-a",
    });

    const assistant = state.byConversation.a.messages.at(-1);
    expect(assistant?.content).toBe("");
    expect(assistant?.run_status).toBe("stopped");
    expect(state.byConversation.a.activeRunId).toBeUndefined();
  });

  it("preserves retry metadata on a retryable failure", () => {
    let state = conversationMessageReducer(
      initialConversationMessageState,
      startRun("a", "run-a", "Retry me"),
    );
    state = conversationMessageReducer(state, {
      type: "run_failed",
      conversationId: "a",
      runId: "run-a",
      error: { code: "ANSWER_STREAM_INCOMPLETE", message: "回答意外中断，请重试。", retryable: true },
    });

    const assistant = state.byConversation.a.messages.at(-1);
    expect(assistant?.run_status).toBe("failed");
    expect(assistant?.run_error?.retryable).toBe(true);
    expect(assistant?.retry_question).toBe("Retry me");
  });

  it("can invalidate an apparent completion after a duplicate terminal error", () => {
    let state = conversationMessageReducer(initialConversationMessageState, startRun("a", "run-a"));
    state = conversationMessageReducer(state, {
      type: "stream_event",
      conversationId: "a",
      runId: "run-a",
      event: {
        type: "done",
        response: {
          request_id: "request-a",
          mode: "direct",
          answer_kind: "insufficient_evidence",
          answer: "Answer",
          evidence: [],
          claims: [],
        },
      },
    });
    expect(state.byConversation.a.messages.at(-1)).toMatchObject({
      answer_kind: "insufficient_evidence",
      claims: [],
      request_id: "request-a",
      run_status: "completed",
    });
    state = conversationMessageReducer(state, {
      type: "run_failed",
      conversationId: "a",
      runId: "run-a",
      error: { code: "ANSWER_STREAM_MULTIPLE_TERMINALS", message: "Protocol error", retryable: true },
    });

    expect(state.byConversation.a.messages.at(-1)?.run_status).toBe("failed");
  });

  it("stores Stage 2 clarification metadata and the persisted assistant id", () => {
    let state = conversationMessageReducer(initialConversationMessageState, startRun("a", "run-a"));
    state = conversationMessageReducer(state, {
      type: "stream_event",
      conversationId: "a",
      runId: "run-a",
      event: {
        type: "done",
        response: {
          request_id: "request-a",
          assistant_message_id: "assistant-persisted",
          mode: "direct",
          contract_version: "v2",
          response_kind: "clarification",
          answer_kind: null,
          answer: "你更想改善激活还是留存？",
          evidence: [],
          claims: [],
          advice: [],
          clarification: {
            coaching_id: "coach-one",
            round: 1,
            prompt: "你更想改善激活还是留存？",
            suggestions: ["激活", "留存"],
            can_skip: true,
          },
        },
      },
    });

    expect(state.byConversation.a.messages.at(-1)).toMatchObject({
      contract_version: "v2",
      response_kind: "clarification",
      assistant_message_id: "assistant-persisted",
      clarification: { coaching_id: "coach-one", round: 1 },
      run_status: "completed",
    });
  });
});
