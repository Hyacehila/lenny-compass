import { afterEach, describe, expect, it, vi } from "vitest";
import { AnswerStreamError, api, streamAnswer } from "../src/api";
import type { DocumentChunkLocator } from "../src/types";

afterEach(() => vi.unstubAllGlobals());

const RUN_ID = "local-run-1";
const DONE_FRAME = 'data: {"type":"done","response":{"request_id":"r1","mode":"direct","answer_kind":"insufficient_evidence","answer":"hello","evidence":[],"claims":[]}}\n\n';

describe("streamAnswer", () => {
  it("parses only public SSE events without requiring an event field", async () => {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('data: {"type":"phase","phase":"retrieval"}\n\n'));
        controller.enqueue(encoder.encode('data: {"type":"answer_delta","delta":"hello"}\n\n'));
        controller.enqueue(encoder.encode(DONE_FRAME));
        controller.close();
      },
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body, { status: 200 })));
    const events: string[] = [];

    await streamAnswer(
      "conversation",
      "question",
      "direct",
      RUN_ID,
      (event) => events.push(event.type),
    );

    expect(events).toEqual(["phase", "answer_delta", "done"]);
    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/conversations/conversation/ask",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          question: "question",
          mode: "direct",
          idempotency_key: RUN_ID,
        }),
      }),
    );
  });

  it("surfaces non-success response bodies", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("facts not ready", { status: 503 })));

    await expect(
      streamAnswer("conversation", "question", "direct", RUN_ID, () => undefined),
    ).rejects.toThrow("facts not ready");
  });

  it("rejects a clean EOF when no done event was received", async () => {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('data: {"type":"answer_delta","delta":"partial"}\n\n'));
        controller.close();
      },
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body, { status: 200 })));

    await expect(
      streamAnswer("conversation", "question", "direct", RUN_ID, () => undefined),
    ).rejects.toMatchObject({
      name: "AnswerStreamError",
      code: "ANSWER_STREAM_INCOMPLETE",
      retryable: true,
    } satisfies Partial<AnswerStreamError>);
  });

  it("rejects duplicate terminal events", async () => {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(DONE_FRAME + DONE_FRAME));
        controller.close();
      },
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body, { status: 200 })));

    await expect(
      streamAnswer("conversation", "question", "direct", RUN_ID, () => undefined),
    ).rejects.toMatchObject({
      code: "ANSWER_STREAM_MULTIPLE_TERMINALS",
      retryable: true,
    });
  });

  it("preserves retryability from an SSE error terminal", async () => {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(
          'data: {"type":"error","error":{"code":"TEMPORARY","message":"try again","retryable":true}}\n\n',
        ));
        controller.close();
      },
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body, { status: 200 })));

    await expect(
      streamAnswer("conversation", "question", "direct", RUN_ID, () => undefined),
    ).rejects.toMatchObject({
      code: "TEMPORARY",
      message: "try again",
      retryable: true,
    });
  });

  it("accepts a Stage 2 clarification and sends its reply context", async () => {
    const encoder = new TextEncoder();
    const clarification = {
      type: "done",
      response: {
        request_id: "r2",
        assistant_message_id: "assistant-2",
        mode: "direct",
        contract_version: "v2",
        response_kind: "clarification",
        answer_kind: null,
        answer: "你更想改善激活还是留存？",
        evidence: [],
        claims: [],
        advice: [],
        user_premises: [],
        clarification: {
          coaching_id: "coach-1",
          round: 1,
          prompt: "你更想改善激活还是留存？",
          suggestions: ["激活", "留存"],
          can_skip: true,
        },
        verification: null,
        insufficient_reason: null,
      },
    };
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(clarification)}\n\n`));
        controller.close();
      },
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body, { status: 200 })));

    const response = await streamAnswer(
      "conversation",
      "留存",
      "direct",
      RUN_ID,
      () => undefined,
      undefined,
      { coachingId: "coach-1", replyToMessageId: "assistant-1", skipClarification: false },
    );

    expect(response.response_kind).toBe("clarification");
    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/conversations/conversation/ask",
      expect.objectContaining({
        body: JSON.stringify({
          question: "留存",
          mode: "direct",
          idempotency_key: RUN_ID,
          coaching_id: "coach-1",
          reply_to_message_id: "assistant-1",
          skip_clarification: false,
        }),
      }),
    );
  });

  it("rejects a malformed Stage 2 grounded answer without verification", async () => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(
          'data: {"type":"done","response":{"request_id":"r2","mode":"direct","contract_version":"v2","response_kind":"grounded_answer","answer_kind":"grounded","answer":"claim","evidence":[{}],"claims":[{"claim_id":"c1","text":"claim","role":"core","evidence_ids":["e1"]}],"advice":[]}}\n\n',
        ));
        controller.close();
      },
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body, { status: 200 })));

    await expect(
      streamAnswer("conversation", "question", "direct", RUN_ID, () => undefined),
    ).rejects.toMatchObject({ code: "ANSWER_STREAM_INVALID_EVENT" });
  });

  it("rejects a done event that omits required response fields", async () => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(
          'data: {"type":"done","response":{"request_id":"r1","mode":"direct","answer":"hello","evidence":[]}}\n\n',
        ));
        controller.close();
      },
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body, { status: 200 })));

    await expect(
      streamAnswer("conversation", "question", "direct", RUN_ID, () => undefined),
    ).rejects.toMatchObject({
      code: "ANSWER_STREAM_INVALID_EVENT",
      retryable: true,
    });
  });

  it("rejects an error event with an invalid retryable field", async () => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(
          'data: {"type":"error","error":{"code":"TEMPORARY","message":"try again","retryable":"yes"}}\n\n',
        ));
        controller.close();
      },
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body, { status: 200 })));

    await expect(
      streamAnswer("conversation", "question", "direct", RUN_ID, () => undefined),
    ).rejects.toMatchObject({
      code: "ANSWER_STREAM_INVALID_EVENT",
      retryable: true,
    });
  });
});

describe("workspace API", () => {
  it("restores Stage 2 fields from persisted response payloads", async () => {
    const stored = [{
      id: "assistant-1",
      conversation_id: "one",
      role: "assistant",
      content: "结论与建议",
      created_at: "2026-09-01T00:00:00Z",
      contract_version: "v2",
      response_kind: "grounded_answer",
      answer_kind: "grounded",
      citations: [],
      claims: [],
      response_payload: {
        advice: [{
          advice_id: "advice-1",
          text: "先确认用户价值。",
          basis: "evidence_synthesis",
          rationale_claim_ids: ["claim-1"],
        }],
        user_premises: [],
        clarification: null,
        verification: {
          policy_version: "claim-support-minimal-v1",
          checked_claim_count: 1,
          omitted_claim_count: 0,
          calibration_level: "uncalibrated",
        },
        insufficient_reason: null,
      },
    }];
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify(stored), { status: 200 }),
    ));

    const [message] = await api.messages("one");

    expect(message.advice?.[0]?.text).toBe("先确认用户价值。");
    expect(message.verification?.checked_claim_count).toBe(1);
    expect("response_payload" in message).toBe(false);
  });

  it("renames conversations and loads system status", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: "one", title: "新标题" }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ build_id: "build-one" }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await api.renameConversation("one", "新标题");
    await api.indexStatus();

    expect(fetchMock).toHaveBeenNthCalledWith(1, "/api/v1/conversations/one", expect.objectContaining({
      method: "PATCH",
      body: JSON.stringify({ title: "新标题" }),
    }));
    expect(fetchMock).toHaveBeenNthCalledWith(2, "/api/v1/index/status", expect.any(Object));
  });

  it("reads a fixed two-chunk source context window", async () => {
    const locator = { kind: "document_chunk", document_id: "doc", chunk_id: "chunk" } as DocumentChunkLocator;
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ evidence: [] }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await api.readSourceWindow(locator);

    expect(fetchMock).toHaveBeenCalledWith("/api/v1/facts/read-source-window", expect.objectContaining({
      method: "POST",
      body: JSON.stringify({ locator, window: 2 }),
    }));
  });
});
