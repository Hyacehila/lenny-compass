import { afterEach, describe, expect, it, vi } from "vitest";
import { streamAnswer } from "../src/api";

afterEach(() => vi.unstubAllGlobals());

describe("streamAnswer", () => {
  it("parses only public SSE events without requiring an event field", async () => {
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('data: {"type":"phase","phase":"retrieval"}\n\n'));
        controller.enqueue(encoder.encode('data: {"type":"answer_delta","delta":"hello"}\n\n'));
        controller.enqueue(encoder.encode('data: {"type":"done","response":{"request_id":"r1","mode":"direct","answer":"hello","evidence":[]}}\n\n'));
        controller.close();
      },
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(body, { status: 200 })));
    const events: string[] = [];

    await streamAnswer("conversation", "question", "direct", (event) => events.push(event.type));

    expect(events).toEqual(["phase", "answer_delta", "done"]);
    expect(fetch).toHaveBeenCalledWith(
      "/api/v1/conversations/conversation/ask",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("surfaces non-success response bodies", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("facts not ready", { status: 503 })));

    await expect(streamAnswer("conversation", "question", "direct", () => undefined)).rejects.toThrow(
      "facts not ready",
    );
  });
});
