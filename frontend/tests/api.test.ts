import { afterEach, describe, expect, it, vi } from "vitest";
import { api, streamAnswer } from "../src/api";
import type { DocumentChunkLocator } from "../src/types";

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

describe("workspace API", () => {
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
