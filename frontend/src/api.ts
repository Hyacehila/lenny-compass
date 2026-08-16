import type { ChatMessage, Conversation, HealthStatus, RagMode, SseEvent } from "./types";

const API_ROOT = "/api/v1";

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

export const api = {
  health: () => jsonRequest<HealthStatus>("/health"),
  conversations: () => jsonRequest<Conversation[]>("/conversations"),
  createConversation: () =>
    jsonRequest<Conversation>("/conversations", {
      method: "POST",
      body: JSON.stringify({ title: "New conversation" }),
    }),
  deleteConversation: (id: string) =>
    jsonRequest<void>(`/conversations/${encodeURIComponent(id)}`, { method: "DELETE" }),
  messages: (id: string) =>
    jsonRequest<ChatMessage[]>(`/conversations/${encodeURIComponent(id)}/messages`),
};

export async function streamAnswer(
  conversationId: string,
  question: string,
  mode: RagMode,
  onEvent: (event: SseEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(
    `${API_ROOT}/conversations/${encodeURIComponent(conversationId)}/ask`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ question, mode }),
      signal,
    },
  );
  if (!response.ok || !response.body) {
    throw new Error((await response.text()) || `${response.status} ${response.statusText}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const payload = frame
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (payload) onEvent(JSON.parse(payload) as SseEvent);
    }
    if (done) break;
  }
}
