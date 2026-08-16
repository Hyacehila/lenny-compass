import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { api, streamAnswer } from "./api";
import type { ChatMessage, Citation, Conversation, HealthStatus, RagMode, SseEvent } from "./types";

function readinessLabel(health: HealthStatus | null): string {
  if (!health) return "Connecting";
  if (health.full_ready) return "Ready";
  if (health.answer_ready) return health.semantic_ready && health.zhipu_ready
    ? "Answer ready"
    : "Answer ready · BM25 only";
  if (health.facts_ready && health.lexical_ready) return "Local facts ready";
  return "Setup required";
}

type CitationImage = NonNullable<Citation["image_observations"]>[number];

export function mergeCitationImages(citation: Citation): CitationImage[] {
  const images: CitationImage[] = (citation.locator.images ?? []).map((image) => {
    const observation = citation.image_observations?.find((item) =>
      item.image_id === image.image_id || Boolean(item.url && item.url === (image.url || image.path)),
    );
    return {
      ...observation,
      image_id: image.image_id,
      url: observation?.url ?? image.url ?? image.path,
      caption: observation?.caption ?? image.caption,
    };
  });
  for (const observation of citation.image_observations ?? []) {
    if (!images.some((image) =>
      image.image_id === observation.image_id || Boolean(image.url && image.url === observation.url),
    )) {
      images.push(observation);
    }
  }
  return images;
}

function EvidenceCard({ citation }: { citation: Citation }) {
  const heading = citation.headings?.join(" › ");
  const locator = citation.locator;
  const images = mergeCitationImages(citation);
  return (
    <details className="evidence-card">
      <summary>
        <span>{citation.title || locator.document_id || locator.path || citation.evidence_id}</span>
        {locator.ordinal !== undefined && <code>chunk {locator.ordinal}</code>}
      </summary>
      {heading && <p className="evidence-heading">{heading}</p>}
      {citation.text && <blockquote>{citation.text}</blockquote>}
      {locator.enhanced_path && <p className="source-path">{locator.enhanced_path}</p>}
      {images.map((image) => (
        <figure key={image.image_id}>
          {image.url && <img src={image.url} alt={image.caption || "Source illustration"} loading="lazy" />}
          <figcaption>
            {image.caption && <span>{image.caption}</span>}
            {image.observation && <span>{image.observation}</span>}
            {image.inspected === false && (
              <span>Image could not be inspected; caption only{image.error ? ` (${image.error})` : ""}.</span>
            )}
            {image.inspected === undefined && <span>Image was not selected for automatic inspection; caption only.</span>}
          </figcaption>
        </figure>
      ))}
    </details>
  );
}

function Message({ message }: { message: ChatMessage }) {
  return (
    <article className={`message ${message.role}`}>
      <div className="message-meta">
        <span>{message.role === "user" ? "You" : "Compass"}</span>
        {message.mode && <span>{message.mode}</span>}
      </div>
      <div className="message-body">
        {message.role === "assistant" ? <ReactMarkdown>{message.content}</ReactMarkdown> : message.content}
      </div>
      {!!message.citations?.length && (
        <div className="evidence-list">
          {message.citations.map((citation) => (
            <EvidenceCard key={citation.evidence_id} citation={citation} />
          ))}
        </div>
      )}
    </article>
  );
}

export default function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [mode, setMode] = useState<RagMode>("direct");
  const [question, setQuestion] = useState("");
  const [phase, setPhase] = useState("");
  const [error, setError] = useState("");
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const refreshConversations = useCallback(async () => {
    const rows = await api.conversations();
    setConversations(rows);
    if (!activeId && rows.length) setActiveId(rows[0].id);
    return rows;
  }, [activeId]);

  useEffect(() => {
    Promise.all([api.health(), api.conversations()])
      .then(([status, rows]) => {
        setHealth(status);
        setConversations(rows);
        if (rows.length) setActiveId(rows[0].id);
      })
      .catch((reason: Error) => setError(reason.message));
    const timer = window.setInterval(() => api.health().then(setHealth).catch(() => undefined), 15_000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!activeId) {
      setMessages([]);
      return;
    }
    api.messages(activeId).then(setMessages).catch((reason: Error) => setError(reason.message));
  }, [activeId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, phase]);

  const agenticAvailable = health?.agentic_enabled === true;
  useEffect(() => {
    if (!agenticAvailable && mode === "agentic") setMode("direct");
  }, [agenticAvailable, mode]);

  const statusItems = useMemo(
    () => [
      ["Facts", health?.facts_ready],
      ["BM25", health?.lexical_ready],
      ["Vector", health?.semantic_ready],
      ["DeepSeek", health?.deepseek_ready],
      ["Vision", health?.zhipu_ready],
    ] as const,
    [health],
  );

  async function createConversation() {
    const created = await api.createConversation();
    setConversations((current) => [created, ...current]);
    setActiveId(created.id);
    setMessages([]);
  }

  async function removeConversation(id: string) {
    await api.deleteConversation(id);
    const remaining = conversations.filter((item) => item.id !== id);
    setConversations(remaining);
    if (activeId === id) setActiveId(remaining[0]?.id ?? null);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || streaming) return;
    setError("");
    let conversationId = activeId;
    if (!conversationId) {
      const created = await api.createConversation();
      setConversations((current) => [created, ...current]);
      setActiveId(created.id);
      conversationId = created.id;
    }
    const now = new Date().toISOString();
    const userMessage: ChatMessage = { id: `local-user-${now}`, role: "user", content: text, mode, created_at: now };
    const assistantMessage: ChatMessage = {
      id: `local-assistant-${now}`,
      role: "assistant",
      content: "",
      mode,
      created_at: now,
      citations: [],
    };
    setMessages((current) => [...current, userMessage, assistantMessage]);
    setQuestion("");
    setStreaming(true);
    setPhase("Preparing query");
    const controller = new AbortController();
    abortRef.current = controller;

    const handleEvent = (streamEvent: SseEvent) => {
      if (streamEvent.type === "phase") setPhase(streamEvent.message || streamEvent.phase);
      if (streamEvent.type === "answer_delta") {
        setMessages((current) => current.map((item) =>
          item.id === assistantMessage.id ? { ...item, content: item.content + streamEvent.delta } : item,
        ));
      }
      if (streamEvent.type === "evidence") {
        setMessages((current) => current.map((item) =>
          item.id === assistantMessage.id
            ? { ...item, citations: [...(item.citations ?? []), streamEvent.evidence] }
            : item,
        ));
      }
      if (streamEvent.type === "done") {
        setPhase("");
        setMessages((current) => current.map((item) =>
          item.id === assistantMessage.id
            ? {
                ...item,
                id: streamEvent.response.request_id,
                content: item.content || streamEvent.response.answer,
                citations: streamEvent.response.evidence,
              }
            : item,
        ));
      }
      if (streamEvent.type === "error") throw new Error(streamEvent.error.message);
    };

    try {
      await streamAnswer(conversationId, text, mode, handleEvent, controller.signal);
      await refreshConversations();
    } catch (reason) {
      if ((reason as Error).name !== "AbortError") {
        setError((reason as Error).message);
        setMessages((current) => current.filter((item) =>
          item.id !== assistantMessage.id || Boolean(item.content || item.citations?.length),
        ));
      }
    } finally {
      setStreaming(false);
      setPhase("");
      abortRef.current = null;
    }
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">LC</div>
          <div><strong>Lenny Compass</strong><span>Private knowledge workspace</span></div>
        </div>
        <button className="new-chat" onClick={() => void createConversation()}>+ New conversation</button>
        <nav className="conversation-list" aria-label="Conversations">
          {conversations.map((conversation) => (
            <div className={`conversation-row ${conversation.id === activeId ? "active" : ""}`} key={conversation.id}>
              <button onClick={() => setActiveId(conversation.id)}>{conversation.title}</button>
              <button className="delete" aria-label="Delete conversation" onClick={() => void removeConversation(conversation.id)}>×</button>
            </div>
          ))}
        </nav>
        <section className="readiness">
          <div className="readiness-title"><span className={`status-dot ${health?.status === "ready" ? "ready" : ""}`} />{readinessLabel(health)}</div>
          <div className="readiness-grid">
            {statusItems.map(([label, ready]) => <span className={ready ? "ready" : ""} key={label}>{label}</span>)}
          </div>
        </section>
      </aside>

      <section className="workspace">
        <header className="toolbar">
          <div>
            <h1>{conversations.find((item) => item.id === activeId)?.title || "Ask the archive"}</h1>
            <p>{health?.document_count ?? 638} private documents · fact-grounded answers</p>
          </div>
          <div className="mode-switch" aria-label="Retrieval mode">
            <button className={mode === "direct" ? "active" : ""} onClick={() => setMode("direct")}>Direct</button>
            <button
              className={mode === "agentic" ? "active" : ""}
              onClick={() => setMode("agentic")}
              disabled={!agenticAvailable}
              title={agenticAvailable ? "Agent chooses searches and evidence windows" : "Enable AGENTIC_RAG_ENABLED in the backend"}
            >Agentic</button>
          </div>
        </header>

        {error && <div className="error-banner"><span>{error}</span><button onClick={() => setError("")}>Dismiss</button></div>}
        <div className="message-list">
          {!messages.length && (
            <section className="empty-state">
              <span className="eyebrow">LOCAL · PRIVATE · TRACEABLE</span>
              <h2>Turn the archive into evidence,<br />not just search results.</h2>
              <p>Ask in Chinese or English. Every answer resolves retrieval hits back to versioned Docling facts.</p>
              <div className="suggestions">
                {["How do great product teams choose a north-star metric?", "总结 Lenny 关于产品市场契合度的核心判断", "What did Eric Ries say about continuous innovation?"].map((item) => (
                  <button key={item} onClick={() => setQuestion(item)}>{item}</button>
                ))}
              </div>
            </section>
          )}
          {messages.map((message) => <Message key={message.id} message={message} />)}
          {phase && <div className="phase"><span /><span /><span />{phase}</div>}
          <div ref={bottomRef} />
        </div>

        <form className="composer" onSubmit={(event) => void submit(event)}>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            placeholder="Ask a question grounded in the archive…"
            rows={2}
          />
          {streaming ? (
            <button type="button" className="send stop" onClick={() => abortRef.current?.abort()}>Stop</button>
          ) : (
            <button type="submit" className="send" disabled={!question.trim()}>Ask</button>
          )}
          <p>Direct uses BM25 + semantic RRF. Answers cite canonical fact chunks; model reasoning is never displayed.</p>
        </form>
      </section>
    </main>
  );
}
