import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { api, streamAnswer } from "./api";
import type { ChatMessage, Citation, Conversation, HealthStatus, RagMode, SseEvent } from "./types";

export function readinessLabel(health: HealthStatus | null): string {
  if (!health) return "正在连接";
  if (health.full_ready) return "系统就绪";
  if (health.answer_ready) return health.semantic_ready && health.zhipu_ready
    ? "问答已就绪"
    : "问答已就绪 · 仅关键词检索";
  if (health.facts_ready && health.lexical_ready) return "本地知识库已就绪";
  return "需要完成配置";
}

const modeLabels: Record<RagMode, string> = {
  direct: "直接回答",
  agentic: "智能检索",
};

export function modeDescription(mode: RagMode): string {
  return mode === "direct"
    ? "自动结合关键词和语义检索，响应更快，适合目标明确的问题。"
    : "由智能体多轮搜索、阅读和筛选证据，覆盖更深入，但需要更长时间。";
}

const phaseLabels: Record<string, string> = {
  rewrite: "正在理解问题",
  retrieve: "正在检索本地知识库",
  resolve: "正在读取原始证据",
  inspect_images: "正在分析相关图片",
  agentic: "智能体正在选择和读取证据",
  answer: "正在根据证据生成回答",
};

export function phaseLabel(phase: string, message?: string): string {
  return phaseLabels[phase] || message || "正在处理";
}

function errorLabel(code: string, message: string): string {
  const labels: Record<string, string> = {
    ANSWER_SERVICE_UNAVAILABLE: "问答服务暂不可用，请稍后重试。",
    ANSWER_STREAM_INCOMPLETE: "回答意外中断，请重试。",
    DEEPSEEK_ANSWER_ERROR: "回答模型请求失败，请稍后重试。",
    DEEPSEEK_AGENTIC_ERROR: "智能检索模型请求失败，请稍后重试。",
    AGENTIC_READ_LIMIT: "智能检索读取的证据过多，请缩小问题范围后重试。",
    AGENTIC_ROUND_LIMIT: "智能检索步骤过多，请简化问题后重试。",
    SEMANTIC_PROVIDER_UNAVAILABLE: "语义检索服务暂不可用。",
  };
  return labels[code] || `请求失败：${message}`;
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
        {locator.ordinal !== undefined && <code>分块 {locator.ordinal}</code>}
      </summary>
      {heading && <p className="evidence-heading">{heading}</p>}
      {citation.text && <blockquote>{citation.text}</blockquote>}
      {locator.enhanced_path && <p className="source-path">{locator.enhanced_path}</p>}
      {images.map((image) => (
        <figure key={image.image_id}>
          {image.url && <img src={image.url} alt={image.caption || "来源图片"} loading="lazy" />}
          <figcaption>
            {image.caption && <span>{image.caption}</span>}
            {image.observation && <span>{image.observation}</span>}
            {image.inspected === false && (
              <span>图片分析失败，仅显示原始说明{image.error ? `（${image.error}）` : ""}。</span>
            )}
            {image.inspected === undefined && <span>该图片未进行自动分析，仅显示原始说明。</span>}
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
        <span>{message.role === "user" ? "你" : "Compass"}</span>
        {message.mode && <span>{modeLabels[message.mode]}</span>}
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
      .catch((reason: Error) => setError(`加载失败：${reason.message}`));
    const timer = window.setInterval(() => api.health().then(setHealth).catch(() => undefined), 15_000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!activeId) {
      setMessages([]);
      return;
    }
    api.messages(activeId).then(setMessages).catch((reason: Error) => setError(`加载对话失败：${reason.message}`));
  }, [activeId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, phase]);

  const agenticAvailable = health?.agentic_enabled !== false;
  useEffect(() => {
    if (!agenticAvailable && mode === "agentic") setMode("direct");
  }, [agenticAvailable, mode]);

  const statusItems = useMemo(
    () => [
      ["知识库", health?.facts_ready],
      ["关键词", health?.lexical_ready],
      ["语义检索", health?.semantic_ready],
      ["回答模型", health?.deepseek_ready],
      ["视觉模型", health?.zhipu_ready],
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
    setPhase("正在准备问题");
    const controller = new AbortController();
    abortRef.current = controller;

    const handleEvent = (streamEvent: SseEvent) => {
      if (streamEvent.type === "phase") setPhase(phaseLabel(streamEvent.phase, streamEvent.message));
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
      if (streamEvent.type === "error") {
        throw new Error(errorLabel(streamEvent.error.code, streamEvent.error.message));
      }
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
          <div><strong>Lenny Compass</strong><span>私有产品知识库</span></div>
        </div>
        <button className="new-chat" onClick={() => void createConversation()}>+ 新建对话</button>
        <nav className="conversation-list" aria-label="对话列表">
          {conversations.map((conversation) => (
            <div className={`conversation-row ${conversation.id === activeId ? "active" : ""}`} key={conversation.id}>
              <button onClick={() => setActiveId(conversation.id)}>{conversation.title}</button>
              <button className="delete" aria-label="删除对话" onClick={() => void removeConversation(conversation.id)}>×</button>
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
            <h1>{conversations.find((item) => item.id === activeId)?.title || "向知识库提问"}</h1>
            <p>{health?.document_count ?? 638} 篇私有文档 · 基于原文证据回答</p>
          </div>
          <div className="mode-control">
            <div className="mode-switch" aria-label="检索模式">
              <button className={mode === "direct" ? "active" : ""} onClick={() => setMode("direct")}>直接回答</button>
              <button
                className={mode === "agentic" ? "active" : ""}
                onClick={() => setMode("agentic")}
                disabled={!agenticAvailable}
                title={agenticAvailable ? "由智能体自主检索并选择证据范围" : "问答服务就绪后才能使用智能检索"}
              >智能检索</button>
            </div>
            <p className="mode-description" aria-live="polite">{modeDescription(mode)}</p>
          </div>
        </header>

        {error && <div className="error-banner"><span>{error}</span><button onClick={() => setError("")}>关闭</button></div>}
        <div className="message-list">
          {!messages.length && (
            <section className="empty-state">
              <span className="eyebrow">本地 · 私有 · 可追溯</span>
              <h2>从知识库中找到证据，<br />而不只是搜索结果。</h2>
              <p>使用中文提问。每个回答都会回溯到经过版本管理的原文证据。</p>
              <div className="suggestions">
                {["优秀的产品团队如何选择北极星指标？", "总结 Lenny 关于产品市场契合度的核心判断", "Eric Ries 如何理解持续创新？"].map((item) => (
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
            placeholder="请输入你想向知识库提问的问题…"
            rows={2}
          />
          {streaming ? (
            <button type="button" className="send stop" onClick={() => abortRef.current?.abort()}>停止</button>
          ) : (
            <button type="submit" className="send" disabled={!question.trim()}>提问</button>
          )}
        </form>
      </section>
    </main>
  );
}
