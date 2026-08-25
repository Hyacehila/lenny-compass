import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, streamAnswer } from "./api";
import { applyTraceEvent, emptyProcessTrace } from "./citations";
import {
  Composer,
  EmptyState,
  MessageThread,
  Sidebar,
  UtilityPanel,
  WorkspaceHeader,
} from "./components";
import { errorLabel, modeDescription, phaseLabel, readinessLabel } from "./labels";
import type {
  ChatMessage,
  Conversation,
  HealthStatus,
  IndexStatus,
  RagMode,
  SseEvent,
  UtilityTab,
} from "./types";

export { mergeCitationImages } from "./components";
export { modeDescription, phaseLabel, readinessLabel } from "./labels";

function isMobileViewport(): boolean {
  return window.matchMedia("(max-width: 767px)").matches;
}

function findLastAssistant(messages: ChatMessage[]): ChatMessage | null {
  return [...messages].reverse().find((message) => message.role === "assistant") ?? null;
}

export default function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null);
  const [diagnosticsError, setDiagnosticsError] = useState("");
  const [diagnosticsLoading, setDiagnosticsLoading] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [mode, setMode] = useState<RagMode>("direct");
  const [question, setQuestion] = useState("");
  const [phase, setPhase] = useState("");
  const [error, setError] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [utilityOpen, setUtilityOpen] = useState(false);
  const [utilityTab, setUtilityTab] = useState<UtilityTab>("evidence");
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null);
  const [selectedCitationId, setSelectedCitationId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const messageListRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const followTailRef = useRef(true);

  const refreshConversations = useCallback(async () => {
    const rows = await api.conversations();
    setConversations(rows);
    return rows;
  }, []);

  const refreshDiagnostics = useCallback(async () => {
    setDiagnosticsLoading(true);
    setDiagnosticsError("");
    const [healthResult, indexResult] = await Promise.allSettled([api.health(), api.indexStatus()]);
    if (healthResult.status === "fulfilled") setHealth(healthResult.value);
    if (indexResult.status === "fulfilled") setIndexStatus(indexResult.value);
    if (healthResult.status === "rejected" || indexResult.status === "rejected") {
      setDiagnosticsError("部分系统状态暂时无法读取，已保留其余可用信息。");
    }
    setDiagnosticsLoading(false);
  }, []);

  useEffect(() => {
    let cancelled = false;
    Promise.all([api.health(), api.conversations()])
      .then(([status, rows]) => {
        if (cancelled) return;
        setHealth(status);
        setConversations(rows);
        if (rows.length) setActiveId(rows[0].id);
      })
      .catch((reason: Error) => { if (!cancelled) setError(`加载失败：${reason.message}`); });
    const timer = window.setInterval(() => api.health().then(setHealth).catch(() => undefined), 15_000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, []);

  useEffect(() => {
    if (!activeId) {
      setMessages([]);
      setSelectedMessageId(null);
      setSelectedCitationId(null);
      setUtilityOpen(false);
      return;
    }
    let cancelled = false;
    api.messages(activeId)
      .then((rows) => {
        if (cancelled) return;
        setMessages(rows);
        const assistant = findLastAssistant(rows);
        setSelectedMessageId(assistant?.id ?? null);
        setSelectedCitationId(assistant?.citations?.[0]?.evidence_id ?? null);
        setUtilityTab("evidence");
        setUtilityOpen(Boolean(assistant?.citations?.length) && !isMobileViewport());
        followTailRef.current = false;
        const lastUser = [...rows].reverse().find((message) => message.role === "user");
        window.requestAnimationFrame(() => window.requestAnimationFrame(() => {
          const container = messageListRef.current;
          if (!container || !lastUser) return;
          const target = [...container.querySelectorAll<HTMLElement>("[data-message-id]")]
            .find((element) => element.dataset.messageId === lastUser.id);
          target?.scrollIntoView({ block: "start" });
        }));
      })
      .catch((reason: Error) => { if (!cancelled) setError(`加载对话失败：${reason.message}`); });
    return () => { cancelled = true; };
  }, [activeId]);

  useEffect(() => {
    if (!(utilityOpen && utilityTab === "system")) return;
    void refreshDiagnostics();
    const timer = window.setInterval(() => void refreshDiagnostics(), 15_000);
    return () => window.clearInterval(timer);
  }, [refreshDiagnostics, utilityOpen, utilityTab]);

  const agenticAvailable = health?.agentic_enabled !== false;
  useEffect(() => {
    if (!agenticAvailable && mode === "agentic") setMode("direct");
  }, [agenticAvailable, mode]);

  const selectedMessage = useMemo(() => {
    const selected = messages.find((message) => message.id === selectedMessageId && message.role === "assistant");
    return selected ?? findLastAssistant(messages);
  }, [messages, selectedMessageId]);

  function scheduleTailFollow() {
    window.requestAnimationFrame(() => {
      const container = messageListRef.current;
      if (container && followTailRef.current) container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
    });
  }

  function handleThreadScroll() {
    const container = messageListRef.current;
    if (!container) return;
    followTailRef.current = container.scrollHeight - container.scrollTop - container.clientHeight < 140;
  }

  async function createConversation() {
    setError("");
    try {
      const created = await api.createConversation();
      setConversations((current) => [created, ...current]);
      setActiveId(created.id);
      setMessages([]);
      setSidebarOpen(false);
      window.requestAnimationFrame(() => textareaRef.current?.focus());
    } catch (reason) {
      setError(`新建会话失败：${(reason as Error).message}`);
    }
  }

  async function renameConversation(id: string, title: string) {
    setError("");
    try {
      const renamed = await api.renameConversation(id, title);
      setConversations((current) => current.map((item) => item.id === id ? renamed : item));
    } catch (reason) {
      setError(`重命名失败：${(reason as Error).message}`);
      throw reason;
    }
  }

  async function removeConversation(id: string) {
    setError("");
    try {
      await api.deleteConversation(id);
      const remaining = conversations.filter((item) => item.id !== id);
      setConversations(remaining);
      if (activeId === id) setActiveId(remaining[0]?.id ?? null);
    } catch (reason) {
      setError(`删除失败：${(reason as Error).message}`);
      throw reason;
    }
  }

  function selectMessage(messageId: string) {
    const message = messages.find((item) => item.id === messageId);
    setSelectedMessageId(messageId);
    setSelectedCitationId(message?.citations?.[0]?.evidence_id ?? null);
  }

  function selectCitation(messageId: string, evidenceId: string) {
    setSelectedMessageId(messageId);
    setSelectedCitationId(evidenceId);
    setUtilityTab("evidence");
    setUtilityOpen(true);
  }

  function changeUtilityTab(tab: UtilityTab) {
    setUtilityTab(tab);
    setUtilityOpen(true);
  }

  function chooseSuggestion(value: string) {
    setQuestion(value);
    window.requestAnimationFrame(() => textareaRef.current?.focus());
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || streaming) return;
    setError("");
    let conversationId = activeId;
    if (!conversationId) {
      try {
        const created = await api.createConversation();
        setConversations((current) => [created, ...current]);
        setActiveId(created.id);
        conversationId = created.id;
      } catch (reason) {
        setError(`新建会话失败：${(reason as Error).message}`);
        return;
      }
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
      process_trace: emptyProcessTrace(),
    };
    setMessages((current) => [...current, userMessage, assistantMessage]);
    setSelectedMessageId(assistantMessage.id);
    setSelectedCitationId(null);
    setQuestion("");
    setStreaming(true);
    setPhase("正在准备问题");
    setUtilityTab("process");
    setUtilityOpen(!isMobileViewport());
    followTailRef.current = true;
    window.requestAnimationFrame(() => window.requestAnimationFrame(() => {
      const target = [...(messageListRef.current?.querySelectorAll<HTMLElement>("[data-message-id]") ?? [])]
        .find((element) => element.dataset.messageId === userMessage.id);
      target?.scrollIntoView({ block: "start" });
    }));
    const controller = new AbortController();
    abortRef.current = controller;

    const handleEvent = (streamEvent: SseEvent) => {
      if (streamEvent.type === "error") throw new Error(errorLabel(streamEvent.error.code, streamEvent.error.message));
      if (streamEvent.type === "phase") setPhase(phaseLabel(streamEvent.phase, streamEvent.message));
      setMessages((current) => current.map((item) => {
        if (item.id !== assistantMessage.id) return item;
        const processTrace = applyTraceEvent(item.process_trace ?? emptyProcessTrace(), streamEvent);
        if (streamEvent.type === "answer_delta") return { ...item, content: item.content + streamEvent.delta, process_trace: processTrace };
        if (streamEvent.type === "evidence") return { ...item, citations: [...(item.citations ?? []), streamEvent.evidence], process_trace: processTrace };
        if (streamEvent.type === "done") return { ...item, id: streamEvent.response.request_id, content: item.content || streamEvent.response.answer, citations: streamEvent.response.evidence, process_trace: processTrace };
        return { ...item, process_trace: processTrace };
      }));
      if (streamEvent.type === "evidence") {
        setSelectedCitationId((current) => current ?? streamEvent.evidence.evidence_id);
      }
      if (streamEvent.type === "done") {
        setPhase("");
        setSelectedMessageId(streamEvent.response.request_id);
        setSelectedCitationId((current) => current ?? streamEvent.response.evidence[0]?.evidence_id ?? null);
      }
      scheduleTailFollow();
    };

    try {
      await streamAnswer(conversationId, text, mode, handleEvent, controller.signal);
      await refreshConversations();
    } catch (reason) {
      if ((reason as Error).name !== "AbortError") {
        setError((reason as Error).message);
        setMessages((current) => current.filter((item) => item.id !== assistantMessage.id || Boolean(item.content || item.citations?.length)));
      }
    } finally {
      setStreaming(false);
      setPhase("");
      abortRef.current = null;
    }
  }

  const activeConversation = conversations.find((item) => item.id === activeId);
  const hasMessages = messages.length > 0;
  const title = hasMessages ? activeConversation?.title || "研究对话" : "向知识库提问";

  return (
    <main className={`app-shell ${sidebarCollapsed ? "sidebar-collapsed" : ""} ${utilityOpen ? "utility-open" : ""}`}>
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        health={health}
        open={sidebarOpen}
        collapsed={sidebarCollapsed}
        onClose={() => setSidebarOpen(false)}
        onToggleCollapse={() => setSidebarCollapsed((current) => !current)}
        onOpenSystem={() => { setUtilityTab("system"); setUtilityOpen(true); setSidebarOpen(false); }}
        onSelect={setActiveId}
        onCreate={createConversation}
        onRename={renameConversation}
        onDelete={removeConversation}
      />
      <section className="workspace">
        <WorkspaceHeader title={title} documentCount={health?.document_count ?? 638} mode={mode} agenticAvailable={agenticAvailable} onModeChange={setMode} onOpenSidebar={() => setSidebarOpen(true)} />
        {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => setError("")}>关闭</button></div>}
        {hasMessages ? (
          <>
            <MessageThread messages={messages} selectedMessageId={selectedMessage?.id ?? null} selectedCitationId={selectedCitationId} phase={phase} containerRef={messageListRef} onScroll={handleThreadScroll} onSelectMessage={selectMessage} onSelectCitation={selectCitation} />
            <Composer question={question} streaming={streaming} textareaRef={textareaRef} onQuestionChange={setQuestion} onSubmit={(event) => void submit(event)} onStop={() => abortRef.current?.abort()} />
          </>
        ) : (
          <EmptyState question={question} streaming={streaming} textareaRef={textareaRef} onQuestionChange={setQuestion} onSubmit={(event) => void submit(event)} onStop={() => abortRef.current?.abort()} onSuggestion={chooseSuggestion} />
        )}
        {!utilityOpen && hasMessages && <button className="utility-trigger" aria-label="打开研究工具" onClick={() => setUtilityOpen(true)}><span>研究工具</span><span aria-hidden="true">‹</span></button>}
      </section>
      <UtilityPanel open={utilityOpen} tab={utilityTab} message={selectedMessage} selectedCitationId={selectedCitationId} health={health} indexStatus={indexStatus} diagnosticsLoading={diagnosticsLoading} diagnosticsError={diagnosticsError} streaming={streaming && selectedMessage?.id.startsWith("local-assistant-") === true} onClose={() => setUtilityOpen(false)} onTabChange={changeUtilityTab} onSelectCitation={setSelectedCitationId} onRefreshDiagnostics={() => void refreshDiagnostics()} />
    </main>
  );
}
