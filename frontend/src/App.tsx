import { FormEvent, useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { AnswerStreamError, api, streamAnswer } from "./api";
import {
  conversationMessageReducer,
  createOptimisticMessages,
  initialConversationMessageState,
  type ConversationMessageEntry,
} from "./conversationState";
import {
  Composer,
  EmptyState,
  GuidePage,
  MessageThread,
  Sidebar,
  UtilityPanel,
  WorkspaceHeader,
} from "./components";
import { errorLabel, modeDescription, phaseLabel, readinessLabel } from "./labels";
import type {
  AnswerRunError,
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

const EMPTY_MESSAGES: ChatMessage[] = [];

type AppRoute = "chat" | "guide";

function routeFromHash(): AppRoute {
  return window.location.hash === "#/guide" ? "guide" : "chat";
}

interface CoachingRequest {
  coachingId: string;
  replyToMessageId: string;
  skipClarification?: boolean;
}

function runError(reason: unknown): AnswerRunError {
  if (reason instanceof AnswerStreamError) {
    return {
      code: reason.code,
      message: errorLabel(reason.code, reason.message),
      retryable: reason.retryable,
    };
  }
  const error = reason as Error;
  return {
    code: "NETWORK_ERROR",
    message: errorLabel("NETWORK_ERROR", error.message || "Network request failed"),
    retryable: true,
  };
}

export default function App() {
  const [route, setRoute] = useState<AppRoute>(routeFromHash);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null);
  const [diagnosticsError, setDiagnosticsError] = useState("");
  const [diagnosticsLoading, setDiagnosticsLoading] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [conversationMessages, dispatchConversationMessage] = useReducer(
    conversationMessageReducer,
    initialConversationMessageState,
  );
  const [mode, setMode] = useState<RagMode>("direct");
  const [question, setQuestion] = useState("");
  const [error, setError] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [utilityOpen, setUtilityOpen] = useState(false);
  const [utilityTab, setUtilityTab] = useState<UtilityTab>("evidence");
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null);
  const [selectedCitationId, setSelectedCitationId] = useState<string | null>(null);
  const activeIdRef = useRef<string | null>(null);
  const conversationMessagesRef = useRef(conversationMessages);
  const controllersRef = useRef(new Map<string, AbortController>());
  const runByConversationRef = useRef(new Map<string, string>());
  const localSequenceRef = useRef(0);
  const messageListRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const followTailRef = useRef(true);

  useEffect(() => {
    conversationMessagesRef.current = conversationMessages;
  }, [conversationMessages]);

  useEffect(() => () => {
    controllersRef.current.forEach((controller) => controller.abort());
    controllersRef.current.clear();
    runByConversationRef.current.clear();
  }, []);

  useEffect(() => {
    const syncRoute = () => setRoute(routeFromHash());
    window.addEventListener("hashchange", syncRoute);
    return () => window.removeEventListener("hashchange", syncRoute);
  }, []);

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

  const applyConversationView = useCallback((entry?: Pick<ConversationMessageEntry, "messages" | "activeRunId">) => {
    const rows = entry?.messages ?? EMPTY_MESSAGES;
    const runningAssistant = entry?.activeRunId
      ? rows.find((message) => message.role === "assistant" && message.run_id === entry.activeRunId)
      : null;
    const assistant = runningAssistant ?? findLastAssistant(rows);
    setSelectedMessageId(assistant?.id ?? null);
    setSelectedCitationId(assistant?.citations?.[0]?.evidence_id ?? null);
    setUtilityTab(runningAssistant ? "process" : "evidence");
    setUtilityOpen(
      runningAssistant
        ? !isMobileViewport()
        : Boolean(assistant?.citations?.length) && !isMobileViewport(),
    );
    followTailRef.current = Boolean(runningAssistant);
  }, []);

  const activateConversation = useCallback((id: string | null) => {
    activeIdRef.current = id;
    setActiveId(id);
    setError("");
    if (!id) {
      applyConversationView();
      return;
    }
    const entry = conversationMessagesRef.current.byConversation[id];
    applyConversationView(entry?.loadStatus === "ready" ? entry : undefined);
  }, [applyConversationView]);

  useEffect(() => {
    let cancelled = false;
    Promise.all([api.health(), api.conversations()])
      .then(([status, rows]) => {
        if (cancelled) return;
        setHealth(status);
        setConversations(rows);
        if (rows.length) activateConversation(rows[0].id);
      })
      .catch((reason: Error) => { if (!cancelled) setError(`加载失败：${reason.message}`); });
    const timer = window.setInterval(() => api.health().then(setHealth).catch(() => undefined), 15_000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [activateConversation]);

  const startConversationLoad = useCallback((conversationId: string) => {
    const current = conversationMessagesRef.current.byConversation[conversationId];
    if (current?.loadStatus === "ready" || current?.loadStatus === "loading" || current?.activeRunId) {
      return () => undefined;
    }

    const requestId = `load-${Date.now()}-${++localSequenceRef.current}`;
    const controller = new AbortController();
    dispatchConversationMessage({ type: "load_started", conversationId, requestId });
    api.messages(conversationId, controller.signal)
      .then((rows) => {
        dispatchConversationMessage({ type: "load_succeeded", conversationId, requestId, messages: rows });
        if (activeIdRef.current !== conversationId) return;
        applyConversationView({ messages: rows, activeRunId: undefined });
        const lastUser = [...rows].reverse().find((message) => message.role === "user");
        window.requestAnimationFrame(() => window.requestAnimationFrame(() => {
          const container = messageListRef.current;
          if (!container || !lastUser || activeIdRef.current !== conversationId) return;
          const target = [...container.querySelectorAll<HTMLElement>("[data-message-id]")]
            .find((element) => element.dataset.messageId === lastUser.id);
          target?.scrollIntoView({ block: "start" });
        }));
      })
      .catch((reason: Error) => {
        if (reason.name === "AbortError") return;
        dispatchConversationMessage({
          type: "load_failed",
          conversationId,
          requestId,
          error: `加载对话失败：${reason.message}`,
        });
      });
    return () => {
      controller.abort();
      dispatchConversationMessage({ type: "load_cancelled", conversationId, requestId });
    };
  }, [applyConversationView]);

  useEffect(() => {
    if (!activeId) return;
    return startConversationLoad(activeId);
  }, [activeId, startConversationLoad]);

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

  const activeEntry = activeId ? conversationMessages.byConversation[activeId] : undefined;
  const messages = activeEntry?.messages ?? EMPTY_MESSAGES;
  const streaming = Boolean(activeEntry?.activeRunId);
  const phase = activeEntry?.phase ?? "";

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
      dispatchConversationMessage({ type: "ensure_conversation", conversationId: created.id, ready: true });
      activateConversation(created.id);
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
    const runId = runByConversationRef.current.get(id);
    if (runId) {
      dispatchConversationMessage({ type: "run_stopped", conversationId: id, runId });
      controllersRef.current.get(runId)?.abort();
    }
    try {
      await api.deleteConversation(id);
      const remaining = conversations.filter((item) => item.id !== id);
      setConversations(remaining);
      dispatchConversationMessage({ type: "remove_conversation", conversationId: id });
      if (activeIdRef.current === id) activateConversation(remaining[0]?.id ?? null);
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

  async function runQuestion(
    conversationId: string,
    text: string,
    requestMode: RagMode,
    coaching?: CoachingRequest,
  ) {
    if (runByConversationRef.current.has(conversationId)) return;

    const runId = `local-run-${Date.now()}-${++localSequenceRef.current}`;
    const createdAt = new Date().toISOString();
    const { userMessage, assistantMessage } = createOptimisticMessages(
      conversationId,
      runId,
      text,
      requestMode,
      createdAt,
    );
    const controller = new AbortController();
    runByConversationRef.current.set(conversationId, runId);
    controllersRef.current.set(runId, controller);
    dispatchConversationMessage({
      type: "run_started",
      conversationId,
      runId,
      userMessage,
      assistantMessage,
    });

    if (activeIdRef.current === conversationId) {
      setSelectedMessageId(assistantMessage.id);
      setSelectedCitationId(null);
      setUtilityTab("process");
      setUtilityOpen(!isMobileViewport());
      followTailRef.current = true;
      window.requestAnimationFrame(() => window.requestAnimationFrame(() => {
        if (activeIdRef.current !== conversationId) return;
        const target = [...(messageListRef.current?.querySelectorAll<HTMLElement>("[data-message-id]") ?? [])]
          .find((element) => element.dataset.messageId === userMessage.id);
        target?.scrollIntoView({ block: "start" });
      }));
    }

    const handleEvent = (streamEvent: SseEvent) => {
      dispatchConversationMessage({
        type: "stream_event",
        conversationId,
        runId,
        event: streamEvent,
        phaseText: streamEvent.type === "phase"
          ? phaseLabel(streamEvent.phase, streamEvent.message)
          : undefined,
      });
      if (activeIdRef.current !== conversationId || !controllersRef.current.has(runId)) return;
      if (streamEvent.type === "evidence") {
        setSelectedCitationId((current) => current ?? streamEvent.evidence.evidence_id);
      }
      if (streamEvent.type === "done") {
        setSelectedMessageId(assistantMessage.id);
        setSelectedCitationId(streamEvent.response.evidence[0]?.evidence_id ?? null);
      }
      scheduleTailFollow();
    };

    let completed = false;
    try {
      await streamAnswer(
        conversationId,
        text,
        requestMode,
        runId,
        handleEvent,
        controller.signal,
        coaching,
      );
      completed = true;
    } catch (reason) {
      if ((reason as Error).name === "AbortError") {
        dispatchConversationMessage({ type: "run_stopped", conversationId, runId });
      } else {
        const normalized = runError(reason);
        dispatchConversationMessage({ type: "run_failed", conversationId, runId, error: normalized });
        if (activeIdRef.current === conversationId) setError(normalized.message);
      }
    } finally {
      controllersRef.current.delete(runId);
      if (runByConversationRef.current.get(conversationId) === runId) {
        runByConversationRef.current.delete(conversationId);
      }
    }

    if (!completed) return;
    try {
      await refreshConversations();
    } catch (reason) {
      if (activeIdRef.current === conversationId) {
        setError(`回答已完成，但会话列表刷新失败：${(reason as Error).message}`);
      }
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || streaming) return;
    setError("");
    let conversationId = activeIdRef.current;
    if (!conversationId) {
      try {
        const created = await api.createConversation();
        setConversations((current) => [created, ...current]);
        dispatchConversationMessage({ type: "ensure_conversation", conversationId: created.id, ready: true });
        activateConversation(created.id);
        conversationId = created.id;
      } catch (reason) {
        setError(`新建会话失败：${(reason as Error).message}`);
        return;
      }
    }
    setQuestion("");
    const lastMessage = messages[messages.length - 1];
    const clarification = lastMessage?.role === "assistant" ? lastMessage.clarification : null;
    await runQuestion(
      conversationId,
      text,
      mode,
      clarification ? {
        coachingId: clarification.coaching_id,
        replyToMessageId: lastMessage.assistant_message_id ?? lastMessage.id,
      } : undefined,
    );
  }

  function stopActiveRun() {
    const conversationId = activeIdRef.current;
    if (!conversationId) return;
    const runId = runByConversationRef.current.get(conversationId);
    if (!runId) return;
    dispatchConversationMessage({ type: "run_stopped", conversationId, runId });
    controllersRef.current.get(runId)?.abort();
    controllersRef.current.delete(runId);
    if (runByConversationRef.current.get(conversationId) === runId) {
      runByConversationRef.current.delete(conversationId);
    }
  }

  function retryQuestion(text: string, requestMode: RagMode) {
    const conversationId = activeIdRef.current;
    if (!conversationId) return;
    setError("");
    void runQuestion(conversationId, text, requestMode);
  }

  function answerClarification(message: ChatMessage, value: string, skip = false) {
    const conversationId = activeIdRef.current;
    const clarification = message.clarification;
    if (!conversationId || !clarification || streaming) return;
    setError("");
    void runQuestion(conversationId, value, message.mode ?? mode, {
      coachingId: clarification.coaching_id,
      replyToMessageId: message.assistant_message_id ?? message.id,
      skipClarification: skip,
    });
  }

  const activeConversation = conversations.find((item) => item.id === activeId);
  const hasMessages = messages.length > 0;
  const conversationLoading = activeEntry?.loadStatus === "loading";
  const conversationLoadError = activeEntry?.loadStatus === "error" ? activeEntry.loadError : "";
  const title = conversationLoading
    ? activeConversation?.title || "正在加载对话"
    : hasMessages
      ? activeConversation?.title || "研究对话"
      : "向知识库提问";

  if (route === "guide") {
    return (
      <GuidePage
        health={health}
        onBack={() => { window.location.hash = "#/"; }}
      />
    );
  }

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
        onOpenGuide={() => { window.location.hash = "#/guide"; setSidebarOpen(false); }}
        onSelect={(id) => activateConversation(id)}
        onCreate={createConversation}
        onRename={renameConversation}
        onDelete={removeConversation}
      />
      <section className="workspace">
        <WorkspaceHeader title={title} documentCount={health?.document_count ?? 638} mode={mode} agenticAvailable={agenticAvailable} onModeChange={setMode} onOpenSidebar={() => setSidebarOpen(true)} onOpenGuide={() => { window.location.hash = "#/guide"; }} />
        {error && <div className="error-banner" role="alert"><span>{error}</span><button onClick={() => setError("")}>关闭</button></div>}
        {conversationLoading ? (
          <div className="conversation-state" role="status">
            <span className="state-spinner" aria-hidden="true" />
            <strong>正在加载对话</strong>
            <p>消息和证据加载完成后会显示在这里。</p>
          </div>
        ) : conversationLoadError && !hasMessages ? (
          <div className="conversation-state error" role="alert">
            <strong>对话暂时无法加载</strong>
            <p>{conversationLoadError}</p>
            {activeId && <button type="button" onClick={() => startConversationLoad(activeId)}>重新加载</button>}
          </div>
        ) : hasMessages ? (
          <>
            <MessageThread messages={messages} selectedMessageId={selectedMessage?.id ?? null} selectedCitationId={selectedCitationId} phase={phase} containerRef={messageListRef} onScroll={handleThreadScroll} onSelectMessage={selectMessage} onSelectCitation={selectCitation} onRetry={retryQuestion} onClarificationReply={answerClarification} retryDisabled={streaming} />
            <Composer question={question} streaming={streaming} textareaRef={textareaRef} onQuestionChange={setQuestion} onSubmit={(event) => void submit(event)} onStop={stopActiveRun} />
          </>
        ) : (
          <EmptyState question={question} streaming={streaming} textareaRef={textareaRef} onQuestionChange={setQuestion} onSubmit={(event) => void submit(event)} onStop={stopActiveRun} onSuggestion={chooseSuggestion} />
        )}
        {!utilityOpen && hasMessages && <button className="utility-trigger" aria-label="打开研究工具" onClick={() => setUtilityOpen(true)}><span>研究工具</span><span aria-hidden="true">‹</span></button>}
      </section>
      <UtilityPanel open={utilityOpen} tab={utilityTab} message={selectedMessage} selectedCitationId={selectedCitationId} health={health} indexStatus={indexStatus} diagnosticsLoading={diagnosticsLoading} diagnosticsError={diagnosticsError} streaming={selectedMessage?.run_status === "streaming"} onClose={() => setUtilityOpen(false)} onTabChange={changeUtilityTab} onSelectCitation={setSelectedCitationId} onRefreshDiagnostics={() => void refreshDiagnostics()} />
    </main>
  );
}
