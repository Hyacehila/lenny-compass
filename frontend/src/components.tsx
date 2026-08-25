import {
  FormEvent,
  RefObject,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import ReactMarkdown from "react-markdown";
import { api } from "./api";
import { annotateCitations } from "./citations";
import { modeLabels, phaseLabel, readinessLabel } from "./labels";
import type {
  AnswerProcessTrace,
  ChatMessage,
  Citation,
  Conversation,
  HealthStatus,
  IndexStatus,
  RagMode,
  UtilityTab,
} from "./types";

type CitationImage = NonNullable<Citation["image_observations"]>[number];

export function mergeCitationImages(citation: Citation): CitationImage[] {
  const sourceImages = citation.locator.kind === "document_chunk" ? citation.locator.images : [];
  const images: CitationImage[] = sourceImages.map((image) => {
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

function PlusIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v14M5 12h14" /></svg>;
}

function MenuIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16" /></svg>;
}

function MoreIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="5" cy="12" r="1.3" /><circle cx="12" cy="12" r="1.3" /><circle cx="19" cy="12" r="1.3" /></svg>;
}

function ChevronIcon({ direction = "right" }: { direction?: "right" | "left" | "down" }) {
  const transform = direction === "left" ? "rotate(180 12 12)" : direction === "down" ? "rotate(90 12 12)" : undefined;
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 7 7-7 7" transform={transform} /></svg>;
}

function CloseIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18" /></svg>;
}

function SendIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m4 4 16 8-16 8 3-8-3-8Z" /><path d="M7 12h13" /></svg>;
}

function RefreshIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11a8 8 0 1 0-2.3 5.7" /><path d="M20 5v6h-6" /></svg>;
}

function formatConversationTime(value: string): string {
  const date = new Date(value);
  const now = new Date();
  const sameDay = date.toDateString() === now.toDateString();
  if (sameDay) return `今天 ${date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}`;
  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  if (date.toDateString() === yesterday.toDateString()) return `昨天 ${date.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}`;
  return date.toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" });
}

interface SidebarProps {
  conversations: Conversation[];
  activeId: string | null;
  health: HealthStatus | null;
  open: boolean;
  collapsed: boolean;
  onClose: () => void;
  onToggleCollapse: () => void;
  onOpenSystem: () => void;
  onSelect: (id: string) => void;
  onCreate: () => Promise<void>;
  onRename: (id: string, title: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
}

export function Sidebar({
  conversations,
  activeId,
  health,
  open,
  collapsed,
  onClose,
  onToggleCollapse,
  onOpenSystem,
  onSelect,
  onCreate,
  onRename,
  onDelete,
}: SidebarProps) {
  const [menuId, setMenuId] = useState<string | null>(null);
  const [renameId, setRenameId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<Conversation | null>(null);
  const [busy, setBusy] = useState(false);

  const statusItems = [
    ["知识库", health?.facts_ready],
    ["关键词", health?.lexical_ready],
    ["语义检索", health?.semantic_ready],
    ["回答模型", health?.deepseek_ready],
    ["视觉模型", health?.zhipu_ready],
  ] as const;

  async function submitRename(event: FormEvent) {
    event.preventDefault();
    if (!renameId || !renameValue.trim()) return;
    setBusy(true);
    try {
      await onRename(renameId, renameValue.trim());
      setRenameId(null);
      setMenuId(null);
    } finally {
      setBusy(false);
    }
  }

  async function confirmDelete() {
    if (!deleteTarget) return;
    setBusy(true);
    try {
      await onDelete(deleteTarget.id);
      setDeleteTarget(null);
      setMenuId(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <button className={`sidebar-backdrop ${open ? "visible" : ""}`} aria-label="关闭会话侧栏" onClick={onClose} />
      <aside className={`sidebar ${open ? "mobile-open" : ""} ${collapsed ? "collapsed" : ""}`}>
        <div className="brand-row">
          <div className="brand-mark">LC</div>
          <div className="brand-copy"><strong>Lenny Compass</strong><span>私有产品知识库</span></div>
          <button className="icon-button mobile-close" aria-label="关闭会话侧栏" onClick={onClose}><CloseIcon /></button>
        </div>
        <button className="new-chat" onClick={() => void onCreate()}><PlusIcon /><span>新建对话</span></button>
        <p className="sidebar-label">对话</p>
        <nav className="conversation-list" aria-label="对话列表">
          {conversations.map((conversation) => {
            const active = conversation.id === activeId;
            const renaming = renameId === conversation.id;
            return (
              <div className={`conversation-row ${active ? "active" : ""}`} key={conversation.id}>
                {renaming ? (
                  <form className="rename-form" onSubmit={(event) => void submitRename(event)}>
                    <input
                      autoFocus
                      value={renameValue}
                      maxLength={200}
                      aria-label="会话标题"
                      onChange={(event) => setRenameValue(event.target.value)}
                      onKeyDown={(event) => { if (event.key === "Escape") setRenameId(null); }}
                    />
                    <div><button disabled={busy || !renameValue.trim()} type="submit">保存</button><button type="button" onClick={() => setRenameId(null)}>取消</button></div>
                  </form>
                ) : (
                  <>
                    <button className="conversation-select" onClick={() => { onSelect(conversation.id); onClose(); }}>
                      <span className="conversation-title"><i />{conversation.title}</span>
                      <span className="conversation-time">{formatConversationTime(conversation.updated_at)}</span>
                    </button>
                    <button
                      className="conversation-more"
                      aria-label={`管理会话：${conversation.title}`}
                      aria-expanded={menuId === conversation.id}
                      onClick={() => setMenuId((current) => current === conversation.id ? null : conversation.id)}
                    ><MoreIcon /></button>
                    {menuId === conversation.id && (
                      <div className="conversation-menu" role="menu">
                        <button role="menuitem" onClick={() => { setRenameId(conversation.id); setRenameValue(conversation.title); }}>重命名</button>
                        <button className="danger" role="menuitem" onClick={() => setDeleteTarget(conversation)}>删除</button>
                      </div>
                    )}
                  </>
                )}
              </div>
            );
          })}
        </nav>
        <button className="readiness" aria-label="打开系统状态" onClick={onOpenSystem}>
          <div className="readiness-title"><span className={`status-dot ${health?.status === "ready" ? "ready" : ""}`} />{readinessLabel(health)}</div>
          <div className="readiness-list">
            {statusItems.map(([label, ready]) => <div key={label}><span className={ready ? "mini-check ready" : "mini-check"}>✓</span><span>{label}</span><b className={ready ? "ready" : ""}>{ready ? "就绪" : "待配置"}</b></div>)}
          </div>
        </button>
        <button className="collapse-sidebar" onClick={onToggleCollapse}><ChevronIcon direction={collapsed ? "right" : "left"} /><span>{collapsed ? "展开" : "收起侧边栏"}</span></button>
      </aside>
      {deleteTarget && (
        <div className="dialog-backdrop" role="presentation">
          <section className="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-title">
            <h2 id="delete-title">删除这个会话？</h2>
            <p>“{deleteTarget.title}”及其中的消息将从本机移除，此操作无法撤销。</p>
            <div><button onClick={() => setDeleteTarget(null)}>取消</button><button className="danger-button" disabled={busy} onClick={() => void confirmDelete()}>确认删除</button></div>
          </section>
        </div>
      )}
    </>
  );
}

interface HeaderProps {
  title: string;
  documentCount: number;
  mode: RagMode;
  agenticAvailable: boolean;
  onModeChange: (mode: RagMode) => void;
  onOpenSidebar: () => void;
}

export function WorkspaceHeader({ title, documentCount, mode, agenticAvailable, onModeChange, onOpenSidebar }: HeaderProps) {
  return (
    <header className="workspace-header">
      <div className="mobile-titlebar">
        <button className="menu-button" aria-label="打开会话侧栏" onClick={onOpenSidebar}><MenuIcon /></button>
        <div className="mobile-mark">LC</div>
        <strong>{title}</strong>
      </div>
      <div className="header-copy"><h1>{title}</h1><p>{documentCount} 篇私有文档 · 基于原文证据回答</p></div>
      <div className="mode-switch" role="group" aria-label="检索模式">
        <button className={mode === "direct" ? "active" : ""} onClick={() => onModeChange("direct")}><span aria-hidden="true">ϟ</span>直接回答</button>
        <button className={mode === "agentic" ? "active" : ""} disabled={!agenticAvailable} title={agenticAvailable ? "由智能体自主检索并选择证据范围" : "问答服务就绪后才能使用智能检索"} onClick={() => onModeChange("agentic")}>智能检索</button>
      </div>
    </header>
  );
}

interface ComposerProps {
  question: string;
  streaming: boolean;
  variant?: "dock" | "empty";
  textareaRef?: RefObject<HTMLTextAreaElement | null>;
  onQuestionChange: (value: string) => void;
  onSubmit: (event: FormEvent) => void;
  onStop: () => void;
}

export function Composer({ question, streaming, variant = "dock", textareaRef, onQuestionChange, onSubmit, onStop }: ComposerProps) {
  return (
    <form className={`composer ${variant}`} onSubmit={onSubmit}>
      <textarea
        ref={textareaRef}
        value={question}
        onChange={(event) => onQuestionChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            event.currentTarget.form?.requestSubmit();
          }
        }}
        placeholder="向知识库提问…"
        aria-label="向知识库提问"
        rows={2}
      />
      {streaming ? (
        <button type="button" className="send stop" onClick={onStop}><span aria-hidden="true" />停止</button>
      ) : (
        <button type="submit" className="send" disabled={!question.trim()} aria-label="发送问题"><SendIcon /></button>
      )}
    </form>
  );
}

const suggestions = [
  "优秀的产品团队如何选择北极星指标？",
  "总结 Lenny 关于产品市场契合度的核心判断",
  "Eric Ries 如何理解持续创新？",
];

interface EmptyStateProps extends Omit<ComposerProps, "variant"> {
  onSuggestion: (value: string) => void;
}

export function EmptyState(props: EmptyStateProps) {
  return (
    <section className="empty-state">
      <div className="empty-copy">
        <h2>从知识库中找到证据，<br />而不只是搜索结果。</h2>
        <p>使用中文提问，每个回答都能回溯到经过版本管理的原文证据。</p>
      </div>
      <Composer {...props} variant="empty" />
      <div className="suggestions" aria-label="示例问题">
        {suggestions.map((item) => <button key={item} onClick={() => props.onSuggestion(item)}><span>{item}</span><ChevronIcon /></button>)}
      </div>
    </section>
  );
}

interface MessageThreadProps {
  messages: ChatMessage[];
  selectedMessageId: string | null;
  selectedCitationId: string | null;
  phase: string;
  containerRef: RefObject<HTMLDivElement | null>;
  onScroll: () => void;
  onSelectMessage: (messageId: string) => void;
  onSelectCitation: (messageId: string, evidenceId: string) => void;
}

export function MessageThread({ messages, selectedMessageId, selectedCitationId, phase, containerRef, onScroll, onSelectMessage, onSelectCitation }: MessageThreadProps) {
  return (
    <div className="message-list" ref={containerRef} onScroll={onScroll}>
      <div className="message-column">
        {messages.map((message) => {
          const citations = message.citations ?? [];
          const markdown = message.role === "assistant" ? annotateCitations(message.content, citations) : message.content;
          return (
            <article
              className={`message ${message.role} ${selectedMessageId === message.id ? "selected" : ""}`}
              data-message-id={message.id}
              key={message.id}
              onClick={() => { if (message.role === "assistant") onSelectMessage(message.id); }}
            >
              <div className="message-meta"><span>{message.role === "user" ? "你" : "Compass"}</span>{message.mode && <span>{modeLabels[message.mode]}</span>}</div>
              <div className="message-body">
                {message.role === "assistant" ? (
                  <ReactMarkdown
                    components={{
                      a: ({ href, children }) => {
                        if (href?.startsWith("#evidence=")) {
                          const evidenceId = decodeURIComponent(href.slice("#evidence=".length));
                          return <button className={`citation-anchor ${selectedCitationId === evidenceId ? "active" : ""}`} aria-label={`查看证据 ${children}`} onClick={(event) => { event.preventDefault(); event.stopPropagation(); onSelectCitation(message.id, evidenceId); }}>{children}</button>;
                        }
                        return <a href={href} target="_blank" rel="noreferrer">{children}</a>;
                      },
                    }}
                  >{markdown}</ReactMarkdown>
                ) : message.content}
              </div>
            </article>
          );
        })}
        {phase && <div className="phase" aria-live="polite"><span /><span /><span />{phase}</div>}
      </div>
    </div>
  );
}

function metadataValue(citation: Citation, key: string): string | undefined {
  const value = citation.metadata[key];
  return typeof value === "string" || typeof value === "number" ? String(value) : undefined;
}

function SourceFigure({ image }: { image: CitationImage }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [image.url]);
  return (
    <figure>
      {image.url && !failed
        ? <img src={image.url} alt={image.caption || "来源图片"} loading="lazy" onError={() => setFailed(true)} />
        : <div className="image-fallback">图片暂时无法加载</div>}
      <figcaption>
        {image.caption && <span>{image.caption}</span>}
        {image.observation && <span>{image.observation}</span>}
        {failed && <span>已保留图片说明和分析状态。</span>}
        {image.inspected === false && <span>图片分析失败，仅显示原始说明。</span>}
        {image.inspected === undefined && <span>该图片未进行自动分析，仅显示原始说明。</span>}
      </figcaption>
    </figure>
  );
}

function EvidenceView({
  message,
  selectedCitationId,
  onSelectCitation,
}: {
  message: ChatMessage | null;
  selectedCitationId: string | null;
  onSelectCitation: (id: string) => void;
}) {
  const citations = message?.citations ?? [];
  const selected = citations.find((citation) => citation.evidence_id === selectedCitationId) ?? citations[0];
  const selectedIndex = selected ? citations.indexOf(selected) : -1;
  const images = selected ? mergeCitationImages(selected) : [];
  const [context, setContext] = useState<Citation[] | null>(null);
  const [contextError, setContextError] = useState("");
  const [contextLoading, setContextLoading] = useState(false);

  useEffect(() => {
    setContext(null);
    setContextError("");
  }, [selected?.evidence_id]);

  async function loadContext() {
    if (!selected || selected.locator.kind !== "document_chunk") return;
    setContextLoading(true);
    setContextError("");
    try {
      const response = await api.readSourceWindow(selected.locator, 2);
      setContext(response.evidence);
    } catch {
      setContextError("上下文暂时无法读取，请稍后重试。");
    } finally {
      setContextLoading(false);
    }
  }

  if (!selected) return <div className="utility-empty"><p>这条回答还没有可查看的原文证据。</p></div>;
  const locator = selected.locator;
  return (
    <div className="evidence-view">
      <p className="utility-kicker">当前来源（{selectedIndex + 1}/{citations.length}）</p>
      <section className="active-evidence">
        <header><span className="source-number">{selectedIndex + 1}</span><div><h3>{selected.title || (locator.kind === "document_chunk" ? locator.document_id : locator.path)}</h3><p>{metadataValue(selected, "document_type") || (selected.evidence_kind === "source_chunk" ? "Lenny Archive" : "LLM Wiki")}{metadataValue(selected, "date") ? ` · ${metadataValue(selected, "date")}` : ""}</p></div></header>
        <div className="source-excerpt"><p>{selected.text}</p></div>
        {locator.kind === "document_chunk" && (
          <dl className="source-meta"><div><dt>路径</dt><dd>{locator.enhanced_path}</dd></div><div><dt>分块</dt><dd>{locator.ordinal}</dd></div>{selected.headings.length > 0 && <div><dt>标题</dt><dd>{selected.headings.at(-1)}</dd></div>}</dl>
        )}
        {images.map((image) => <SourceFigure image={image} key={image.image_id} />)}
        {locator.kind === "document_chunk" && (
          <button className="expand-context" disabled={contextLoading} onClick={() => void loadContext()}>{contextLoading ? "正在读取…" : context ? "重新读取上下文" : "展开上下文"}<ChevronIcon direction="down" /></button>
        )}
        {contextError && <p className="inline-error">{contextError}</p>}
        {context && <div className="source-context">{context.map((item) => <article key={item.evidence_id}><span>分块 {item.locator.kind === "document_chunk" ? item.locator.ordinal : "—"}</span><p>{item.text}</p></article>)}</div>}
      </section>
      <p className="source-list-label">所有来源</p>
      <div className="source-list">
        {citations.map((citation, index) => <button className={citation.evidence_id === selected.evidence_id ? "active" : ""} key={citation.evidence_id} onClick={() => onSelectCitation(citation.evidence_id)}><span className="source-index">{index + 1}</span><span>{citation.title || (citation.locator.kind === "document_chunk" ? citation.locator.document_id : citation.locator.path)}</span><small>{citation.locator.kind === "document_chunk" ? `分块 ${citation.locator.ordinal}` : "Wiki"}</small></button>)}
      </div>
    </div>
  );
}

function ProcessView({ trace, mode, streaming }: { trace: AnswerProcessTrace | null | undefined; mode?: RagMode; streaming: boolean }) {
  if (!trace || trace.phases.length === 0) return <div className="utility-empty"><p>该回答没有保存过程记录。</p><span>新完成的回答会在这里保留公开的检索阶段与汇总。</span></div>;
  const counts = Object.entries(trace.provider_counts);
  return (
    <div className="process-view">
      <p className="utility-kicker">本次请求 <strong>{mode ? modeLabels[mode] : "回答"}</strong></p>
      <ol className="phase-timeline">
        {trace.phases.map((item, index) => {
          const current = streaming && index === trace.phases.length - 1;
          return <li className={current ? "current" : "complete"} key={item.phase}><span>{current ? "" : "✓"}</span><div><strong>{phaseLabel(item.phase, item.message)}</strong>{item.message && phaseLabel(item.phase, item.message) !== item.message && <small>{item.message}</small>}</div></li>;
        })}
      </ol>
      {counts.length > 0 && <section className="trace-summary"><h3>检索摘要</h3><div>{counts.map(([provider, count]) => <p key={provider}><span>{provider === "keyword" ? "关键词" : provider === "vector" ? "语义" : provider}</span><strong>{count}</strong></p>)}{trace.fused_count !== null && trace.fused_count > 0 && <p><span>融合</span><strong>{trace.fused_count}</strong></p>}</div></section>}
      <section className="trace-summary"><h3>已接收证据</h3><div><p><span>本次回答</span><strong>{trace.evidence_count}</strong></p></div></section>
      <p className="trace-note">过程记录仅包含公开阶段与检索汇总，不包含模型思维链。</p>
    </div>
  );
}

function statusText(value: boolean | undefined): string {
  return value ? "就绪" : "待配置";
}

function SystemView({ health, indexStatus, loading, error, onRefresh }: { health: HealthStatus | null; indexStatus: IndexStatus | null; loading: boolean; error: string; onRefresh: () => void }) {
  const groups = [
    { title: "知识与事实层", rows: [["文档", (health?.document_count ?? indexStatus?.document_count ?? 0).toLocaleString("zh-CN")], ["分块", (health?.chunk_count ?? indexStatus?.chunk_count ?? 0).toLocaleString("zh-CN")], ["事实层", statusText(health?.facts_ready)]] },
    { title: "检索索引", rows: [["关键词索引", statusText(health?.lexical_ready)], ["语义索引", statusText(health?.semantic_ready)], ["嵌入模型", indexStatus?.embedding_model || "—"], ["向量维度", indexStatus?.embedding_dimensions ? String(indexStatus.embedding_dimensions) : "—"]] },
    { title: "索引构建", rows: [["Schema", indexStatus ? `v${indexStatus.schema_version}` : "—"], ["Build", indexStatus?.build_id ? `${indexStatus.build_id.slice(0, 8)}…` : "—"], ["SQLite", indexStatus?.sqlite_version || "—"], ["sqlite-vec", indexStatus?.sqlite_vec_version || "—"]] },
    { title: "模型服务", rows: [["回答模型", statusText(health?.deepseek_ready)], ["智能检索", health?.agentic_enabled ? "可用" : "待配置"], ["视觉模型", statusText(health?.zhipu_ready)]] },
  ];
  return (
    <div className="system-view">
      <div className="system-heading"><p><span className={`status-dot ${health?.full_ready ? "ready" : ""}`} /><strong>{readinessLabel(health)}</strong></p><small>每 15 秒自动更新</small></div>
      {groups.map((group) => <section className="diagnostic-group" key={group.title}><h3>{group.title}</h3><dl>{group.rows.map(([label, value]) => <div key={label}><dt>{label}</dt><dd className={value === "就绪" || value === "可用" ? "ready" : value === "待配置" ? "warning" : ""}>{value}</dd></div>)}</dl></section>)}
      {error && <p className="inline-error">{error}</p>}
      <button className="refresh-button" disabled={loading} onClick={onRefresh}><RefreshIcon />{loading ? "正在刷新…" : "立即刷新"}</button>
    </div>
  );
}

interface UtilityPanelProps {
  open: boolean;
  tab: UtilityTab;
  message: ChatMessage | null;
  selectedCitationId: string | null;
  health: HealthStatus | null;
  indexStatus: IndexStatus | null;
  diagnosticsLoading: boolean;
  diagnosticsError: string;
  streaming: boolean;
  onClose: () => void;
  onTabChange: (tab: UtilityTab) => void;
  onSelectCitation: (id: string) => void;
  onRefreshDiagnostics: () => void;
}

export function UtilityPanel({ open, tab, message, selectedCitationId, health, indexStatus, diagnosticsLoading, diagnosticsError, streaming, onClose, onTabChange, onSelectCitation, onRefreshDiagnostics }: UtilityPanelProps) {
  const citationCount = message?.citations?.length ?? 0;
  const contentRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (open && contentRef.current) contentRef.current.scrollTop = 0;
  }, [message?.id, open, selectedCitationId, tab]);
  if (!open) return null;
  return (
    <>
      <button className="utility-backdrop" aria-label="关闭工具面板" onClick={onClose} />
      <aside className="utility-panel" aria-label="研究工具面板">
        <div className="sheet-handle" aria-hidden="true" />
        <header className="utility-header">
          <div className="utility-tabs" role="tablist" aria-label="研究工具">
            {(["evidence", "process", "system"] as UtilityTab[]).map((item) => <button role="tab" aria-selected={tab === item} className={tab === item ? "active" : ""} key={item} onClick={() => onTabChange(item)}>{item === "evidence" ? `证据 ${citationCount || ""}` : item === "process" ? "过程" : "系统"}</button>)}
          </div>
          <button className="utility-close" aria-label="关闭工具面板" onClick={onClose}><span className="desktop-collapse" aria-hidden="true">‹‹</span><span className="mobile-close-sheet"><CloseIcon /></span></button>
        </header>
        <div className="utility-content" ref={contentRef} role="tabpanel">
          {tab === "evidence" ? <EvidenceView message={message} selectedCitationId={selectedCitationId} onSelectCitation={onSelectCitation} /> : tab === "process" ? <ProcessView trace={message?.process_trace} mode={message?.mode} streaming={streaming} /> : <SystemView health={health} indexStatus={indexStatus} loading={diagnosticsLoading} error={diagnosticsError} onRefresh={onRefreshDiagnostics} />}
        </div>
      </aside>
    </>
  );
}
