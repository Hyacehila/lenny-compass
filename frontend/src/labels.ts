import type { HealthStatus, RagMode } from "./types";

export const modeLabels: Record<RagMode, string> = {
  direct: "直接回答",
  agentic: "智能检索",
};

const phaseLabels: Record<string, string> = {
  rewrite: "正在理解问题",
  retrieve: "正在检索本地知识库",
  resolve: "正在读取原文证据",
  inspect_images: "正在分析相关图片",
  agentic: "智能体正在选择和读取证据",
  answer: "正在根据证据生成回答",
};

export function readinessLabel(health: HealthStatus | null): string {
  if (!health) return "正在连接";
  if (health.full_ready) return "系统就绪";
  if (health.answer_ready) return health.semantic_ready && health.zhipu_ready
    ? "问答已就绪"
    : "问答已就绪 · 关键词模式";
  if (health.facts_ready && health.lexical_ready) return "本地知识库已就绪";
  return "需要完成配置";
}

export function modeDescription(mode: RagMode): string {
  return mode === "direct"
    ? "自动结合关键词和语义检索，响应更快，适合目标明确的问题。"
    : "由智能体多轮搜索、阅读和筛选证据，覆盖更深入，但需要更长时间。";
}

export function phaseLabel(phase: string, message?: string): string {
  return phaseLabels[phase] || message || "正在处理";
}

export function errorLabel(code: string, message: string): string {
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
