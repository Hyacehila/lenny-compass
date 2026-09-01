import type { HealthStatus, RagMode } from "./types";

export const modeLabels: Record<RagMode, string> = {
  direct: "快速模式",
  agentic: "深入模式",
};

const phaseLabels: Record<string, string> = {
  plan: "正在判断问题和所需信息",
  rewrite: "正在理解问题",
  retrieve: "正在检索本地知识库",
  resolve: "正在读取原文证据",
  inspect_images: "正在分析相关图片",
  agentic: "智能体正在选择和读取证据",
  answer: "正在根据证据生成回答",
  verify: "正在独立核验结论",
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
    ? "一次检索后直接组织答案，响应更快，适合定义、方法和目标明确的问题。"
    : "由智能体多轮搜索、阅读和筛选证据，适合比较与复杂场景，需要更长时间且模型调用更多。";
}

export function phaseLabel(phase: string, message?: string): string {
  return phaseLabels[phase] || message || "正在处理";
}

export function errorLabel(code: string, message: string): string {
  const labels: Record<string, string> = {
    ANSWER_SERVICE_UNAVAILABLE: "问答服务暂不可用，请稍后重试。",
    ANSWER_STREAM_INCOMPLETE: "回答意外中断，请重试。",
    ANSWER_STREAM_MULTIPLE_TERMINALS: "回答协议异常：收到重复完成事件，请重试。",
    ANSWER_STREAM_EVENT_AFTER_TERMINAL: "回答协议异常：完成后仍收到数据，请重试。",
    ANSWER_STREAM_INVALID_EVENT: "回答数据格式异常，请重试。",
    ANSWER_REQUEST_FAILED: "回答请求失败，请稍后重试。",
    NETWORK_ERROR: "网络连接中断，请检查服务后重试。",
    DEEPSEEK_ANSWER_ERROR: "回答模型请求失败，请稍后重试。",
    DEEPSEEK_AGENTIC_ERROR: "智能检索模型请求失败，请稍后重试。",
    DEEPSEEK_PLAN_ERROR: "问题规划暂时失败，请稍后重试。",
    SEMANTIC_VERIFICATION_UNAVAILABLE: "结论核验暂时不可用，本次未发布未经核验的结论。",
    INVALID_COACHING_CONTEXT: "这条追问已失效，请从当前对话重新提问。",
    AGENTIC_READ_LIMIT: "智能检索读取的证据过多，请缩小问题范围后重试。",
    AGENTIC_ROUND_LIMIT: "智能检索步骤过多，请简化问题后重试。",
    SEMANTIC_PROVIDER_UNAVAILABLE: "语义检索服务暂不可用。",
  };
  return labels[code] || `请求失败：${message}`;
}
