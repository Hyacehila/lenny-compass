import type { AnswerProcessTrace, Citation, SseEvent } from "./types";

const PLACEHOLDER = "@@LENNY_CITATION_";

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function citationAliases(citation: Citation): string[] {
  const aliases = [citation.evidence_id];
  if (citation.locator.kind === "document_chunk") {
    aliases.push(`source:${citation.locator.chunk_id}`);
  }
  return [...new Set(aliases)];
}

export function annotateCitations(markdown: string, citations: Citation[]): string {
  if (!citations.length) return markdown;

  const aliasToIndex = new Map<string, number>();
  citations.forEach((citation, index) => {
    citationAliases(citation).forEach((alias) => aliasToIndex.set(alias, index));
  });
  const aliases = [...aliasToIndex.keys()].sort((left, right) => right.length - left.length);
  if (!aliases.length) return markdown;

  const pattern = new RegExp(aliases.map(escapeRegExp).join("|"), "g");
  let tokenized = markdown.replace(pattern, (alias) => `${PLACEHOLDER}${aliasToIndex.get(alias)}@@`);
  const marker = `${escapeRegExp(PLACEHOLDER)}\\d+@@`;
  tokenized = tokenized.replace(new RegExp(`evidence:\\s*(?=${marker})`, "gi"), "");
  const group = `(${marker}(?:\\s*[,，、]\\s*${marker})*)`;
  tokenized = tokenized
    .replace(new RegExp(`\\(\\s*${group}\\s*\\)`, "g"), "$1")
    .replace(new RegExp(`\\[\\s*${group}\\s*\\]`, "g"), "$1");

  return tokenized.replace(new RegExp(`${escapeRegExp(PLACEHOLDER)}(\\d+)@@`, "g"), (_, rawIndex) => {
    const index = Number(rawIndex);
    const citation = citations[index];
    return `[${index + 1}](#evidence=${encodeURIComponent(citation.evidence_id)})`;
  });
}

export function emptyProcessTrace(): AnswerProcessTrace {
  return { phases: [], provider_counts: {}, fused_count: null, evidence_count: 0 };
}

export function applyTraceEvent(trace: AnswerProcessTrace, event: SseEvent): AnswerProcessTrace {
  if (event.type === "phase") {
    const phases = trace.phases.some((item) => item.phase === event.phase)
      ? trace.phases.map((item) => item.phase === event.phase ? { phase: event.phase, message: event.message } : item)
      : [...trace.phases, { phase: event.phase, message: event.message }];
    return { ...trace, phases };
  }
  if (event.type === "retrieval_summary") {
    return {
      ...trace,
      provider_counts: event.provider_counts,
      fused_count: event.fused_count,
    };
  }
  if (event.type === "evidence") {
    return { ...trace, evidence_count: trace.evidence_count + 1 };
  }
  if (event.type === "done") {
    return { ...trace, evidence_count: Math.max(trace.evidence_count, event.response.evidence.length) };
  }
  return trace;
}
