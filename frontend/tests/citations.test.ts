import { describe, expect, it } from "vitest";
import { annotateCitations, applyTraceEvent, emptyProcessTrace } from "../src/citations";
import type { Citation } from "../src/types";

function citation(evidenceId: string, chunkId: string): Citation {
  return {
    evidence_id: evidenceId,
    evidence_kind: "source_chunk",
    headings: [],
    text: "source",
    metadata: {},
    doc_item_refs: [],
    locator: { kind: "document_chunk", document_id: "doc", chunk_id: chunkId } as Citation["locator"],
  };
}

describe("annotateCitations", () => {
  it("maps exact evidence ids and source chunk aliases to stable answer-local numbers", () => {
    const citations = [citation("evidence:alpha", "alpha"), citation("evidence:beta", "beta")];
    const content = "First [evidence:beta], again (source:beta), then evidence:alpha.";

    expect(annotateCitations(content, citations)).toBe(
      "First [2](#evidence=evidence%3Abeta), again [2](#evidence=evidence%3Abeta), then [1](#evidence=evidence%3Aalpha).",
    );
  });

  it("does not partially replace overlapping identifiers", () => {
    const citations = [citation("source:chunk-1", "chunk-1"), citation("source:chunk-10", "chunk-10")];
    expect(annotateCitations("source:chunk-10 source:chunk-1", citations)).toBe(
      "[2](#evidence=source%3Achunk-10) [1](#evidence=source%3Achunk-1)",
    );
  });

  it("consumes the evidence wrapper used by persisted answers", () => {
    const citations = [citation("source:document:chunk-1", "chunk-1")];
    expect(annotateCitations("Claim [evidence:source:document:chunk-1].", citations)).toBe(
      "Claim [1](#evidence=source%3Adocument%3Achunk-1).",
    );
  });
});

describe("process trace reducer", () => {
  it("merges phases, provider counts, fusion and evidence count", () => {
    let trace = applyTraceEvent(emptyProcessTrace(), { type: "phase", phase: "retrieve", message: "Search" });
    trace = applyTraceEvent(trace, {
      type: "retrieval_summary",
      provider_counts: { keyword: 5, vector: 3 },
      fused_count: 6,
    });
    trace = applyTraceEvent(trace, { type: "evidence", evidence: citation("one", "one") });

    expect(trace).toEqual({
      phases: [{ phase: "retrieve", message: "Search" }],
      provider_counts: { keyword: 5, vector: 3 },
      fused_count: 6,
      evidence_count: 1,
    });
  });
});
