export type RagMode = "direct" | "agentic";

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface Citation {
  evidence_id: string;
  evidence_kind: "source_chunk" | "wiki_fact";
  title?: string;
  headings: string[];
  text: string;
  metadata: Record<string, unknown>;
  doc_item_refs: string[];
  locator: {
    kind: "document_chunk" | "llm_wiki_page";
    document_id?: string;
    ordinal?: number;
    raw_path?: string;
    enhanced_path?: string;
    project_id?: string;
    path?: string;
    images?: Array<{
      url?: string;
      path?: string;
      image_id: string;
      self_ref?: string;
      sha256?: string;
      caption?: string;
    }>;
  };
  image_observations?: Array<{
    image_id: string;
    url?: string;
    caption?: string;
    observation?: string;
    inspected?: boolean;
    error?: string;
  }>;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  mode?: RagMode;
  created_at: string;
  citations?: Citation[];
}

export interface HealthStatus {
  status: string;
  agentic_enabled?: boolean;
  facts_ready?: boolean;
  lexical_ready?: boolean;
  semantic_ready?: boolean;
  deepseek_ready?: boolean;
  zhipu_ready?: boolean;
  answer_ready?: boolean;
  full_ready?: boolean;
  document_count?: number;
  chunk_count?: number;
  detail?: Record<string, unknown>;
}

export type SseEvent =
  | { type: "phase"; phase: string; message?: string }
  | { type: "retrieval_summary"; provider_counts: Record<string, number>; fused_count: number }
  | { type: "evidence"; evidence: Citation }
  | { type: "answer_delta"; delta: string }
  | {
      type: "done";
      response: { request_id: string; mode: RagMode; answer: string; evidence: Citation[] };
    }
  | { type: "error"; error: { code: string; message: string; retryable: boolean } };
