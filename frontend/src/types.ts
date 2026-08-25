export type RagMode = "direct" | "agentic";
export type UtilityTab = "evidence" | "process" | "system";

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ImageReference {
  url?: string;
  path?: string;
  image_id: string;
  self_ref?: string;
  sha256?: string;
  caption?: string;
}

export interface DocumentChunkLocator {
  kind: "document_chunk";
  config_version: string;
  document_id: string;
  chunk_id: string;
  ordinal: number;
  segment_ordinal: number;
  fact_document_path: string;
  raw_path: string;
  raw_sha256: string;
  enhanced_path: string;
  enhanced_sha256: string;
  docling_path: string;
  docling_sha256: string;
  chunk_sha256: string;
  segment: {
    first_self_ref: string;
    last_self_ref: string;
    item_count: number;
  };
  self_refs: string[];
  headings: string[];
  images: ImageReference[];
}

export interface LlmWikiPageLocator {
  kind: "llm_wiki_page";
  project_id: string;
  path: string;
  revision?: string;
  fact_id?: string;
  provider_uri?: string;
}

export type EvidenceLocator = DocumentChunkLocator | LlmWikiPageLocator;

export interface Citation {
  evidence_id: string;
  evidence_kind: "source_chunk" | "wiki_fact";
  title?: string;
  headings: string[];
  text: string;
  metadata: Record<string, unknown>;
  doc_item_refs: string[];
  locator: EvidenceLocator;
  image_observations?: Array<{
    image_id: string;
    url?: string;
    caption?: string;
    observation?: string;
    inspected?: boolean;
    error?: string;
  }>;
}

export interface AnswerProcessPhase {
  phase: string;
  message?: string;
}

export interface AnswerProcessTrace {
  phases: AnswerProcessPhase[];
  provider_counts: Record<string, number>;
  fused_count: number | null;
  evidence_count: number;
}

export interface ChatMessage {
  id: string;
  conversation_id?: string;
  role: "user" | "assistant";
  content: string;
  mode?: RagMode;
  created_at: string;
  citations?: Citation[];
  process_trace?: AnswerProcessTrace | null;
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

export interface IndexStatus {
  build_id: string;
  chunk_count: number;
  created_at: string;
  document_count: number;
  embedding_dimensions: number | null;
  embedding_model: string | null;
  facts_manifest_file: string;
  facts_manifest_sha256: string;
  lexical_database_file: string;
  lexical_database_sha256: string;
  lexical_status: "ready" | "building" | "test_only";
  schema_version: string;
  semantic_database_file: string;
  semantic_database_sha256?: string | null;
  semantic_reason?: string | null;
  semantic_status: "ready" | "building" | "test_only";
  sqlite_vec_version?: string | null;
  sqlite_version: string;
}

export interface SourceWindowResponse {
  locator: DocumentChunkLocator;
  window: number;
  evidence: Citation[];
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
