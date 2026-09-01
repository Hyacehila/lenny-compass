export type RagMode = "direct" | "agentic";
export type AnswerKind = "grounded" | "insufficient_evidence";
export type AnswerContractVersion = "v1" | "v2";
export type ResponseKind =
  | "grounded_answer"
  | "general_guidance"
  | "clarification"
  | "insufficient_evidence";
export type ClaimRole = "core" | "supporting" | "advice_rationale";
export type UtilityTab = "evidence" | "process" | "system";
export type AnswerRunStatus = "streaming" | "completed" | "stopped" | "failed";

export interface AnswerClaim {
  claim_id: string;
  text: string;
  evidence_ids: string[];
  role?: ClaimRole;
}

export interface AdviceItem {
  advice_id: string;
  text: string;
  basis: "evidence_synthesis" | "general_pm_guidance";
  rationale_claim_ids: string[];
}

export interface UserPremise {
  premise_id: string;
  text: string;
  source_message_id: string;
}

export interface Clarification {
  coaching_id: string;
  round: number;
  prompt: string;
  suggestions: string[];
  can_skip: true;
}

export interface AnswerVerification {
  status: "passed";
  scope: "claim_evidence_support";
  policy_version: string;
  checked_claim_count: number;
  omitted_claim_count: number;
  calibration_level: "uncalibrated" | "minimal";
}

export interface AnswerRunError {
  code: string;
  message: string;
  retryable: boolean;
}

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
  answer_kind?: AnswerKind | null;
  contract_version?: AnswerContractVersion;
  response_kind?: ResponseKind | null;
  claims?: AnswerClaim[];
  advice?: AdviceItem[];
  user_premises?: UserPremise[];
  clarification?: Clarification | null;
  verification?: AnswerVerification | null;
  insufficient_reason?: "no_evidence" | "semantic_support_failed" | "semantic_support_uncertain" | null;
  process_trace?: AnswerProcessTrace | null;
  run_id?: string;
  request_id?: string;
  assistant_message_id?: string;
  run_status?: AnswerRunStatus;
  run_error?: AnswerRunError;
  retry_question?: string;
}

export interface HealthStatus {
  status: string;
  agentic_enabled?: boolean;
  facts_ready?: boolean;
  lexical_ready?: boolean;
  semantic_ready?: boolean;
  deepseek_ready?: boolean;
  claim_verifier_ready?: boolean;
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
      response: {
        request_id: string;
        assistant_message_id?: string;
        mode: RagMode;
        contract_version?: AnswerContractVersion;
        response_kind?: ResponseKind | null;
        answer_kind: AnswerKind | null;
        answer: string;
        evidence: Citation[];
        claims: AnswerClaim[];
        advice?: AdviceItem[];
        user_premises?: UserPremise[];
        clarification?: Clarification | null;
        verification?: AnswerVerification | null;
        insufficient_reason?: "no_evidence" | "semantic_support_failed" | "semantic_support_uncertain" | null;
      };
    }
  | { type: "error"; error: { code: string; message: string; retryable: boolean } };
