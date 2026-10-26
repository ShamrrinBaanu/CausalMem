export type View =
  | "chat"
  | "memories"
  | "timeline"
  | "evidence"
  | "diagnostics"
  | "settings";

export type CitationKind = "memory" | "state" | "transition";

export interface Citation {
  kind: CitationKind;
  id: string;
}

export interface QueryAnalysis {
  category: string;
  rationale: string;
  is_question: boolean;
}

export interface MemoryRecord {
  id: string | null;
  memory: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  attribution?: string | null;
}

export interface StateRecord {
  state_id: string;
  user_id: string;
  state_type: string;
  state_value: string;
  status: string;
  valid_from: string;
  valid_until: string | null;
  confidence: number;
  source_memory_ids: string[];
  metadata: Record<string, unknown>;
}

export interface TransitionRecord {
  transition_id: string;
  user_id: string;
  from_state_id: string;
  to_state_id: string;
  transition_type: string;
  relation_kind: string;
  trigger: string | null;
  reason: string | null;
  timestamp: string;
  confidence: number;
  source_memory_ids: string[];
  metadata: Record<string, unknown>;
}

export interface CausalPathStep {
  transition_id?: string;
  user_id?: string;
  from_state_id?: string;
  to_state_id?: string;
  relation_kind?: string;
  from?: string;
  to?: string;
  trigger?: string | null;
  reason?: string | null;
}

export interface EvidenceBundle {
  query_category: string | null;
  retrieval_sources: string[];
  retrieval_reason?: string | null;
  requested_timestamp: string | null;
  temporal_context: Record<string, unknown> | null;
  ambiguity: Record<string, unknown> | null;
  state_focus: Record<string, unknown> | null;
  memories: MemoryRecord[];
  states: StateRecord[];
  transitions: TransitionRecord[];
  causal_path: CausalPathStep[];
}

export interface CitationValidation {
  citation_kind: CitationKind | "invalid";
  evidence_id_resolves_owned: boolean;
  kind_matches_resolved_type: boolean | null;
  reason_code: string;
}

export interface DiagnosticsSummary {
  causal_gate_passed: boolean | null;
  required_transition_citation_accepted: boolean | null;
  answer_reason: string;
  model_response_status: string;
  accepted_claim_count: number;
  rejected_claim_count: number;
  rejection_categories: Record<string, number>;
  rejection_reasons: Record<string, number>;
  citation_validation?: CitationValidation[];
}

export interface Mem0StageTiming {
  stage: string;
  duration_ms?: number;
  status?: string;
  success?: boolean;
  error_category?: string;
}

export interface IngestionTiming {
  add_memory_ms: number;
  memory_read_ms: number;
  state_extraction_ms: number;
  extractor_skipped: boolean;
  mem0_infer?: boolean;
  mem0_stages: Mem0StageTiming[];
}

export interface RetrievalTiming {
  mem0_search_ms?: number;
  timeline_db_ms?: number;
  graph_ms?: number;
  total_ms: number;
}

export interface GenerationTiming {
  model_ms?: number;
  validation_ms?: number;
  render_ms?: number;
  total_ms: number;
}

export interface MeasuredTiming {
  total_ms: number;
  classification_ms?: number;
  retrieval_ms?: number;
  answer_generation_ms?: number;
  retrieval?: RetrievalTiming;
  generation?: GenerationTiming;
  ingestion?: IngestionTiming;
  scope: "measured";
}

export interface AnswerResponse {
  request_id?: string;
  response_type: "answer";
  answer: string;
  citations: Citation[];
  query_analysis: QueryAnalysis;
  evidence: EvidenceBundle;
  evidence_counts: {
    memories: number;
    states: number;
    transitions: number;
    causal_path_steps: number;
  };
  diagnostics: DiagnosticsSummary;
  timings: MeasuredTiming;
}

export interface IngestResponse {
  request_id?: string;
  response_type: "ingest";
  ingest_status: "state_recorded" | "no_new_state" | "not_ingestable";
  answer: string;
  citations: Citation[];
  query_analysis: QueryAnalysis;
  memories_added: Array<Record<string, unknown>>;
  states: StateRecord[];
  transitions: TransitionRecord[];
  diagnostics: null;
  timings: MeasuredTiming;
}

export type ChatResponse = AnswerResponse | IngestResponse;

export interface Config {
  provider: string;
  model: string;
  default_user: string;
  has_api_key: boolean;
  temporal_rule: string;
}

export interface DiagnosticSnapshot {
  user_id: string;
  provider: string;
  model: string;
  counts: {
    memories: number;
    states: number;
    active_states: number;
    transitions: number;
    explicit_causes: number;
    temporal_successions: number;
  };
  store_status: Record<string, string>;
  errors: string[];
  metrics_note: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  response?: ChatResponse;
  error?: string;
}
