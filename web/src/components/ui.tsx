import type { ReactNode } from "react";
import type { StateRecord, TransitionRecord } from "../types";

export type IconName =
  | "chat"
  | "memories"
  | "timeline"
  | "evidence"
  | "diagnostics"
  | "settings"
  | "search"
  | "database"
  | "branch"
  | "check"
  | "alert"
  | "arrow"
  | "clock"
  | "layers"
  | "spark"
  | "shield";

const paths: Record<IconName, ReactNode> = {
  chat: (
    <>
      <path d="M21 12a8 8 0 0 1-8 8H5l-2 2V12a8 8 0 0 1 8-8h2a8 8 0 0 1 8 8Z" />
      <path d="M8 11h8M8 15h5" />
    </>
  ),
  memories: (
    <>
      <path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v18H6.5A2.5 2.5 0 0 1 4 18.5Z" />
      <path d="M8 7h8M8 11h8M8 15h5" />
    </>
  ),
  timeline: (
    <>
      <path d="M6 3v18M6 6h10a2 2 0 0 1 2 2v2a2 2 0 0 1-2 2H6" />
      <path d="M6 14h7a2 2 0 0 1 2 2v2a2 2 0 0 1-2 2H6" />
      <circle cx="6" cy="3" r="1.5" />
    </>
  ),
  evidence: (
    <>
      <path d="M12 3 4 6.5V12c0 5 3.4 7.8 8 9 4.6-1.2 8-4 8-9V6.5Z" />
      <path d="m8.5 12 2.3 2.3 4.7-4.6" />
    </>
  ),
  diagnostics: (
    <>
      <path d="M4 19V5M4 19h16" />
      <path d="m7 15 3-3 3 2 4-6" />
    </>
  ),
  settings: (
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.7 1.7 0 0 0 .34 1.88l.06.06-2.12 2.12-.06-.06a1.7 1.7 0 0 0-1.88-.34 1.7 1.7 0 0 0-1.03 1.56V20h-3v-.08a1.7 1.7 0 0 0-1.03-1.56 1.7 1.7 0 0 0-1.88.34l-.06.06-2.12-2.12.06-.06A1.7 1.7 0 0 0 5.6 15a1.7 1.7 0 0 0-1.56-1.03H4v-3h.04A1.7 1.7 0 0 0 5.6 9a1.7 1.7 0 0 0-.34-1.88l-.06-.06 2.12-2.12.06.06A1.7 1.7 0 0 0 9.26 5.3 1.7 1.7 0 0 0 10.3 3.74V3h3v.74a1.7 1.7 0 0 0 1.03 1.56 1.7 1.7 0 0 0 1.88-.34l.06-.06 2.12 2.12-.06.06A1.7 1.7 0 0 0 18 9c.14.62.66 1.09 1.3 1.13H20v3h-.7A1.7 1.7 0 0 0 19.4 15Z" />
    </>
  ),
  search: (
    <>
      <circle cx="11" cy="11" r="6.5" />
      <path d="m16 16 4 4" />
    </>
  ),
  database: (
    <>
      <ellipse cx="12" cy="5.5" rx="7" ry="3" />
      <path d="M5 5.5v6c0 1.66 3.13 3 7 3s7-1.34 7-3v-6" />
      <path d="M5 11.5v6c0 1.66 3.13 3 7 3s7-1.34 7-3v-6" />
    </>
  ),
  branch: (
    <>
      <circle cx="6" cy="5" r="2" />
      <circle cx="18" cy="6" r="2" />
      <circle cx="6" cy="19" r="2" />
      <path d="M6 7v10M8 15c6 0 8-2 8-7" />
    </>
  ),
  check: <path d="m5 12 4.5 4.5L19 7" />,
  alert: (
    <>
      <path d="M12 4 3.5 20h17Z" />
      <path d="M12 9v5M12 17h.01" />
    </>
  ),
  arrow: <path d="M5 12h14M13 6l6 6-6 6" />,
  clock: (
    <>
      <circle cx="12" cy="12" r="8" />
      <path d="M12 7v5l3.5 2" />
    </>
  ),
  layers: (
    <>
      <path d="m12 3 9 5-9 5-9-5Z" />
      <path d="m3 13 9 5 9-5M3 17l9 5 9-5" />
    </>
  ),
  spark: (
    <>
      <path d="M12 3v4M12 17v4M3 12h4M17 12h4" />
      <path d="m6.3 6.3 2.8 2.8M14.9 14.9l2.8 2.8M17.7 6.3l-2.8 2.8M9.1 14.9l-2.8 2.8" />
    </>
  ),
  shield: (
    <>
      <path d="M12 3 4 6.5V12c0 5 3.4 7.8 8 9 4.6-1.2 8-4 8-9V6.5Z" />
      <path d="M12 7v6M12 16h.01" />
    </>
  ),
};

export function Icon({
  name,
  size = 18,
  className,
}: {
  name: IconName;
  size?: number;
  className?: string;
}) {
  return (
    <svg
      aria-hidden="true"
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {paths[name]}
    </svg>
  );
}

export function Badge({
  children,
  tone = "neutral",
  className = "",
  title,
}: {
  children: ReactNode;
  tone?: "neutral" | "accent" | "violet" | "warning" | "danger" | "muted";
  className?: string;
  title?: string;
}) {
  return (
    <span className={`badge badge-${tone} ${className}`.trim()} title={title}>
      {children}
    </span>
  );
}

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className = "",
}: {
  title?: string;
  subtitle?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`.trim()}>
      {(title || subtitle || actions) && (
        <header className="panel-header">
          <div>
            {title && <h2>{title}</h2>}
            {subtitle && <p>{subtitle}</p>}
          </div>
          {actions && <div className="panel-actions">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

export function EmptyState({
  icon = "layers",
  title,
  body,
}: {
  icon?: IconName;
  title: string;
  body: string;
}) {
  return (
    <div className="empty-state">
      <span className="empty-icon">
        <Icon name={icon} size={22} />
      </span>
      <h3>{title}</h3>
      <p>{body}</p>
    </div>
  );
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <span className="spinner-wrap" role="status">
      <span className="spinner" />
      <span>{label}</span>
    </span>
  );
}

export function ErrorBanner({ message, onDismiss }: { message: string; onDismiss?: () => void }) {
  return (
    <div className="error-banner" role="alert">
      <Icon name="alert" size={17} />
      <span>{message}</span>
      {onDismiss && (
        <button type="button" className="icon-button" onClick={onDismiss} aria-label="Dismiss error">
          ×
        </button>
      )}
    </div>
  );
}

export function StatCard({
  label,
  value,
  hint,
  tone = "neutral",
}: {
  label: string;
  value: string | number;
  hint?: string;
  tone?: "neutral" | "accent" | "violet";
}) {
  return (
    <div className={`stat-card stat-${tone}`}>
      <span className="stat-label">{label}</span>
      <strong>{value}</strong>
      {hint && <span className="stat-hint">{hint}</span>}
    </div>
  );
}

export function formatDate(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function formatInterval(from?: string | null, until?: string | null): string {
  return `${formatDate(from)} → ${until ? formatDate(until) : "open"}`;
}

export function titleCase(value: string): string {
  return value
    .toLowerCase()
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

const REASON_LABELS: Record<string, string> = {
  validated_claims: "Validated against evidence",
  no_validated_claims: "No validated claims",
  no_validated_causal_claim: "No validated causal claim",
  empty_evidence: "No retrieved evidence",
  ambiguous_evidence: "Ambiguous evidence",
  explicit_cause_gate_rejected: "Explicit-cause gate rejected",
  missing_explicit_cause_transition: "No explicit cause transition",
  missing_predecessor_transition: "No predecessor transition",
  ambiguous_predecessor_transition: "Ambiguous predecessor transition",
  structured_answer_method_unavailable: "Structured answer method unavailable",
  malformed_structured_response: "Malformed model response",
  unavailable: "Diagnostics unavailable",
  not_called: "Model not called",
  structured_response: "Structured response",
  malformed_claim: "Malformed claim",
  malformed_evidence_reference: "Malformed evidence reference",
  unknown_or_unowned_evidence_id: "Unknown or unowned evidence",
  claim_not_supported_by_source: "Claim not supported by source",
  causal_relation_or_support_rejected: "Causal relation or support rejected",
  temporal_relation_or_support_rejected: "Temporal relation or support rejected",
  claim_type_not_supported_by_transition: "Claim type not supported by transition",
  no_valid_evidence_reference: "No valid evidence reference",
  evidence_not_in_model_bundle: "Evidence not in model bundle",
  citation_id_unknown: "Citation ID unknown",
  citation_ref_unknown: "Citation reference unknown",
  citation_ref_conflict: "Citation reference conflict",
  citation_kind_mismatch: "Citation kind mismatch",
  citation_foreign_user: "Citation owned by another user",
  citation_owner_unverified: "Citation owner unverified",
  citation_not_in_model_bundle: "Citation not in model bundle",
  citation_resolved: "Citation resolved",
  citation_structure_malformed: "Citation structure malformed",
  fact_lexical_support_failed: "Fact lexical support failed",
  claim_not_relevant_to_question: "Claim not relevant to question",
  temporal_validity_failed: "Temporal validity failed",
  state_memory_claim_type_not_supported: "State/memory claim type unsupported",
  causal_relation_not_explicit: "Causal relation not explicit",
  causal_recorded_cause_missing: "Recorded cause missing",
  causal_query_support_missing: "Causal query support missing",
  causal_claim_missing_cause_terms: "Causal claim missing cause terms",
  causal_claim_lexical_support_failed: "Causal claim lexical support failed",
};

export function reasonLabel(value?: string | null): string {
  if (!value) return "Not available";
  return REASON_LABELS[value] ?? titleCase(value);
}

export function isLegacyState(
  state: Pick<StateRecord, "state_type" | "metadata">,
): boolean {
  return (
    state.state_type === "OTHER" ||
    state.metadata?.legacy === true ||
    state.metadata?.review_status === "legacy"
  );
}

export function isLegacyTransition(
  transition: Pick<TransitionRecord, "relation_kind" | "metadata">,
): boolean {
  return (
    transition.relation_kind === "UNKNOWN" ||
    transition.metadata?.legacy === true ||
    transition.metadata?.review_status === "legacy"
  );
}
