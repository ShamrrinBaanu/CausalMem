import type { ReactNode } from "react";
import type { ChatResponse } from "../types";
import {
  Badge,
  EmptyState,
  Icon,
  Panel,
  StatCard,
  formatDate,
  formatInterval,
  reasonLabel,
  titleCase,
} from "../components/ui";

export default function EvidencePage({
  response,
  userId,
}: {
  response: ChatResponse | null;
  userId: string;
}) {
  if (!response) {
    return (
      <Panel title="Evidence inspector" subtitle={`No validated response is loaded for “${userId}”.`}>
        <EmptyState
          icon="evidence"
          title="Nothing to inspect yet"
          body="Ask a question in Chat. The inspector will show the server-side evidence bundle, citation checks, and claim rejection counts."
        />
      </Panel>
    );
  }

  if (response.response_type === "ingest") {
    return (
      <div className="page-stack">
        <Panel
          title="Evidence inspector"
          subtitle="This turn was treated as a statement. It writes records but makes no answer claims."
        >
          <div className="ingest-summary">
            <Badge tone="violet">
              {response.ingest_status === "state_recorded"
                ? "Statement ingested"
                : response.ingest_status === "no_new_state"
                  ? "Memory stored, no new state"
                  : "Not ingestable"}
            </Badge>
            <span>{response.memories_added.length} Mem0 write(s)</span>
            <span>{response.states.length} state(s)</span>
            <span>{response.transitions.length} transition(s)</span>
          </div>
        </Panel>
        {response.states.length > 0 && (
          <Panel title="States recorded from this statement">
            <div className="detail-list">
              {response.states.map((state) => (
                <article className="detail-row" key={state.state_id}>
                  <div>
                    <Badge tone="accent">{titleCase(state.state_type)}</Badge>
                    <h3>{state.state_value}</h3>
                    <p>{formatInterval(state.valid_from, state.valid_until)}</p>
                  </div>
                  <code>{state.state_id}</code>
                </article>
              ))}
            </div>
          </Panel>
        )}
      </div>
    );
  }

  const { evidence, diagnostics, citations } = response;
  const accepted = diagnostics.accepted_claim_count;
  const rejected = diagnostics.rejected_claim_count;
  const validated = diagnostics.answer_reason === "validated_claims";
  const citationChecks = diagnostics.citation_validation ?? [];

  return (
    <div className="page-stack">
      <Panel
        title="Evidence inspector"
        subtitle="Only evidence owned by this namespace is shown. Citation checks are performed server-side."
        actions={
          <Badge tone={validated ? "accent" : rejected > 0 ? "warning" : "muted"}>
            {validated ? "Validated claims" : reasonLabel(diagnostics.answer_reason)}
          </Badge>
        }
      >
        <div className="stat-grid">
          <StatCard label="Mem0 memories" value={evidence.memories.length} hint="retrieved" tone="violet" />
          <StatCard label="Structured states" value={evidence.states.length} hint="retrieved" tone="accent" />
          <StatCard label="Transitions" value={evidence.transitions.length} hint="retrieved" />
          <StatCard label="Causal path" value={evidence.causal_path.length} hint="recorded steps" />
          <StatCard label="Accepted claims" value={accepted} hint="validated" tone="accent" />
          <StatCard label="Rejected claims" value={rejected} hint="blocked" tone={rejected > 0 ? "neutral" : "accent"} />
        </div>
      </Panel>

      <div className="two-column">
        <Panel title="Retrieval context">
          <dl className="facts">
            <Fact label="Query category" value={response.query_analysis.category} />
            <Fact label="Question routed" value={response.query_analysis.is_question ? "Yes" : "No"} />
            <Fact label="Requested time" value={evidence.requested_timestamp ? formatDate(evidence.requested_timestamp) : "Current state"} />
            <Fact label="Retrieval sources" value={evidence.retrieval_sources.join(", ") || "—"} />
            {evidence.retrieval_reason && (
              <Fact label="Retrieval reason" value={titleCase(evidence.retrieval_reason)} />
            )}
            <Fact label="Rationale" value={response.query_analysis.rationale} />
          </dl>
        </Panel>
        <Panel title="Validation gates">
          <dl className="facts">
            <Fact
              label="Causal gate"
              value={
                diagnostics.causal_gate_passed === null
                  ? "Not applicable"
                  : diagnostics.causal_gate_passed
                    ? "Passed"
                    : "Failed"
              }
            />
            <Fact
              label="Required transition citation"
              value={
                diagnostics.required_transition_citation_accepted === null
                  ? "Not applicable"
                  : diagnostics.required_transition_citation_accepted
                    ? "Accepted"
                    : "Missing or rejected"
              }
            />
            <Fact label="Model response status" value={titleCase(diagnostics.model_response_status)} />
            <Fact label="Answer reason" value={reasonLabel(diagnostics.answer_reason)} />
          </dl>
          {Object.keys(diagnostics.rejection_reasons).length > 0 && (
            <div className="reason-list">
              <span className="section-label">Rejection reasons</span>
              {Object.entries(diagnostics.rejection_reasons).map(([reason, count]) => (
                <div className="reason-pill" key={reason} title={reason}>
                  <span>{reasonLabel(reason)}</span>
                  <span>{count}</span>
                </div>
              ))}
            </div>
          )}
        </Panel>
      </div>

      <Panel
        title="Citation validation"
        subtitle="Short model citation refs are resolved to real evidence IDs before ownership and bundle checks. Checks are per model reference; rendered citations are deduplicated."
      >
        {citationChecks.length === 0 ? (
          <EmptyState icon="shield" title="No citation checks emitted" body="This response did not include any model evidence references to validate." />
        ) : (
          <div className="citation-table">
            <div className="citation-table-head">
              <span>Ref</span>
              <span>Kind</span>
              <span>Resolved &amp; owned</span>
              <span>Kind matches</span>
              <span>Reason</span>
            </div>
            {citationChecks.map((check, index) => {
              const ok = Boolean(check.evidence_id_resolves_owned && check.kind_matches_resolved_type);
              return (
                <div className="citation-table-row" key={`citation-check-${index}`}>
                  <span className="mono citation-id">#{index + 1}</span>
                  <span className="citation-kind">{check.citation_kind}</span>
                  <span>
                    <Badge tone={check.evidence_id_resolves_owned ? "accent" : "danger"}>
                      {check.evidence_id_resolves_owned ? "Yes" : "No"}
                    </Badge>
                  </span>
                  <span>
                    <Badge tone={check.kind_matches_resolved_type ? "accent" : "warning"}>
                      {check.kind_matches_resolved_type === null
                        ? "N/A"
                        : check.kind_matches_resolved_type
                          ? "Yes"
                          : "No"}
                    </Badge>
                  </span>
                  <span className={ok ? "reason-ok" : "reason-bad"}>{check.reason_code}</span>
                </div>
              );
            })}
          </div>
        )}
        <div className="rendered-citations">
          <span className="section-label">Rendered answer citations</span>
          {citations.length === 0 ? (
            <p className="muted-small">This answer did not cite any evidence.</p>
          ) : (
            <div className="citation-row">
              {citations.map((citation) => (
                <Badge
                  key={`${citation.kind}-${citation.id}`}
                  tone={citation.kind === "transition" ? "warning" : citation.kind === "state" ? "accent" : "violet"}
                  className="citation-badge"
                >
                  {citation.kind}:{citation.id}
                </Badge>
              ))}
            </div>
          )}
        </div>
      </Panel>

      <EvidenceSection title="Mem0 memories" count={evidence.memories.length}>
        {evidence.memories.map((memory, index) => (
          <article className="evidence-item" key={memory.id ?? `memory-${index}`}>
            <div className="evidence-item-top">
              <Badge tone="violet"><Icon name="database" size={12} /> Mem0</Badge>
              <code>{memory.id ?? "—"}</code>
            </div>
            <p>{memory.memory ?? "—"}</p>
            <span className="muted-small">Created {formatDate(memory.created_at)}</span>
          </article>
        ))}
      </EvidenceSection>

      <EvidenceSection title="Structured states" count={evidence.states.length}>
        {evidence.states.map((state) => (
          <article className="evidence-item" key={state.state_id}>
            <div className="evidence-item-top">
              <Badge tone="accent"><Icon name="layers" size={12} /> {titleCase(state.state_type)}</Badge>
              <code>{state.state_id}</code>
            </div>
            <p>{state.state_value}</p>
            <span className="muted-small">
              {state.status} · {formatInterval(state.valid_from, state.valid_until)} · {Math.round(state.confidence * 100)}% confidence
            </span>
          </article>
        ))}
      </EvidenceSection>

      <EvidenceSection title="Transitions" count={evidence.transitions.length}>
        {evidence.transitions.map((transition) => (
          <article className="evidence-item" key={transition.transition_id}>
            <div className="evidence-item-top">
              <Badge tone={transition.relation_kind === "EXPLICIT_CAUSE" ? "warning" : "muted"}>
                <Icon name="branch" size={12} /> {titleCase(transition.relation_kind)}
              </Badge>
              <code>{transition.transition_id}</code>
            </div>
            <p className="mono">
              {transition.from_state_id} <Icon name="arrow" size={12} /> {transition.to_state_id}
            </p>
            <p>{transition.trigger || transition.reason || "No trigger recorded"}</p>
            <span className="muted-small">{formatDate(transition.timestamp)}</span>
          </article>
        ))}
      </EvidenceSection>

      <EvidenceSection title="Recorded causal path" count={evidence.causal_path.length}>
        {evidence.causal_path.map((step, index) => (
          <article className="evidence-item" key={step.transition_id ?? `path-${index}`}>
            <div className="evidence-item-top">
              <Badge tone="warning">Step {index + 1}</Badge>
              <code>{step.transition_id ?? "—"}</code>
            </div>
            <p>{step.from ?? step.from_state_id} <Icon name="arrow" size={12} /> {step.to ?? step.to_state_id}</p>
            <span className="muted-small">{step.relation_kind ? titleCase(step.relation_kind) : "—"}</span>
          </article>
        ))}
      </EvidenceSection>

      <Panel title="Timing">
        <dl className="facts inline-facts">
          <Fact label="Total" value={`${response.timings.total_ms.toFixed(1)} ms`} />
          <Fact label="Intent routing" value={response.timings.classification_ms !== undefined ? `${response.timings.classification_ms.toFixed(1)} ms` : "n/a"} />
          <Fact label="Retrieval" value={response.timings.retrieval_ms !== undefined ? `${response.timings.retrieval_ms.toFixed(1)} ms` : "—"} />
          <Fact label="Answer generation" value={response.timings.answer_generation_ms !== undefined ? `${response.timings.answer_generation_ms.toFixed(1)} ms` : "—"} />
          <Fact label="Request ID" value={response.request_id ?? "n/a"} />
          <Fact label="Scope" value={response.timings.scope} />
        </dl>
      </Panel>
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function EvidenceSection({
  title,
  count,
  children,
}: {
  title: string;
  count: number;
  children: ReactNode;
}) {
  if (count === 0) {
    return (
      <Panel title={title}>
        <p className="muted-small">No records of this type were retrieved.</p>
      </Panel>
    );
  }
  return (
    <Panel title={title} subtitle={`${count} record(s) from the validated bundle`}>
      <div className="detail-list">{children}</div>
    </Panel>
  );
}
