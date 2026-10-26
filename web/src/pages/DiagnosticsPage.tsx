import { useEffect, useState } from "react";
import { api, ApiError } from "../api";
import type { ChatResponse, DiagnosticSnapshot } from "../types";
import { IngestTrace, LatencyTrace, RetrievalTrace } from "../components/Timing";
import {
  Badge,
  EmptyState,
  ErrorBanner,
  Panel,
  Spinner,
  StatCard,
  reasonLabel,
  titleCase,
} from "../components/ui";

export default function DiagnosticsPage({
  userId,
  lastResponse,
}: {
  userId: string;
  lastResponse: ChatResponse | null;
}) {
  const [snapshot, setSnapshot] = useState<DiagnosticSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .getDiagnostics(userId)
      .then((response) => {
        if (!cancelled) setSnapshot(response);
      })
      .catch((caught) => {
        if (!cancelled) setError(caught instanceof ApiError ? caught.message : "Could not load diagnostics.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [userId, reload]);

  const diagnostics = lastResponse?.response_type === "answer" ? lastResponse.diagnostics : null;

  return (
    <div className="page-stack">
      <Panel
        title="Diagnostics"
        subtitle="Aggregate store health and the last validated response. Raw prompts and model output are never exposed."
        actions={
          <button type="button" className="ghost-button" onClick={() => setReload((value) => value + 1)}>
            Refresh
          </button>
        }
      >
        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}
        {loading ? (
          <div className="centered-block">
            <Spinner label="Loading diagnostics" />
          </div>
        ) : snapshot ? (
          <>
            <div className="stat-grid">
              <StatCard label="Mem0 memories" value={snapshot.counts.memories} hint="namespace total" tone="violet" />
              <StatCard label="Structured states" value={snapshot.counts.states} hint="all intervals" tone="accent" />
              <StatCard label="Current states" value={snapshot.counts.active_states} hint="valid now" tone="accent" />
              <StatCard label="Transitions" value={snapshot.counts.transitions} hint="recorded links" />
              <StatCard label="Explicit causes" value={snapshot.counts.explicit_causes} hint="asserted causal links" />
              <StatCard label="Temporal successions" value={snapshot.counts.temporal_successions} hint="order only" />
            </div>
            <div className="two-column compact">
              <dl className="facts">
                <dt>Provider</dt>
                <dd><Badge tone="neutral">{snapshot.provider}</Badge></dd>
                <dt>Model</dt>
                <dd><code>{snapshot.model}</code></dd>
                <dt>Namespace</dt>
                <dd><code>{snapshot.user_id}</code></dd>
              </dl>
              <dl className="facts">
                {Object.entries(snapshot.store_status).map(([store, status]) => (
                  <div key={store}>
                    <dt>{titleCase(store)}</dt>
                    <dd>
                      <Badge tone={status === "ok" ? "accent" : "danger"}>{status}</Badge>
                    </dd>
                  </div>
                ))}
              </dl>
            </div>
            {snapshot.errors.length > 0 && (
              <div className="inline-errors">
                {snapshot.errors.map((message) => (
                  <ErrorBanner key={message} message={message} />
                ))}
              </div>
            )}
            <p className="metrics-note">{snapshot.metrics_note}</p>
          </>
        ) : (
          <EmptyState icon="diagnostics" title="No diagnostic snapshot" body="The backend did not return store status." />
        )}
      </Panel>

      <Panel
        title="Last response integrity"
        subtitle="Claim-level validation result from the most recent answer in this session."
      >
        {!lastResponse ? (
          <EmptyState icon="shield" title="No response this session" body="Ask a question to populate claim and citation diagnostics." />
        ) : lastResponse.response_type === "ingest" ? (
          <>
            <div className="facts inline-facts">
              <div>
                <dt>Response type</dt>
                <dd><Badge tone="violet">Statement ingested</Badge></dd>
              </div>
              <div>
                <dt>States recorded</dt>
                <dd>{lastResponse.states.length}</dd>
              </div>
              <div>
                <dt>Transitions recorded</dt>
                <dd>{lastResponse.transitions.length}</dd>
              </div>
              <div>
                <dt>Total time</dt>
                <dd>{lastResponse.timings.total_ms.toFixed(1)} ms</dd>
              </div>
              {lastResponse.request_id && (
                <div>
                  <dt>Request ID</dt>
                  <dd className="mono">{lastResponse.request_id}</dd>
                </div>
              )}
            </div>
            {lastResponse.timings.ingestion && (
              <IngestTrace timing={lastResponse.timings.ingestion} />
            )}
          </>
        ) : diagnostics ? (
          <>
            <div className="fact-grid">
              <div>
                <span className="section-label">Answer reason</span>
                <p>{reasonLabel(diagnostics.answer_reason)}</p>
              </div>
              <div>
                <span className="section-label">Causal gate</span>
                <p>
                  {diagnostics.causal_gate_passed === null
                    ? "Not applicable"
                    : diagnostics.causal_gate_passed
                      ? "Passed"
                      : "Failed"}
                </p>
              </div>
              <div>
                <span className="section-label">Accepted / rejected</span>
                <p>{diagnostics.accepted_claim_count} / {diagnostics.rejected_claim_count}</p>
              </div>
              <div>
                <span className="section-label">Model status</span>
                <p>{reasonLabel(diagnostics.model_response_status)}</p>
              </div>
              {lastResponse.request_id && (
                <div>
                  <span className="section-label">Request ID</span>
                  <p className="mono">{lastResponse.request_id}</p>
                </div>
              )}
            </div>
            <div className="trace-block">
              <span className="section-label">Turn latency</span>
              <LatencyTrace
                total={lastResponse.timings.total_ms}
                classification={lastResponse.timings.classification_ms}
                retrieval={lastResponse.timings.retrieval_ms}
                generation={lastResponse.timings.answer_generation_ms}
              />
            </div>
            {lastResponse.timings.retrieval && (
              <div className="trace-block">
                <span className="section-label">Retrieval breakdown</span>
                <RetrievalTrace timing={lastResponse.timings.retrieval} />
              </div>
            )}
            {Object.keys(diagnostics.rejection_categories).length > 0 && (
              <div className="reason-list">
                <span className="section-label">Rejection categories</span>
                {Object.entries(diagnostics.rejection_categories).map(([category, count]) => (
                  <div className="reason-pill" key={category} title={category}>
                    <span>{reasonLabel(category)}</span>
                    <span>{count}</span>
                  </div>
                ))}
              </div>
            )}
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
            <p className="metrics-note">
              Timings are measured on the server for this turn; they are not an accuracy metric.
              Last response at session level only.
            </p>
          </>
        ) : null}
      </Panel>
    </div>
  );
}
