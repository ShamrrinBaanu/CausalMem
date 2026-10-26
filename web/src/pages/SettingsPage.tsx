import { useEffect, useState } from "react";
import type { Config } from "../types";
import { Badge, Icon, Panel } from "../components/ui";

export default function SettingsPage({
  userId,
  setUserId,
  config,
}: {
  userId: string;
  setUserId: (value: string) => void;
  config: Config | null;
}) {
  const [draft, setDraft] = useState(userId);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setDraft(userId);
    setSaved(false);
  }, [userId]);

  const normalized = draft.trim();
  const valid = normalized.length > 0 && normalized.length <= 128;

  return (
    <div className="page-stack">
      <Panel
        title="Workspace settings"
        subtitle="Choose the memory namespace. The namespace is an isolation boundary, not an authentication mechanism."
      >
        <div className="settings-form">
          <label htmlFor="namespace-input">Active namespace</label>
          <div className="inline-form">
            <input
              id="namespace-input"
              className="text-field"
              value={draft}
              maxLength={128}
              onChange={(event) => {
                setDraft(event.target.value);
                setSaved(false);
              }}
              placeholder="default"
            />
            <button
              type="button"
              className="primary-button"
              disabled={!valid || normalized === userId}
              onClick={() => {
                setUserId(normalized);
                setSaved(true);
              }}
            >
              Use namespace
            </button>
          </div>
          <p className="field-hint">
            User IDs are namespaces, not authentication. Do not rely on them for access control.
          </p>
          {saved && <p className="success-note"><Icon name="check" size={14} /> Namespace updated.</p>}
        </div>

        <div className="namespace-card">
          <span className="section-label">Currently active</span>
          <code>{userId}</code>
          <p>All chat, memory, timeline, evidence, and diagnostic queries are scoped to this namespace.</p>
        </div>
      </Panel>

      <Panel title="Model and storage configuration" subtitle="Read-only values reported by the backend configuration endpoint.">
        <dl className="facts settings-facts">
          <div>
            <dt>Provider</dt>
            <dd>{config ? <Badge tone="neutral">{config.provider}</Badge> : "—"}</dd>
          </div>
          <div>
            <dt>Model</dt>
            <dd><code>{config?.model ?? "—"}</code></dd>
          </div>
          <div>
            <dt>Default namespace</dt>
            <dd><code>{config?.default_user ?? "—"}</code></dd>
          </div>
          <div>
            <dt>Hosted API key</dt>
            <dd>{config ? <Badge tone={config.has_api_key ? "accent" : "muted"}>{config.has_api_key ? "Configured" : "Not configured"}</Badge> : "—"}</dd>
          </div>
          <div>
            <dt>Temporal rule</dt>
            <dd>{config?.temporal_rule ?? "—"}</dd>
          </div>
        </dl>
        <p className="metrics-note">
          Provider credentials and raw prompts are never returned by the API. Local-only operation is supported with Ollama.
        </p>
      </Panel>

      <Panel title="API" subtitle="The React UI talks only to the local FastAPI service.">
        <div className="api-links">
          <a href="/docs" target="_blank" rel="noreferrer">
            OpenAPI documentation <Icon name="arrow" size={14} />
          </a>
          <span className="muted-small">Interactive API docs are served by FastAPI at <code>/docs</code>.</span>
        </div>
      </Panel>
    </div>
  );
}
