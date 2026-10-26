import { useEffect, useMemo, useState } from "react";
import { api, ApiError } from "../api";
import type { MemoryRecord, StateRecord } from "../types";
import {
  Badge,
  EmptyState,
  ErrorBanner,
  Icon,
  Panel,
  Spinner,
  formatDate,
  formatInterval,
  titleCase,
} from "../components/ui";

type Tab = "memories" | "states";
type StateFilter = "all" | "active" | "historical";

export default function MemoryPage({ userId }: { userId: string }) {
  const [tab, setTab] = useState<Tab>("memories");
  const [memories, setMemories] = useState<MemoryRecord[]>([]);
  const [states, setStates] = useState<StateRecord[]>([]);
  const [search, setSearch] = useState("");
  const [stateFilter, setStateFilter] = useState<StateFilter>("all");
  const [typeFilter, setTypeFilter] = useState("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    Promise.all([api.getMemories(userId), api.getStates(userId, true)])
      .then(([memoryResponse, stateResponse]) => {
        if (cancelled) return;
        setMemories(memoryResponse.memories);
        setStates(stateResponse.states);
      })
      .catch((caught) => {
        if (cancelled) return;
        setError(caught instanceof ApiError ? caught.message : "Could not load memory records.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [userId, reload]);

  const needle = search.trim().toLowerCase();
  const visibleMemories = useMemo(
    () =>
      memories.filter((record) =>
        !needle
          ? true
          : [record.memory, record.id, record.attribution]
              .filter(Boolean)
              .some((value) => String(value).toLowerCase().includes(needle)),
      ),
    [memories, needle],
  );
  const stateTypes = useMemo(
    () => Array.from(new Set(states.map((state) => state.state_type))).sort(),
    [states],
  );
  const visibleStates = useMemo(
    () =>
      states.filter((state) => {
        if (stateFilter === "active" && state.status !== "ACTIVE") return false;
        if (stateFilter === "historical" && state.status === "ACTIVE") return false;
        if (typeFilter !== "all" && state.state_type !== typeFilter) return false;
        if (!needle) return true;
        return [state.state_value, state.state_type, state.state_id, ...state.source_memory_ids]
          .filter(Boolean)
          .some((value) => String(value).toLowerCase().includes(needle));
      }),
    [states, needle, stateFilter, typeFilter],
  );

  return (
    <div className="page-stack">
      <Panel
        title="Memory explorer"
        subtitle={`Browsing namespace “${userId}”. Mem0 memories and structured states remain distinct records.`}
        actions={
          <button type="button" className="ghost-button" onClick={() => setReload((value) => value + 1)}>
            <Icon name="spark" size={15} />
            Refresh
          </button>
        }
      >
        <div className="toolbar">
          <div className="segmented-control" role="tablist" aria-label="Memory record type">
            <button
              type="button"
              role="tab"
              aria-selected={tab === "memories"}
              className={tab === "memories" ? "active" : ""}
              onClick={() => setTab("memories")}
            >
              Mem0 memories <span>{memories.length}</span>
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={tab === "states"}
              className={tab === "states" ? "active" : ""}
              onClick={() => setTab("states")}
            >
              Structured states <span>{states.length}</span>
            </button>
          </div>
          <label className="search-field">
            <Icon name="search" size={16} />
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder={tab === "memories" ? "Search memory text or ID" : "Search value, type, or source memory"}
              aria-label="Search memory records"
            />
          </label>
          {tab === "states" && (
            <>
              <select
                className="select-field"
                value={stateFilter}
                onChange={(event) => setStateFilter(event.target.value as StateFilter)}
                aria-label="Filter state lifecycle"
              >
                <option value="all">All states</option>
                <option value="active">Active only</option>
                <option value="historical">Historical only</option>
              </select>
              <select
                className="select-field"
                value={typeFilter}
                onChange={(event) => setTypeFilter(event.target.value)}
                aria-label="Filter state type"
              >
                <option value="all">All types</option>
                {stateTypes.map((type) => (
                  <option value={type} key={type}>{titleCase(type)}</option>
                ))}
              </select>
            </>
          )}
        </div>

        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}
        {loading ? (
          <div className="centered-block">
            <Spinner label="Loading memory records" />
          </div>
        ) : tab === "memories" ? (
          visibleMemories.length === 0 ? (
            <EmptyState
              icon="database"
              title="No Mem0 memories match"
              body="Tell the chat something personal, or clear the search filter."
            />
          ) : (
            <div className="record-grid">
              {visibleMemories.map((record, index) => (
                <article className="record-card" key={record.id ?? `memory-${index}`}>
                  <div className="record-card-top">
                    <Badge tone="violet">
                      <Icon name="database" size={12} /> Mem0 memory
                    </Badge>
                    {record.attribution && <Badge tone="muted">{record.attribution}</Badge>}
                  </div>
                  <p className="record-text">{record.memory ?? "—"}</p>
                  <dl className="record-meta">
                    <div>
                      <dt>ID</dt>
                      <dd className="mono">{record.id ?? "—"}</dd>
                    </div>
                    <div>
                      <dt>Created</dt>
                      <dd>{formatDate(record.created_at)}</dd>
                    </div>
                  </dl>
                </article>
              ))}
            </div>
          )
        ) : visibleStates.length === 0 ? (
          <EmptyState
            icon="layers"
            title="No structured states match"
            body="States appear when CausalMem extracts a grounded personal fact or explicit change."
          />
        ) : (
          <div className="record-grid">
            {visibleStates.map((state) => (
              <article className="record-card" key={state.state_id}>
                <div className="record-card-top">
                  <Badge tone="accent">
                    <Icon name="layers" size={12} /> {titleCase(state.state_type)}
                  </Badge>
                  <Badge tone={state.status === "ACTIVE" ? "accent" : "muted"}>
                    {state.status === "ACTIVE" ? "Current" : "Historical"}
                  </Badge>
                </div>
                <p className="record-text">{state.state_value}</p>
                <dl className="record-meta">
                  <div>
                    <dt>Validity</dt>
                    <dd>{formatInterval(state.valid_from, state.valid_until)}</dd>
                  </div>
                  <div>
                    <dt>Confidence</dt>
                    <dd>{Math.round(state.confidence * 100)}%</dd>
                  </div>
                  <div>
                    <dt>State ID</dt>
                    <dd className="mono">{state.state_id}</dd>
                  </div>
                  <div>
                    <dt>Source memories</dt>
                    <dd className="mono">
                      {state.source_memory_ids.length > 0 ? state.source_memory_ids.join(", ") : "No recorded links"}
                    </dd>
                  </div>
                </dl>
              </article>
            ))}
          </div>
        )}
      </Panel>
    </div>
  );
}
