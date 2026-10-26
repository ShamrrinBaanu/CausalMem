import { useEffect, useMemo, useState } from "react";
import { api, ApiError } from "../api";
import type { StateRecord, TransitionRecord } from "../types";
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

type TimelineEvent =
  | { kind: "state"; time: string; state: StateRecord }
  | { kind: "transition"; time: string; transition: TransitionRecord };

export default function TimelinePage({ userId }: { userId: string }) {
  const [states, setStates] = useState<StateRecord[]>([]);
  const [transitions, setTransitions] = useState<TransitionRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  const [search, setSearch] = useState("");
  const [stateFilter, setStateFilter] = useState("all");
  const [relationFilter, setRelationFilter] = useState("all");
  const [typeFilter, setTypeFilter] = useState("all");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .getTimeline(userId)
      .then((response) => {
        if (cancelled) return;
        setStates(response.states);
        setTransitions(response.transitions);
      })
      .catch((caught) => {
        if (!cancelled) setError(caught instanceof ApiError ? caught.message : "Could not load the timeline.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [userId, reload]);

  const stateById = useMemo(
    () => new Map(states.map((state) => [state.state_id, state])),
    [states],
  );
  const events = useMemo<TimelineEvent[]>(() => {
    const merged: TimelineEvent[] = [
      ...states.map((state) => ({ kind: "state" as const, time: state.valid_from, state })),
      ...transitions.map((transition) => ({
        kind: "transition" as const,
        time: transition.timestamp,
        transition,
      })),
    ];
    return merged.sort((left, right) => {
      const leftTime = new Date(left.time).getTime();
      const rightTime = new Date(right.time).getTime();
      if (Number.isNaN(leftTime) || Number.isNaN(rightTime)) return left.time.localeCompare(right.time);
      return leftTime - rightTime;
    });
  }, [states, transitions]);

  const stateTypes = useMemo(
    () => Array.from(new Set(states.map((state) => state.state_type))).sort(),
    [states],
  );
  const needle = search.trim().toLowerCase();
  const visibleEvents = useMemo(
    () =>
      events.filter((event) => {
        if (event.kind === "state") {
          if (stateFilter === "active" && event.state.status !== "ACTIVE") return false;
          if (stateFilter === "historical" && event.state.status === "ACTIVE") return false;
          if (typeFilter !== "all" && event.state.state_type !== typeFilter) return false;
          if (relationFilter !== "all") return false;
          if (!needle) return true;
          return [event.state.state_value, event.state.state_type, event.state.state_id]
            .some((value) => value.toLowerCase().includes(needle));
        }
        if (relationFilter === "explicit" && event.transition.relation_kind !== "EXPLICIT_CAUSE") {
          return false;
        }
        if (relationFilter === "temporal" && event.transition.relation_kind !== "TEMPORAL_SUCCESSION") {
          return false;
        }
        if (relationFilter === "other") {
          const relation = event.transition.relation_kind;
          if (relation === "EXPLICIT_CAUSE" || relation === "TEMPORAL_SUCCESSION") return false;
        }
        if (stateFilter !== "all" || typeFilter !== "all") return false;
        if (!needle) return true;
        const source = stateById.get(event.transition.from_state_id)?.state_value ?? "";
        const destination = stateById.get(event.transition.to_state_id)?.state_value ?? "";
        return [
          source,
          destination,
          event.transition.transition_type,
          event.transition.relation_kind,
          event.transition.trigger,
          event.transition.reason,
        ]
          .filter(Boolean)
          .some((value) => String(value).toLowerCase().includes(needle));
      }),
    [events, needle, relationFilter, stateById, stateFilter, typeFilter],
  );
  const filtersActive =
    Boolean(needle) || stateFilter !== "all" || relationFilter !== "all" || typeFilter !== "all";

  return (
    <div className="page-stack">
      <Panel
        title="State and transition timeline"
        subtitle={`Chronological recorded history for “${userId}”. Causality is shown only where an EXPLICIT_CAUSE transition exists.`}
        actions={
          <button type="button" className="ghost-button" onClick={() => setReload((value) => value + 1)}>
            <Icon name="spark" size={15} />
            Refresh
          </button>
        }
      >
        <div className="legend-row">
          <span className="legend-item"><i className="legend-dot state" /> State interval</span>
          <span className="legend-item"><i className="legend-dot cause" /> Explicit cause</span>
          <span className="legend-item"><i className="legend-dot sequence" /> Temporal succession</span>
          <span className="legend-item"><i className="legend-dot unknown" /> Unknown relation</span>
        </div>

        <div className="toolbar timeline-toolbar">
          <label className="search-field">
            <Icon name="search" size={16} />
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search values, causes, or IDs"
              aria-label="Search timeline"
            />
          </label>
          <select
            className="select-field"
            value={stateFilter}
            onChange={(event) => setStateFilter(event.target.value)}
            aria-label="Filter by state lifecycle"
          >
            <option value="all">All records</option>
            <option value="active">Current states</option>
            <option value="historical">Historical states</option>
          </select>
          <select
            className="select-field"
            value={typeFilter}
            onChange={(event) => setTypeFilter(event.target.value)}
            aria-label="Filter by state type"
          >
            <option value="all">All state types</option>
            {stateTypes.map((type) => (
              <option value={type} key={type}>{titleCase(type)}</option>
            ))}
          </select>
          <select
            className="select-field"
            value={relationFilter}
            onChange={(event) => setRelationFilter(event.target.value)}
            aria-label="Filter by relation"
          >
            <option value="all">All relations</option>
            <option value="explicit">Explicit cause</option>
            <option value="temporal">Temporal succession</option>
            <option value="other">Other</option>
          </select>
          {filtersActive && (
            <button
              type="button"
              className="ghost-button"
              onClick={() => {
                setSearch("");
                setStateFilter("all");
                setTypeFilter("all");
                setRelationFilter("all");
              }}
            >
              Clear
            </button>
          )}
          <span className="timeline-count">
            {visibleEvents.length} / {events.length} events
          </span>
        </div>

        {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}
        {loading ? (
          <div className="centered-block">
            <Spinner label="Loading recorded timeline" />
          </div>
        ) : visibleEvents.length === 0 ? (
          <EmptyState
            icon="timeline"
            title={events.length === 0 ? "No recorded timeline" : "No events match"}
            body={
              events.length === 0
                ? "State changes appear here after a grounded personal update is ingested."
                : "Clear or broaden the timeline filters to see recorded history."
            }
          />
        ) : (
          <div className="timeline">
            {visibleEvents.map((event, index) =>
              event.kind === "state" ? (
                <StateEvent key={`state-${event.state.state_id}-${index}`} state={event.state} />
              ) : (
                <TransitionEvent
                  key={`transition-${event.transition.transition_id}-${index}`}
                  transition={event.transition}
                  source={stateById.get(event.transition.from_state_id)}
                  destination={stateById.get(event.transition.to_state_id)}
                />
              ),
            )}
          </div>
        )}
      </Panel>
    </div>
  );
}

function StateEvent({ state }: { state: StateRecord }) {
  return (
    <article className="timeline-event">
      <span className={`timeline-marker ${state.status === "ACTIVE" ? "marker-active" : "marker-history"}`} />
      <div className="timeline-card">
        <div className="timeline-card-top">
          <Badge tone="accent">
            <Icon name="layers" size={12} /> {titleCase(state.state_type)}
          </Badge>
          <Badge tone={state.status === "ACTIVE" ? "accent" : "muted"}>
            {state.status === "ACTIVE" ? "Current" : "Historical"}
          </Badge>
          <span className="timeline-time">{formatDate(state.valid_from)}</span>
        </div>
        <h3>{state.state_value}</h3>
        <p className="timeline-meta">{formatInterval(state.valid_from, state.valid_until)}</p>
        <p className="timeline-meta mono">{state.state_id}</p>
      </div>
    </article>
  );
}

function TransitionEvent({
  transition,
  source,
  destination,
}: {
  transition: TransitionRecord;
  source?: StateRecord;
  destination?: StateRecord;
}) {
  const relation = transition.relation_kind;
  const explicitCause = relation === "EXPLICIT_CAUSE";
  const unknown = relation !== "EXPLICIT_CAUSE" && relation !== "TEMPORAL_SUCCESSION";
  const marker = explicitCause ? "marker-cause" : unknown ? "marker-unknown" : "marker-sequence";
  return (
    <article className="timeline-event">
      <span className={`timeline-marker ${marker}`} />
      <div className={`timeline-card transition-card ${explicitCause ? "transition-cause" : ""}`}>
        <div className="timeline-card-top">
          <Badge tone={explicitCause ? "warning" : unknown ? "muted" : "neutral"}>
            <Icon name="branch" size={12} />
            {titleCase(relation)}
          </Badge>
          <span className="timeline-time">{formatDate(transition.timestamp)}</span>
        </div>
        <div className="transition-path">
          <span>{source?.state_value ?? transition.from_state_id}</span>
          <Icon name="arrow" size={16} />
          <span>{destination?.state_value ?? transition.to_state_id}</span>
        </div>
        {explicitCause ? (
          <div className="cause-block">
            <span className="cause-label">Recorded cause</span>
            <p>{transition.trigger || transition.reason || "Recorded explicit cause"}</p>
            {transition.reason && transition.trigger && (
              <p className="timeline-meta">{transition.reason}</p>
            )}
          </div>
        ) : (
          <p className="timeline-meta">
            {unknown
              ? "Relationship not sufficiently established."
              : "Sequence only. No cause is recorded for this transition."}
          </p>
        )}
        <p className="timeline-meta mono">{transition.transition_id}</p>
      </div>
    </article>
  );
}
