import type { IngestionTiming, Mem0StageTiming, RetrievalTiming } from "../types";
import { Badge, Icon, titleCase } from "./ui";

function formatMs(value?: number): string {
  if (value === undefined || Number.isNaN(value)) return "—";
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`;
  return `${value.toFixed(value >= 100 ? 0 : 1)} ms`;
}

function stageTone(stage: Mem0StageTiming): "accent" | "warning" | "muted" {
  if (stage.error_category || stage.success === false) return "warning";
  if (stage.status) return "muted";
  return "accent";
}

export function IngestTrace({
  timing,
  compact = false,
}: {
  timing: IngestionTiming;
  compact?: boolean;
}) {
  const stages = timing.mem0_stages.filter((stage) => stage.stage !== "total_mem0_add");
  const maxStage = Math.max(
    ...stages.map((stage) => stage.duration_ms ?? 0),
    timing.add_memory_ms,
    timing.memory_read_ms,
    timing.state_extraction_ms,
    1,
  );
  const cells = [
    { label: "Mem0 write", value: timing.add_memory_ms, icon: "database" as const },
    { label: "Memory read", value: timing.memory_read_ms, icon: "search" as const },
    { label: "State extraction", value: timing.state_extraction_ms, icon: "layers" as const },
  ];

  return (
    <div className={`ingest-trace ${compact ? "trace-compact" : ""}`}>
      <div className="trace-head">
        <div>
          <span className="section-label">Ingestion trace</span>
          <p>
            Server-measured stages for this statement. No prompt or memory text is
            included in timing diagnostics.
          </p>
        </div>
        <Badge tone={timing.extractor_skipped ? "accent" : "violet"}>
          <Icon name={timing.extractor_skipped ? "check" : "spark"} size={12} />
          {timing.extractor_skipped ? "Rule parser" : "LLM extractor"}
        </Badge>
      </div>

      <div className="trace-cells">
        {cells.map((cell) => (
          <div className="trace-cell" key={cell.label}>
            <span className="trace-cell-label">
              <Icon name={cell.icon} size={12} />
              {cell.label}
            </span>
            <strong>{formatMs(cell.value)}</strong>
            <span className="trace-meter" aria-hidden="true">
              <i style={{ width: `${Math.max(2, (cell.value / maxStage) * 100)}%` }} />
            </span>
          </div>
        ))}
      </div>

      {stages.length > 0 && (
        <div className="trace-stages">
          {stages.map((stage, index) => {
            const duration = stage.duration_ms ?? 0;
            return (
              <div className="trace-stage" key={`${stage.stage}-${index}`}>
                <span className="trace-stage-name">{titleCase(stage.stage)}</span>
                <span className="trace-stage-meter" aria-hidden="true">
                  <i style={{ width: `${Math.max(2, (duration / maxStage) * 100)}%` }} />
                </span>
                <Badge tone={stageTone(stage)}>
                  {stage.status ?? formatMs(stage.duration_ms)}
                </Badge>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export function LatencyTrace({
  total,
  classification,
  retrieval,
  generation,
}: {
  total: number;
  classification?: number;
  retrieval?: number;
  generation?: number;
}) {
  const rows = [
    { label: "Intent routing", value: classification, tone: "neutral" },
    { label: "Retrieval", value: retrieval, tone: "violet" },
    { label: "Answer generation", value: generation, tone: "accent" },
    { label: "Total turn", value: total, tone: "neutral" },
  ].filter((row) => row.label === "Total turn" || row.value !== undefined);
  const max = Math.max(total, classification ?? 0, retrieval ?? 0, generation ?? 0, 1);
  return (
    <div className="latency-trace">
      {rows.map((row) => (
        <div className="latency-row" key={row.label}>
          <span>{row.label}</span>
          <span className="latency-meter" aria-hidden="true">
            <i
              className={`latency-${row.tone}`}
              style={{ width: `${Math.max(2, ((row.value ?? 0) / max) * 100)}%` }}
            />
          </span>
          <strong>{formatMs(row.value)}</strong>
        </div>
      ))}
    </div>
  );
}

export function RetrievalTrace({ timing }: { timing: RetrievalTiming }) {
  const rows = [
    { label: "Mem0 search", value: timing.mem0_search_ms },
    { label: "Timeline SQLite", value: timing.timeline_db_ms },
    { label: "Causal graph", value: timing.graph_ms },
  ];
  const max = Math.max(...rows.map((row) => row.value ?? 0), timing.total_ms, 1);
  return (
    <div className="latency-trace">
      {rows.map((row) => (
        <div className="latency-row" key={row.label}>
          <span>{row.label}</span>
          <span className="latency-meter" aria-hidden="true">
            <i
              className="latency-violet"
              style={{ width: `${Math.max(2, ((row.value ?? 0) / max) * 100)}%` }}
            />
          </span>
          <strong>{formatMs(row.value)}</strong>
        </div>
      ))}
    </div>
  );
}

export { formatMs };
