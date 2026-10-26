"""UI-independent chat workflow shared by the HTTP API and legacy helpers."""
from __future__ import annotations

import logging
import re
import time
import uuid
from datetime import datetime, timezone
from collections.abc import Mapping

from answer_generator import generate_answer_with_diagnostics, safe_answer_diagnostic_summary
from hybrid_retriever import retrieve
from mem0_engine import add_memory, get_memories, memory_ids_from_add_response
from models import QueryAnalysis
from query_router import classify_query
from state_extractor import (
    can_extract_deterministically,
    ingest_message,
    is_ingestable_message,
)

logger = logging.getLogger(__name__)

_CITATION_BLOCK = re.compile(r"\[Sources:\s*([^\]]+)\]")
_CITATION_TOKEN = re.compile(r"^(memory|state|transition):(.+)$")


def parse_citations(answer: str) -> list[dict[str, str]]:
    """Extract the validated citations rendered into an answer."""
    citations: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    if not isinstance(answer, str):
        return citations
    for block in _CITATION_BLOCK.findall(answer):
        for token in block.split(","):
            match = _CITATION_TOKEN.match(token.strip())
            if match is None:
                continue
            kind, identifier = match.group(1), match.group(2).strip()
            if not identifier or (kind, identifier) in seen:
                continue
            seen.add((kind, identifier))
            citations.append({"kind": kind, "id": identifier})
    return citations


def _visible_answer(answer: str) -> str:
    """Remove machine-readable source markers from chat text.

    Citation IDs remain available in the separate response ``citations`` field
    and the full evidence bundle, where the Evidence Inspector displays them.
    """
    cleaned = re.sub(r"\s*\[Sources:\s*[^\]]*\]", "", answer)
    return re.sub(r"\s+([,.;:!?])", r"\1", cleaned).strip()


def _public_memory(record: Mapping) -> dict:
    """Return only UI-relevant Mem0 fields for the requesting namespace."""
    return {
        "id": record.get("id") or record.get("memory_id"),
        "memory": record.get("memory") or record.get("text") or record.get("content"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "attribution": record.get("attributed_to") or record.get("attribution"),
    }


def public_evidence(evidence: Mapping) -> dict:
    """Shape a retrieval bundle for display without dropping provenance."""
    return {
        "query_category": evidence.get("query_category"),
        "retrieval_sources": evidence.get("retrieval_sources", []),
        "retrieval_reason": evidence.get("retrieval_reason"),
        "requested_timestamp": evidence.get("requested_timestamp"),
        "temporal_context": evidence.get("temporal_context"),
        "ambiguity": evidence.get("ambiguity"),
        "state_focus": evidence.get("state_focus"),
        "memories": [_public_memory(record) for record in evidence.get("memories", [])
                     if isinstance(record, Mapping)],
        "states": [dict(state) for state in evidence.get("states", [])
                   if isinstance(state, Mapping)],
        "transitions": [dict(transition) for transition in evidence.get("transitions", [])
                        if isinstance(transition, Mapping)],
        "causal_path": [dict(path) for path in evidence.get("causal_path", [])
                        if isinstance(path, Mapping)],
    }


def evidence_counts(evidence: Mapping) -> dict:
    """Count only the evidence the server actually bundled for this response."""
    return {
        "memories": len(evidence.get("memories", []) or []),
        "states": len(evidence.get("states", []) or []),
        "transitions": len(evidence.get("transitions", []) or []),
        "causal_path_steps": len(evidence.get("causal_path", []) or []),
    }


def _measured_timing(started: float, retrieval_started: float | None,
                     generation_started: float | None) -> dict:
    finished = time.perf_counter()
    timing = {"total_ms": round((finished - started) * 1000, 1), "scope": "measured"}
    if retrieval_started is not None:
        end = generation_started if generation_started is not None else finished
        timing["retrieval_ms"] = round((end - retrieval_started) * 1000, 1)
    if generation_started is not None:
        timing["answer_generation_ms"] = round((finished - generation_started) * 1000, 1)
    return timing


def _public_mem0_stages(records: list[dict]) -> list[dict]:
    """Expose only low-cardinality timing fields, never Mem0 arguments."""
    stages: list[dict] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        stage = record.get("stage")
        if not isinstance(stage, str) or not stage:
            continue
        item = {"stage": stage[:80]}
        duration = record.get("duration_seconds")
        if isinstance(duration, (int, float)):
            item["duration_ms"] = round(duration * 1000, 1)
        status = record.get("status")
        if isinstance(status, str):
            item["status"] = status[:40]
        success = record.get("success")
        if isinstance(success, bool):
            item["success"] = success
        error_category = record.get("error_category")
        if isinstance(error_category, str):
            item["error_category"] = error_category[:40]
        stages.append(item)
    return stages


def _ingestion_timing(add_memory_ms: float, memory_read_ms: float,
                      state_extraction_ms: float, extractor_skipped: bool,
                      mem0_stages: list[dict]) -> dict:
    return {
        "add_memory_ms": round(add_memory_ms, 1),
        "memory_read_ms": round(memory_read_ms, 1),
        "state_extraction_ms": round(state_extraction_ms, 1),
        "extractor_skipped": extractor_skipped,
        "mem0_infer": not extractor_skipped,
        "mem0_stages": mem0_stages,
    }


def _write_failure_response(analysis, started: float, status: str,
                            errors: list[str], memory_ids: list[str],
                            add_memory_ms: float, memory_read_ms: float,
                            state_extraction_ms: float, extractor_skipped: bool,
                            mem0_stages: list[dict], request_id: str | None = None) -> dict:
    """Return an explicit partial-write outcome instead of hiding a store error."""
    timings = _measured_timing(started, None, None)
    timings["ingestion"] = _ingestion_timing(
        add_memory_ms, memory_read_ms, state_extraction_ms,
        extractor_skipped, mem0_stages,
    )
    return {
        "request_id": request_id,
        "response_type": "ingest",
        "ingest_status": status,
        "answer": (
            "The memory write was only partially completed. "
            "Check diagnostics before retrying."
        ),
        "citations": [],
        "query_analysis": analysis.model_dump(),
        "memories_added": [],
        "states": [],
        "transitions": [],
        "diagnostics": None,
        "storage_warnings": [],
        "storage_errors": errors,
        "memory_ids": memory_ids,
        "timings": timings,
    }


def _ignored_ingest_response(analysis, started: float,
                             request_id: str | None = None) -> dict:
    """Abstain from writes when the message is not a user assertion."""
    return {
        "request_id": request_id,
        "response_type": "ingest",
        "ingest_status": "not_ingestable",
        "answer": (
            "Nothing was stored because the message is not a clear first-person "
            "statement about you. Rephrase it as a fact or event you want "
            "remembered."
        ),
        "citations": [],
        "query_analysis": analysis.model_dump(),
        "memories_added": [],
        "states": [],
        "transitions": [],
        "diagnostics": None,
        "timings": _measured_timing(started, None, None),
    }


def process_chat(message: str, user_id: str, requested_timestamp: str | None = None,
                 source_role: str = "user") -> dict:
    """Run one message through the actual CausalMem memory and answer path."""
    started = time.perf_counter()
    source_timestamp = datetime.now(timezone.utc)
    request_id = f"req_{uuid.uuid4().hex}"

    # Assistant output and other non-user roles are never durable user facts.
    # Reject them before classification or any model/store call.
    if source_role != "user":
        analysis = QueryAnalysis(
            category="OTHER",
            rationale="Non-user source role",
            is_question=False,
        )
        return _ignored_ingest_response(analysis, started, request_id)

    classification_started = time.perf_counter()
    deterministic = can_extract_deterministically(message)
    if deterministic:
        # A deterministic statement cannot be a question: its durable clauses
        # were already parsed by the local extractor. Avoid a redundant model
        # classification pass before the write.
        analysis = QueryAnalysis(
            is_question=False,
            category="OTHER",
            rationale="Deterministic statement; intent classification skipped",
        )
    else:
        analysis = classify_query(message)
    classification_ms = (time.perf_counter() - classification_started) * 1000

    if analysis.is_question:
        retrieval_started = time.perf_counter()
        retrieval_timings: dict = {}
        evidence = retrieve(message, user_id, analysis.category,
                            requested_timestamp, timings=retrieval_timings)
        generation_started = time.perf_counter()
        answer, diagnostics = generate_answer_with_diagnostics(message, evidence)
        summary = safe_answer_diagnostic_summary(diagnostics)
        timings = _measured_timing(started, retrieval_started, generation_started)
        timings["classification_ms"] = classification_ms
        timings["retrieval"] = retrieval_timings
        generation_timings = diagnostics.get("timings")
        if isinstance(generation_timings, Mapping):
            timings["generation"] = dict(generation_timings)
        return {
            "request_id": request_id,
            "response_type": "answer",
            "citations": parse_citations(answer),
            "answer": _visible_answer(answer),
            "query_analysis": analysis.model_dump(),
            "evidence": public_evidence(evidence),
            "evidence_counts": evidence_counts(evidence),
            "diagnostics": summary,
            "timings": timings,
        }

    if not is_ingestable_message(message):
        return _ignored_ingest_response(analysis, started, request_id)

    mem0_stage_records: list[dict] = []
    add_started = time.perf_counter()
    try:
        result = add_memory(
            message,
            user_id,
            diagnostics_callback=mem0_stage_records.append,
            # The deterministic state extractor is authoritative for these
            # messages. Store the raw memory for provenance without paying for
            # Mem0's second LLM fact-extraction pass.
            infer=not deterministic,
        )
    except Exception as exc:
        add_memory_ms = (time.perf_counter() - add_started) * 1000
        logger.error("Mem0 write failed (%s)", type(exc).__name__)
        return _write_failure_response(
            analysis, started, "mem0_write_failed", ["mem0_write_failed"], [],
            add_memory_ms, 0.0, 0.0, deterministic,
            _public_mem0_stages(mem0_stage_records), request_id,
        )
    add_memory_ms = (time.perf_counter() - add_started) * 1000

    memory_read_started = time.perf_counter()
    memories = [] if deterministic else get_memories(user_id)
    memory_read_ms = (time.perf_counter() - memory_read_started) * 1000

    memory_ids = memory_ids_from_add_response(result)
    extraction_started = time.perf_counter()
    try:
        states, transitions = ingest_message(
            message,
            user_id,
            memory_ids,
            memories,
            source_role=source_role,
            allow_llm=not deterministic,
            source_timestamp=source_timestamp,
        )
    except Exception as exc:
        state_extraction_ms = (time.perf_counter() - extraction_started) * 1000
        logger.error("Structured state write failed (%s)", type(exc).__name__)
        return _write_failure_response(
            analysis, started, "state_write_failed", ["state_write_failed"],
            memory_ids, add_memory_ms, memory_read_ms, state_extraction_ms,
            deterministic, _public_mem0_stages(mem0_stage_records), request_id,
        )
    state_extraction_ms = (time.perf_counter() - extraction_started) * 1000

    added = result.get("results", []) if isinstance(result, dict) else []
    warnings: list[str] = []
    if not memory_ids:
        warnings.append("mem0_no_memory_id")
        logger.warning("Mem0 write returned no memory id for a user assertion")
    answer = "Memory stored in Mem0."
    if states:
        ingest_status = "state_recorded"
        answer += " State recorded: " + ", ".join(state.state_value for state in states) + "."
    else:
        ingest_status = "no_new_state"
        answer += " No new structured state was added."
    timings = _measured_timing(started, None, None)
    timings["classification_ms"] = classification_ms
    timings["ingestion"] = _ingestion_timing(
        add_memory_ms,
        memory_read_ms,
        state_extraction_ms,
        deterministic,
        _public_mem0_stages(mem0_stage_records),
    )
    return {
        "request_id": request_id,
        "response_type": "ingest",
        "ingest_status": ingest_status,
        "answer": answer,
        "citations": [],
        "query_analysis": analysis.model_dump(),
        "memories_added": [record for record in added if isinstance(record, Mapping)],
        "states": [state.model_dump(mode="json") for state in states],
        "transitions": [transition.model_dump(mode="json") for transition in transitions],
        "diagnostics": None,
        "storage_warnings": warnings,
        "storage_errors": [],
        "memory_ids": memory_ids,
        "timings": timings,
    }
