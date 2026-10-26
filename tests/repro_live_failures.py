"""Reproduce the live failures through the same function used by POST /api/chat.

The script always uses a fresh temporary SQLite database, Mem0/Qdrant store, and
test namespace. It never reads or writes the production store.

Run:
    .\\.venv\\Scripts\\python.exe tests\\repro_live_failures.py
    .\\.venv\\Scripts\\python.exe tests\\repro_live_failures.py --scenario A
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path


def _prepare_isolated_environment() -> Path:
    root = Path(tempfile.mkdtemp(prefix="causalmem-repro-"))
    os.environ["CAUSALMEM_DB"] = str(root / "causalmem.sqlite3")
    os.environ["MEM0_PATH"] = str(root / "mem0")
    os.environ["MEM0_DIR"] = str(root / "mem0-config")
    os.environ["CAUSALMEM_USER"] = f"test_user_{uuid.uuid4().hex[:10]}"
    return root


ROOT = _prepare_isolated_environment()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chat_service import process_chat  # noqa: E402

USER_ID = os.environ["CAUSALMEM_USER"]

SCENARIOS = {
    "A": {
        "name": "causal wording",
        "ingest": [
            "I love running.",
            "I switched from running to swimming because of a knee injury.",
        ],
        "queries": [
            "Why did I switch from running to swimming?",
            "Did I switch because of a knee injury or is that unknown?",
        ],
    },
    "B": {
        "name": "conflicting programming preferences",
        "ingest": [
            "I prefer Python for programming.",
            "I prefer Java for programming.",
        ],
        "queries": [
            "What programming language do I prefer now?",
            "what i prefer for programming",
        ],
    },
    "C": {
        "name": "overlapping tea/coffee preferences",
        "ingest": [
            "I prefer tea.",
            "I prefer coffee.",
        ],
        "queries": [
            "What do I prefer?",
            "Do I prefer tea?",
        ],
    },
    "D": {
        "name": "location move",
        "ingest": [
            "I live in Chennai.",
            "I moved from Chennai to Coimbatore.",
        ],
        "queries": [
            "Where do I live now?",
            "Where did I live before?",
            "What location change did I tell you about?",
        ],
    },
    "E": {
        "name": "goal and completion",
        "ingest": [
            "My new goal is to learn system design.",
            "I completed my machine learning project.",
        ],
        "queries": [
            "What is my current goal?",
            "What did I complete?",
            "What was my previous goal?",
        ],
    },
    "F": {
        "name": "goal replacement",
        "ingest": [
            "My goal is to learn system design.",
            "My new goal is to learn data structures.",
        ],
        "queries": [
            "What is my current goal?",
            "What was my previous goal?",
            "What goal did I have before?",
        ],
    },
}


def _brief(value: str, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _state_map(response: dict) -> dict[str, dict]:
    return {item["state_id"]: item for item in response.get("states", [])}


def _print_ingest(label: str, message: str, response: dict, elapsed: float) -> None:
    states = response.get("states", [])
    transitions = response.get("transitions", [])
    state_by_id = _state_map(response)
    print(f"\n  INGEST {label}: {message!r}")
    print(
        "    intent={category} question={is_question} status={status} elapsed={elapsed:.1f}s".format(
            category=response.get("query_analysis", {}).get("category"),
            is_question=response.get("query_analysis", {}).get("is_question"),
            status=response.get("ingest_status"),
            elapsed=elapsed,
        )
    )
    print(f"    request_id={response.get('request_id')}")
    print(f"    timings={json.dumps(response.get('timings') or {}, default=str)}")
    print(
        "    states="
        + json.dumps(
            [
                {
                    "type": state.get("state_type"),
                    "value": state.get("state_value"),
                    "status": state.get("status"),
                    "valid_from": state.get("valid_from"),
                    "valid_until": state.get("valid_until"),
                    "source_memory_ids": state.get("source_memory_ids"),
                    "valid_from": state.get("valid_from"),
                    "valid_until": state.get("valid_until"),
                    "source_memory_ids": state.get("source_memory_ids"),
                }
                for state in states
            ],
            ensure_ascii=False,
        )
    )
    print(
        "    transitions="
        + json.dumps(
            [
                {
                    "relation": edge.get("relation_kind"),
                    "trigger": edge.get("trigger"),
                    "timestamp": edge.get("timestamp"),
                    "from_state_id": edge.get("from_state_id"),
                    "to_state_id": edge.get("to_state_id"),
                    "timestamp": edge.get("timestamp"),
                    "from_state_id": edge.get("from_state_id"),
                    "to_state_id": edge.get("to_state_id"),
                    "from": state_by_id.get(edge.get("from_state_id"), {}).get("state_value"),
                    "to": state_by_id.get(edge.get("to_state_id"), {}).get("state_value"),
                }
                for edge in transitions
            ],
            ensure_ascii=False,
        )
    )


def _print_answer(question: str, response: dict, elapsed: float) -> None:
    diagnostics = response.get("diagnostics") or {}
    counts = response.get("evidence_counts") or {}
    evidence = response.get("evidence") or {}
    state_by_id = {
        state.get("state_id"): state for state in evidence.get("states", [])
    }
    print(f"\n  QUERY: {question!r}")
    print(
        "    intent={category} question={is_question} elapsed={elapsed:.1f}s".format(
            category=response.get("query_analysis", {}).get("category"),
            is_question=response.get("query_analysis", {}).get("is_question"),
            elapsed=elapsed,
        )
    )
    print(f"    request_id={response.get('request_id')}")
    print(f"    timings={json.dumps(response.get('timings') or {}, default=str)}")
    print(
        "    retrieved="
        + json.dumps(
            {
                "memories": counts.get("memories"),
                "states": counts.get("states"),
                "transitions": counts.get("transitions"),
            }
        )
    )
    print(
        "    states="
        + json.dumps(
            [
                {
                    "type": state.get("state_type"),
                    "value": state.get("state_value"),
                    "status": state.get("status"),
                }
                for state in evidence.get("states", [])
            ],
            ensure_ascii=False,
        )
    )
    print(
        "    transitions="
        + json.dumps(
            [
                {
                    "relation": edge.get("relation_kind"),
                    "trigger": edge.get("trigger"),
                    "from": state_by_id.get(edge.get("from_state_id"), {}).get("state_value"),
                    "to": state_by_id.get(edge.get("to_state_id"), {}).get("state_value"),
                }
                for edge in evidence.get("transitions", [])
            ],
            ensure_ascii=False,
        )
    )
    print(
        "    gate={gate} model={model} accepted={accepted} rejected={rejected}".format(
            gate=diagnostics.get("causal_gate_passed"),
            model=diagnostics.get("model_response_status"),
            accepted=diagnostics.get("accepted_claim_count"),
            rejected=diagnostics.get("rejected_claim_count"),
        )
    )
    print(f"    answer_reason={diagnostics.get('answer_reason')}")
    print(f"    rejection_reasons={diagnostics.get('rejection_reasons')}")
    print(f"    answer={_brief(response.get('answer'))!r}")


def run(scenario_key: str) -> None:
    scenario = SCENARIOS[scenario_key]
    # Keep each end-to-end case independent even though the temporary stores
    # are shared by this one process.
    user_id = f"{USER_ID}_{scenario_key.lower()}"
    print("=" * 88)
    print(f"SCENARIO {scenario_key}: {scenario['name']} | user={user_id}")
    print("=" * 88)
    for index, message in enumerate(scenario["ingest"], start=1):
        started = time.perf_counter()
        response = process_chat(message, user_id)
        _print_ingest(index, message, response, time.perf_counter() - started)
    for question in scenario["queries"]:
        started = time.perf_counter()
        response = process_chat(question, user_id)
        _print_answer(question, response, time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), action="append")
    args = parser.parse_args()
    print(f"Temporary isolated root: {ROOT}")
    for key in args.scenario or sorted(SCENARIOS):
        run(key)


if __name__ == "__main__":
    main()
