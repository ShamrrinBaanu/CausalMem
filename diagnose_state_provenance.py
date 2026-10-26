"""Read-only structural inspection of one SQLite state and linked transitions.

The report intentionally omits user, state, transition, and memory identifiers,
as well as state values and memory content. It never creates a schema or writes
to the supplied database.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def _memory_ids(serialized: Any) -> list[str] | None:
    try:
        value = json.loads(serialized) if isinstance(serialized, str) else serialized
    except (TypeError, ValueError):
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        return None
    return value


def _safe_status(value: Any) -> str:
    return value if value in {"ACTIVE", "INACTIVE"} else "UNKNOWN"


def _safe_relation(value: Any) -> str:
    allowed = {"EXPLICIT_CAUSE", "TEMPORAL_SUCCESSION", "UNKNOWN", "INFERRED_RELATIONSHIP"}
    return value if value in allowed else "OTHER"


def inspect_state_provenance(database_path: str | Path, user_id: str,
                             state_id: str) -> dict[str, Any]:
    """Return redacted structural facts for a user-scoped state lookup.

    SQLite opens with ``mode=ro`` and ``query_only``. Errors are reduced to
    stable categories by the CLI; no supplied identifier is returned.
    """
    db_uri = Path(database_path).expanduser().resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(db_uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only = ON")
        state = connection.execute(
            "SELECT state_id,user_id,state_type,status,source_memory_ids "
            "FROM states WHERE user_id=? AND state_id=?",
            (user_id, state_id),
        ).fetchone()
        if state is None:
            return {"state_found": False}

        state_ids = _memory_ids(state["source_memory_ids"])
        state_id_set = set(state_ids or [])
        transitions = connection.execute(
            "SELECT transition_id,from_state_id,to_state_id,relation_kind,user_id,source_memory_ids "
            "FROM transitions WHERE user_id=? AND (from_state_id=? OR to_state_id=?) "
            "ORDER BY timestamp,transition_id",
            (user_id, state_id, state_id),
        ).fetchall()

        linked = []
        for edge in transitions:
            is_source = edge["from_state_id"] == state_id
            other_id = edge["to_state_id"] if is_source else edge["from_state_id"]
            other = connection.execute(
                "SELECT status,source_memory_ids FROM states "
                "WHERE user_id=? AND state_id=?", (user_id, other_id),
            ).fetchone()
            edge_ids = _memory_ids(edge["source_memory_ids"])
            other_ids = _memory_ids(other["source_memory_ids"]) if other else None
            linked.append({
                "relation_kind": _safe_relation(edge["relation_kind"]),
                "direction": "outgoing" if is_source else "incoming",
                "transition_source_memory_id_count": len(edge_ids) if edge_ids is not None else None,
                "state_source_ids_shared_with_transition_count": (
                    len(state_id_set.intersection(edge_ids)) if state_ids is not None and edge_ids is not None else None
                ),
                "state_source_ids_shared_with_other_endpoint_count": (
                    len(state_id_set.intersection(other_ids))
                    if state_ids is not None and other_ids is not None else None
                ),
                "other_endpoint_found": other is not None,
                "other_endpoint_status": _safe_status(other["status"]) if other else None,
            })

        expected_close_path = any(
            edge["direction"] == "outgoing"
            and (edge["relation_kind"] == "EXPLICIT_CAUSE"
                 or (edge["relation_kind"] == "TEMPORAL_SUCCESSION"
                     and state["state_type"] == "GOAL"))
            for edge in linked
        )
        status = _safe_status(state["status"])
        if expected_close_path and status == "ACTIVE":
            path_assessment = "active_source_conflicts_with_current_close_path"
        elif expected_close_path and status == "INACTIVE":
            path_assessment = "matches_current_source_close_path"
        elif linked:
            path_assessment = "linked_transition_does_not_identify_current_close_path"
        else:
            path_assessment = "no_linked_transition_path_identified"

        return {
            "state_found": True,
            "state_status": status,
            "state_source_memory_id_count": len(state_ids) if state_ids is not None else None,
            "state_source_memory_ids_well_formed": state_ids is not None,
            "associated_transition_count": len(linked),
            "associated_transitions": linked,
            "current_code_path_assessment": path_assessment,
        }
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, help="Existing SQLite database file")
    parser.add_argument("--user-id", required=True, help="User namespace used for lookup")
    parser.add_argument("--state-id", required=True, help="State identifier used for lookup")
    args = parser.parse_args()
    try:
        report = inspect_state_provenance(args.database, args.user_id, args.state_id)
    except (OSError, sqlite3.Error, ValueError):
        print(json.dumps({"error": "database_unavailable_or_invalid"}, sort_keys=True))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0 if report["state_found"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
