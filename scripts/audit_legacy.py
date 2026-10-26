"""Read-only audit for legacy or inconsistent CausalMem records.

This script never writes, migrates, repairs, or deletes data. It opens the
configured SQLite database with SQLite's ``mode=ro`` URI flag and prints a JSON
report of suspicious rows for manual review.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\audit_legacy.py
    .\\.venv\\Scripts\\python.exe scripts\\audit_legacy.py --db data\\causalmem.sqlite3
    .\\.venv\\Scripts\\python.exe scripts\\audit_legacy.py --output audit.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402

SINGULAR_TYPES = {
    "PREFERENCE",
    "GOAL",
    "LOCATION",
    "STATUS",
    "EMOTION",
    "PROJECT",
}
REPLACEMENT_TYPES = {
    "PREFERENCE_CHANGE",
    "GOAL_CHANGE",
    "LOCATION_CHANGE",
    "GOAL_COMPLETION",
    "ACTIVITY_CHANGE",
}


def _read_only_connection(path: Path) -> sqlite3.Connection:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _parse_json(value):
    try:
        return json.loads(value or "[]")
    except (TypeError, ValueError):
        return []


def _parse_metadata(value):
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def _parse_instant(value):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalized(value):
    return " ".join(str(value or "").split()).casefold()


def _state_summary(row):
    metadata = _parse_metadata(row["metadata"])
    return {
        "state_id": row["state_id"],
        "user_id": row["user_id"],
        "state_type": row["state_type"],
        "state_value": row["state_value"],
        "status": row["status"],
        "valid_from": row["valid_from"],
        "valid_until": row["valid_until"],
        "source_memory_ids": _parse_json(row["source_memory_ids"]),
        "review_status": metadata.get("review_status"),
    }


def _transition_summary(row):
    metadata = _parse_metadata(row["metadata"])
    return {
        "transition_id": row["transition_id"],
        "user_id": row["user_id"],
        "from_state_id": row["from_state_id"],
        "to_state_id": row["to_state_id"],
        "transition_type": row["transition_type"],
        "relation_kind": row["relation_kind"],
        "trigger": row["trigger"],
        "reason": row["reason"],
        "timestamp": row["timestamp"],
        "source_memory_ids": _parse_json(row["source_memory_ids"]),
        "review_status": metadata.get("review_status"),
    }


def audit(database_path: Path, limit: int) -> dict:
    if not database_path.exists():
        return {
            "database": str(database_path),
            "read_only": True,
            "error": "database not found",
        }
    try:
        connection = _read_only_connection(database_path)
    except sqlite3.Error as exc:
        return {
            "database": str(database_path),
            "read_only": True,
            "error": f"cannot open database read-only: {exc}",
        }
    try:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not {"states", "transitions"}.issubset(tables):
            return {
                "database": str(database_path),
                "read_only": True,
                "error": "states and transitions tables are required",
                "tables": sorted(tables),
            }
        states = connection.execute("SELECT * FROM states ORDER BY user_id,valid_from").fetchall()
        transitions = connection.execute(
            "SELECT * FROM transitions ORDER BY user_id,timestamp"
        ).fetchall()
    finally:
        connection.close()

    state_by_id = {row["state_id"]: row for row in states}
    active = [row for row in states if str(row["status"] or "").upper() == "ACTIVE"]

    active_by_type_value = defaultdict(list)
    active_by_type = defaultdict(list)
    for row in active:
        active_by_type_value[
            (row["user_id"], row["state_type"], _normalized(row["state_value"]))
        ].append(row)
        active_by_type[(row["user_id"], row["state_type"])].append(row)

    other_active = [_state_summary(row) for row in active
                    if str(row["state_type"] or "").upper() == "OTHER"]

    duplicate_active = []
    for (user_id, state_type, value), rows in sorted(active_by_type_value.items()):
        if len(rows) > 1:
            duplicate_active.append({
                "user_id": user_id,
                "state_type": state_type,
                "state_value": rows[0]["state_value"],
                "count": len(rows),
                "state_ids": sorted(row["state_id"] for row in rows),
            })

    conflicting_active = []
    for (user_id, state_type), rows in sorted(active_by_type.items()):
        values = {_normalized(row["state_value"]) for row in rows}
        if state_type in SINGULAR_TYPES and len(values) > 1:
            conflicting_active.append({
                "user_id": user_id,
                "state_type": state_type,
                "count": len(rows),
                "values": sorted({row["state_value"] for row in rows}),
                "state_ids": sorted(row["state_id"] for row in rows),
            })

    dangling_transitions = []
    self_transitions = []
    duplicate_transitions = []
    inconsistent_replacement_sources = []
    explicit_cause_missing_trigger = []
    transition_groups = defaultdict(list)

    for row in transitions:
        transition = _transition_summary(row)
        source = state_by_id.get(row["from_state_id"])
        destination = state_by_id.get(row["to_state_id"])
        if source is None or destination is None:
            dangling_transitions.append(transition)
            continue
        if source["user_id"] != row["user_id"] or destination["user_id"] != row["user_id"]:
            dangling_transitions.append(transition)
            continue
        if row["from_state_id"] == row["to_state_id"]:
            self_transitions.append(transition)
        if (row["transition_type"] in REPLACEMENT_TYPES
                and str(source["status"] or "").upper() == "ACTIVE"):
            item = transition | {
                "source_status": source["status"],
                "source_valid_until": source["valid_until"],
            }
            inconsistent_replacement_sources.append(item)
        if (row["relation_kind"] == "EXPLICIT_CAUSE"
                and not (str(row["trigger"] or "").strip()
                         or str(row["reason"] or "").strip())):
            explicit_cause_missing_trigger.append(transition)
        transition_groups[
            (
                row["user_id"],
                row["from_state_id"],
                row["to_state_id"],
                row["transition_type"],
                row["relation_kind"],
                row["timestamp"],
            )
        ].append(row)

    for key, rows in sorted(transition_groups.items()):
        if len(rows) > 1:
            duplicate_transitions.append({
                "user_id": key[0],
                "from_state_id": key[1],
                "to_state_id": key[2],
                "transition_type": key[3],
                "relation_kind": key[4],
                "timestamp": key[5],
                "count": len(rows),
                "transition_ids": sorted(row["transition_id"] for row in rows),
            })

    invalid_intervals = []
    for row in states:
        valid_from = _parse_instant(row["valid_from"])
        valid_until = _parse_instant(row["valid_until"])
        invalid = valid_from is None
        if not invalid and row["valid_until"] is not None:
            invalid = valid_until is None or valid_until <= valid_from
        if invalid:
            invalid_intervals.append(_state_summary(row))

    findings = {
        "other_active": other_active,
        "duplicate_active": duplicate_active,
        "conflicting_active": conflicting_active,
        "dangling_transitions": dangling_transitions,
        "self_transitions": self_transitions,
        "duplicate_transitions": duplicate_transitions,
        "inconsistent_replacement_sources": inconsistent_replacement_sources,
        "explicit_cause_missing_trigger": explicit_cause_missing_trigger,
        "invalid_intervals": invalid_intervals,
    }
    summary = {name: len(items) for name, items in findings.items()}
    limited = {
        name: items[:limit] for name, items in findings.items()
    }
    return {
        "database": str(database_path),
        "read_only": True,
        "total_states": len(states),
        "total_transitions": len(transitions),
        "active_states": len(active),
        "summary": summary,
        "findings": limited,
        "truncated": {
            name: len(items) > limit for name, items in findings.items()
        },
        "counts_by_type": dict(sorted(Counter(
            row["state_type"] for row in states
        ).items(), key=lambda item: str(item[0]))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(settings.database_path),
        help="SQLite database to audit (opened read-only)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the JSON report to this path instead of stdout",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="maximum examples per finding category (default: 20)",
    )
    args = parser.parse_args()
    if not args.db.exists():
        print(f"database not found: {args.db}", file=sys.stderr)
        return 2
    report = audit(args.db, max(0, args.limit))
    text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
