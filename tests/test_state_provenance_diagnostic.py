import hashlib
import json
import sqlite3

from diagnose_state_provenance import inspect_state_provenance


def _db_with_linked_state(path):
    connection = sqlite3.connect(path)
    connection.executescript("""
        CREATE TABLE states(
            state_id TEXT PRIMARY KEY,user_id TEXT NOT NULL,state_type TEXT,
            state_value TEXT,status TEXT,valid_from TEXT,valid_until TEXT,
            confidence REAL,source_memory_ids TEXT,metadata TEXT);
        CREATE TABLE transitions(
            transition_id TEXT PRIMARY KEY,user_id TEXT NOT NULL,
            from_state_id TEXT,to_state_id TEXT,transition_type TEXT,
            relation_kind TEXT,trigger TEXT,reason TEXT,timestamp TEXT,
            confidence REAL,source_memory_ids TEXT,metadata TEXT);
    """)
    connection.execute(
        "INSERT INTO states VALUES(?,?,?,?,?,?,?,?,?,?)",
        ("state-secret", "user-secret", "ACTIVITY", "private-state-value", "ACTIVE",
         "2026-01-01T00:00:00+00:00", None, .9,
         json.dumps(["original-memory", "switch-memory"]), "{}"),
    )
    connection.execute(
        "INSERT INTO states VALUES(?,?,?,?,?,?,?,?,?,?)",
        ("destination-secret", "user-secret", "ACTIVITY", "private-destination", "ACTIVE",
         "2026-02-01T00:00:00+00:00", None, .9,
         json.dumps(["switch-memory"]), "{}"),
    )
    connection.execute(
        "INSERT INTO transitions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        ("transition-secret", "user-secret", "state-secret", "destination-secret",
         "ACTIVITY_CHANGE", "EXPLICIT_CAUSE", "private trigger", "private reason",
         "2026-02-01T00:00:00+00:00", .9, json.dumps(["switch-memory"]), "{}"),
    )
    connection.commit()
    connection.close()


def test_state_provenance_diagnostic_is_read_only_and_redacts_identifiers(tmp_path):
    path = tmp_path / "diagnostic.sqlite3"
    _db_with_linked_state(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    report = inspect_state_provenance(path, "user-secret", "state-secret")

    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after
    assert report == {
        "state_found": True,
        "state_status": "ACTIVE",
        "state_source_memory_id_count": 2,
        "state_source_memory_ids_well_formed": True,
        "associated_transition_count": 1,
        "associated_transitions": [{
            "relation_kind": "EXPLICIT_CAUSE",
            "direction": "outgoing",
            "transition_source_memory_id_count": 1,
            "state_source_ids_shared_with_transition_count": 1,
            "state_source_ids_shared_with_other_endpoint_count": 1,
            "other_endpoint_found": True,
            "other_endpoint_status": "ACTIVE",
        }],
        "current_code_path_assessment": "active_source_conflicts_with_current_close_path",
    }
    rendered = repr(report)
    for private in ("user-secret", "state-secret", "destination-secret",
                    "transition-secret", "original-memory", "switch-memory",
                    "private-state-value", "private-destination", "private trigger", "private reason"):
        assert private not in rendered


def test_state_provenance_diagnostic_scopes_lookup_to_supplied_user(tmp_path):
    path = tmp_path / "diagnostic.sqlite3"
    _db_with_linked_state(path)

    report = inspect_state_provenance(path, "another-user", "state-secret")

    assert report == {"state_found": False}


def test_state_provenance_diagnostic_redacts_unrecognized_structural_values(tmp_path):
    path = tmp_path / "diagnostic.sqlite3"
    _db_with_linked_state(path)
    connection = sqlite3.connect(path)
    connection.execute("UPDATE states SET status=? WHERE state_id=?",
                       ("private-status", "state-secret"))
    connection.execute("UPDATE transitions SET relation_kind=? WHERE transition_id=?",
                        ("private-relation", "transition-secret"))
    connection.commit()
    connection.close()

    report = inspect_state_provenance(path, "user-secret", "state-secret")

    assert report["state_status"] == "UNKNOWN"
    assert report["associated_transitions"][0]["relation_kind"] == "OTHER"
    assert "private" not in repr(report)
