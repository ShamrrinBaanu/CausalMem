from pathlib import Path
from database import Database
from models import State, Transition
from scripts.audit_legacy import audit


def test_audit_reports_legacy_conflicts_without_writing(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    store = Database(str(path))
    store.save_state(State(user_id="alice", state_type="OTHER",
                           state_value="I run regularly.", status="ACTIVE",
                           source_memory_ids=["m1"]))
    store.save_state(State(user_id="alice", state_type="PREFERENCE",
                           state_value="tea", status="ACTIVE",
                           source_memory_ids=["m2"]))
    store.save_state(State(user_id="alice", state_type="PREFERENCE",
                           state_value="tea", status="ACTIVE",
                           source_memory_ids=["m3"]))
    source = State(user_id="alice", state_type="PREFERENCE",
                   state_value="coffee", status="ACTIVE",
                   source_memory_ids=["m4"])
    target = State(user_id="alice", state_type="PREFERENCE",
                   state_value="juice", status="ACTIVE",
                   source_memory_ids=["m5"])
    store.save_state(source)
    store.save_state(target)
    store.save_transition(Transition(
        user_id="alice", from_state_id=source.state_id,
        to_state_id=target.state_id, transition_type="PREFERENCE_CHANGE",
        relation_kind="TEMPORAL_SUCCESSION", timestamp=target.valid_from,
        source_memory_ids=["m5"],
    ))
    before = path.read_bytes()

    report = audit(path, limit=10)

    assert path.read_bytes() == before
    assert report["read_only"] is True
    assert report["summary"]["other_active"] == 1
    assert report["summary"]["duplicate_active"] == 1
    assert report["summary"]["conflicting_active"] == 1
    assert report["summary"]["inconsistent_replacement_sources"] == 1
    assert report["findings"]["other_active"][0]["state_value"] == "I run regularly."


def test_audit_reports_missing_database(tmp_path):
    report = audit(tmp_path / "missing.sqlite3", limit=10)
    assert report["read_only"] is True
    assert "error" in report
