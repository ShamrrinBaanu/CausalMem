from datetime import datetime, timedelta, timezone

from database import Database
from models import State, Transition
import answer_generator
import hybrid_retriever
import query_router
import state_extractor
import temporal_engine


def _state(user_id, value, *, status="ACTIVE", key=None, source_role=None,
           source_timestamp=None, source_memory_ids=(), valid_from=None):
    metadata = {}
    if key:
        metadata["preference_key"] = key
    if source_role:
        metadata["source_role"] = source_role
    if source_timestamp:
        metadata["source_timestamp"] = source_timestamp
    return State(
        user_id=user_id,
        state_type="PREFERENCE",
        state_value=value,
        status=status,
        valid_from=valid_from or datetime.now(timezone.utc),
        valid_until=None,
        source_memory_ids=list(source_memory_ids),
        metadata=metadata,
    )


def test_named_programming_preference_replaces_only_programming_state_atomically(
        tmp_path, monkeypatch):
    store = Database(str(tmp_path / "preferences.sqlite3"))
    python = _state("alice", "Python", key="favorite:programming language",
                    source_memory_ids=["python-memory"])
    tea = _state("alice", "tea", key="favorite:drink",
                 source_memory_ids=["tea-memory"])
    store.save_state(python)
    store.save_state(tea)
    monkeypatch.setattr(state_extractor, "db", store)

    created, transitions = state_extractor.ingest_message(
        "My desired current programming preference is Java.",
        "alice", ["java-memory"], [], allow_llm=False)

    history = {state.state_id: state for state in store.states("alice")}
    assert [(state.state_type, state.state_value) for state in created] == [
        ("PREFERENCE", "Java")]
    assert created[0].metadata["preference_key"] == "favorite:programming language"
    assert created[0].metadata["source_role"] == "user"
    assert created[0].source_memory_ids == ["java-memory"]
    assert history[python.state_id].status == "INACTIVE"
    assert history[python.state_id].source_memory_ids == ["python-memory"]
    assert history[tea.state_id].status == "ACTIVE"
    assert len(transitions) == 1
    assert transitions[0].relation_kind == "TEMPORAL_SUCCESSION"
    assert transitions[0].trigger is None and transitions[0].reason is None
    assert [state.state_value for state in store.states("alice", include_inactive=False)
            if state.metadata.get("preference_key") == "favorite:programming language"] == [
        "Java"]

    repeated, repeated_edges = state_extractor.ingest_message(
        "My desired current programming preference is Java.",
        "alice", ["same-java-memory"], [], allow_llm=False)
    assert repeated == [] and repeated_edges == []


def test_assistant_authored_preference_cannot_update_user_state(tmp_path, monkeypatch):
    store = Database(str(tmp_path / "assistant-preference.sqlite3"))
    java = _state("alice", "Java", key="favorite:programming language")
    store.save_state(java)
    monkeypatch.setattr(state_extractor, "db", store)

    states, transitions = state_extractor.ingest_message(
        "I now prefer Python for programming.", "alice", ["assistant-python"], [],
        source_role="assistant", allow_llm=False)

    assert states == [] and transitions == []
    assert [(state.state_value, state.status) for state in store.states("alice")
            if state.metadata.get("preference_key") == "favorite:programming language"] == [
        ("Java", "ACTIVE")]


def test_preference_state_keeps_the_original_user_statement_timestamp(
        tmp_path, monkeypatch):
    store = Database(str(tmp_path / "preference-timestamp.sqlite3"))
    monkeypatch.setattr(state_extractor, "db", store)
    source_timestamp = datetime(2026, 10, 24, 12, 30, tzinfo=timezone.utc)

    states, transitions = state_extractor.ingest_message(
        "I prefer Java for programming.", "alice", ["java-memory"], [],
        allow_llm=False, source_timestamp=source_timestamp)

    assert transitions == []
    assert states[0].valid_from == source_timestamp
    assert states[0].metadata["source_timestamp"] == source_timestamp.isoformat()
    assert states[0].metadata["source_role"] == "user"


def test_current_programming_preference_is_domain_scoped_and_citable(
        tmp_path, monkeypatch):
    store = Database(str(tmp_path / "current-preference.sqlite3"))
    java = _state("alice", "Java", key="favorite:programming language",
                  source_role="user", source_memory_ids=["java-memory"])
    tea = _state("alice", "tea", key="favorite:drink",
                 source_memory_ids=["tea-memory"])
    store.save_state(java)
    store.save_state(tea)
    records = [
        {"id": "java-memory", "user_id": "alice",
         "memory": "User now prefers Java for programming", "attribution": "user"},
        {"id": "tea-memory", "user_id": "alice",
         "memory": "User prefers tea", "attribution": "user"},
    ]
    monkeypatch.setattr(hybrid_retriever, "db", store)
    monkeypatch.setattr(temporal_engine, "db", store)
    monkeypatch.setattr(hybrid_retriever, "search_memory", lambda *_args: records)
    evidence = hybrid_retriever.retrieve(
        "What is my current programming preference?", "alice", "STATE")

    assert [state["state_value"] for state in evidence["states"]] == ["Java"]
    assert [memory["id"] for memory in evidence["memories"]] == ["java-memory"]

    class JavaAnswer:
        def answer_with_provenance(self, _question, _evidence):
            return {"claims": [{"text": "You prefer Java.", "claim_type": "fact",
                                 "evidence": [{"kind": "state", "id": java.state_id}]}]}

    monkeypatch.setattr(answer_generator, "llm", JavaAnswer())
    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "What is my current programming preference?", evidence)
    assert answer == f"You prefer Java. [Sources: state:{java.state_id}]"
    assert diagnostics["accepted_claim_count"] == 1


def test_most_recent_programming_preference_uses_user_provenance_and_source_time(
        tmp_path, monkeypatch):
    store = Database(str(tmp_path / "recent-preference.sqlite3"))
    base = datetime(2026, 10, 25, tzinfo=timezone.utc)
    python = _state(
        "alice", "Python", status="INACTIVE", key="favorite:programming language",
        source_role="user", source_timestamp=(base - timedelta(days=1)).isoformat(),
        source_memory_ids=["python-memory"], valid_from=base - timedelta(days=1))
    assistant_java = _state(
        "alice", "Rust", key="favorite:programming language", source_role="assistant",
        source_timestamp=(base + timedelta(days=2)).isoformat(),
        source_memory_ids=["assistant-rust-memory"], valid_from=base + timedelta(days=2))
    java = _state(
        "alice", "Java", key="favorite:programming language", source_role="user",
        source_timestamp=base.isoformat(), source_memory_ids=["java-memory"],
        valid_from=base)
    for state in (python, assistant_java, java):
        store.save_state(state)
    records = [
        {"id": "python-memory", "user_id": "alice", "memory": "User prefers Python for programming"},
        {"id": "java-memory", "user_id": "alice", "memory": "User now prefers Java for programming"},
        {"id": "assistant-rust-memory", "user_id": "alice", "memory": "User prefers Rust for programming"},
    ]
    monkeypatch.setattr(hybrid_retriever, "db", store)
    monkeypatch.setattr(temporal_engine, "db", store)
    monkeypatch.setattr(hybrid_retriever, "search_memory", lambda *_args: records)

    question = "What programming language did I most recently say I prefer?"
    route = query_router._deterministic_category(question)
    evidence = hybrid_retriever.retrieve(question, "alice", "HISTORY")

    assert route == "HISTORY"
    assert [state["state_value"] for state in evidence["states"]] == ["Java"]
    assert [record["id"] for record in evidence["memories"]] == ["java-memory"]


def test_programming_preference_history_keeps_older_records_and_ignores_drinks(
        tmp_path, monkeypatch):
    store = Database(str(tmp_path / "preference-history.sqlite3"))
    python = _state("alice", "Python", status="INACTIVE",
                    key="favorite:programming language",
                    source_memory_ids=["python-memory"])
    java = _state("alice", "Java", key="favorite:programming language",
                  source_memory_ids=["java-memory"])
    tea = _state("alice", "tea", key="favorite:drink", source_memory_ids=["tea-memory"])
    for state in (python, java, tea):
        store.save_state(state)
    records = [
        {"id": "python-memory", "user_id": "alice", "memory": "User preferred Python for programming"},
        {"id": "java-memory", "user_id": "alice", "memory": "User prefers Java for programming"},
        {"id": "tea-memory", "user_id": "alice", "memory": "User prefers tea"},
    ]
    monkeypatch.setattr(hybrid_retriever, "db", store)
    monkeypatch.setattr(temporal_engine, "db", store)
    monkeypatch.setattr(hybrid_retriever, "search_memory", lambda *_args: records)

    evidence = hybrid_retriever.retrieve(
        "What programming languages have I preferred in the past?", "alice", "HISTORY")

    assert {state["state_value"] for state in evidence["states"]} == {"Python", "Java"}
    assert {memory["id"] for memory in evidence["memories"]} == {
        "python-memory", "java-memory"}


def test_programming_preference_change_answer_requires_owned_supersession_edge(
        monkeypatch):
    boundary = datetime(2026, 10, 25, 6, 0, tzinfo=timezone.utc)
    python = _state(
        "alice", "Python", status="INACTIVE",
        key="favorite:programming language", source_memory_ids=["python-memory"],
        valid_from=boundary - timedelta(days=1))
    python.valid_until = boundary
    java = _state(
        "alice", "Java", key="favorite:programming language",
        source_role="user", source_timestamp=boundary.isoformat(),
        source_memory_ids=["java-memory"], valid_from=boundary)
    edge = Transition(
        user_id="alice", from_state_id=python.state_id, to_state_id=java.state_id,
        transition_type="PREFERENCE_CHANGE", relation_kind="TEMPORAL_SUCCESSION",
        timestamp=boundary, source_memory_ids=["java-memory"])
    evidence = {
        "user_id": "alice", "query_category": "EVOLUTION", "memories": [],
        "states": [python.model_dump(mode="json"), java.model_dump(mode="json")],
        "transitions": [edge.model_dump(mode="json")], "causal_path": [],
    }

    class EmptyAnswer:
        def answer_with_provenance(self, _question, _evidence):
            return {"claims": []}

    monkeypatch.setattr(answer_generator, "llm", EmptyAnswer())
    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "How did my programming preference change?", evidence)
    assert answer == f"Your programming preference changed from Python to Java. [Sources: transition:{edge.transition_id}]", diagnostics
    assert diagnostics["fallback_claim_used"] is True
    assert diagnostics["accepted_claim_count"] == 1


def test_past_programming_preference_query_routes_to_history_without_classifier(
        monkeypatch):
    question = "What programming languages have I preferred in the past?"
    monkeypatch.setattr(query_router.llm, "classify",
                        lambda *_args: (_ for _ in ()).throw(AssertionError("LLM routing used")))
    assert query_router.classify_query(question).category == "HISTORY"


def test_most_recently_stated_programming_preference_routes_to_history(
        monkeypatch):
    question = "What was my most recently stated programming preference?"
    monkeypatch.setattr(query_router.llm, "classify",
                        lambda *_args: (_ for _ in ()).throw(AssertionError("LLM routing used")))
    assert query_router.classify_query(question).category == "HISTORY"


def test_programming_preference_history_answers_from_inactive_state_with_citation(
        monkeypatch):
    boundary = datetime(2026, 10, 25, 6, 0, tzinfo=timezone.utc)
    python = _state(
        "alice", "Python", status="INACTIVE",
        key="favorite:programming language", source_memory_ids=["python-memory"],
        valid_from=boundary - timedelta(days=1))
    python.valid_until = boundary
    java = _state(
        "alice", "Java", key="favorite:programming language",
        source_role="user", source_timestamp=boundary.isoformat(),
        source_memory_ids=["java-memory"], valid_from=boundary)
    evidence = {
        "user_id": "alice", "query_category": "HISTORY", "memories": [],
        "states": [python.model_dump(mode="json"), java.model_dump(mode="json")],
        "transitions": [], "causal_path": [],
    }

    class EmptyAnswer:
        def answer_with_provenance(self, _question, _evidence):
            return {"claims": []}

    monkeypatch.setattr(answer_generator, "llm", EmptyAnswer())
    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "What programming languages have I preferred in the past?", evidence)
    assert answer == f"You previously preferred Python for programming. [Sources: state:{python.state_id}]"
    assert diagnostics["fallback_claim_used"] is True
    assert diagnostics["accepted_claim_count"] == 1

