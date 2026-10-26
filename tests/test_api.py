from types import SimpleNamespace

from fastapi.testclient import TestClient

import api
import chat_service
from hybrid_retriever import RetrievalFailure
from llm_service import LLMUnavailable


def test_visible_answer_hides_source_markers_but_keeps_separate_citations():
    answer = "I prefer coffee. [Sources: state:pref-123, transition:edge-456]"

    assert chat_service._visible_answer(answer) == "I prefer coffee."
    assert chat_service.parse_citations(answer) == [
        {"kind": "state", "id": "pref-123"},
        {"kind": "transition", "id": "edge-456"},
    ]


def test_config_endpoint_exposes_no_credentials(monkeypatch):
    monkeypatch.setattr(
        api,
        "settings",
        SimpleNamespace(
            provider="ollama",
            model="llama3.2:3b",
            default_user="local-user",
            api_key="sk-do-not-expose",
        ),
    )

    payload = api.config()

    assert payload == {
        "provider": "ollama",
        "model": "llama3.2:3b",
        "default_user": "local-user",
        "has_api_key": True,
        "temporal_rule": "valid_from <= t < valid_until",
    }
    assert "sk-do-not-expose" not in str(payload)


def test_chat_request_default_user_comes_from_settings(monkeypatch):
    monkeypatch.setattr(
        api,
        "settings",
        SimpleNamespace(default_user="namespace-from-config", provider="ollama",
                        model="model", api_key=""),
    )

    request = api.ChatRequest(message="hello")

    assert request.user_id == "namespace-from-config"


def test_chat_delegates_trimmed_message_and_normalized_timestamp(monkeypatch):
    calls = {}

    def fake_process_chat(message, user_id, requested_timestamp=None, source_role=None):
        calls.update(message=message, user_id=user_id,
                     requested_timestamp=requested_timestamp,
                     source_role=source_role)
        return {"response_type": "answer", "answer": "ok"}

    monkeypatch.setattr(api, "process_chat", fake_process_chat)
    client = TestClient(api.app)

    response = client.post(
        "/api/chat",
        json={
            "message": "  what do you remember?  ",
            "user_id": "alice",
            "requested_timestamp": "2024-01-02T03:04:05Z",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"response_type": "answer", "answer": "ok"}
    assert calls == {
        "message": "what do you remember?",
        "user_id": "alice",
        "requested_timestamp": "2024-01-02T03:04:05+00:00",
        "source_role": "user",
    }


def test_chat_rejects_blank_message_and_naive_timestamp():
    client = TestClient(api.app)

    blank = client.post("/api/chat", json={"message": "   ", "user_id": "alice"})
    naive = client.post(
        "/api/chat",
        json={
            "message": "hello",
            "user_id": "alice",
            "requested_timestamp": "2024-01-02T03:04:05",
        },
    )

    assert blank.status_code == 422
    assert naive.status_code == 422


def test_chat_maps_retrieval_failure_without_leaking_cause(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RetrievalFailure("mem0", "secret-provider-payload")

    monkeypatch.setattr(api, "process_chat", fail)
    client = TestClient(api.app)

    response = client.post(
        "/api/chat",
        json={"message": "what do you remember?", "user_id": "alice"},
    )

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "Could not retrieve mem0 evidence" in detail
    assert "secret-provider-payload" not in response.text


def test_chat_maps_value_error_without_echoing_internal_text(monkeypatch):
    def fail(*_args, **_kwargs):
        raise ValueError("raw internal text must not be echoed")

    monkeypatch.setattr(api, "process_chat", fail)
    client = TestClient(api.app)

    response = client.post(
        "/api/chat",
        json={"message": "what do you remember?", "user_id": "alice"},
    )

    assert response.status_code == 400
    assert "raw internal text" not in response.text
    assert response.json()["detail"] == (
        "The memory operation failed. Check the provider configuration and the "
        "application terminal for details."
    )


def test_chat_maps_llm_unavailable_to_503(monkeypatch):
    def fail(*_args, **_kwargs):
        raise LLMUnavailable("The local model is unavailable.")

    monkeypatch.setattr(api, "process_chat", fail)
    client = TestClient(api.app)

    response = client.post(
        "/api/chat",
        json={"message": "what do you remember?", "user_id": "alice"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "The local model is unavailable."


def test_memories_are_scoped_and_filtered_to_public_fields(monkeypatch):
    seen = []

    def fake_memories(user_id):
        seen.append(user_id)
        return [
            {
                "id": "m-1",
                "memory": "likes tea",
                "user_id": "alice",
                "metadata": {"secret": "internal-only"},
                "created_at": "2024-01-01T00:00:00+00:00",
            }
        ]

    monkeypatch.setattr(api, "get_memories", fake_memories)
    client = TestClient(api.app)

    response = client.get("/api/memories?user_id=alice")
    payload = response.json()

    assert response.status_code == 200
    assert seen == ["alice"]
    assert payload["user_id"] == "alice"
    assert payload["memories"] == [
        {
            "id": "m-1",
            "memory": "likes tea",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": None,
            "attribution": None,
        }
    ]
    assert "internal-only" not in response.text


def test_timeline_scopes_both_states_and_transitions(monkeypatch):
    seen = []

    def fake_states(user_id, **kwargs):
        seen.append(("states", user_id, kwargs.get("include_quarantined")))
        return []

    def fake_transitions(user_id, **kwargs):
        seen.append(("transitions", user_id, kwargs.get("include_quarantined")))
        return []

    monkeypatch.setattr(api, "timeline", fake_states)
    monkeypatch.setattr(api.db, "transitions", fake_transitions)
    client = TestClient(api.app)

    response = client.get("/api/timeline?user_id=bob")

    assert response.status_code == 200
    assert response.json() == {
        "user_id": "bob",
        "states": [],
        "transitions": [],
    }
    assert seen == [("states", "bob", True), ("transitions", "bob", True)]


def test_forensic_endpoints_expose_quarantined_rows_on_request(monkeypatch, tmp_path):
    from database import Database
    from models import State, Transition

    store = Database(str(tmp_path / "api_forensics.sqlite"))
    valid = State(user_id="alice", state_type="ACTIVITY", state_value="swimming")
    quarantined = State(user_id="alice", state_type="OTHER",
                        state_value="why i switched", status="INACTIVE",
                        valid_from=valid.valid_from, valid_until=valid.valid_from,
                        metadata={"review_status": "quarantined"})
    store.save_state(valid)
    store.save_state(quarantined)
    edge = Transition(user_id="alice", from_state_id=quarantined.state_id,
                      to_state_id=valid.state_id, relation_kind="UNKNOWN",
                      metadata={"review_status": "quarantined"})
    store.save_transition(edge)
    monkeypatch.setattr(api, "db", store)
    client = TestClient(api.app)

    visible = client.get("/api/states?user_id=alice")
    hidden = client.get("/api/states?user_id=alice&include_quarantined=false")
    visible_edges = client.get("/api/transitions?user_id=alice")
    hidden_edges = client.get("/api/transitions?user_id=alice&include_quarantined=false")

    assert {row["state_id"] for row in visible.json()["states"]} == {
        valid.state_id, quarantined.state_id}
    assert [row["state_id"] for row in hidden.json()["states"]] == [valid.state_id]
    assert [row["transition_id"] for row in visible_edges.json()["transitions"]] == [
        edge.transition_id]
    assert hidden_edges.json()["transitions"] == []


def test_process_chat_does_not_write_none_assertions_or_assistant_content(monkeypatch):
    import chat_service
    from models import QueryAnalysis

    writes = []
    monkeypatch.setattr(
        chat_service, "classify_query",
        lambda _message: QueryAnalysis(is_question=False, category="OTHER",
                                       rationale="test"),
    )
    monkeypatch.setattr(
        chat_service, "add_memory",
        lambda *_args, **_kwargs: writes.append("mem0") or {"results": []},
    )
    monkeypatch.setattr(
        chat_service, "ingest_message",
        lambda *_args, **_kwargs: writes.append("structured") or ([], []),
    )

    cases = [
        ("Please explain why I switched to swimming.", "user"),
        ("You switched from running to swimming because of a knee injury.", "user"),
        ("I switched to swimming because of a knee injury.", "assistant"),
    ]
    for message, role in cases:
        result = chat_service.process_chat(message, "alice", source_role=role)
        assert result["response_type"] == "ingest"
        assert result["memories_added"] == []
        assert result["states"] == [] and result["transitions"] == []

    assert writes == []


def test_process_chat_retrieval_query_never_writes(monkeypatch):
    import chat_service
    from models import QueryAnalysis

    writes = []
    monkeypatch.setattr(
        chat_service, "classify_query",
        lambda _message: QueryAnalysis(is_question=True, category="HISTORY",
                                       rationale="test"),
    )
    monkeypatch.setattr(
        chat_service, "add_memory",
        lambda *_args, **_kwargs: writes.append("mem0") or {"results": []},
    )
    evidence = {"query_category": "HISTORY", "memories": [], "states": [],
                "transitions": [], "causal_path": []}
    monkeypatch.setattr(chat_service, "retrieve", lambda *_args, **_kwargs: evidence)
    monkeypatch.setattr(
        chat_service, "generate_answer_with_diagnostics",
        lambda *_args, **_kwargs: ("No answer.", {"answer_reason": "no_validated_claims"}),
    )
    monkeypatch.setattr(
        chat_service, "safe_answer_diagnostic_summary",
        lambda _diagnostics: {"answer_reason": "no_validated_claims"},
    )

    result = chat_service.process_chat("Why did I switch to swimming?", "alice")

    assert result["response_type"] == "answer"
    assert writes == []


def test_diagnostics_hides_store_exception_and_reports_unavailable(monkeypatch):
    monkeypatch.setattr(
        api.db, "states",
        lambda _user_id, include_inactive=True, include_quarantined=False: [])
    monkeypatch.setattr(
        api.db, "transitions",
        lambda _user_id, include_quarantined=False: [])

    def fail(_user_id):
        raise RuntimeError("postgres://secret-connection-string")

    monkeypatch.setattr(api, "get_memories", fail)
    client = TestClient(api.app)

    response = client.get("/api/diagnostics?user_id=alice")
    payload = response.json()

    assert response.status_code == 200
    assert payload["user_id"] == "alice"
    assert payload["store_status"] == {"mem0": "unavailable"}
    assert payload["errors"] == ["Mem0 unavailable"]
    assert "secret-connection-string" not in response.text
