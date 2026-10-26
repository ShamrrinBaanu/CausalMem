"""End-to-end state, transition, temporal, retrieval, citation, and API matrix.

The suite uses temporary SQLite files and deterministic doubles. It never calls a
live LLM, Mem0, Ollama, or any production store.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from database import Database
from models import Extraction, QueryAnalysis, State, Transition
import answer_generator
import causal_graph
import chat_service
import hybrid_retriever
import state_extractor
import temporal_engine
import query_router


STATE_QUERY_TERMS = {
    "FACT": "fact",
    "PREFERENCE": "preference",
    "ACTIVITY": "activity",
    "GOAL": "goal",
    "PLAN": "plan",
    "SKILL": "skill",
    "INTEREST": "interest",
    "RELATIONSHIP": "relationship",
    "LOCATION": "location",
    "LIFE_EVENT": "life-event",
    "STATUS": "status",
    "EMOTION": "emotion",
    "PROJECT": "project",
    "OTHER": None,
}


@pytest.mark.parametrize(("question", "category"), [
    ("Which city did I move from?", "EVOLUTION"),
    ("What am I trying to learn now?", "STATE"),
    ("What are my stored programming preferences?", "HISTORY"),
    ("What did I complete?", "HISTORY"),
    ("Why did I move from one city to another?", "CAUSAL"),
    ("Which place was I living in before my relocation?", "HISTORY"),
])
def test_clear_personal_recall_intents_route_without_llm(question, category, monkeypatch):
    monkeypatch.setattr(query_router.llm, "classify",
                        lambda *_args: pytest.fail("clear intent should skip classifier"))
    analysis=query_router.classify_query(question)
    assert analysis.is_question is True
    assert analysis.category==category


class EmptyExtractor:
    def extract(self, *_args, **_kwargs):
        return Extraction()


class TableExtractor:
    def __init__(self, states_by_message):
        self.states_by_message = states_by_message

    def extract(self, message, *_args, **_kwargs):
        return Extraction(states=list(self.states_by_message.get(message, [])))


class StructuredModel:
    def __init__(self, claims):
        self.claims = claims
        self.calls = 0

    def answer_with_provenance(self, question, evidence):
        self.calls += 1
        return {"claims": self.claims}


def prepared_store(tmp_path, monkeypatch, name, extractor=None, memories=None):
    store = Database(str(tmp_path / f"{name}.sqlite3"))
    for module in (state_extractor, hybrid_retriever, temporal_engine, causal_graph):
        monkeypatch.setattr(module, "db", store)
    monkeypatch.setattr(state_extractor, "llm", extractor or EmptyExtractor())
    monkeypatch.setattr(
        hybrid_retriever, "search_memory",
        lambda _question, _user_id: list((memories or {}).get(_user_id, [])),
    )
    return store


def snapshot(store, user_id):
    return {
        "states": [state.model_dump(mode="json") for state in store.states(user_id)],
        "transitions": [
            transition.model_dump(mode="json")
            for transition in store.transitions(user_id)
        ],
    }


def state_ingest(store, message, user_id="alice", memory_id="m1"):
    states, edges = state_extractor.ingest_message(message, user_id, [memory_id], [])
    return states, edges


def memory_evidence(source_text, query_category="PREFERENCE"):
    return {
        "user_id": "alice",
        "query_category": query_category,
        "memories": [
            {"id": "source-memory", "user_id": "alice", "memory": source_text}
        ],
        "states": [],
        "transitions": [],
        "causal_path": [],
    }


def claim(text, kind, ref_kind, ref_id):
    return {
        "claims": [
            {
                "text": text,
                "claim_type": kind,
                "evidence": [{"kind": ref_kind, "id": ref_id}],
            }
        ]
    }


# A. Every supported state type -------------------------------------------------


@pytest.mark.parametrize("state_type", sorted(state_extractor.STATE_TYPES))
def test_every_state_type_extracts_and_resolves_a_current_typed_query(
    tmp_path, monkeypatch, state_type
):
    value = f"{state_type.replace('_', ' ').lower()} sample"
    message = f"I have a {value}."
    extractor = TableExtractor(
        {message: [{"state_type": state_type, "state_value": value, "confidence": 0.8}]}
    )
    store = prepared_store(tmp_path, monkeypatch, f"type-{state_type}", extractor)

    created, edges = state_ingest(store, message, memory_id=f"{state_type}-memory")

    assert [(state.state_type, state.state_value) for state in created] == [
        (state_type, value)
    ]
    assert edges == []
    persisted = store.states("alice")
    assert len(persisted) == 1
    assert persisted[0].status == "ACTIVE"
    assert persisted[0].source_memory_ids == [f"{state_type}-memory"]
    assert persisted[0].valid_until is None
    assert persisted[0].valid_from.tzinfo is not None

    unrelated_type = "SKILL" if state_type != "SKILL" else "FACT"
    unrelated = State(
        user_id="alice",
        state_type=unrelated_type,
        state_value=f"unrelated {unrelated_type.lower()}",
        source_memory_ids=["unrelated-memory"],
    )
    store.save_state(unrelated)

    term = STATE_QUERY_TERMS[state_type]
    if term is None:
        result = hybrid_retriever.retrieve("What is current?", "alice", "STATE")
        assert persisted[0].state_id in {
            item["state_id"] for item in result["states"]
        }
        return

    result = hybrid_retriever.retrieve(
        f"What is my current {term}?", "alice", "STATE"
    )
    assert result["query_category"] == "STATE"
    assert result["state_focus"]["candidate_state_ids"] == [persisted[0].state_id]
    assert persisted[0].state_id in {
        item["state_id"] for item in result["states"]
    }
    assert unrelated.state_id not in result["state_focus"]["candidate_state_ids"]


@pytest.mark.parametrize("state_type", sorted(state_extractor.STATE_TYPES))
def test_every_state_type_allows_independent_values_and_noop_reingest(
    tmp_path, monkeypatch, state_type
):
    alpha = f"alpha {state_type.replace('_', ' ').lower()}"
    beta = f"beta {state_type.replace('_', ' ').lower()}"
    first_message = f"I have {alpha}."
    second_message = f"I have {beta}."
    extractor = TableExtractor(
        {
            first_message: [
                {"state_type": state_type, "state_value": alpha, "confidence": 0.8}
            ],
            second_message: [
                {"state_type": state_type, "state_value": beta, "confidence": 0.8}
            ],
        }
    )
    store = prepared_store(tmp_path, monkeypatch, f"independent-{state_type}", extractor)

    state_ingest(store, first_message, memory_id="alpha-memory")
    state_ingest(store, second_message, memory_id="beta-memory")
    active = [state for state in store.states("alice") if state.status == "ACTIVE"]
    assert [(state.state_type, state.state_value) for state in active] == [
        (state_type, alpha),
        (state_type, beta),
    ]
    assert store.transitions("alice") == []
    before = snapshot(store, "alice")

    repeated, repeated_edges = state_ingest(
        store, second_message, memory_id="beta-memory-retry"
    )

    assert repeated == [] and repeated_edges == []
    assert snapshot(store, "alice") == before


@pytest.mark.parametrize("state_type", sorted(state_extractor.STATE_TYPES))
def test_every_state_type_as_of_query_returns_history_not_current_truth(
    tmp_path, monkeypatch, state_type
):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    boundary = start + timedelta(days=1)
    old_value = f"old {state_type.replace('_', ' ').lower()}"
    new_value = f"new {state_type.replace('_', ' ').lower()}"
    old = State(
        user_id="alice",
        state_type=state_type,
        state_value=old_value,
        status="INACTIVE",
        valid_from=start,
        valid_until=boundary,
        source_memory_ids=["old-memory"],
    )
    new = State(
        user_id="alice",
        state_type=state_type,
        state_value=new_value,
        status="ACTIVE",
        valid_from=boundary,
        source_memory_ids=["new-memory"],
    )
    store = prepared_store(tmp_path, monkeypatch, f"asof-{state_type}")
    store.save_state(old)
    store.save_state(new)

    term = STATE_QUERY_TERMS[state_type]
    question = (
        f"What was my {term} as of 2025-01-01T12:00:00Z?"
        if term
        else "What was current as of 2025-01-01T12:00:00Z?"
    )
    result = hybrid_retriever.retrieve(question, "alice", "HISTORY")

    assert [state["state_id"] for state in result["states"]] == [old.state_id]
    assert new.state_id not in {state["state_id"] for state in result["states"]}
    if term:
        assert result["state_focus"]["candidate_state_ids"] == [old.state_id]


# B. Preference and comparison semantics ---------------------------------------


def test_plain_preference_closes_the_prior_global_preference(
    tmp_path, monkeypatch
):
    store = prepared_store(tmp_path, monkeypatch, "simple-preference")

    first, first_edges = state_ingest(store, "I prefer tea.", memory_id="tea-memory")
    second, second_edges = state_ingest(
        store, "I prefer coffee.", memory_id="coffee-memory"
    )

    assert [(state.state_type, state.state_value) for state in first] == [
        ("PREFERENCE", "tea")
    ]
    assert [(state.state_type, state.state_value) for state in second] == [
        ("PREFERENCE", "coffee")
    ]
    assert first_edges == []
    assert len(second_edges) == 1
    assert second_edges[0].transition_type == "PREFERENCE_CHANGE"
    assert second_edges[0].from_state_id == first[0].state_id
    assert second_edges[0].to_state_id == second[0].state_id
    active = [state for state in store.states("alice") if state.status == "ACTIVE"]
    assert [(state.state_type, state.state_value) for state in active] == [
        ("PREFERENCE", "coffee")
    ]
    history = {state.state_id: state for state in store.states("alice")}
    assert history[first[0].state_id].status == "INACTIVE"
    before = snapshot(store, "alice")

    repeated, repeated_edges = state_ingest(
        store, "I really prefer coffee.", memory_id="coffee-retry"
    )

    assert repeated == [] and repeated_edges == []
    assert snapshot(store, "alice") == before


def test_programming_preference_paraphrase_uses_current_state_only(
    tmp_path, monkeypatch
):
    store = prepared_store(tmp_path, monkeypatch, "programming-preference")
    python, _ = state_ingest(
        store, "I prefer Python for programming.", memory_id="python-memory"
    )
    java, edges = state_ingest(
        store, "I prefer Java for programming.", memory_id="java-memory"
    )

    evidence = hybrid_retriever.retrieve(
        "what i prefer for programming", "alice", "HISTORY"
    )
    assert [state["state_id"] for state in evidence["states"]] == [java[0].state_id]
    assert python[0].state_id not in {
        state["state_id"] for state in evidence["states"]
    }

    model = StructuredModel(
        claim(
            "I prefer Java for programming.",
            "fact",
            "state",
            java[0].state_id,
        )["claims"]
    )
    monkeypatch.setattr(answer_generator, "llm", model)
    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "what i prefer for programming", evidence
    )
    assert f"state:{java[0].state_id}" in answer
    assert diagnostics["accepted_claim_count"] == 1


def test_legacy_conflicting_active_preferences_abstain_on_yes_no_query(
    tmp_path, monkeypatch
):
    store = prepared_store(tmp_path, monkeypatch, "legacy-preference-conflict")
    store.save_state(
        State(user_id="alice", state_type="PREFERENCE", state_value="tea",
              status="ACTIVE", source_memory_ids=["tea-memory"])
    )
    store.save_state(
        State(user_id="alice", state_type="PREFERENCE", state_value="coffee",
              status="ACTIVE", source_memory_ids=["coffee-memory"])
    )

    evidence = hybrid_retriever.retrieve("Do I prefer tea?", "alice", "PREFERENCE")
    assert evidence["ambiguity"]["kind"] == "multiple_state_candidates"

    model = StructuredModel([])
    monkeypatch.setattr(answer_generator, "llm", model)
    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "Do I prefer tea?", evidence
    )
    assert answer == answer_generator.ABSTENTION
    assert diagnostics["answer_reason"] == "ambiguous_evidence"
    assert model.calls == 0


@pytest.mark.parametrize(
    ("message", "expected_target", "dispreferred"),
    [
        ("I prefer coffee rather than tea.", "coffee", "tea"),
        ("I like tea more than coffee.", "tea", "coffee"),
        ("I prefer coffee less than tea.", "tea", "coffee"),
        ("I love coffee more than tea.", "coffee", "tea"),
        ("I prefer coffee over tea.", "coffee", "tea"),
    ],
)
def test_comparative_preference_forms_normalize_preferred_direction(
    tmp_path, monkeypatch, message, expected_target, dispreferred
):
    store = prepared_store(tmp_path, monkeypatch, f"comparative-{expected_target}")
    old = State(
        user_id="alice",
        state_type="PREFERENCE",
        state_value=dispreferred,
        source_memory_ids=["old-memory"],
    )
    store.save_state(old)

    created, edges = state_ingest(store, message, memory_id="new-memory")

    assert [(state.state_type, state.state_value) for state in created] == [
        ("PREFERENCE", expected_target)
    ]
    assert len(edges) == 1
    assert edges[0].transition_type == "PREFERENCE_CHANGE"
    assert edges[0].relation_kind == "TEMPORAL_SUCCESSION"
    assert edges[0].from_state_id == old.state_id
    assert edges[0].to_state_id == created[0].state_id
    history = {state.state_id: state for state in store.states("alice")}
    assert history[old.state_id].status == "INACTIVE"
    assert [state.state_value for state in history.values() if state.status == "ACTIVE"] == [
        expected_target
    ]


def test_preference_retraction_and_historical_only_statement(tmp_path, monkeypatch):
    store = prepared_store(tmp_path, monkeypatch, "preference-retraction")
    state_ingest(store, "I prefer coffee.", memory_id="coffee-memory")
    active = next(state for state in store.states("alice") if state.status == "ACTIVE")

    retracted, edges = state_ingest(
        store, "I no longer prefer coffee.", memory_id="retraction-memory"
    )

    assert retracted == [] and edges == []
    history = {state.state_id: state for state in store.states("alice")}
    assert history[active.state_id].status == "INACTIVE"
    assert history[active.state_id].valid_until is not None
    assert not any(state.status == "ACTIVE" for state in history.values())

    historical_store = prepared_store(
        tmp_path, monkeypatch, "preference-historical-only"
    )
    created, historical_edges = state_ingest(
        historical_store, "I used to prefer coffee.", memory_id="used-to-memory"
    )
    assert created == [] and historical_edges == []
    assert historical_store.states("alice") == []


def test_current_marker_replaces_unique_preference_and_keeps_history(
    tmp_path, monkeypatch
):
    store = prepared_store(tmp_path, monkeypatch, "current-marker-preference")
    old, _ = state_ingest(store, "I prefer coffee.", memory_id="coffee-memory")

    created, edges = state_ingest(
        store, "Currently, I prefer tea.", memory_id="tea-memory"
    )

    assert len(created) == len(edges) == 1
    history = {state.state_id: state for state in store.states("alice")}
    assert history[old[0].state_id].status == "INACTIVE"
    assert history[old[0].state_id].valid_until == edges[0].timestamp
    assert [state.state_value for state in history.values() if state.status == "ACTIVE"] == [
        "tea"
    ]
    assert edges[0].transition_type == "PREFERENCE_CHANGE"


def test_independent_preference_domains_remain_active_without_transition(
    tmp_path, monkeypatch
):
    store = prepared_store(tmp_path, monkeypatch, "preference-domains")

    state_ingest(
        store,
        "I prefer Python for backend development.",
        memory_id="python-memory",
    )
    state_ingest(store, "I prefer tea for drinks.", memory_id="tea-memory")

    active = [state for state in store.states("alice") if state.status == "ACTIVE"]
    assert [(state.state_type, state.state_value) for state in active] == [
        ("PREFERENCE", "Python"),
        ("PREFERENCE", "tea"),
    ]
    assert {state.metadata.get("preference_key") for state in active} == {
        "favorite:backend development",
        "favorite:drinks",
    }
    assert store.transitions("alice") == []


def test_multiple_preference_updates_preserve_a_directional_chain(tmp_path, monkeypatch):
    store = prepared_store(tmp_path, monkeypatch, "preference-chain")
    state_ingest(store, "I prefer tea.", memory_id="tea-memory")
    state_ingest(store, "I now prefer coffee over tea.", memory_id="coffee-memory")
    state_ingest(store, "I now prefer juice over coffee.", memory_id="juice-memory")

    states = store.states("alice")
    edges = store.transitions("alice")
    assert [state.state_value for state in states] == ["tea", "coffee", "juice"]
    assert [state.status for state in states] == ["INACTIVE", "INACTIVE", "ACTIVE"]
    assert [edge.transition_type for edge in edges] == [
        "PREFERENCE_CHANGE",
        "PREFERENCE_CHANGE",
    ]
    by_id = {state.state_id: state for state in states}
    assert [by_id[edge.from_state_id].state_value for edge in edges] == [
        "tea",
        "coffee",
    ]
    assert [by_id[edge.to_state_id].state_value for edge in edges] == [
        "coffee",
        "juice",
    ]
    for edge in edges:
        assert by_id[edge.from_state_id].valid_until == edge.timestamp
        assert by_id[edge.to_state_id].valid_from == edge.timestamp


def test_preference_replacement_closes_an_already_active_target(tmp_path, monkeypatch):
    store = prepared_store(tmp_path, monkeypatch, "preference-active-target")
    old_tea = State(
        user_id="alice",
        state_type="PREFERENCE",
        state_value="tea",
        source_memory_ids=["old-tea"],
    )
    coffee = State(
        user_id="alice",
        state_type="PREFERENCE",
        state_value="coffee",
        source_memory_ids=["coffee-memory"],
    )
    store.save_state(old_tea)
    store.save_state(coffee)

    created, edges = state_ingest(
        store, "I prefer tea over coffee.", memory_id="new-tea"
    )

    assert [(state.state_type, state.state_value) for state in created] == [
        ("PREFERENCE", "tea")
    ]
    assert len(edges) == 1
    edge = edges[0]
    assert edge.transition_type == "PREFERENCE_CHANGE"
    assert edge.relation_kind == "TEMPORAL_SUCCESSION"
    assert edge.from_state_id == coffee.state_id
    assert edge.to_state_id == created[0].state_id

    history = {state.state_id: state for state in store.states("alice")}
    assert history[old_tea.state_id].status == "INACTIVE"
    assert history[coffee.state_id].status == "INACTIVE"
    assert history[old_tea.state_id].valid_until == edge.timestamp
    assert history[coffee.state_id].valid_until == edge.timestamp
    assert created[0].valid_from == edge.timestamp
    assert [state.state_id for state in history.values() if state.status == "ACTIVE"] == [
        created[0].state_id
    ]

    before = snapshot(store, "alice")
    repeated, repeated_edges = state_ingest(
        store, "I prefer tea over coffee.", memory_id="new-tea-retry"
    )
    assert repeated == [] and repeated_edges == []
    assert snapshot(store, "alice") == before


def test_location_move_is_typed_queryable_and_replaces_the_origin(tmp_path, monkeypatch):
    memories = {
        "alice": [
            {"id": "chennai-memory", "user_id": "alice", "memory": "User lived in Chennai."},
            {
                "id": "move-memory",
                "user_id": "alice",
                "memory": "User moved from Chennai to Coimbatore.",
            },
        ]
    }
    store = prepared_store(
        tmp_path, monkeypatch, "location-move", memories=memories
    )
    old, _ = state_ingest(store, "I live in Chennai.", memory_id="chennai-memory")

    created, edges = state_ingest(
        store, "I moved from Chennai to Coimbatore.", memory_id="move-memory"
    )

    assert [(state.state_type, state.state_value) for state in created] == [
        ("LOCATION", "Coimbatore")
    ]
    assert len(edges) == 1
    edge = edges[0]
    assert edge.transition_type == "LOCATION_CHANGE"
    assert edge.relation_kind == "TEMPORAL_SUCCESSION"
    assert edge.from_state_id == old[0].state_id
    assert edge.to_state_id == created[0].state_id
    history = {state.state_id: state for state in store.states("alice")}
    assert history[old[0].state_id].status == "INACTIVE"
    assert history[old[0].state_id].valid_until == edge.timestamp
    assert created[0].valid_from == edge.timestamp

    current = hybrid_retriever.retrieve("Where do I live now?", "alice", "STATE")
    assert [state["state_value"] for state in current["states"]] == ["Coimbatore"]
    assert current["state_focus"]["candidate_state_ids"] == [created[0].state_id]

    previous = hybrid_retriever.retrieve("Where did I live before?", "alice", "HISTORY")
    assert [state["state_value"] for state in previous["states"]] == ["Chennai"]
    assert previous["transitions"][0]["transition_type"] == "LOCATION_CHANGE"


@pytest.mark.parametrize(
    "move_message",
    [
        "I moved from Chennai to Coimbatore.",
        "I relocated to Coimbatore from Chennai.",
        "I have shifted from Chennai to Coimbatore.",
        "I now live in Coimbatore.",
    ],
)
def test_location_move_paraphrases_replace_the_origin(tmp_path, monkeypatch, move_message):
    store = prepared_store(tmp_path, monkeypatch, "location-paraphrase")
    old, _ = state_ingest(store, "I live in Chennai.", memory_id="origin-memory")

    created, edges = state_ingest(store, move_message, memory_id="move-memory")

    assert [(state.state_type, state.state_value) for state in created] == [
        ("LOCATION", "Coimbatore")
    ]
    assert len(edges) == 1
    assert edges[0].transition_type == "LOCATION_CHANGE"
    assert edges[0].from_state_id == old[0].state_id
    history = {state.state_id: state for state in store.states("alice")}
    assert history[old[0].state_id].status == "INACTIVE"
    assert history[created[0].state_id].status == "ACTIVE"


@pytest.mark.parametrize(
    "goal_message",
    [
        "My new goal is to learn data structures.",
        "My current goal is to learn data structures.",
        "I am now trying to learn data structures.",
        "I am now planning to learn data structures.",
    ],
)
def test_goal_paraphrases_replace_the_previous_goal(tmp_path, monkeypatch, goal_message):
    store = prepared_store(tmp_path, monkeypatch, "goal-paraphrase")
    old, _ = state_ingest(
        store, "My goal is to learn system design.", memory_id="old-goal"
    )

    created, edges = state_ingest(store, goal_message, memory_id="new-goal")

    assert [(state.state_type, state.state_value) for state in created] == [
        ("GOAL", "learn data structures")
    ]
    assert len(edges) == 1
    assert edges[0].transition_type == "GOAL_CHANGE"
    assert edges[0].relation_kind == "TEMPORAL_SUCCESSION"
    assert edges[0].from_state_id == old[0].state_id
    history = {state.state_id: state for state in store.states("alice")}
    assert history[old[0].state_id].status == "INACTIVE"
    assert history[created[0].state_id].status == "ACTIVE"


@pytest.mark.parametrize(
    "completion_message",
    [
        "I completed my machine learning project.",
        "I have finished my machine learning project.",
        "I just finished my machine learning project.",
        "I already achieved my machine learning project.",
    ],
)
def test_completion_paraphrases_create_only_a_life_event(
    tmp_path, monkeypatch, completion_message
):
    extractor = TableExtractor(
        {
            completion_message: [
                {"state_type": "INTEREST", "state_value": "machine learning project"},
                {"state_type": "GOAL", "state_value": "machine learning project"},
            ]
        }
    )
    store = prepared_store(tmp_path, monkeypatch, "completion-paraphrase", extractor)

    created, edges = state_ingest(store, completion_message, memory_id="completion")

    assert [(state.state_type, state.state_value) for state in created] == [
        ("LIFE_EVENT", "completed my machine learning project")
    ]
    assert edges == []
    assert [state.state_type for state in store.states("alice")] == ["LIFE_EVENT"]


@pytest.mark.parametrize(
    "question",
    [
        "Where did I live before?",
        "What was my previous location?",
        "Where was I living previously?",
    ],
)
def test_previous_location_paraphrases_return_only_the_closed_state(
    tmp_path, monkeypatch, question
):
    store = prepared_store(tmp_path, monkeypatch, "previous-location-paraphrase")
    old, _ = state_ingest(store, "I live in Chennai.", memory_id="origin")
    new, edges = state_ingest(
        store, "I moved from Chennai to Coimbatore.", memory_id="move"
    )

    evidence = hybrid_retriever.retrieve(question, "alice", "HISTORY")

    assert [state["state_id"] for state in evidence["states"]] == [old[0].state_id]
    assert [state["status"] for state in evidence["states"]] == ["INACTIVE"]
    assert evidence["transitions"][0]["transition_id"] == edges[0].transition_id
    assert new[0].state_id not in {
        state["state_id"] for state in evidence["states"]
    }


@pytest.mark.parametrize(
    "question",
    [
        "What was my previous goal?",
        "What goal did I have before?",
        "What was my goal previously?",
    ],
)
def test_previous_goal_paraphrases_return_only_the_closed_goal(
    tmp_path, monkeypatch, question
):
    store = prepared_store(tmp_path, monkeypatch, "previous-goal-paraphrase")
    old, _ = state_ingest(
        store, "My goal is to learn system design.", memory_id="old-goal"
    )
    new, edges = state_ingest(
        store, "My new goal is to learn data structures.", memory_id="new-goal"
    )

    evidence = hybrid_retriever.retrieve(question, "alice", "HISTORY")

    assert [state["state_id"] for state in evidence["states"]] == [old[0].state_id]
    assert [state["status"] for state in evidence["states"]] == ["INACTIVE"]
    assert evidence["transitions"][0]["transition_id"] == edges[0].transition_id
    assert new[0].state_id not in {
        state["state_id"] for state in evidence["states"]
    }


@pytest.mark.parametrize(
    "question",
    [
        "What did I complete?",
        "What have I completed?",
        "What did I finish?",
    ],
)
def test_completion_paraphrases_route_to_the_life_event(
    tmp_path, monkeypatch, question
):
    store = prepared_store(tmp_path, monkeypatch, "completion-query-paraphrase")
    completed, _ = state_ingest(
        store, "I completed my machine learning project.", memory_id="completion"
    )

    evidence = hybrid_retriever.retrieve(question, "alice", "HISTORY")

    assert evidence["state_focus"]["candidate_state_ids"] == [completed[0].state_id]
    assert evidence["state_focus"]["state_types"] == ["LIFE_EVENT"]


@pytest.mark.parametrize(
    "question",
    [
        "What location change did I tell you about?",
        "What was my location change?",
        "Did my location change?",
    ],
)
def test_location_change_paraphrases_retrieve_the_transition(
    tmp_path, monkeypatch, question
):
    store = prepared_store(tmp_path, monkeypatch, "location-change-paraphrase")
    state_ingest(store, "I live in Chennai.", memory_id="origin")
    _, edges = state_ingest(
        store, "I moved from Chennai to Coimbatore.", memory_id="move"
    )

    evidence = hybrid_retriever.retrieve(question, "alice", "HISTORY")

    assert [edge["transition_id"] for edge in evidence["transitions"]] == [
        edges[0].transition_id
    ]


@pytest.mark.parametrize(
    "question",
    ["Which city did I move from?", "Which city did I move to?"],
)
def test_relocation_origin_and_destination_paraphrases_keep_both_endpoints(
    tmp_path, monkeypatch, question
):
    store = prepared_store(tmp_path, monkeypatch, "relocation-endpoint")
    old, _ = state_ingest(store, "I live in Chennai.", memory_id="origin-memory")
    new, edges = state_ingest(
        store, "I moved from Chennai to Coimbatore.", memory_id="move-memory"
    )

    evidence = hybrid_retriever.retrieve(question, "alice", "HISTORY")

    assert [edge["transition_id"] for edge in evidence["transitions"]] == [
        edges[0].transition_id
    ]
    assert {state["state_id"] for state in evidence["states"]} == {
        old[0].state_id,
        new[0].state_id,
    }


def test_place_living_before_relocation_uses_verified_predecessor(
    tmp_path, monkeypatch
):
    store = prepared_store(tmp_path, monkeypatch, "place-before-relocation")
    old, _ = state_ingest(store, "I live in Chennai.", memory_id="origin-memory")
    state_ingest(
        store, "I moved from Chennai to Coimbatore.", memory_id="move-memory"
    )

    question = "Which place was I living in before my relocation?"
    evidence = hybrid_retriever.retrieve(question, "alice", "HISTORY")

    assert [state["state_id"] for state in evidence["states"]] == [old[0].state_id]
    assert evidence["transitions"][0]["from_state_id"] == old[0].state_id


def test_trying_to_learn_now_focuses_the_current_goal(tmp_path, monkeypatch):
    store = prepared_store(tmp_path, monkeypatch, "trying-to-learn-goal")
    goal, _ = state_ingest(
        store, "My new goal is to learn system design.", memory_id="goal-memory"
    )

    evidence = hybrid_retriever.retrieve(
        "What am I trying to learn now?", "alice", "STATE"
    )

    assert evidence["query_category"] == "STATE"
    assert evidence["state_focus"]["candidate_state_ids"] == [goal[0].state_id]
    assert [state["state_id"] for state in evidence["states"]] == [goal[0].state_id]


def test_project_have_i_finished_focuses_completion_record(tmp_path, monkeypatch):
    store = prepared_store(tmp_path, monkeypatch, "project-finished-question")
    completed, _ = state_ingest(
        store, "I completed my machine learning project.", memory_id="completion-memory"
    )

    evidence = hybrid_retriever.retrieve(
        "Which project have I finished?", "alice", "HISTORY"
    )

    assert evidence["state_focus"]["candidate_state_ids"] == [completed[0].state_id]
    assert evidence["state_focus"]["state_types"] == ["LIFE_EVENT"]


def test_sport_exercise_before_swimming_uses_verified_activity_transition(
    tmp_path, monkeypatch
):
    from datetime import datetime, timezone

    store = prepared_store(tmp_path, monkeypatch, "sport-before-swimming")
    boundary = datetime(2026, 10, 22, 15, 56, 44, tzinfo=timezone.utc)
    running = State(
        user_id="alice", state_type="ACTIVITY", state_value="running",
        status="INACTIVE", valid_until=boundary, source_memory_ids=["running-memory"],
    )
    swimming = State(
        user_id="alice", state_type="ACTIVITY", state_value="swimming",
        valid_from=boundary, source_memory_ids=["swimming-memory"],
    )
    store.save_state(running)
    store.save_state(swimming)
    edge = Transition(
        user_id="alice", from_state_id=running.state_id,
        to_state_id=swimming.state_id, transition_type="ACTIVITY_CHANGE",
        relation_kind="TEMPORAL_SUCCESSION", timestamp=boundary,
        source_memory_ids=["swimming-memory"],
    )
    store.save_transition(edge)

    evidence = hybrid_retriever.retrieve(
        "What sport or exercise was I doing before swimming?", "alice", "HISTORY"
    )

    assert [state["state_id"] for state in evidence["states"]] == [running.state_id]
    assert [transition["transition_id"] for transition in evidence["transitions"]] == [
        edge.transition_id
    ]


@pytest.mark.parametrize(("question", "memory_text", "expected_type"), [
    ("What am I trying to learn now?",
     "User's new goal is to learn system design", "GOAL"),
    ("Which project have I finished?",
     "User completed their machine learning project", "LIFE_EVENT"),
])
def test_legacy_other_state_falls_back_to_owned_memory_after_bad_model_ref(
    tmp_path, monkeypatch, question, memory_text, expected_type
):
    store=prepared_store(tmp_path,monkeypatch,"legacy-memory-fallback")
    state=State(user_id="alice",state_type="OTHER",
                state_value=memory_text,source_memory_ids=[])
    store.save_state(state)
    memory_id="legacy-supported-memory"
    monkeypatch.setattr(
        hybrid_retriever,"search_memory",
        lambda _query,_user:[{"id":memory_id,"user_id":"alice",
                              "memory":memory_text}],
    )
    category="STATE" if expected_type=="GOAL" else "HISTORY"
    evidence=hybrid_retriever.retrieve(question,"alice",category)
    assert evidence["state_focus"]["state_types"]==[expected_type]
    assert evidence["state_focus"]["memory_fallback"] is True
    assert evidence["state_focus"]["source_memory_ids"]==[memory_id]

    class BadCitationModel:
        def answer_with_provenance(self,_question,_evidence):
            return {"claims":[{"text":"unsupported answer","claim_type":"fact",
                "evidence":[{"kind":"memory","ref":"M999"}]}]}

    monkeypatch.setattr(answer_generator,"llm",BadCitationModel())
    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(question,evidence)

    assert answer.startswith(memory_text)
    assert f"memory:{memory_id}" in answer
    assert diagnostics["fallback_claim_used"] is True
    assert diagnostics["model_claim_rejections"]["rejection_details"]=={
        "citation_ref_unknown":1,
    }
    assert diagnostics["accepted_claim_count"]==1
    assert diagnostics["citation_validation"][0]["reason_code"]=="citation_resolved"


def test_unique_relocation_direction_uses_validated_transition_if_model_ref_is_bad(
    tmp_path,monkeypatch
):
    store=prepared_store(tmp_path,monkeypatch,"relocation-transition-fallback")
    state_ingest(store,"I live in Chennai.",memory_id="chennai-memory")
    _,edges=state_ingest(store,"I moved from Chennai to Coimbatore.",
                         memory_id="relocation-memory")
    question="Which city did I move from?"
    evidence=hybrid_retriever.retrieve(question,"alice","EVOLUTION")
    assert [edge["transition_id"] for edge in evidence["transitions"]]==[
        edges[0].transition_id
    ]

    class BadCitationModel:
        def answer_with_provenance(self,_question,_evidence):
            return {"claims":[{"text":"Chennai was the origin","claim_type":"temporal",
                "evidence":[{"kind":"transition","ref":"T999"}]}]}

    monkeypatch.setattr(answer_generator,"llm",BadCitationModel())
    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(question,evidence)
    assert answer.startswith("You moved from Chennai")
    assert f"transition:{edges[0].transition_id}" in answer
    assert diagnostics["fallback_claim_used"] is True
    assert diagnostics["accepted_claim_count"]==1


def test_stored_programming_preferences_preserve_conflict_evidence(
    tmp_path, monkeypatch
):
    store = prepared_store(tmp_path, monkeypatch, "stored-programming-preferences")
    java = State(
        user_id="alice", state_type="PREFERENCE", state_value="Java",
        source_memory_ids=["java-memory"], metadata={"preference_key": "favorite:programming"},
    )
    python = State(
        user_id="alice", state_type="PREFERENCE", state_value="Python",
        source_memory_ids=["python-memory"], metadata={"preference_key": "favorite:programming"},
    )
    store.save_state(java)
    store.save_state(python)
    memories = {
        "alice": [
            {"id": "java-memory", "user_id": "alice", "memory": "I prefer Java for programming."},
            {"id": "python-memory", "user_id": "alice", "memory": "I prefer Python for programming."},
        ]
    }
    monkeypatch.setattr(
        hybrid_retriever, "search_memory",
        lambda _question, user_id: list(memories.get(user_id, [])),
    )

    evidence = hybrid_retriever.retrieve(
        "What are my stored programming preferences?", "alice", "HISTORY"
    )

    assert {state["state_id"] for state in evidence["states"]} == {
        java.state_id, python.state_id,
    }
    assert {memory["id"] for memory in evidence["memories"]} == {
        "java-memory", "python-memory",
    }
    assert "ambiguity" not in evidence
    assert set(evidence.get("preference_conflict", {}).get("state_ids", [])) == {
        java.state_id, python.state_id,
    }
    from llm_service import model_evidence_projection
    projected, _citation_refs = model_evidence_projection(evidence)
    assert set(projected["preference_conflict"]["values"]) == {"Java", "Python"}
    assert len(projected["preference_conflict"]["state_refs"]) == 2
    assert set(evidence["preference_conflict"]["memory_ids"]) == {
        "java-memory", "python-memory",
    }
    assert evidence["state_focus"]["stored_preference_inventory"] is True

    class ConflictAnswer:
        def answer_with_provenance(self, _question, model_evidence):
            refs=model_evidence["preference_conflict"]["state_ids"]
            memory_ids=model_evidence["preference_conflict"]["memory_ids"]
            return {"claims":[{
                "text":"Your stored programming preferences conflict: Java and Python.",
                "claim_type":"fact",
                "evidence":([{"kind":"state","id":ref} for ref in refs]
                            +[{"kind":"memory","id":ref} for ref in memory_ids]),
            }]}

    monkeypatch.setattr(answer_generator, "llm", ConflictAnswer())
    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "What are my stored programming preferences?", evidence
    )
    assert diagnostics["rejected_claim_count"] == 0, diagnostics["rejected_claims"]
    assert "conflict" in answer.casefold()
    assert "Java" in answer and "Python" in answer
    assert f"state:{java.state_id}" in answer and f"state:{python.state_id}" in answer
    assert diagnostics["accepted_claim_count"] == 1
    assert diagnostics["rejected_claim_count"] == 0


def test_previous_location_without_a_transition_abstains_specifically(
    tmp_path, monkeypatch
):
    store = prepared_store(tmp_path, monkeypatch, "missing-previous-location")
    state_ingest(store, "I live in Chennai.", memory_id="only-location")

    evidence = hybrid_retriever.retrieve(
        "Where did I live before?", "alice", "HISTORY"
    )
    assert evidence["states"] == [] and evidence["transitions"] == []
    assert evidence["retrieval_reason"] == "missing_predecessor_transition"

    model = StructuredModel([])
    monkeypatch.setattr(answer_generator, "llm", model)
    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "Where did I live before?", evidence
    )
    assert answer == answer_generator.ABSTENTION
    assert diagnostics["answer_reason"] == "missing_predecessor_transition"
    assert model.calls == 0


def test_goal_completion_and_replacement_are_typed_and_historical(tmp_path, monkeypatch):
    store = prepared_store(tmp_path, monkeypatch, "goal-completion")
    old, _ = state_ingest(
        store,
        "My current goal is to complete my machine learning project.",
        memory_id="goal-memory",
    )

    created, edges = state_ingest(
        store,
        "I completed my machine learning project, and my new goal is to learn system design.",
        memory_id="completion-memory",
    )

    assert {(state.state_type, state.state_value) for state in created} == {
        ("GOAL", "learn system design"),
        ("LIFE_EVENT", "completed my machine learning project"),
    }
    assert len(edges) == 1
    edge = edges[0]
    assert edge.transition_type == "GOAL_CHANGE"
    assert edge.relation_kind == "TEMPORAL_SUCCESSION"
    assert edge.from_state_id == old[0].state_id
    new_goal = next(state for state in created if state.state_type == "GOAL")
    completion = next(state for state in created if state.state_type == "LIFE_EVENT")
    assert edge.to_state_id == new_goal.state_id

    history = {state.state_id: state for state in store.states("alice")}
    assert history[old[0].state_id].status == "INACTIVE"
    assert history[old[0].state_id].valid_until == edge.timestamp
    assert new_goal.valid_from == edge.timestamp
    assert history[completion.state_id].status == "ACTIVE"
    assert not any(state.state_type == "OTHER" for state in history.values())

    current = hybrid_retriever.retrieve("What is my current goal?", "alice", "STATE")
    assert current["state_focus"]["candidate_state_ids"] == [new_goal.state_id]

    previous = hybrid_retriever.retrieve("What was my previous goal?", "alice", "HISTORY")
    assert [state["state_value"] for state in previous["states"]] == [
        "complete my machine learning project"
    ]

    completed = hybrid_retriever.retrieve("What did I complete?", "alice", "HISTORY")
    assert completed["state_focus"]["candidate_state_ids"] == [completion.state_id]
    assert completed["state_focus"]["state_types"] == ["LIFE_EVENT"]


def test_completion_without_a_new_goal_closes_the_goal_and_keeps_the_event(
    tmp_path, monkeypatch
):
    store = prepared_store(tmp_path, monkeypatch, "goal-completion-only")
    old, _ = state_ingest(
        store,
        "My goal is to complete my machine learning project.",
        memory_id="goal-memory",
    )

    created, edges = state_ingest(
        store,
        "I completed my machine learning project.",
        memory_id="completion-memory",
    )

    assert [(state.state_type, state.state_value) for state in created] == [
        ("LIFE_EVENT", "completed my machine learning project")
    ]
    assert len(edges) == 1
    assert edges[0].transition_type == "GOAL_COMPLETION"
    assert edges[0].relation_kind == "TEMPORAL_SUCCESSION"
    assert edges[0].from_state_id == old[0].state_id
    history = {state.state_id: state for state in store.states("alice")}
    assert history[old[0].state_id].status == "INACTIVE"
    assert history[created[0].state_id].status == "ACTIVE"

    previous = hybrid_retriever.retrieve("What was my previous goal?", "alice", "HISTORY")
    assert [state["state_value"] for state in previous["states"]] == [
        "complete my machine learning project"
    ]


# C. Every supported transition type -------------------------------------------


@pytest.mark.parametrize(
    (
        "name",
        "transition_type",
        "relation_kind",
        "initial_type",
        "initial_value",
        "message",
        "extracted_states",
        "new_value",
        "trigger_fragment",
    ),
    [
        (
            "preference",
            "PREFERENCE_CHANGE",
            "TEMPORAL_SUCCESSION",
            "PREFERENCE",
            "tea",
            "I now prefer coffee over tea.",
            [],
            "coffee",
            None,
        ),
        (
            "goal",
            "GOAL_CHANGE",
            "TEMPORAL_SUCCESSION",
            "GOAL",
            "learn guitar",
            "I want to learn piano instead.",
            [{"state_type": "GOAL", "state_value": "learn piano", "confidence": 0.9}],
            "learn piano",
            None,
        ),
        (
            "location",
            "LOCATION_CHANGE",
            "TEMPORAL_SUCCESSION",
            "LOCATION",
            "London",
            "I moved to Paris.",
            [],
            "Paris",
            None,
        ),
        (
            "activity",
            "ACTIVITY_CHANGE",
            "EXPLICIT_CAUSE",
            "ACTIVITY",
            "running",
            "Due to a knee injury I switched from running to swimming.",
            [
                {
                    "state_type": "ACTIVITY",
                    "state_value": "swimming",
                    "confidence": 0.9,
                }
            ],
            "swimming",
            "knee injury",
        ),
    ],
)
def test_every_transition_type_has_exact_direction_boundaries_and_provenance(
    tmp_path,
    monkeypatch,
    name,
    transition_type,
    relation_kind,
    initial_type,
    initial_value,
    message,
    extracted_states,
    new_value,
    trigger_fragment,
):
    extractor = TableExtractor({message: extracted_states})
    store = prepared_store(tmp_path, monkeypatch, f"transition-{name}", extractor)
    source = State(
        user_id="alice",
        state_type=initial_type,
        state_value=initial_value,
        source_memory_ids=["old-memory"],
    )
    store.save_state(source)

    created, edges = state_ingest(store, message, memory_id="new-memory")

    assert len(created) == len(edges) == 1
    edge = edges[0]
    destination = created[0]
    assert edge.transition_type == transition_type
    assert edge.relation_kind == relation_kind
    assert edge.from_state_id == source.state_id
    assert edge.to_state_id == destination.state_id
    assert edge.source_memory_ids == ["new-memory"]
    assert destination.source_memory_ids == ["new-memory"]
    if trigger_fragment:
        assert trigger_fragment in (edge.trigger or "")
    else:
        assert edge.trigger is None and edge.reason is None

    history = {state.state_id: state for state in store.states("alice")}
    assert history[source.state_id].status == "INACTIVE"
    assert history[source.state_id].valid_until == edge.timestamp
    assert destination.status == "ACTIVE"
    assert destination.valid_from == edge.timestamp

    term = {
        "PREFERENCE": "preference",
        "GOAL": "goal",
        "LOCATION": "location",
        "ACTIVITY": "activity",
    }[initial_type]
    before = hybrid_retriever.retrieve(
        f"What was my {term}?",
        "alice",
        "HISTORY",
        (edge.timestamp - timedelta(microseconds=1)).isoformat(),
    )
    at_boundary = hybrid_retriever.retrieve(
        f"What was my {term}?",
        "alice",
        "HISTORY",
        edge.timestamp.isoformat(),
    )
    assert [state["state_id"] for state in before["states"]] == [source.state_id]
    assert [state["state_id"] for state in at_boundary["states"]] == [
        destination.state_id
    ]

    before_snapshot = snapshot(store, "alice")
    repeated, repeated_edges = state_ingest(
        store, message, memory_id="new-memory-retry"
    )
    assert repeated == [] and repeated_edges == []
    assert snapshot(store, "alice") == before_snapshot


@pytest.mark.parametrize(
    ("message", "active_values", "extracted_states"),
    [
        (
            "Due to a knee injury I switched from yoga to swimming.",
            ["running"],
            [{"state_type": "ACTIVITY", "state_value": "swimming", "confidence": 0.9}],
        ),
        (
            "Due to a knee injury I switched to swimming.",
            ["running", "cycling"],
            [{"state_type": "ACTIVITY", "state_value": "swimming", "confidence": 0.9}],
        ),
        (
            "Due to a knee injury I switched from running to running.",
            ["running"],
            [{"state_type": "ACTIVITY", "state_value": "running", "confidence": 0.9}],
        ),
    ],
)
def test_unknown_ambiguous_or_self_transition_never_creates_an_edge(
    tmp_path, monkeypatch, message, active_values, extracted_states
):
    extractor = TableExtractor({message: extracted_states})
    store = prepared_store(tmp_path, monkeypatch, "invalid-cause", extractor)
    for index, value in enumerate(active_values):
        store.save_state(
            State(
                user_id="alice",
                state_type="ACTIVITY",
                state_value=value,
                source_memory_ids=[f"old-{index}"],
            )
        )

    created, edges = state_ingest(store, message, memory_id="new-memory")

    assert edges == []
    assert store.transitions("alice") == []
    assert not any(
        state.state_type == "OTHER" and state.state_value == message
        for state in created
    )


# D. Temporal and historical behavior ------------------------------------------


def test_state_defaults_are_timezone_aware_and_open_ended(tmp_path, monkeypatch):
    store = prepared_store(tmp_path, monkeypatch, "state-defaults")
    message = "I have a fact sample."
    extractor = TableExtractor(
        {
            message: [
                {
                    "state_type": "FACT",
                    "state_value": "fact sample",
                    "confidence": 0.8,
                }
            ]
        }
    )
    monkeypatch.setattr(state_extractor, "llm", extractor)

    created, _ = state_ingest(store, message, memory_id="fact-memory")

    assert created[0].valid_from.tzinfo is not None
    assert created[0].valid_until is None
    assert store.states_at("alice", datetime.now(timezone.utc)) == created


def test_conflicting_natural_and_iso_timestamps_fail_closed(monkeypatch):
    with pytest.raises(ValueError, match="conflicting timestamps"):
        temporal_engine.resolve_requested_timestamp(
            "What was my goal on January 1, 2025 UTC at 2025-02-01T00:00:00Z?"
        )


# E. Retrieval and ambiguity ---------------------------------------------------


def test_broad_present_tense_typed_query_abstains_on_multiple_current_candidates(
    tmp_path, monkeypatch
):
    store = prepared_store(tmp_path, monkeypatch, "broad-current")
    tea = State(
        user_id="alice",
        state_type="PREFERENCE",
        state_value="tea",
        source_memory_ids=["tea-memory"],
    )
    coffee = State(
        user_id="alice",
        state_type="PREFERENCE",
        state_value="coffee",
        source_memory_ids=["coffee-memory"],
    )
    store.save_state(tea)
    store.save_state(coffee)
    monkeypatch.setattr(
        hybrid_retriever,
        "search_memory",
        lambda _question, _user_id: [
            {"id": "tea-memory", "user_id": "alice", "memory": "I prefer tea."},
            {"id": "coffee-memory", "user_id": "alice", "memory": "I prefer coffee."},
        ],
    )

    result = hybrid_retriever.retrieve("What do I prefer?", "alice", "PREFERENCE")

    assert result["query_category"] == "STATE"
    assert result["ambiguity"]["kind"] == "multiple_state_candidates"
    assert set(result["ambiguity"]["candidate_state_ids"]) == {
        tea.state_id,
        coffee.state_id,
    }
    model = StructuredModel(
        claim(
            "I prefer coffee.",
            "fact",
            "memory",
            "coffee-memory",
        )["claims"]
    )
    monkeypatch.setattr(answer_generator, "llm", model)
    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "What do I prefer?", result
    )
    assert answer == answer_generator.ABSTENTION
    assert diagnostics["answer_reason"] == "ambiguous_evidence"
    assert model.calls == 0


def test_single_current_typed_state_excludes_unlinked_same_type_memory(
    tmp_path, monkeypatch
):
    store = prepared_store(tmp_path, monkeypatch, "single-current-memory")
    stale = State(
        user_id="alice",
        state_type="PREFERENCE",
        state_value="tea",
        status="INACTIVE",
        source_memory_ids=["tea-memory"],
    )
    current = State(
        user_id="alice",
        state_type="PREFERENCE",
        state_value="coffee",
        source_memory_ids=["coffee-memory"],
    )
    store.save_state(stale)
    store.save_state(current)
    monkeypatch.setattr(
        hybrid_retriever,
        "search_memory",
        lambda _question, _user_id: [
            {"id": "tea-memory", "user_id": "alice", "memory": "I prefer tea."},
            {"id": "coffee-memory", "user_id": "alice", "memory": "I prefer coffee."},
            {
                "id": "unlinked-stale-memory",
                "user_id": "alice",
                "memory": "I prefer tea over coffee.",
            },
        ],
    )

    result = hybrid_retriever.retrieve("What do I prefer?", "alice", "PREFERENCE")

    assert result["query_category"] == "STATE"
    assert result["state_focus"]["candidate_state_ids"] == [current.state_id]
    assert [record["id"] for record in result["memories"]] == ["coffee-memory"]
    assert "ambiguity" not in result


@pytest.mark.parametrize(
    "question",
    [
        "What did I prefer?",
        "Which preference existed previously?",
        "What was my preference previously?",
    ],
)
def test_historical_preference_questions_resolve_the_unique_predecessor(
    tmp_path, monkeypatch, question
):
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    boundary = start + timedelta(days=1)
    store = prepared_store(tmp_path, monkeypatch, "historical-preference")
    old = State(
        user_id="alice",
        state_type="PREFERENCE",
        state_value="tea",
        status="INACTIVE",
        valid_from=start,
        valid_until=boundary,
        source_memory_ids=["tea-memory"],
    )
    new = State(
        user_id="alice",
        state_type="PREFERENCE",
        state_value="coffee",
        status="ACTIVE",
        valid_from=boundary,
        source_memory_ids=["coffee-memory"],
    )
    store.save_state(old)
    store.save_state(new)
    edge = Transition(
        user_id="alice",
        from_state_id=old.state_id,
        to_state_id=new.state_id,
        relation_kind="TEMPORAL_SUCCESSION",
        timestamp=boundary,
        source_memory_ids=["coffee-memory"],
    )
    store.save_transition(edge)
    monkeypatch.setattr(
        hybrid_retriever,
        "search_memory",
        lambda _question, _user_id: [
            {"id": "tea-memory", "user_id": "alice", "memory": "I used to prefer tea."},
            {"id": "coffee-memory", "user_id": "alice", "memory": "I now prefer coffee."},
        ],
    )

    result = hybrid_retriever.retrieve(question, "alice", "PREFERENCE")

    assert result["query_category"] == "HISTORY"
    assert [state["state_id"] for state in result["states"]] == [old.state_id]
    assert [transition["transition_id"] for transition in result["transitions"]] == [
        edge.transition_id
    ]
    assert [record["id"] for record in result["memories"]] == ["tea-memory"]
    assert result.get("retrieval_reason") is None


# F. Citation and claim validation ---------------------------------------------


@pytest.mark.parametrize(
    ("source", "text"),
    [
        (
            "User now prefers tea over coffee.",
            "User now prefers coffee over tea.",
        ),
        (
            "User prefers tea rather than coffee.",
            "User prefers coffee rather than tea.",
        ),
        (
            "User likes tea more than coffee.",
            "User likes coffee more than tea.",
        ),
        (
            "User prefers coffee less than tea.",
            "User prefers coffee more than tea.",
        ),
        ("I used to prefer coffee.", "I prefer coffee."),
        ("I was a teacher.", "I am a teacher."),
        ("I have lived in Paris.", "I live in Paris."),
    ],
)
def test_reversed_comparison_or_historical_source_cannot_support_present_claim(
    monkeypatch, source, text
):
    model = StructuredModel(claim(text, "fact", "memory", "source-memory")["claims"])
    monkeypatch.setattr(answer_generator, "llm", model)

    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "What do I prefer?", memory_evidence(source)
    )

    assert answer == answer_generator.ABSTENTION
    assert diagnostics["answer_reason"] == "no_validated_claims"


@pytest.mark.parametrize(
    ("source", "text"),
    [
        (
            "User likes tea more than coffee.",
            "User prefers tea over coffee.",
        ),
        (
            "User prefers coffee less than tea.",
            "User prefers tea over coffee.",
        ),
        ("I used to prefer coffee.", "I used to prefer coffee."),
        ("I was a teacher.", "I was a teacher."),
        ("I have lived in Paris.", "I have lived in Paris."),
    ],
)
def test_correct_comparison_and_historical_paraphrase_are_supported(
    monkeypatch, source, text
):
    model = StructuredModel(claim(text, "fact", "memory", "source-memory")["claims"])
    monkeypatch.setattr(answer_generator, "llm", model)

    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "What do I prefer?", memory_evidence(source)
    )

    assert "memory:source-memory" in answer
    assert diagnostics["accepted_claim_count"] == 1


def _transition_evidence(relation="TEMPORAL_SUCCESSION", trigger=None):
    running = {
        "user_id": "alice",
        "state_id": "running-id",
        "state_type": "ACTIVITY",
        "state_value": "running",
        "status": "INACTIVE",
        "source_memory_ids": ["running-memory"],
    }
    swimming = {
        "user_id": "alice",
        "state_id": "swimming-id",
        "state_type": "ACTIVITY",
        "state_value": "swimming",
        "status": "ACTIVE",
        "source_memory_ids": ["switch-memory"],
    }
    edge = {
        "user_id": "alice",
        "transition_id": "edge-id",
        "from_state_id": "running-id",
        "to_state_id": "swimming-id",
        "transition_type": "ACTIVITY_CHANGE",
        "relation_kind": relation,
        "trigger": trigger,
        "reason": trigger,
        "source_memory_ids": ["switch-memory"],
    }
    return {
        "user_id": "alice",
        "query_category": "CAUSAL" if relation == "EXPLICIT_CAUSE" else "HISTORY",
        "memories": [],
        "states": [running, swimming],
        "transitions": [edge],
        "causal_path": [],
    }


def test_location_state_citations_validate_current_and_predecessor_evidence(
    tmp_path, monkeypatch
):
    memories = {
        "alice": [
            {"id": "chennai-memory", "user_id": "alice", "memory": "User lived in Chennai."},
            {
                "id": "move-memory",
                "user_id": "alice",
                "memory": "User moved from Chennai to Coimbatore.",
            },
        ]
    }
    store = prepared_store(
        tmp_path, monkeypatch, "location-citations", memories=memories
    )
    old, _ = state_ingest(store, "I live in Chennai.", memory_id="chennai-memory")
    new, _ = state_ingest(
        store, "I moved from Chennai to Coimbatore.", memory_id="move-memory"
    )

    current = hybrid_retriever.retrieve("Where do I live now?", "alice", "STATE")
    model = StructuredModel(
        claim("User lives in Coimbatore.", "fact", "state", new[0].state_id)["claims"]
    )
    monkeypatch.setattr(answer_generator, "llm", model)
    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "Where do I live now?", current
    )
    assert f"state:{new[0].state_id}" in answer
    assert diagnostics["accepted_claim_count"] == 1

    previous = hybrid_retriever.retrieve("Where did I live before?", "alice", "HISTORY")
    model = StructuredModel(
        claim("User lived in Chennai.", "fact", "state", old[0].state_id)["claims"]
    )
    monkeypatch.setattr(answer_generator, "llm", model)
    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "Where did I live before?", previous
    )
    assert f"state:{old[0].state_id}" in answer
    assert diagnostics["accepted_claim_count"] == 1


def test_location_change_transition_citation_and_predecessor_bundle(
    tmp_path, monkeypatch
):
    memories = {
        "alice": [
            {"id": "chennai-memory", "user_id": "alice", "memory": "User lived in Chennai."},
            {
                "id": "move-memory",
                "user_id": "alice",
                "memory": "User moved from Chennai to Coimbatore.",
            },
        ]
    }
    store = prepared_store(
        tmp_path, monkeypatch, "location-change-citation", memories=memories
    )
    old, _ = state_ingest(store, "I live in Chennai.", memory_id="chennai-memory")
    new, edges = state_ingest(
        store, "I moved from Chennai to Coimbatore.", memory_id="move-memory"
    )

    previous = hybrid_retriever.retrieve("Where did I live before?", "alice", "HISTORY")
    model = StructuredModel(
        claim(
            "User previously lived in Chennai.",
            "temporal",
            "state",
            old[0].state_id,
        )["claims"]
    )
    monkeypatch.setattr(answer_generator, "llm", model)
    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "Where did I live before?", previous
    )
    assert f"state:{old[0].state_id}" in answer
    assert diagnostics["accepted_claim_count"] == 1

    change = hybrid_retriever.retrieve(
        "What location change did I tell you about?", "alice", "HISTORY"
    )
    model = StructuredModel(
        claim(
            "User moved from Chennai to Coimbatore.",
            "temporal",
            "transition",
            edges[0].transition_id,
        )["claims"]
    )
    monkeypatch.setattr(answer_generator, "llm", model)
    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "What location change did I tell you about?", change
    )
    assert f"transition:{edges[0].transition_id}" in answer
    assert diagnostics["accepted_claim_count"] == 1


def test_predecessor_state_claim_accepts_inflected_live_verb(
    tmp_path, monkeypatch
):
    memories = {
        "alice": [
            {"id": "chennai-memory", "user_id": "alice", "memory": "User lives in Chennai."},
            {
                "id": "move-memory",
                "user_id": "alice",
                "memory": "User moved from Chennai to Coimbatore.",
            },
        ]
    }
    store = prepared_store(
        tmp_path, monkeypatch, "predecessor-live-inflection", memories=memories
    )
    old, _ = state_ingest(store, "I live in Chennai.", memory_id="chennai-memory")
    state_ingest(
        store, "I moved from Chennai to Coimbatore.", memory_id="move-memory"
    )

    previous = hybrid_retriever.retrieve("Where did I live before?", "alice", "HISTORY")
    model = StructuredModel(
        claim("I lived in Chennai before.", "temporal", "state", old[0].state_id)["claims"]
    )
    monkeypatch.setattr(answer_generator, "llm", model)
    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "Where did I live before?", previous
    )

    assert f"state:{old[0].state_id}" in answer
    assert diagnostics["accepted_claim_count"] == 1
    assert diagnostics["temporal_validation"][0]["lexical_support_passed"] is True


def test_predecessor_claim_accepts_historical_paraphrase_with_verified_transition(
    tmp_path, monkeypatch
):
    store = prepared_store(tmp_path, monkeypatch, "predecessor-used-to")
    old, _ = state_ingest(store, "I live in Chennai.", memory_id="chennai-memory")
    state_ingest(
        store, "I moved from Chennai to Coimbatore.", memory_id="move-memory"
    )
    previous = hybrid_retriever.retrieve(
        "Where did I live before?", "alice", "HISTORY"
    )
    model = StructuredModel(
        claim("I used to live in Chennai", "temporal", "state", old[0].state_id)["claims"]
    )
    monkeypatch.setattr(answer_generator, "llm", model)

    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "Where did I live before?", previous
    )

    assert f"state:{old[0].state_id}" in answer
    assert diagnostics["accepted_claim_count"] == 1


def test_goal_and_completion_citations_validate_typed_lifecycle_evidence(
    tmp_path, monkeypatch
):
    memories = {
        "alice": [
            {
                "id": "goal-memory",
                "user_id": "alice",
                "memory": "User's current goal is to complete my machine learning project.",
            },
            {
                "id": "completion-memory",
                "user_id": "alice",
                "memory": (
                    "User completed my machine learning project, and now wants "
                    "to learn system design."
                ),
            },
        ]
    }
    store = prepared_store(
        tmp_path, monkeypatch, "goal-citations", memories=memories
    )
    old, _ = state_ingest(
        store,
        "My current goal is to complete my machine learning project.",
        memory_id="goal-memory",
    )
    created, _ = state_ingest(
        store,
        "I completed my machine learning project, and my new goal is to learn system design.",
        memory_id="completion-memory",
    )
    new_goal = next(state for state in created if state.state_type == "GOAL")
    completion = next(state for state in created if state.state_type == "LIFE_EVENT")

    current = hybrid_retriever.retrieve("What is my current goal?", "alice", "STATE")
    model = StructuredModel(
        claim(
            "User's current goal is to learn system design.",
            "fact",
            "state",
            new_goal.state_id,
        )["claims"]
    )
    monkeypatch.setattr(answer_generator, "llm", model)
    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "What is my current goal?", current
    )
    assert f"state:{new_goal.state_id}" in answer
    assert diagnostics["accepted_claim_count"] == 1

    completed = hybrid_retriever.retrieve("What did I complete?", "alice", "HISTORY")
    model = StructuredModel(
        claim(
            "User completed their machine learning project.",
            "fact",
            "state",
            completion.state_id,
        )["claims"]
    )
    monkeypatch.setattr(answer_generator, "llm", model)
    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "What did I complete?", completed
    )
    assert f"state:{completion.state_id}" in answer
    assert diagnostics["accepted_claim_count"] == 1

    previous = hybrid_retriever.retrieve("What was my previous goal?", "alice", "HISTORY")
    model = StructuredModel(
        claim(
            "User's previous goal was to complete their machine learning project.",
            "fact",
            "state",
            old[0].state_id,
        )["claims"]
    )
    monkeypatch.setattr(answer_generator, "llm", model)
    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "What was my previous goal?", previous
    )
    assert f"state:{old[0].state_id}" in answer
    assert diagnostics["accepted_claim_count"] == 1


@pytest.mark.parametrize(
    ("text", "accepted"),
    [
        ("I switched from running to swimming.", True),
        ("I switched from swimming to running.", False),
        ("I switched to running.", False),
        ("I switched to swimming.", True),
        ("I switched from running to running.", False),
    ],
)
def test_transition_citation_enforces_exact_direction(monkeypatch, text, accepted):
    model = StructuredModel(claim(text, "temporal", "transition", "edge-id")["claims"])
    monkeypatch.setattr(answer_generator, "llm", model)

    answer = answer_generator.generate_answer(
        "What change was recorded from running to swimming?",
        _transition_evidence(),
    )

    if accepted:
        assert "transition:edge-id" in answer
    else:
        assert answer == answer_generator.ABSTENTION


@pytest.mark.parametrize(
    "question",
    [
        "Why did I switch from running to swimming?",
        "Did I switch because of a knee injury or is that unknown?",
        "Was it because of a knee injury?",
        "What caused the change from running to swimming?",
        "Did running change because of a knee injury?",
    ],
)
def test_causal_paraphrases_retrieve_the_explicit_cause_edge(
    tmp_path, monkeypatch, question
):
    extractor = TableExtractor(
        {
            "Due to a knee injury I switched from running to swimming.": [
                {
                    "state_type": "ACTIVITY",
                    "state_value": "swimming",
                    "confidence": 0.9,
                }
            ]
        }
    )
    store = prepared_store(tmp_path, monkeypatch, "causal-paraphrases", extractor)
    source = State(
        user_id="alice",
        state_type="ACTIVITY",
        state_value="running",
        source_memory_ids=["old-memory"],
    )
    store.save_state(source)
    _, edges = state_ingest(
        store,
        "Due to a knee injury I switched from running to swimming.",
        memory_id="switch-memory",
    )

    evidence = hybrid_retriever.retrieve(question, "alice", "CAUSAL")

    assert [edge["transition_id"] for edge in evidence["transitions"]] == [
        edges[0].transition_id
    ]
    assert evidence["transitions"][0]["relation_kind"] == "EXPLICIT_CAUSE"


@pytest.mark.parametrize(
    "question",
    [
        "Did I switch because of a broken ankle?",
        "Was it because of bad weather?",
    ],
)
def test_unsupported_cause_paraphrases_retrieve_no_causal_edge(
    tmp_path, monkeypatch, question
):
    extractor = TableExtractor(
        {
            "Due to a knee injury I switched from running to swimming.": [
                {
                    "state_type": "ACTIVITY",
                    "state_value": "swimming",
                    "confidence": 0.9,
                }
            ]
        }
    )
    store = prepared_store(tmp_path, monkeypatch, "unsupported-cause", extractor)
    store.save_state(
        State(
            user_id="alice",
            state_type="ACTIVITY",
            state_value="running",
            source_memory_ids=["old-memory"],
        )
    )
    state_ingest(
        store,
        "Due to a knee injury I switched from running to swimming.",
        memory_id="switch-memory",
    )

    evidence = hybrid_retriever.retrieve(question, "alice", "CAUSAL")

    assert evidence["transitions"] == []
    model = StructuredModel([])
    monkeypatch.setattr(answer_generator, "llm", model)
    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        question, evidence
    )
    assert answer == answer_generator.ABSTENTION
    assert diagnostics["answer_reason"] == "missing_explicit_cause_transition"
    assert model.calls == 0


def test_causal_claim_with_recorded_trigger_is_accepted(monkeypatch):
    evidence = _transition_evidence("EXPLICIT_CAUSE", "knee injury")
    model = StructuredModel(
        claim(
            "A knee injury made me switch from running to swimming.",
            "causal",
            "transition",
            "edge-id",
        )["claims"]
    )
    monkeypatch.setattr(answer_generator, "llm", model)

    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "Did I switch because of a knee injury?", evidence
    )

    assert "transition:edge-id" in answer
    assert diagnostics["causal_gate_passed"] is True
    assert diagnostics["causal_transition_citation_accepted"] is True


def test_causal_claim_with_unrecorded_trigger_abstains(monkeypatch):
    evidence = _transition_evidence("EXPLICIT_CAUSE", "knee injury")
    model = StructuredModel(
        claim(
            "A broken ankle made me switch from running to swimming.",
            "causal",
            "transition",
            "edge-id",
        )["claims"]
    )
    monkeypatch.setattr(answer_generator, "llm", model)

    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "Did I switch because of a broken ankle?", evidence
    )

    assert answer == answer_generator.ABSTENTION
    assert diagnostics["causal_gate_passed"] is False
    assert diagnostics["answer_reason"] == "explicit_cause_gate_rejected"


def test_causal_transition_citation_rejects_reversed_endpoint_direction(monkeypatch):
    evidence = _transition_evidence("EXPLICIT_CAUSE", "knee injury")
    model = StructuredModel(
        claim(
            "A knee injury made me switch from swimming to running.",
            "causal",
            "transition",
            "edge-id",
        )["claims"]
    )
    monkeypatch.setattr(answer_generator, "llm", model)

    answer = answer_generator.generate_answer(
        "Why did running change to swimming?", evidence
    )

    assert answer == answer_generator.ABSTENTION


@pytest.mark.parametrize(
    ("source", "text"),
    [
        ("I don't prefer coffee.", "I prefer coffee."),
        ("I prefer coffee.", "I don't prefer coffee."),
        ("I no longer prefer coffee.", "I prefer coffee."),
    ],
)
def test_citation_rejects_negated_positive_mismatch(monkeypatch, source, text):
    model = StructuredModel(claim(text, "fact", "memory", "source-memory")["claims"])
    monkeypatch.setattr(answer_generator, "llm", model)

    answer = answer_generator.generate_answer(
        "What do I prefer?", memory_evidence(source)
    )

    assert answer == answer_generator.ABSTENTION


def test_multiple_citations_must_each_support_the_complete_claim(monkeypatch):
    evidence = {
        "user_id": "alice",
        "query_category": "PREFERENCE",
        "memories": [
            {"id": "coffee-memory", "user_id": "alice", "memory": "I prefer coffee."},
            {
                "id": "backend-memory",
                "user_id": "alice",
                "memory": "I work on backend services.",
            },
        ],
        "states": [],
        "transitions": [],
        "causal_path": [],
    }
    model = StructuredModel(
        {
            "claims": [
                {
                    "text": "I prefer coffee for backend services.",
                    "claim_type": "fact",
                    "evidence": [
                        {"kind": "memory", "id": "coffee-memory"},
                        {"kind": "memory", "id": "backend-memory"},
                    ],
                }
            ]
        }
    )
    monkeypatch.setattr(answer_generator, "llm", model)

    answer = answer_generator.generate_answer("What do I prefer?", evidence)

    assert answer == answer_generator.ABSTENTION


def test_mixed_valid_and_invalid_citations_render_only_valid_evidence(monkeypatch):
    evidence = memory_evidence("I prefer coffee.")
    evidence["memories"].append(
        {"id": "second-memory", "user_id": "alice", "memory": "Coffee is warm."}
    )
    model = StructuredModel(
        [
            {
                "text": "I prefer coffee.",
                "claim_type": "fact",
                "evidence": [
                    {"kind": "memory", "id": "source-memory"},
                    {"kind": "memory", "id": "unknown-memory"},
                ],
            }
        ]
    )
    monkeypatch.setattr(answer_generator, "llm", model)

    answer, diagnostics = answer_generator.generate_answer_with_diagnostics(
        "What do I prefer?", evidence
    )

    assert answer == "I prefer coffee. [Sources: memory:source-memory]"
    assert "unknown-memory" not in answer
    assert diagnostics["rejection_categories"] == {"unknown_or_unowned_evidence_id": 1}


# G. Multi-user isolation and ownership ----------------------------------------


@pytest.mark.parametrize(
    ("ref_kind", "ref_id"),
    [("memory", "bob-memory"), ("state", "bob-state"), ("transition", "bob-edge")],
)
def test_foreign_evidence_kinds_never_become_citations(monkeypatch, ref_kind, ref_id):
    evidence = {
        "user_id": "alice",
        "query_category": "HISTORY",
        "memories": [
            {"id": "bob-memory", "user_id": "bob", "memory": "I prefer tea."}
        ],
        "states": [
            {
                "user_id": "bob",
                "state_id": "bob-state",
                "state_type": "PREFERENCE",
                "state_value": "tea",
                "source_memory_ids": ["bob-memory"],
            }
        ],
        "transitions": [
            {
                "user_id": "bob",
                "transition_id": "bob-edge",
                "from_state_id": "bob-state",
                "to_state_id": "bob-state",
                "relation_kind": "TEMPORAL_SUCCESSION",
            }
        ],
        "causal_path": [],
    }
    kind = "causal" if ref_kind == "transition" else "fact"
    model = StructuredModel(claim("I prefer tea.", kind, ref_kind, ref_id)["claims"])
    monkeypatch.setattr(answer_generator, "llm", model)

    answer = answer_generator.generate_answer("What do I prefer?", evidence)

    assert answer == answer_generator.ABSTENTION


# H. Ingestion and idempotency -------------------------------------------------


def test_reingest_does_not_change_existing_state_provenance(tmp_path, monkeypatch):
    store = prepared_store(tmp_path, monkeypatch, "provenance-idempotency")
    first, edges = state_ingest(
        store, "I prefer tea over coffee.", memory_id="first-memory"
    )
    before = snapshot(store, "alice")
    original_state = first[0]

    repeated, repeated_edges = state_ingest(
        store, "I prefer tea over coffee.", memory_id="second-memory"
    )

    assert edges == [] and repeated == [] and repeated_edges == []
    assert snapshot(store, "alice") == before
    persisted = store.states("alice")[0]
    assert persisted.state_id == original_state.state_id
    assert persisted.source_memory_ids == ["first-memory"]
    assert persisted.valid_from == original_state.valid_from
    assert persisted.metadata == original_state.metadata


# I. API and frontend contract -------------------------------------------------


def test_process_chat_routes_ingest_and_question_without_cross_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        chat_service,
        "classify_query",
        lambda _message: QueryAnalysis(
            is_question=False, category="OTHER", rationale="test"
        ),
    )
    monkeypatch.setattr(
        chat_service,
        "add_memory",
        lambda *_args, **_kwargs: calls.append("mem0") or {"results": []},
    )
    monkeypatch.setattr(chat_service, "get_memories", lambda _user: [])
    monkeypatch.setattr(
        chat_service, "memory_ids_from_add_response", lambda _result: ["memory-id"]
    )
    monkeypatch.setattr(
        chat_service,
        "ingest_message",
        lambda *_args, **_kwargs: calls.append("structured") or ([], []),
    )
    monkeypatch.setattr(
        chat_service,
        "retrieve",
        lambda *_args, **_kwargs: calls.append("retrieve") or {},
    )
    monkeypatch.setattr(
        chat_service,
        "generate_answer_with_diagnostics",
        lambda *_args, **_kwargs: calls.append("answer") or ("", {}),
    )

    ingest = chat_service.process_chat("I prefer coffee.", "alice")

    assert ingest["response_type"] == "ingest"
    assert calls == ["mem0", "structured"]
    assert "retrieve" not in calls and "answer" not in calls

    calls.clear()
    monkeypatch.setattr(
        chat_service,
        "classify_query",
        lambda _message: QueryAnalysis(
            is_question=True, category="PREFERENCE", rationale="test"
        ),
    )
    monkeypatch.setattr(
        chat_service,
        "safe_answer_diagnostic_summary",
        lambda _diagnostics: {"answer_reason": "no_validated_claims"},
    )

    answer = chat_service.process_chat("What do I prefer?", "alice")

    assert answer["response_type"] == "answer"
    assert calls == ["retrieve", "answer"]
    assert "mem0" not in calls and "structured" not in calls


def test_process_chat_surfaces_mem0_and_state_write_failures(monkeypatch):
    monkeypatch.setattr(
        chat_service, "classify_query",
        lambda _message: QueryAnalysis(category="PREFERENCE", is_question=False),
    )
    monkeypatch.setattr(
        chat_service, "can_extract_deterministically", lambda _message: True
    )

    def fail_mem0(*_args, **_kwargs):
        raise RuntimeError("private provider detail")

    monkeypatch.setattr(chat_service, "add_memory", fail_mem0)
    failed_mem0 = chat_service.process_chat("I prefer tea.", "alice")
    assert failed_mem0["ingest_status"] == "mem0_write_failed"
    assert failed_mem0["storage_errors"] == ["mem0_write_failed"]
    assert failed_mem0["storage_warnings"] == []

    monkeypatch.setattr(
        chat_service, "add_memory",
        lambda *_args, **_kwargs: {"results": [{"id": "memory-1"}]},
    )
    monkeypatch.setattr(
        chat_service, "ingest_message",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("private sqlite detail")
        ),
    )
    failed_state = chat_service.process_chat("I prefer tea.", "alice")
    assert failed_state["ingest_status"] == "state_write_failed"
    assert failed_state["storage_errors"] == ["state_write_failed"]
    assert failed_state["memory_ids"] == ["memory-1"]
    assert "private" not in str(failed_state)


def test_process_chat_reports_ingestion_timings_and_skips_duplicate_extractor(
    monkeypatch,
):
    calls = []
    ingest_kwargs = []
    add_kwargs = []

    def fake_add(*_args, diagnostics_callback=None, **kwargs):
        calls.append("mem0")
        add_kwargs.append(kwargs)
        if diagnostics_callback is not None:
            diagnostics_callback(
                {
                    "stage": "total_mem0_add",
                    "duration_seconds": 0.01,
                    "success": True,
                }
            )
            diagnostics_callback(
                {
                    "stage": "llm_memory_extraction",
                    "duration_seconds": 0.005,
                    "success": True,
                }
            )
        return {"results": []}

    def fake_ingest(*_args, **kwargs):
        calls.append("structured")
        ingest_kwargs.append(kwargs)
        return [], []

    monkeypatch.setattr(
        chat_service,
        "classify_query",
        lambda _message: QueryAnalysis(
            is_question=False, category="OTHER", rationale="test"
        ),
    )
    monkeypatch.setattr(chat_service, "add_memory", fake_add)
    monkeypatch.setattr(
        chat_service,
        "get_memories",
        lambda _user: calls.append("memory_read") or [],
    )
    monkeypatch.setattr(
        chat_service, "memory_ids_from_add_response", lambda _result: ["memory-id"]
    )
    monkeypatch.setattr(chat_service, "ingest_message", fake_ingest)

    deterministic = chat_service.process_chat("I prefer tea over coffee.", "alice")

    assert calls == ["mem0", "structured"]
    assert add_kwargs[-1]["infer"] is False
    assert deterministic["request_id"].startswith("req_")
    assert len(deterministic["request_id"]) == 36
    assert ingest_kwargs[-1]["allow_llm"] is False
    ingestion = deterministic["timings"]["ingestion"]
    assert ingestion["extractor_skipped"] is True
    assert ingestion["add_memory_ms"] >= 0
    assert ingestion["memory_read_ms"] >= 0
    assert ingestion["state_extraction_ms"] >= 0
    assert {stage["stage"] for stage in ingestion["mem0_stages"]} == {
        "total_mem0_add",
        "llm_memory_extraction",
    }
    assert all("duration_ms" in stage for stage in ingestion["mem0_stages"])

    calls.clear()
    ingest_kwargs.clear()
    non_deterministic = chat_service.process_chat("I love running.", "alice")

    assert calls == ["mem0", "memory_read", "structured"]
    assert add_kwargs[-1]["infer"] is True
    assert ingest_kwargs[-1]["allow_llm"] is True
    assert non_deterministic["timings"]["ingestion"]["extractor_skipped"] is False


def test_process_chat_skips_classifier_for_deterministic_statement(monkeypatch):
    monkeypatch.setattr(
        chat_service, "can_extract_deterministically", lambda _message: True
    )
    monkeypatch.setattr(
        chat_service,
        "classify_query",
        lambda _message: (_ for _ in ()).throw(
            AssertionError("deterministic statements must not call the classifier")
        ),
    )
    monkeypatch.setattr(
        chat_service, "add_memory", lambda *_args, **_kwargs: {"results": []}
    )
    monkeypatch.setattr(chat_service, "ingest_message", lambda *_args, **_kwargs: ([], []))

    response = chat_service.process_chat("I live in Chennai.", "alice")

    assert response["response_type"] == "ingest"
    assert response["query_analysis"]["is_question"] is False
    assert response["query_analysis"]["category"] == "OTHER"
    assert response["timings"]["classification_ms"] >= 0


def test_process_chat_answer_reports_request_id_and_timing_breakdown(monkeypatch):
    evidence = {
        "user_id": "alice",
        "query_category": "STATE",
        "memories": [],
        "states": [],
        "transitions": [],
        "causal_path": [],
        "retrieval_sources": ["Timeline"],
    }

    monkeypatch.setattr(
        chat_service,
        "classify_query",
        lambda _message: QueryAnalysis(
            is_question=True, category="STATE", rationale="test"
        ),
    )

    def fake_retrieve(_message, _user_id, _category, _timestamp, timings=None):
        if timings is not None:
            timings.update({
                "mem0_search_ms": 1.0,
                "timeline_db_ms": 2.0,
                "graph_ms": 3.0,
                "total_ms": 6.0,
            })
        return evidence

    monkeypatch.setattr(chat_service, "retrieve", fake_retrieve)
    monkeypatch.setattr(
        chat_service,
        "generate_answer_with_diagnostics",
        lambda _message, _evidence: (
            "answer",
            {
                "answer_reason": "validated_claims",
                "timings": {
                    "model_ms": 4.0,
                    "validation_ms": 1.0,
                    "render_ms": 0.5,
                    "total_ms": 5.5,
                },
            },
        ),
    )

    response = chat_service.process_chat("What do I prefer?", "alice")

    assert response["request_id"].startswith("req_")
    assert len(response["request_id"]) == 36
    assert response["timings"]["retrieval"] == {
        "mem0_search_ms": 1.0,
        "timeline_db_ms": 2.0,
        "graph_ms": 3.0,
        "total_ms": 6.0,
    }
    assert response["timings"]["generation"]["model_ms"] == 4.0
    assert response["timings"]["generation"]["validation_ms"] == 1.0
    assert response["timings"]["generation"]["render_ms"] == 0.5


def test_process_chat_reports_no_new_state_and_not_ingestable_distinctly(monkeypatch):
    monkeypatch.setattr(
        chat_service,
        "classify_query",
        lambda _message: QueryAnalysis(
            is_question=False, category="OTHER", rationale="test"
        ),
    )
    monkeypatch.setattr(chat_service, "add_memory", lambda *_args, **_kwargs: {"results": []})
    monkeypatch.setattr(chat_service, "get_memories", lambda _user: [])
    monkeypatch.setattr(
        chat_service, "memory_ids_from_add_response", lambda _result: ["memory-id"]
    )
    monkeypatch.setattr(
        chat_service, "ingest_message", lambda *_args, **_kwargs: ([], [])
    )

    no_state = chat_service.process_chat("I prefer coffee.", "alice")

    assert no_state["ingest_status"] == "no_new_state"
    assert "No new structured state was added" in no_state["answer"]

    writes = []
    monkeypatch.setattr(
        chat_service,
        "add_memory",
        lambda *_args, **_kwargs: writes.append("mem0") or {"results": []},
    )
    not_ingestable = chat_service.process_chat("Swimming after the injury", "alice")

    assert not_ingestable["ingest_status"] == "not_ingestable"
    assert writes == []


def test_process_chat_never_renders_a_rejected_claim(monkeypatch):
    monkeypatch.setattr(
        chat_service,
        "classify_query",
        lambda _message: QueryAnalysis(
            is_question=True, category="PREFERENCE", rationale="test"
        ),
    )
    evidence = memory_evidence("User now prefers tea over coffee.")
    monkeypatch.setattr(chat_service, "retrieve", lambda *_args, **_kwargs: evidence)
    model = StructuredModel(
        claim(
            "User now prefers coffee over tea.",
            "fact",
            "memory",
            "source-memory",
        )["claims"]
    )
    monkeypatch.setattr(answer_generator, "llm", model)

    result = chat_service.process_chat("What do I prefer?", "alice")

    assert result["response_type"] == "answer"
    assert result["answer"] == answer_generator.ABSTENTION
    assert result["citations"] == []
    assert "coffee over tea" not in result["answer"]
