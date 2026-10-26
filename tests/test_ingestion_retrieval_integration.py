"""Isolated ingestion-to-retrieval integration tests.

SQLite is real and temporary. Extraction, Mem0 search, and answer generation are
deterministic test doubles, so these tests never contact Ollama or Qdrant.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from models import Extraction, State


def _isolated_pipeline(tmp_path, monkeypatch, extractor, memories=None):
    # Set storage environment before importing config/database in standalone
    # runs, since the application DB singleton is initialized on import.
    root = tmp_path.resolve()
    db_path = root / "sqlite" / "causalmem.sqlite3"
    qdrant_path = root / "mem0-qdrant"
    mem0_config_path = root / "mem0-config"
    db_path.parent.mkdir(parents=True)
    qdrant_path.mkdir()
    mem0_config_path.mkdir()
    monkeypatch.setenv("CAUSALMEM_DB", str(db_path))
    monkeypatch.setenv("MEM0_PATH", str(qdrant_path))
    monkeypatch.setenv("MEM0_DIR", str(mem0_config_path))

    from database import Database
    import causal_graph
    import hybrid_retriever
    import state_extractor
    import temporal_engine

    store = Database(str(db_path))
    for module in (state_extractor, hybrid_retriever, temporal_engine, causal_graph):
        monkeypatch.setattr(module, "db", store)
    monkeypatch.setattr(state_extractor, "llm", extractor)
    requested_users = []
    records_by_user = memories or {}

    def search_memory(query, user_id):
        requested_users.append(user_id)
        return list(records_by_user.get(user_id, []))

    monkeypatch.setattr(hybrid_retriever, "search_memory", search_memory)
    modules = SimpleNamespace(state=state_extractor, hybrid=hybrid_retriever,
                              temporal=temporal_engine, graph=causal_graph)
    return store, f"phase6e-{uuid4().hex}", requested_users, db_path, qdrant_path, modules


class GoalReplacementExtractor:
    def extract(self, message, *_args):
        if "guitar" in message.casefold():
            return Extraction(states=[{"state_type": "GOAL", "state_value": "learn guitar", "confidence": .9}])
        return Extraction(
            # Model output intentionally links the previous goal to the new one.
            states=[{"state_type": "GOAL", "state_value": "learn piano", "confidence": .9}],
            transitions=[{"from_state_value": "learn guitar", "to_state_value": "learn piano",
                          "relation_kind": "UNKNOWN", "trigger": "not user supplied"}],
        )


def test_conversation_goal_replacement_retrieves_before_boundary_and_after(tmp_path, monkeypatch):
    store, user_id, _, db_path, qdrant_path, modules = _isolated_pipeline(
        tmp_path, monkeypatch, GoalReplacementExtractor())
    old_id = f"{user_id}:old-goal"
    new_id = f"{user_id}:new-goal"

    modules.state.ingest_message("My goal is to learn guitar.", user_id, [old_id], [])
    created, transitions = modules.state.ingest_message(
        "I want to learn piano instead.", user_id, [new_id], [])

    assert db_path.is_relative_to(tmp_path.resolve())
    assert qdrant_path.is_relative_to(tmp_path.resolve())
    assert len(created) == len(transitions) == 1
    edge = transitions[0]
    history = {state.state_value: state for state in store.states(user_id)}
    old, new = history["learn guitar"], history["learn piano"]
    assert edge.relation_kind == "TEMPORAL_SUCCESSION"
    assert edge.from_state_id == old.state_id and edge.to_state_id == new.state_id
    assert edge.trigger is None and edge.reason is None
    assert old.status == "INACTIVE" and old.valid_until == edge.timestamp
    assert new.status == "ACTIVE" and new.valid_from == edge.timestamp
    assert old.source_memory_ids == [old_id]
    assert new.source_memory_ids == [new_id]
    assert edge.source_memory_ids == [new_id]

    # Keep this comfortably outside SQLite julianday floating-point resolution;
    # exact-boundary inclusion is checked separately at the transition instant.
    before = old.valid_from + (edge.timestamp - old.valid_from) / 2
    at_boundary = edge.timestamp
    after = edge.timestamp + timedelta(seconds=1)
    for instant, expected in ((before, "learn guitar"), (at_boundary, "learn piano"),
                              (after, "learn piano")):
        result = modules.hybrid.retrieve(
            "What was my goal?", user_id, "HISTORY", requested_timestamp=instant)
        assert [state["state_value"] for state in result["states"]] == [expected]
        assert all(state["user_id"] == user_id for state in result["states"])


def test_clear_instead_goal_replacement_does_not_depend_on_model_transition_output(tmp_path,monkeypatch):
    class SmallModelReplacementExtractor:
        def extract(self,message,*_args):
            if "guitar" in message.casefold():
                return Extraction(states=[{"state_type":"GOAL","state_value":"learn guitar","confidence":.9}])
            # Reproduces the live model behavior: it extracts a grounded noun
            # with the wrong type and omits the transition altogether.
            return Extraction(states=[{"state_type":"INTEREST","state_value":"piano","confidence":.8}])

    store,user_id,_,_,_,modules=_isolated_pipeline(
        tmp_path,monkeypatch,SmallModelReplacementExtractor())
    old_memory=f"{user_id}:goal-guitar"
    new_memory=f"{user_id}:goal-piano"
    modules.state.ingest_message("My goal is to learn guitar.",user_id,[old_memory],[])
    created,edges=modules.state.ingest_message(
        "I want to learn piano instead.",user_id,[new_memory],[])

    assert len(created)==len(edges)==1
    assert created[0].state_type=="GOAL" and created[0].state_value=="learn piano"
    assert created[0].source_memory_ids==[new_memory]
    assert edges[0].relation_kind=="TEMPORAL_SUCCESSION"
    assert edges[0].trigger is None and edges[0].reason is None
    assert edges[0].from_state_id!=edges[0].to_state_id
    states={state.state_value:state for state in store.states(user_id)}
    old,new=states["learn guitar"],states["learn piano"]
    assert old.status=="INACTIVE" and old.valid_until==edges[0].timestamp
    assert old.source_memory_ids==[old_memory]
    assert new.status=="ACTIVE" and new.valid_from==edges[0].timestamp
    assert new.source_memory_ids==[new_memory]
    assert store.transitions(user_id)==edges
    for instant,expected in ((edges[0].timestamp-timedelta(microseconds=1),"learn guitar"),
                             (edges[0].timestamp,"learn piano"),
                             (edges[0].timestamp+timedelta(microseconds=1),"learn piano")):
        result=modules.hybrid.retrieve("What was my goal?",user_id,"HISTORY",instant)
        assert [state["state_value"] for state in result["states"]]==[expected]


def test_goal_replacement_keeps_source_interval_when_clock_timestamps_collide(tmp_path, monkeypatch):
    store, user_id, _, _, _, modules = _isolated_pipeline(
        tmp_path, monkeypatch, GoalReplacementExtractor())
    fixed_time=datetime(2026,10,23,12,0,tzinfo=timezone.utc)
    original_state_model=modules.state.State
    monkeypatch.setattr(modules.state,"State",
                        lambda **kwargs: original_state_model(valid_from=fixed_time,**kwargs))
    modules.state.ingest_message("My goal is to learn guitar.",user_id,[f"{user_id}:guitar"],[])
    created,edges=modules.state.ingest_message(
        "I want to learn piano instead.",user_id,[f"{user_id}:piano"],[])

    assert len(created)==len(edges)==1
    states={state.state_value:state for state in store.states(user_id)}
    source,destination=states["learn guitar"],states["learn piano"]
    edge=edges[0]
    assert source.valid_from==fixed_time
    assert source.valid_until==edge.timestamp==destination.valid_from
    assert destination.valid_from==fixed_time+timedelta(microseconds=1)
    before=modules.hybrid.retrieve("What was my goal?",user_id,"HISTORY",fixed_time)
    at_boundary=modules.hybrid.retrieve("What was my goal?",user_id,"HISTORY",edge.timestamp)
    assert [state["state_value"] for state in before["states"]]==["learn guitar"]
    assert [state["state_value"] for state in at_boundary["states"]]==["learn piano"]


class CauseExtractor:
    def extract(self, message, *_args):
        if "enjoy running" in message.casefold():
            return Extraction(states=[{"state_type": "ACTIVITY", "state_value": "running", "confidence": .9}])
        return Extraction(
            states=[{"state_type": "ACTIVITY", "state_value": "swimming", "confidence": .9}],
            transitions=[{"from_state_value": "running", "to_state_value": "swimming",
                          "relation_kind": "UNKNOWN", "trigger": "unsupported model cause"}],
        )


def test_conversation_explicit_cause_reaches_retrieval_and_causal_answer_gate(tmp_path, monkeypatch):
    user_id = f"phase6e-{uuid4().hex}"
    store, user_id, _, _, _, modules = _isolated_pipeline(tmp_path, monkeypatch, CauseExtractor())
    source_memory = f"{user_id}:running-interest"
    cause_memory = f"{user_id}:switch-cause"
    modules.state.ingest_message("I enjoy running.", user_id, [source_memory], [])
    _, transitions = modules.state.ingest_message(
        "I switched from running to swimming because a knee injury made running painful.",
        user_id, [cause_memory], [])

    edge = transitions[0]
    states_by_value = {state.state_value: state for state in store.states(user_id)}
    assert edge.relation_kind == "EXPLICIT_CAUSE"
    assert edge.trigger == "a knee injury made running painful"
    assert edge.reason == "I switched from running to swimming because a knee injury made running painful."
    assert edge.from_state_id == states_by_value["running"].state_id
    assert edge.to_state_id == states_by_value["swimming"].state_id
    assert states_by_value["running"].status == "INACTIVE"
    assert states_by_value["running"].source_memory_ids == [source_memory]
    assert states_by_value["swimming"].status == "ACTIVE"
    assert states_by_value["swimming"].source_memory_ids == [cause_memory]
    assert edge.source_memory_ids == [cause_memory]

    result = modules.hybrid.retrieve(
        "Why did I switch from running to swimming?", user_id, "CAUSAL",
        requested_timestamp=edge.timestamp)
    assert result["transitions"] == [edge.model_dump(mode="json")]
    assert result["causal_path"][0]["from"] == "running"
    assert result["causal_path"][0]["to"] == "swimming"
    assert result["causal_path"][0]["source_memory_ids"] == [cause_memory]
    assert [state["state_value"] for state in result["states"]] == ["swimming"]

    class DeterministicAnswerer:
        def answer_with_provenance(self, question, evidence):
            assert question.startswith("Why")
            assert evidence["transitions"][0]["relation_kind"] == "EXPLICIT_CAUSE"
            edge=evidence["transitions"][0]
            return {"claims":[{"text":edge["trigger"],"claim_type":"causal",
                "evidence":[{"kind":"transition","id":edge["transition_id"]}]}]}

    import answer_generator
    monkeypatch.setattr(answer_generator, "llm", DeterministicAnswerer())
    answer = answer_generator.generate_answer(
        "Why did I switch from running to swimming?", result)
    assert "knee injury" in answer.casefold()
    assert "swimming" in answer.casefold()


def test_explicit_cause_closes_cross_type_source_and_preserves_provenance(tmp_path, monkeypatch):
    class CrossTypeCauseExtractor:
        def extract(self, *_args):
            return Extraction(states=[{
                "state_type": "ACTIVITY",
                "state_value": "swimming",
                "confidence": .9,
            }])

    store, user_id, _, _, _, modules = _isolated_pipeline(
        tmp_path, monkeypatch, CrossTypeCauseExtractor())
    original_memory = f"{user_id}:running-interest"
    change_memory = f"{user_id}:switch-cause"
    source = State(
        user_id=user_id,
        state_type="INTEREST",
        state_value="running",
        source_memory_ids=[original_memory],
    )
    store.save_state(source)

    created, transitions = modules.state.ingest_message(
        "I switched from running to swimming because a knee injury made running painful.",
        user_id,
        [change_memory],
        [],
    )

    assert len(created) == len(transitions) == 1
    destination = created[0]
    transition = transitions[0]
    persisted = {state.state_id: state for state in store.states(user_id)}
    persisted_source = persisted[source.state_id]
    persisted_destination = persisted[destination.state_id]

    assert persisted_source.state_type == "INTEREST"
    assert persisted_source.status == "INACTIVE"
    assert persisted_source.valid_until == transition.timestamp
    assert persisted_source.source_memory_ids == [original_memory]
    assert persisted_destination.state_type == "ACTIVITY"
    assert persisted_destination.state_value == "swimming"
    assert persisted_destination.status == "ACTIVE"
    assert persisted_destination.valid_from == transition.timestamp
    assert persisted_destination.source_memory_ids == [change_memory]
    assert transition.relation_kind == "EXPLICIT_CAUSE"
    assert transition.from_state_id == source.state_id
    assert transition.to_state_id == destination.state_id
    assert transition.source_memory_ids == [change_memory]


def test_additive_interest_does_not_supersede_goal_in_retrieved_states(tmp_path, monkeypatch):
    class AdditiveExtractor:
        def extract(self, message, *_args):
            if "guitar" in message.casefold():
                return Extraction(states=[{"state_type": "GOAL", "state_value": "learn guitar", "confidence": .9}])
            return Extraction(
                states=[{"state_type": "INTEREST", "state_value": "swimming", "confidence": .9}],
                transitions=[{"from_state_value": "learn guitar", "to_state_value": "swimming",
                              "relation_kind": "TEMPORAL_SUCCESSION"}],
            )

    store, user_id, _, _, _, modules = _isolated_pipeline(tmp_path, monkeypatch, AdditiveExtractor())
    old_id = f"{user_id}:old-goal"
    new_id = f"{user_id}:swimming-interest"
    modules.state.ingest_message("My goal is to learn guitar.", user_id, [old_id], [])
    _, edges = modules.state.ingest_message(
        "I am also interested in swimming.", user_id, [new_id], [])

    states = {state.state_value: state for state in store.states(user_id)}
    assert edges == [] and store.transitions(user_id) == []
    assert states["learn guitar"].status == states["swimming"].status == "ACTIVE"
    assert states["learn guitar"].source_memory_ids == [old_id]
    assert states["swimming"].source_memory_ids == [new_id]
    result = modules.hybrid.retrieve("What is my goal?", user_id, "STATE")
    assert {state["state_value"] for state in result["states"]} == {"learn guitar", "swimming"}


def test_similar_second_user_facts_do_not_leak_into_retrieval(tmp_path, monkeypatch):
    class UserActivityExtractor:
        def extract(self, message, *_args):
            if "enjoy running" in message.casefold():
                return Extraction(states=[{"state_type": "ACTIVITY", "state_value": "running", "confidence": .9}])
            target = "swimming" if "swimming" in message.casefold() else "cycling"
            return Extraction(states=[{"state_type": "ACTIVITY", "state_value": target, "confidence": .9}])

    alice = f"phase6e-alice-{uuid4().hex}"
    bob = f"phase6e-bob-{uuid4().hex}"
    store, _, search_users, _, _, modules = _isolated_pipeline(
        tmp_path, monkeypatch, UserActivityExtractor(),
        memories={alice: [{"id": f"{alice}:memory", "user_id": alice, "memory": "swimming"}],
                  bob: [{"id": f"{bob}:memory", "user_id": bob, "memory": "cycling"}]})
    modules.state.ingest_message("I enjoy running.", alice, [f"{alice}:running"], [])
    modules.state.ingest_message("I switched from running to swimming because of a knee injury.",
                                    alice, [f"{alice}:swimming"], [])
    modules.state.ingest_message("I enjoy running.", bob, [f"{bob}:running"], [])
    modules.state.ingest_message("I switched from running to cycling because of a knee injury.",
                                    bob, [f"{bob}:cycling"], [])

    alice_result = modules.hybrid.retrieve(
        "Why did I switch from running to swimming?", alice, "CAUSAL")
    assert search_users == [alice]
    assert {state["user_id"] for state in alice_result["states"]} == {alice}
    assert {state["state_value"] for state in alice_result["states"]} == {"running", "swimming"}
    assert len(alice_result["transitions"]) == 1
    assert alice_result["transitions"][0]["user_id"] == alice
    assert alice_result["transitions"][0]["to_state_id"] == next(
        state.state_id for state in store.states(alice) if state.state_value == "swimming")
    assert all("cycling" not in str(value) for value in alice_result.values())


def test_transition_write_failure_rolls_back_ingestion_and_retrieval(tmp_path, monkeypatch):
    import pytest

    store, user_id, _, _, _, modules = _isolated_pipeline(tmp_path, monkeypatch, CauseExtractor())
    source_memory = f"{user_id}:running"
    modules.state.ingest_message("I enjoy running.", user_id, [source_memory], [])
    original_save_transition = store.save_transition

    def insert_then_fail(transition, connection=None):
        original_save_transition(transition, connection=connection)
        raise RuntimeError("injected transition persistence failure")

    monkeypatch.setattr(store, "save_transition", insert_then_fail)
    with pytest.raises(RuntimeError, match="transition persistence failure"):
        modules.state.ingest_message(
            "I switched from running to swimming because a knee injury made running painful.",
            user_id, [f"{user_id}:switch"], [])

    # These calls use fresh SQLite connections after the failed transaction.
    persisted = store.states(user_id)
    assert len(persisted) == 1
    assert persisted[0].state_value == "running"
    assert persisted[0].status == "ACTIVE" and persisted[0].valid_until is None
    assert persisted[0].source_memory_ids == [source_memory]
    assert store.transitions(user_id) == []
    result = modules.hybrid.retrieve(
        "Why did I switch from running to swimming?", user_id, "CAUSAL")
    # Causal retrieval returns states only when they belong to a selected path;
    # the failed transition must therefore leave no causal state evidence.
    assert result["states"] == []
    assert result["transitions"] == [] and result["causal_path"] == []
