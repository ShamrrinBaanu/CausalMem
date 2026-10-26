from database import Database
from models import Extraction, State
import state_extractor

class EmptyExtractor:
    def extract(self,*args): return Extraction()

class SwimmingExtractor:
    def extract(self,*args):
        return Extraction(states=[{"state_type":"ACTIVITY","state_value":"running","confidence":.8}],transitions=[{"from_state_value":"running","to_state_value":"swimming","relation_kind":"TEMPORAL_SUCCESSION"}])

def test_empty_model_output_only_falls_back_for_first_person_assertion(tmp_path,monkeypatch):
    store=Database(str(tmp_path/"fallback.sqlite"));monkeypatch.setattr(state_extractor,"db",store);monkeypatch.setattr(state_extractor,"llm",EmptyExtractor())
    states,edges=state_extractor.ingest_message("I love running","u",["m1"],[])
    assert states[0].state_value == "I love running"
    assert states[0].state_type == "OTHER"
    assert states[0].source_memory_ids == ["m1"]
    assert edges == []
    assert [s.state_value for s in store.states("u")] == ["I love running"]

def test_question_with_empty_extraction_creates_no_state_or_graph_node(tmp_path,monkeypatch):
    import causal_graph
    store=Database(str(tmp_path/"question.sqlite"));monkeypatch.setattr(state_extractor,"db",store);monkeypatch.setattr(state_extractor,"llm",EmptyExtractor());monkeypatch.setattr(causal_graph,"db",store)
    states,edges=state_extractor.ingest_message("Why did I switch to swimming?","u",["m1"],[])
    assert states == [] and edges == []
    assert store.states("u") == [] and store.transitions("u") == []
    assert causal_graph.build_graph("u").number_of_nodes() == 0

def test_supported_statement_can_be_extracted_without_fallback(tmp_path,monkeypatch):
    class StatementExtractor:
        def extract(self,*args): return Extraction(states=[{"state_type":"ACTIVITY","state_value":"swimming","confidence":.8}])
    store=Database(str(tmp_path/"statement.sqlite"));monkeypatch.setattr(state_extractor,"db",store);monkeypatch.setattr(state_extractor,"llm",StatementExtractor())
    states,edges=state_extractor.ingest_message("I switched to swimming last year.","u",["m2"],[])
    assert [s.state_value for s in states] == ["swimming"]
    assert [s.state_value for s in store.states("u")] == ["swimming"]
    assert store.transitions("u") == [] and edges == []

def test_ambiguous_utterance_and_extraction_failure_have_no_side_effects(tmp_path,monkeypatch):
    import pytest
    store=Database(str(tmp_path/"ambiguous.sqlite"));monkeypatch.setattr(state_extractor,"db",store);monkeypatch.setattr(state_extractor,"llm",EmptyExtractor())
    states,edges=state_extractor.ingest_message("Swimming after the injury","u",["m3"],[])
    assert states == [] and edges == []
    class FailedExtractor:
        def extract(self,*args): raise RuntimeError("provider unavailable")
    monkeypatch.setattr(state_extractor,"llm",FailedExtractor())
    with pytest.raises(RuntimeError): state_extractor.ingest_message("I switched to swimming last year.","u",["m4"],[])
    assert store.states("u") == [] and store.transitions("u") == []

def test_cause_statement_keeps_grounded_state_and_edge(tmp_path,monkeypatch):
    store=Database(str(tmp_path/"cause.sqlite"));monkeypatch.setattr(state_extractor,"db",store);monkeypatch.setattr(state_extractor,"llm",SwimmingExtractor())
    source=State(user_id="u",state_type="ACTIVITY",state_value="swimming");store.save_state(source)
    states,edges=state_extractor.ingest_message("I started running because swimming was inconvenient.","u",["m5"],[])
    assert store.states("u")
    assert [s.state_value for s in store.states("u") if s.state_id != source.state_id] == ["running"]
    assert not any(e.relation_kind == "EXPLICIT_CAUSE" for e in store.transitions("u"))

def test_explicit_cause_is_grounded_and_not_downgraded(tmp_path,monkeypatch):
    store=Database(str(tmp_path/"cause.sqlite"));monkeypatch.setattr(state_extractor,"db",store);monkeypatch.setattr(state_extractor,"llm",SwimmingExtractor())
    source=State(user_id="u",state_type="ACTIVITY",state_value="running",source_memory_ids=["m1"]);store.save_state(source)
    states,edges=state_extractor.ingest_message("Due to knee injury I switched to swimming","u",["m2"],[])
    assert any(s.state_value=="swimming" for s in states)
    assert len(edges)==1
    assert edges[0].relation_kind=="EXPLICIT_CAUSE"
    assert edges[0].trigger=="knee injury"
    assert edges[0].source_memory_ids==["m2"]
    history=store.states("u")
    old=next(s for s in history if s.state_id==source.state_id)
    new=next(s for s in history if s.state_value=="swimming")
    assert old.status=="INACTIVE" and old.valid_until==edges[0].timestamp
    assert new.status=="ACTIVE" and new.valid_from <= edges[0].timestamp

def test_transition_failure_rolls_back_prior_state_write(tmp_path,monkeypatch):
    store=Database(str(tmp_path/"rollback_transition.sqlite"));monkeypatch.setattr(state_extractor,"db",store);monkeypatch.setattr(state_extractor,"llm",SwimmingExtractor())
    source=State(user_id="u",state_type="ACTIVITY",state_value="running");store.save_state(source)
    original=store.save_transition
    def fail_after_transition_insert(edge,connection=None):
        original(edge,connection=connection)
        raise RuntimeError("forced failure after transition insert")
    monkeypatch.setattr(store,"save_transition",fail_after_transition_insert)
    import pytest
    with pytest.raises(RuntimeError,match="forced failure"):
        state_extractor.ingest_message("Due to knee injury I switched to swimming","u",["m2"],[])
    assert [s.state_value for s in store.states("u")] == ["running"]
    assert store.transitions("u") == []

def test_failure_after_source_close_rolls_back_entire_causal_ingestion(tmp_path,monkeypatch):
    import pytest
    store=Database(str(tmp_path/"rollback_after_close.sqlite"));monkeypatch.setattr(state_extractor,"db",store);monkeypatch.setattr(state_extractor,"llm",SwimmingExtractor())
    source=State(user_id="u",state_type="ACTIVITY",state_value="running");store.save_state(source)
    original_close=store.close_state
    def close_then_fail(state_id,when,connection=None):
        original_close(state_id,when,connection=connection)
        raise RuntimeError("forced failure after source close")
    monkeypatch.setattr(store,"close_state",close_then_fail)
    with pytest.raises(RuntimeError,match="after source close"):
        state_extractor.ingest_message("Due to knee injury I switched to swimming","u",["m2"],[])
    # Read after the ingestion connection has exited and rolled back.
    persisted=store.states("u")
    assert len(persisted)==1
    assert persisted[0].state_id==source.state_id
    assert persisted[0].status=="ACTIVE"
    assert persisted[0].valid_until is None
    assert store.transitions("u")==[]

def test_state_failure_rolls_back_operation(tmp_path,monkeypatch):
    store=Database(str(tmp_path/"rollback_state.sqlite"));monkeypatch.setattr(state_extractor,"db",store);monkeypatch.setattr(state_extractor,"llm",SwimmingExtractor())
    original=store.save_state
    def fail_after_state_insert(state,connection=None):
        original(state,connection=connection)
        raise RuntimeError("forced state failure")
    monkeypatch.setattr(store,"save_state",fail_after_state_insert)
    import pytest
    with pytest.raises(RuntimeError,match="forced state"):
        state_extractor.ingest_message("I enjoy running","u",["m1"],[])
    assert store.states("u") == []
    assert store.transitions("u") == []

def test_later_state_failure_rolls_back_earlier_state_and_transition(tmp_path,monkeypatch):
    class MultiWriteExtractor:
        def extract(self,*args):
            return Extraction(
                states=[
                    {"state_type":"ACTIVITY","state_value":"swimming","confidence":.8},
                    {"state_type":"ACTIVITY","state_value":"cycling","confidence":.8},
                ],
                transitions=[{"from_state_value":"running","to_state_value":"swimming","relation_kind":"TEMPORAL_SUCCESSION"}],
            )

    store=Database(str(tmp_path/"rollback_later_state.sqlite"));monkeypatch.setattr(state_extractor,"db",store);monkeypatch.setattr(state_extractor,"llm",MultiWriteExtractor())
    source=State(user_id="u",state_type="ACTIVITY",state_value="running");store.save_state(source)
    original=store.save_state; calls=0
    def fail_on_second_state(state,connection=None):
        nonlocal calls
        calls+=1
        original(state,connection=connection)
        if calls==2: raise RuntimeError("forced later state failure")
    monkeypatch.setattr(store,"save_state",fail_on_second_state)
    import pytest
    with pytest.raises(RuntimeError,match="later state failure"):
        state_extractor.ingest_message("I switched to swimming and cycling","u",["m2"],[])
    assert [s.state_value for s in store.states("u")] == ["running"]
    assert store.transitions("u") == []


class PlausibleQuestionExtractor:
    def extract(self,*args):
        return Extraction(states=[{"state_type":"ACTIVITY","state_value":"swimming","confidence":.9}],transitions=[{"from_state_value":"running","to_state_value":"swimming","relation_kind":"TEMPORAL_SUCCESSION"}])

def test_question_with_plausible_extraction_has_no_persistence_or_graph_node(tmp_path,monkeypatch):
    import causal_graph
    store=Database(str(tmp_path/"question_model.sqlite"));monkeypatch.setattr(state_extractor,"db",store);monkeypatch.setattr(state_extractor,"llm",PlausibleQuestionExtractor());monkeypatch.setattr(causal_graph,"db",store)
    source=State(user_id="u",state_type="ACTIVITY",state_value="running");store.save_state(source)
    states,edges=state_extractor.ingest_message("Why did I switch to swimming?","u",["m1"],[])
    assert states == [] and edges == []
    assert [s.state_value for s in store.states("u")] == ["running"]
    assert store.transitions("u") == []
    graph=causal_graph.build_graph("u")
    assert [data["state_value"] for _,data in graph.nodes(data=True)] == ["running"]

def test_ascii_and_curly_im_fallbacks_persist_first_person_assertions(tmp_path,monkeypatch):
    store=Database(str(tmp_path/"contractions.sqlite"));monkeypatch.setattr(state_extractor,"db",store);monkeypatch.setattr(state_extractor,"llm",EmptyExtractor())
    apostrophe="\u2019"
    utterances=["I'm training for a marathon.",f"I{apostrophe}m recovering from an injury!","I prefer swimming."]
    expected=[
        ("OTHER",utterances[0]),
        ("OTHER",utterances[1]),
        ("PREFERENCE","swimming"),
    ]
    for n,text in enumerate(utterances):
        states,edges=state_extractor.ingest_message(text,"u",[f"m{n}"],[])
        assert len(states)==1 and edges==[]
        assert (states[0].state_type,states[0].state_value)==expected[n]
    assert [(s.state_type,s.state_value) for s in store.states("u")] == expected


def test_first_person_question_mark_is_not_a_durable_assertion(tmp_path,monkeypatch):
    store=Database(str(tmp_path/"first_person_question.sqlite"));monkeypatch.setattr(state_extractor,"db",store);monkeypatch.setattr(state_extractor,"llm",PlausibleQuestionExtractor())
    states,edges=state_extractor.ingest_message("I prefer swimming?","u",["m1"],[])
    assert states == [] and edges == []
    assert store.states("u") == [] and store.transitions("u") == []
def test_explicit_change_separates_target_from_cause_and_preserves_sources(
    tmp_path, monkeypatch
):
    class CombinedStateExtractor:
        def extract(self, *args):
            return Extraction(
                states=[
                    {
                        "state_type": "ACTIVITY",
                        "state_value": "running",
                        "confidence": 0.8,
                    },
                    {
                        "state_type": "ACTIVITY",
                        "state_value": (
                            "swimming because a knee injury made running painful"
                        ),
                        "confidence": 0.85,
                    },
                ]
            )

    store = Database(str(tmp_path / "normalized_cause.sqlite"))
    monkeypatch.setattr(state_extractor, "db", store)
    monkeypatch.setattr(state_extractor, "llm", CombinedStateExtractor())

    source = State(
        user_id="u",
        state_type="ACTIVITY",
        state_value="running",
        source_memory_ids=["m1"],
    )
    store.save_state(source)

    message = (
        "I switched from running to swimming because "
        "a knee injury made running painful."
    )

    states, edges = state_extractor.ingest_message(
        message, "u", ["m2"], []
    )

    persisted = store.states("u")
    old = next(s for s in persisted if s.state_id == source.state_id)
    new = next(s for s in persisted if s.state_id != source.state_id)

    assert new.state_value == "swimming"
    assert new.source_memory_ids == ["m2"]

    # The switch message must not be attached to the old state's sources.
    assert old.source_memory_ids == ["m1"]

    assert old.status == "INACTIVE"
    assert old.valid_until == edges[0].timestamp
    assert new.status == "ACTIVE"

    assert len(edges) == 1
    assert edges[0].relation_kind == "EXPLICIT_CAUSE"
    assert edges[0].trigger == "a knee injury made running painful"

def test_lowercase_and_request_question_forms_create_no_state_or_transition(tmp_path,monkeypatch):
    store=Database(str(tmp_path/"non_assertions.sqlite"));monkeypatch.setattr(state_extractor,"db",store);monkeypatch.setattr(state_extractor,"llm",PlausibleQuestionExtractor())
    source=State(user_id="u",state_type="ACTIVITY",state_value="running");store.save_state(source)
    messages=[
        "why i switched to swimming",
        "Please tell me why I switched to swimming",
        "Can you explain why I switched to swimming?",
        "Do you remember what activity I switched to?",
        "I asked how Java streams work.",
        "I wonder why I switched to swimming.",
        "You switched from running to swimming because of a knee injury.",
    ]
    for message in messages:
        states,edges=state_extractor.ingest_message(message,"u",["m1"],[])
        assert states==[] and edges==[]
    assert [s.state_value for s in store.states("u")]==["running"]
    assert store.transitions("u")==[]


def test_assistant_role_is_rejected_before_extraction_or_persistence(tmp_path,monkeypatch):
    class SpyExtractor:
        def __init__(self): self.calls=0
        def extract(self,*args):
            self.calls+=1
            return Extraction(states=[{"state_type":"ACTIVITY","state_value":"swimming","confidence":.9}])
    spy=SpyExtractor();store=Database(str(tmp_path/"assistant_role.sqlite"));monkeypatch.setattr(state_extractor,"db",store);monkeypatch.setattr(state_extractor,"llm",spy)
    source=State(user_id="u",state_type="ACTIVITY",state_value="running");store.save_state(source)
    states,edges=state_extractor.ingest_message("I switched from running to swimming because of a knee injury.","u",["m1"],[],source_role="assistant")
    assert states==[] and edges==[] and spy.calls==0
    assert [s.state_value for s in store.states("u")]==["running"]
    assert store.transitions("u")==[]


def test_conversational_explicit_cause_is_retained(tmp_path,monkeypatch):
    store=Database(str(tmp_path/"conversational_cause.sqlite"));monkeypatch.setattr(state_extractor,"db",store);monkeypatch.setattr(state_extractor,"llm",EmptyExtractor())
    source=State(user_id="u",state_type="ACTIVITY",state_value="running");store.save_state(source)
    states,edges=state_extractor.ingest_message(
        "Just so you know, I switched from running to swimming because of a knee injury.","u",["m1"],[])
    assert [s.state_value for s in states]==["swimming"]
    assert len(edges)==1 and edges[0].relation_kind=="EXPLICIT_CAUSE"
    assert edges[0].trigger=="a knee injury"
    assert store.transitions("u")==edges


def test_repeated_explicit_cause_ingestion_is_idempotent(tmp_path,monkeypatch):
    store=Database(str(tmp_path/"explicit_cause_idempotent.sqlite"))
    monkeypatch.setattr(state_extractor,"db",store)
    monkeypatch.setattr(state_extractor,"llm",EmptyExtractor())
    source=State(user_id="u",state_type="ACTIVITY",state_value="running",
                 source_memory_ids=["m1"])
    store.save_state(source)
    message=("I switched from running to swimming because of a knee injury.")
    first_states,first_edges=state_extractor.ingest_message(
        message,"u",["m2"],[])
    second_states,second_edges=state_extractor.ingest_message(
        message,"u",["m3"],[])
    assert [state.state_value for state in first_states]==["swimming"]
    assert len(first_edges)==1 and first_edges[0].relation_kind=="EXPLICIT_CAUSE"
    assert second_states==[] and second_edges==[]
    history=store.states("u")
    assert len(history)==2
    assert [state.state_value for state in history if state.status=="ACTIVE"]==[
        "swimming"]
    assert len(store.transitions("u"))==1


def test_explicit_cause_does_not_substitute_unrelated_active_source(
        tmp_path,monkeypatch):
    store=Database(str(tmp_path/"explicit_cause_unrelated_source.sqlite"))
    monkeypatch.setattr(state_extractor,"db",store)
    monkeypatch.setattr(state_extractor,"llm",EmptyExtractor())
    unrelated=State(user_id="u",state_type="ACTIVITY",state_value="cooking")
    store.save_state(unrelated)
    states,edges=state_extractor.ingest_message(
        "I switched from running to swimming because of a knee injury.",
        "u",["m1"],[])
    assert states==[] and edges==[]
    assert [state.state_value for state in store.states("u")]==["cooking"]
    assert store.transitions("u")==[]


def test_source_less_explicit_cause_does_not_reopen_its_own_target(
        tmp_path,monkeypatch):
    store=Database(str(tmp_path/"explicit_cause_self_target.sqlite"))
    monkeypatch.setattr(state_extractor,"db",store)
    monkeypatch.setattr(state_extractor,"llm",EmptyExtractor())
    active=State(user_id="u",state_type="ACTIVITY",state_value="swimming")
    store.save_state(active)
    states,edges=state_extractor.ingest_message(
        "I switched to swimming because of a knee injury.","u",["m1"],[])
    assert states==[] and edges==[]
    assert [(state.state_id,state.status) for state in store.states("u")]==[
        (active.state_id,"ACTIVE")]
    assert store.transitions("u")==[]


def test_favorite_statement_is_normalized_to_preference_with_domain_key(
        tmp_path,monkeypatch):
    class FavoriteExtractor:
        def extract(self,*args):
            return Extraction(states=[{"state_type":"INTEREST",
                                       "state_value":"Java","confidence":.8}])
    store=Database(str(tmp_path/"favorite_preference.sqlite"))
    monkeypatch.setattr(state_extractor,"db",store)
    monkeypatch.setattr(state_extractor,"llm",FavoriteExtractor())
    states,edges=state_extractor.ingest_message(
        "My favorite programming language is Java.","u",["m1"],[])
    assert edges==[]
    assert [(state.state_type,state.state_value) for state in states]==[
        ("PREFERENCE","Java")]
    assert states[0].metadata["preference_key"]=="favorite:programming language"


def test_explicit_preference_replacement_closes_old_values_temporally(
        tmp_path,monkeypatch):
    from datetime import datetime, timedelta, timezone
    class PythonExtractor:
        def extract(self,*args):
            return Extraction(states=[
                {"state_type":"INTEREST","state_value":"Java","confidence":.8},
                {"state_type":"INTEREST","state_value":"Python","confidence":.8},
            ])
    store=Database(str(tmp_path/"preference_replacement.sqlite"))
    monkeypatch.setattr(state_extractor,"db",store)
    monkeypatch.setattr(state_extractor,"llm",PythonExtractor())
    now=datetime.now(timezone.utc)
    favorite_java=State(
        user_id="u",state_type="OTHER",
        state_value="My favorite programming language is Java.",
        valid_from=now-timedelta(minutes=5),source_memory_ids=["old-favorite"])
    interest_java=State(
        user_id="u",state_type="INTEREST",state_value="Java",
        valid_from=now-timedelta(minutes=4),source_memory_ids=["old-interest"])
    store.save_state(favorite_java);store.save_state(interest_java)

    states,edges=state_extractor.ingest_message(
        "I now prefer Python over Java.","u",["new-python"],[])

    assert [(state.state_type,state.state_value) for state in states]==[
        ("PREFERENCE","Python")]
    assert len(edges)==1
    assert edges[0].transition_type=="PREFERENCE_CHANGE"
    assert edges[0].relation_kind=="TEMPORAL_SUCCESSION"
    assert edges[0].trigger is None and edges[0].reason is None
    assert edges[0].from_state_id==interest_java.state_id
    assert edges[0].to_state_id==states[0].state_id
    assert states[0].metadata["preference_key"]=="favorite:programming language"
    history={state.state_id:state for state in store.states("u")}
    assert history[favorite_java.state_id].status=="INACTIVE"
    assert history[interest_java.state_id].status=="INACTIVE"
    assert history[interest_java.state_id].valid_until==edges[0].timestamp
    assert [state.state_value for state in history.values()
            if state.status=="ACTIVE"]==["Python"]


def test_repeated_preference_change_is_idempotent(tmp_path,monkeypatch):
    class PythonExtractor:
        def extract(self,*args):
            return Extraction(states=[
                {"state_type":"INTEREST","state_value":"Java","confidence":.8},
                {"state_type":"INTEREST","state_value":"Python","confidence":.8},
            ])
    store=Database(str(tmp_path/"preference_idempotent.sqlite"))
    monkeypatch.setattr(state_extractor,"db",store)
    monkeypatch.setattr(state_extractor,"llm",PythonExtractor())
    old=State(user_id="u",state_type="INTEREST",state_value="Java",
              source_memory_ids=["old-java"])
    store.save_state(old)
    first_states,first_edges=state_extractor.ingest_message(
        "I now prefer Python over Java.","u",["new-python"],[])
    second_states,second_edges=state_extractor.ingest_message(
        "I now prefer Python over Java.","u",["new-python-again"],[])
    assert [state.state_value for state in first_states]==["Python"]
    assert second_states==[] and second_edges==[]
    assert len(first_edges)==1
    assert len(store.transitions("u"))==1
    active=[state.state_value for state in store.states("u")
            if state.status=="ACTIVE"]
    assert active==["Python"]


def test_model_transitions_without_message_evidence_are_not_persisted(tmp_path,monkeypatch):
    class UnsupportedCauseExtractor:
        def extract(self,*args):
            return Extraction(
                states=[{"state_type":"ACTIVITY","state_value":"swimming","confidence":.9}],
                transitions=[
                    {"from_state_value":"running","to_state_value":"swimming",
                     "relation_kind":"EXPLICIT_CAUSE","trigger":"knee injury","confidence":.9},
                    {"from_state_value":"running","to_state_value":"swimming",
                     "relation_kind":"TEMPORAL_SUCCESSION","trigger":None,"confidence":.9},
                    {"from_state_value":"running","to_state_value":"swimming",
                     "relation_kind":"UNKNOWN","trigger":None,"confidence":.9},
                ],
            )
    store=Database(str(tmp_path/"unsupported_cause.sqlite"));monkeypatch.setattr(state_extractor,"db",store);monkeypatch.setattr(state_extractor,"llm",UnsupportedCauseExtractor())
    source=State(user_id="u",state_type="ACTIVITY",state_value="running");store.save_state(source)
    states,edges=state_extractor.ingest_message("I switched from running to swimming.","u",["m1"],[])
    assert [s.state_value for s in states]==["swimming"]
    assert edges==[] and store.transitions("u")==[]
