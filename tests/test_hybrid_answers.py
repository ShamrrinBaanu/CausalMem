import pytest

from database import Database
from models import State, Transition
import answer_generator
import causal_graph
import hybrid_retriever


class AnswerSpy:
    def __init__(self, result="grounded answer"):
        self.calls=[]
        self.result=result
        self.structured_result=None

    def answer_with_provenance(self,question,evidence):
        self.calls.append((question,evidence))
        if self.structured_result is not None:
            return self.structured_result
        if self.result==answer_generator.ABSTENTION:
            return {"claims":[]}
        import re
        words={word.casefold() for word in re.findall(r"[\w'-]+",self.result)
               if len(word)>2 and word.casefold() not in {"you","your","the","a","an","is","are","was","were","to","of","in","on","for","from","with","and","it","my"}}
        causal=bool(answer_generator._CAUSAL_CLAIM.search(self.result))
        kind="causal" if causal else "fact"
        refs=[]
        for record in evidence.get("memories",[]):
            text=record.get("memory") or record.get("text") or ""
            if words and words.issubset({w.casefold() for w in re.findall(r"[\w'-]+",text)}):
                memory_id=record.get("id") or record.get("memory_id")
                if memory_id: refs.append({"kind":"memory","id":memory_id})
        for state in evidence.get("states",[]):
            text=state.get("state_value","")
            if words and words.issubset({w.casefold() for w in re.findall(r"[\w'-]+",text)}):
                if state.get("state_id"): refs.append({"kind":"state","id":state["state_id"]})
        for edge in evidence.get("transitions",[]):
            text=" ".join(str(edge.get(key) or "") for key in ("trigger","reason"))
            if words and words.issubset({w.casefold() for w in re.findall(r"[\w'-]+",text)}):
                if edge.get("transition_id"): refs.append({"kind":"transition","id":edge["transition_id"]})
        return {"claims":[{"text":self.result,"claim_type":kind,"evidence":refs}]}


def test_retrieval_deduplicates_only_exact_records_with_stable_ids(tmp_path,monkeypatch):
    store=Database(str(tmp_path/"memories.sqlite"))
    duplicate={"id":"m-1","memory":"likes tea","user_id":"alice"}
    records=[duplicate,dict(duplicate),
        {"id":"m-2","memory":"likes tea","user_id":"alice"},
        {"id":"m-3","memory":"likes coffee","user_id":"alice"},
        {"id":"m-1","memory":"likes coffee","user_id":"alice"},
        {"memory":"legacy note","user_id":"alice"},
        {"memory":"legacy note","user_id":"alice"}]
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda query,user_id: records)

    result=hybrid_retriever.retrieve("What do I like?","alice","FACTUAL")

    assert result["memories"]==[records[i] for i in (0,2,3,4,5,6)]
    assert len(result["memories"])==6
    assert result["user_id"]=="alice"


def test_retrieval_fails_closed_on_mixed_ownership_without_timestamp(tmp_path,monkeypatch):
    store=Database(str(tmp_path/"memory_owner.sqlite"))
    records=[
        {"id":"alice-top","memory":"owned by alice","user_id":"alice"},
        {"id":"bob-top","memory":"owned by bob","user_id":"bob"},
        {"id":"unknown","memory":"owner missing"},
        {"id":"alice-nested","memory":"nested owner","metadata":{"user_id":"alice"}},
        {"id":"conflict","memory":"conflicting owners","user_id":"alice","metadata":{"user_id":"bob"}},
    ]
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _query,_user_id:records)

    import pytest
    with pytest.raises(hybrid_retriever.RetrievalFailure,match="Mem0"):
        hybrid_retriever.retrieve("What do I remember?","alice","FACTUAL")


@pytest.mark.parametrize("bad_record",[None,{"id":"no-owner"},{"id":"foreign","user_id":"bob"}])
def test_malformed_or_foreign_mem0_record_fails_closed(tmp_path,monkeypatch,bad_record):
    import pytest
    store=Database(str(tmp_path/"malformed_mem0.sqlite"))
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _q,_u:[
        {"id":"alice-valid","user_id":"alice","memory":"valid"},bad_record])

    with pytest.raises(hybrid_retriever.RetrievalFailure) as error:
        hybrid_retriever.retrieve("What do I remember?","alice","FACTUAL")
    assert error.value.source=="Mem0 ownership validation"


def test_retrieval_fails_closed_on_foreign_or_unknown_memories_with_as_of_timestamp(tmp_path,monkeypatch):
    from datetime import datetime, timezone
    store=Database(str(tmp_path/"memory_owner_asof.sqlite"))
    records=[
        {"id":"alice-old","memory":"past alice memory","user_id":"alice",
         "created_at":"2025-01-01T00:00:00Z"},
        {"id":"bob-old","memory":"past bob memory","user_id":"bob",
         "created_at":"2025-01-01T00:00:00Z"},
        {"id":"alice-future","memory":"future alice memory","user_id":"alice",
         "created_at":"2025-03-01T00:00:00Z"},
        {"id":"owner-missing","memory":"unknown owner","created_at":"2025-01-01T00:00:00Z"},
        {"id":"nested-alice-old","memory":"nested past alice","metadata":{"user_id":"alice"},
         "created_at":"2025-01-01T00:00:00Z"},
    ]
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _query,_user_id:records)

    import pytest
    with pytest.raises(hybrid_retriever.RetrievalFailure,match="Mem0"):
        hybrid_retriever.retrieve("What did I remember?","alice","HISTORY",
                                  requested_timestamp=datetime(2025,2,1,tzinfo=timezone.utc))


def test_empty_mem0_result_still_allows_valid_user_scoped_sqlite_evidence(tmp_path,monkeypatch):
    store=Database(str(tmp_path/"empty_mem0.sqlite"))
    state=State(user_id="alice",state_type="GOAL",state_value="learn guitar")
    store.save_state(state)
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _q,_u:[])

    evidence=hybrid_retriever.retrieve("What is my current goal?","alice","STATE")

    assert evidence["memories"]==[]
    assert [item["state_id"] for item in evidence["states"]]==[state.state_id]


def test_asof_retrieval_fails_closed_when_mem0_timestamp_is_missing_or_malformed(tmp_path,monkeypatch):
    from datetime import datetime,timezone
    import pytest
    store=Database(str(tmp_path/"bad_mem0_time.sqlite"))
    monkeypatch.setattr(hybrid_retriever,"db",store)
    for timestamp in (None,"not-a-time"):
        record={"id":"alice-record","user_id":"alice","memory":"old record"}
        if timestamp is not None:
            record["created_at"]=timestamp
        monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _q,_u,record=record:[record])
        with pytest.raises(hybrid_retriever.RetrievalFailure) as error:
            hybrid_retriever.retrieve("What did I remember?","alice","HISTORY",
                                      datetime(2025,1,1,tzinfo=timezone.utc))
        assert error.value.source=="Mem0 temporal metadata"


@pytest.mark.parametrize(("offset","expected_state","expected_memory_ids"),[
    (-1,"learn guitar",{"guitar-memory","unlinked-before"}),
    (0,"learn piano",{"piano-memory","unlinked-before"}),
    (1,"learn piano",{"piano-memory","unlinked-before"}),
])
def test_asof_retrieval_aligns_linked_mem0_with_state_validity_and_passes_time_to_answer(
        tmp_path,monkeypatch,offset,expected_state,expected_memory_ids):
    from datetime import datetime,timedelta,timezone
    import temporal_engine

    store=Database(str(tmp_path/f"asof_linked_memories_{offset}.sqlite"))
    start=datetime(2025,1,1,tzinfo=timezone.utc)
    boundary=datetime(2025,6,1,tzinfo=timezone.utc)
    old=State(user_id="alice",state_type="GOAL",state_value="learn guitar",
              status="INACTIVE",valid_from=start,valid_until=boundary,
              source_memory_ids=["guitar-memory"])
    new=State(user_id="alice",state_type="GOAL",state_value="learn piano",
              valid_from=boundary,source_memory_ids=["piano-memory"])
    store.save_state(old);store.save_state(new)
    store.save_transition(Transition(user_id="alice",from_state_id=old.state_id,
        to_state_id=new.state_id,relation_kind="TEMPORAL_SUCCESSION",timestamp=boundary,
        source_memory_ids=["piano-memory"]))
    records=[
        # Mem0 writes happen before SQLite closes the source state, so both
        # linked records can have created_at earlier than the edge boundary.
        {"id":"guitar-memory","user_id":"alice","memory":"learn guitar",
         "created_at":(boundary-timedelta(seconds=2)).isoformat()},
        {"id":"piano-memory","user_id":"alice","memory":"learn piano",
         "created_at":(boundary-timedelta(seconds=1)).isoformat()},
        {"id":"unlinked-before","user_id":"alice","memory":"unlinked old note",
         "created_at":(boundary-timedelta(seconds=3)).isoformat()},
        {"id":"unlinked-after","user_id":"alice","memory":"unlinked future note",
         "created_at":(boundary+timedelta(seconds=1)).isoformat()},
    ]
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(temporal_engine,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _q,_u:records)

    instant=boundary+timedelta(microseconds=offset)
    evidence=hybrid_retriever.retrieve("What was my goal?","alice","TEMPORAL",instant)

    assert [state["state_value"] for state in evidence["states"]]==[expected_state]
    assert {memory["id"] for memory in evidence["memories"]}==expected_memory_ids
    assert evidence["requested_timestamp"]==instant.isoformat()
    assert evidence["temporal_context"]=={
        "mode":"as_of",
        "requested_timestamp":instant.isoformat(),
        "state_validity_rule":"valid_from <= requested_timestamp < valid_until; null valid_until is open-ended",
        "selected_states_valid_at_requested_timestamp":True,
    }

    spy=AnswerSpy(result=answer_generator.ABSTENTION)
    monkeypatch.setattr(answer_generator,"llm",spy)
    answer_generator.generate_answer_with_diagnostics("What was my goal?",evidence)
    assert spy.calls[0][1]["requested_timestamp"]==instant.isoformat()
    assert spy.calls[0][1]["temporal_context"]["selected_states_valid_at_requested_timestamp"] is True


@pytest.mark.parametrize(("source","category","query"),[
    ("Mem0","FACTUAL","What do I remember?"),
    ("SQLite states","STATE","What is my current goal?"),
    ("SQLite transitions","CAUSAL","Why did running change to swimming?"),
    ("causal path","COMPARISON","Compare running with swimming."),
])
def test_retrieval_source_failures_propagate_with_context_and_do_not_answer(
        tmp_path,monkeypatch,source,category,query):
    import pytest
    store=Database(str(tmp_path/f"failure-{source.replace(' ','_')}.sqlite"))
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(causal_graph,"db",store)
    import temporal_engine
    monkeypatch.setattr(temporal_engine,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _q,_u: (
        (_ for _ in ()).throw(OSError("private low-level details")) if source=="Mem0" else []))
    if source=="SQLite states":
        class BrokenStates:
            def states(self,*_args,**_kwargs): raise OSError("db detail")
        monkeypatch.setattr(hybrid_retriever,"db",BrokenStates())
    if source=="SQLite transitions":
        class BrokenTransitions:
            def states(self,*_args,**_kwargs): return []
            def transitions(self,_user): raise OSError("db detail")
        monkeypatch.setattr(causal_graph,"db",BrokenTransitions())
    if source=="causal path":
        monkeypatch.setattr(hybrid_retriever,"relevant_path",
                            lambda *_args,**_kwargs:(_ for _ in ()).throw(OSError("graph detail")))
    spy=AnswerSpy()
    monkeypatch.setattr(answer_generator,"llm",spy)

    with pytest.raises(hybrid_retriever.RetrievalFailure) as error:
        hybrid_retriever.retrieve(query,"alice",category)
    assert error.value.source==source
    if source=="Mem0":
        assert error.value.cause_type=="OSError"
    assert "private low-level details" not in str(error.value)
    assert spy.calls==[]


def test_retrieval_failure_has_clear_sanitized_user_message():
    from llm_service import user_facing_error
    error=hybrid_retriever.RetrievalFailure("Mem0")
    message=user_facing_error(error)
    assert "Could not retrieve Mem0 evidence" in message
    assert "No answer was generated" in message


def test_retrieval_keeps_causal_and_state_source_evidence_user_scoped(tmp_path,monkeypatch):
    store=Database(str(tmp_path/"causal_evidence.sqlite"))
    running=State(user_id="alice",state_value="running",source_memory_ids=["m-running"])
    swimming=State(user_id="alice",state_value="swimming",source_memory_ids=["m-swimming"])
    bob_running=State(user_id="bob",state_value="running")
    bob_swimming=State(user_id="bob",state_value="swimming")
    for state in (running,swimming,bob_running,bob_swimming): store.save_state(state)
    alice_edge=Transition(user_id="alice",from_state_id=running.state_id,to_state_id=swimming.state_id,
        relation_kind="EXPLICIT_CAUSE",trigger="knee injury",reason="switched due to knee injury",
        source_memory_ids=["m-cause"])
    bob_edge=Transition(user_id="bob",from_state_id=bob_running.state_id,to_state_id=bob_swimming.state_id,
        relation_kind="EXPLICIT_CAUSE",trigger="schedule",source_memory_ids=["bob-cause"])
    store.save_transition(alice_edge);store.save_transition(bob_edge)
    searched=[]
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(causal_graph,"db",store)
    monkeypatch.setattr("temporal_engine.db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda query,user_id: searched.append(user_id) or [{"id":"m-cause","memory":"knee injury caused switch","user_id":user_id}])

    result=hybrid_retriever.retrieve("Why did running change to swimming","alice","HISTORY")

    assert searched==["alice"]
    assert result["query_category"]=="HISTORY" and result["user_id"]=="alice"
    assert result["transitions"]==[alice_edge.model_dump(mode="json")]
    assert result["causal_path"][0]["source_memory_ids"]==["m-cause"]
    assert {state["source_memory_ids"][0] for state in result["states"]}=={"m-running","m-swimming"}


def _evidence(relation="EXPLICIT_CAUSE",trigger="injury",reason=None):
    transition={"user_id":"alice","transition_id":"edge-id","from_state_id":"running-id","to_state_id":"swimming-id",
        "relation_kind":relation,"trigger":trigger,"reason":reason,"source_memory_ids":["cause-memory"],"timestamp":"2025-01-01T00:00:00+00:00"}
    return {"user_id":"alice","query_category":"CAUSAL","memories":[],
        "states":[{"user_id":"alice","state_id":"running-id","state_value":"running","source_memory_ids":["m1"]},
                  {"user_id":"alice","state_id":"swimming-id","state_value":"swimming","source_memory_ids":["m2"]}],
        "transitions":[transition],"causal_path":[]}


@pytest.mark.parametrize("relation",["TEMPORAL_SUCCESSION","UNKNOWN","INFERRED_RELATIONSHIP"])
def test_noncausal_transition_kinds_do_not_pass_causal_gate(relation,monkeypatch):
    spy=AnswerSpy();monkeypatch.setattr(answer_generator,"llm",spy)
    evidence=_evidence(relation=relation,trigger="injury")

    answer=answer_generator.generate_answer("What caused running to change to swimming?",evidence)

    assert answer==answer_generator.ABSTENTION
    assert spy.calls==[]


def test_causal_gate_rejects_missing_cause_relevance_endpoints_and_wrong_user(monkeypatch):
    spy=AnswerSpy();monkeypatch.setattr(answer_generator,"llm",spy)
    cases=[]
    no_reason=_evidence(trigger=" ",reason="")
    cases.append(("missing cause",no_reason,"Why did running change to swimming?"))
    no_endpoints=_evidence();no_endpoints["states"]=[]
    cases.append(("missing endpoints",no_endpoints,"Why did running change to swimming?"))
    wrong_user=_evidence();wrong_user["transitions"][0]["user_id"]="bob"
    cases.append(("foreign transition",wrong_user,"Why did running change to swimming?"))
    no_anchor=_evidence()
    cases.append(("no anchor",no_anchor,"Why did this change?"))

    for _,evidence,question in cases:
        assert answer_generator.generate_answer(question,evidence)==answer_generator.ABSTENTION
    assert spy.calls==[]


def test_generic_current_state_overlap_does_not_authorize_unrelated_cause(monkeypatch):
    spy=AnswerSpy();monkeypatch.setattr(answer_generator,"llm",spy)
    evidence=_evidence()
    evidence["states"]=[
        {"user_id":"alice","state_id":"running-id","state_value":"current goal"},
        {"user_id":"alice","state_id":"swimming-id","state_value":"future goal"},
    ]

    assert answer_generator.generate_answer("Why did the current state change?",evidence)==answer_generator.ABSTENTION
    assert answer_generator.generate_answer("Why did my goal change?",evidence)==answer_generator.ABSTENTION
    assert spy.calls==[]


def test_bob_owned_endpoint_states_cannot_validate_alice_transition(monkeypatch):
    spy=AnswerSpy();monkeypatch.setattr(answer_generator,"llm",spy)
    evidence=_evidence()
    evidence["states"]=[
        {"user_id":"bob","state_id":"running-id","state_value":"running"},
        {"user_id":"bob","state_id":"swimming-id","state_value":"swimming"},
    ]
    evidence["causal_path"]=[{"user_id":"alice","from_state_id":"running-id","to_state_id":"swimming-id",
        "from":"running","to":"swimming","relation_kind":"EXPLICIT_CAUSE"}]

    assert answer_generator.generate_answer("Why did running change to swimming?",evidence)==answer_generator.ABSTENTION
    assert spy.calls==[]


def test_relevant_explicit_cause_passes_gate_and_preserves_answer_path(monkeypatch):
    spy=AnswerSpy("changed because knee injury");monkeypatch.setattr(answer_generator,"llm",spy)
    evidence=_evidence(reason="changed because of a knee injury")

    answer=answer_generator.generate_answer("Why did running change to swimming?",evidence)

    assert answer == "You switched from running to swimming because of a knee injury. [Sources: transition:edge-id]"
    assert len(spy.calls)==1
    model_evidence=spy.calls[0][1]
    assert model_evidence["query_category"]=="CAUSAL"
    assert model_evidence["memories"]==[]
    assert model_evidence["states"]==[]
    assert model_evidence["transitions"]==[{
        "transition_id":"edge-id","transition_type":None,
        "relation_kind":"EXPLICIT_CAUSE","from":"running","to":"swimming",
        "trigger":"injury","reason":"changed because of a knee injury",
        "timestamp":"2025-01-01T00:00:00+00:00",
    }]
    assert "user_id" not in model_evidence
    assert "source_memory_ids" not in model_evidence["transitions"][0]


def test_causal_answer_adds_recorded_destination_when_llm_omits_it(monkeypatch):
    spy=AnswerSpy("a knee injury made running painful")
    monkeypatch.setattr(answer_generator,"llm",spy)
    evidence=_evidence(trigger="a knee injury made running painful")

    answer=answer_generator.generate_answer("Why did running change to swimming?",evidence)

    assert len(spy.calls)==1
    assert answer=="You switched from running to swimming because a knee injury made running painful. [Sources: transition:edge-id]"
    assert "swimming" in answer and "knee injury" in answer


def test_category_marks_causal_query_even_without_why_pattern(monkeypatch):
    spy=AnswerSpy();monkeypatch.setattr(answer_generator,"llm",spy)
    evidence=_evidence();evidence["query_category"]="CAUSAL"

    spy.result="change because injury"
    evidence["transitions"][0]["reason"]="change because injury"
    answer=answer_generator.generate_answer("Explain the running to swimming transition",evidence)
    assert "transition:edge-id" in answer and "swimming" in answer
    assert len(spy.calls)==1


@pytest.mark.parametrize("relation",["TEMPORAL_SUCCESSION","UNKNOWN","INFERRED_RELATIONSHIP"])
def test_noncausal_multihop_answer_cannot_turn_relation_into_cause(relation,monkeypatch):
    spy=AnswerSpy("Running changed to swimming because an injury made it difficult.")
    monkeypatch.setattr(answer_generator,"llm",spy)
    evidence=_evidence(relation=relation,trigger="injury")
    evidence["query_category"]="MULTI_HOP"

    answer=answer_generator.generate_answer(
        "How did running change to swimming?",evidence)

    assert answer==answer_generator.ABSTENTION
    assert len(spy.calls)==1


def test_reason_question_is_causal_and_requires_explicit_evidence(monkeypatch):
    spy=AnswerSpy("The transition happened because of an injury.")
    monkeypatch.setattr(answer_generator,"llm",spy)
    evidence=_evidence(relation="TEMPORAL_SUCCESSION",trigger="injury")
    evidence["query_category"]="HISTORY"

    assert answer_generator.generate_answer(
        "What was the reason running changed to swimming?",evidence)==answer_generator.ABSTENTION
    assert spy.calls==[]


def test_future_explicit_transition_cannot_support_asof_causal_answer(tmp_path,monkeypatch):
    from datetime import datetime,timedelta,timezone
    import temporal_engine
    store=Database(str(tmp_path/"asof_causal_answer.sqlite"))
    start=datetime(2025,1,1,tzinfo=timezone.utc)
    boundary=start+timedelta(days=2)
    running=State(user_id="alice",state_type="ACTIVITY",state_value="running",
                  valid_from=start,valid_until=boundary,status="INACTIVE")
    swimming=State(user_id="alice",state_type="ACTIVITY",state_value="swimming",
                   valid_from=boundary)
    store.save_state(running);store.save_state(swimming)
    edge=Transition(user_id="alice",from_state_id=running.state_id,to_state_id=swimming.state_id,
                    relation_kind="EXPLICIT_CAUSE",trigger="knee injury",timestamp=boundary)
    store.save_transition(edge)
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(causal_graph,"db",store)
    monkeypatch.setattr(temporal_engine,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _q,_u:[])
    spy=AnswerSpy("A knee injury caused the change.")
    monkeypatch.setattr(answer_generator,"llm",spy)

    evidence=hybrid_retriever.retrieve("Why did running change to swimming?","alice","CAUSAL",boundary-timedelta(microseconds=1))

    assert evidence["transitions"]==[]
    assert answer_generator.generate_answer("Why did running change to swimming?",evidence)==answer_generator.ABSTENTION
    assert spy.calls==[]


def test_answer_prompt_defines_relation_semantics_without_live_llm(monkeypatch):
    from llm_service import LLMService
    captured={}
    def fake_json(prompt,payload):
        captured["prompt"]=prompt
        captured["payload"]=payload
        return {"claims":[{"text":"recorded ordering","claim_type":"temporal",
                           "evidence":[]}]}
    service=LLMService.__new__(LLMService)
    monkeypatch.setattr(service,"_json",fake_json)

    result=service.answer_with_provenance("How did it change?",{"transitions":[]})
    assert result["claims"][0]["text"]=="recorded ordering"
    prompt=captured["prompt"]
    assert 'claim_type must be exactly one of "fact", "temporal", or "causal"' in prompt
    assert 'evidence reference kind must be exactly one of "memory", "state", or "transition"' in prompt
    assert '"claim_type":"temporal"' in prompt and '"kind":"state"' in prompt
    assert "marked INACTIVE can still be valid historical evidence" in prompt
    assert "For a causal claim, cite the supplied EXPLICIT_CAUSE transition citation_ref" in prompt
    assert '"claim_type":"causal","evidence":[{"kind":"transition"' in prompt
    assert "never substitute a memory or endpoint-state citation" in prompt
    assert "short citation_ref" in prompt
    assert "Do not copy or invent long IDs or UUIDs" in prompt
    assert '"ref":"S1"' in prompt and '"ref":"T1"' in prompt
    assert "TEMPORAL_SUCCESSION supports ordering only, not causation" in prompt
    assert "UNKNOWN" in prompt and "INFERRED_RELATIONSHIP" in prompt


def test_asof_temporal_claim_can_cite_state_only_when_its_interval_contains_request(monkeypatch):
    from datetime import datetime,timezone
    instant=datetime(2025,5,31,23,59,59,tzinfo=timezone.utc)
    state={"user_id":"alice","state_id":"old-goal","state_type":"GOAL",
           "state_value":"learn guitar","status":"INACTIVE",
           "valid_from":"2025-01-01T00:00:00+00:00",
           "valid_until":"2025-06-01T00:00:00+00:00"}
    evidence={"user_id":"alice","query_category":"TEMPORAL",
        "requested_timestamp":instant.isoformat(),
        "temporal_context":{"mode":"as_of","requested_timestamp":instant.isoformat()},
        "memories":[],"states":[state],"transitions":[],"causal_path":[]}
    spy=AnswerSpy()
    spy.structured_result={"claims":[{"text":"My goal was to learn guitar before the change",
        "claim_type":"temporal","evidence":[{"kind":"state","id":"old-goal"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics("What was my goal?",evidence)

    assert answer=="My goal was to learn guitar before the change [Sources: state:old-goal]"
    assert diagnostics["accepted_claim_count"]==1


def test_asof_temporal_state_claim_is_rejected_outside_its_validity_interval(monkeypatch):
    from datetime import datetime,timezone
    instant=datetime(2025,6,1,tzinfo=timezone.utc)
    state={"user_id":"alice","state_id":"old-goal","state_type":"GOAL",
           "state_value":"learn guitar","status":"INACTIVE",
           "valid_from":"2025-01-01T00:00:00+00:00",
           "valid_until":"2025-06-01T00:00:00+00:00"}
    evidence={"user_id":"alice","query_category":"TEMPORAL",
        "requested_timestamp":instant.isoformat(),"memories":[],"states":[state],
        "transitions":[],"causal_path":[]}
    spy=AnswerSpy()
    spy.structured_result={"claims":[{"text":"My goal was to learn guitar",
        "claim_type":"temporal","evidence":[{"kind":"state","id":"old-goal"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics("What was my goal?",evidence)

    assert answer==answer_generator.ABSTENTION
    assert diagnostics["rejection_categories"]=={"claim_not_supported_by_source":1}


def test_predecessor_temporal_claim_requires_transition_boundary_link(monkeypatch):
    evidence={"user_id":"alice","query_category":"HISTORY","retrieval_sources":["Timeline"],
        "memories":[],"states":[
            {"user_id":"alice","state_id":"old","state_type":"GOAL","state_value":"learn guitar",
             "status":"INACTIVE","valid_from":"2025-01-01T00:00:00+00:00",
             "valid_until":"2025-06-01T00:00:00+00:00"},
            {"user_id":"alice","state_id":"new","state_type":"GOAL","state_value":"learn piano",
             "status":"ACTIVE","valid_from":"2025-06-01T00:00:00+00:00","valid_until":None}],
        "transitions":[{"user_id":"alice","transition_id":"succession","from_state_id":"old",
             "to_state_id":"new","relation_kind":"TEMPORAL_SUCCESSION",
             "timestamp":"2025-06-01T00:00:00+00:00"}],"causal_path":[]}
    spy=AnswerSpy()
    spy.structured_result={"claims":[{"text":"My goal was to learn guitar before the change",
        "claim_type":"temporal","evidence":[{"kind":"state","id":"old"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer=answer_generator.generate_answer("What goal did I have before I changed it?",evidence)

    assert answer=="My goal was to learn guitar before the change [Sources: state:old]"


def test_temporal_state_claim_cannot_use_another_users_state(monkeypatch):
    evidence={"user_id":"alice","query_category":"TEMPORAL",
        "requested_timestamp":"2025-05-01T00:00:00+00:00","memories":[],
        "states":[{"user_id":"bob","state_id":"bob-goal","state_type":"GOAL",
                    "state_value":"learn guitar","valid_from":"2025-01-01T00:00:00+00:00",
                    "valid_until":None}],"transitions":[],"causal_path":[]}
    spy=AnswerSpy()
    spy.structured_result={"claims":[{"text":"My goal was to learn guitar",
        "claim_type":"temporal","evidence":[{"kind":"state","id":"bob-goal"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)
    assert answer_generator.generate_answer("What was my goal?",evidence)==answer_generator.ABSTENTION


def test_direct_fact_accepts_common_inflection_mismatch_but_not_unsupported_fact(monkeypatch):
    evidence={"user_id":"alice","query_category":"FACTUAL","states":[],"transitions":[],
        "memories":[{"id":"toronto-memory","user_id":"alice","memory":"User lives in Toronto."}]}
    spy=AnswerSpy();spy.structured_result={"claims":[{"text":"You live in Toronto.",
        "claim_type":"fact","evidence":[{"kind":"memory","id":"toronto-memory"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)
    assert answer_generator.generate_answer("Where do I live?",evidence)==(
        "You live in Toronto. [Sources: memory:toronto-memory]")

    spy.structured_result={"claims":[{"text":"You live in Vancouver.",
        "claim_type":"fact","evidence":[{"kind":"memory","id":"toronto-memory"}]}]}
    assert answer_generator.generate_answer("Where do I live?",evidence)==answer_generator.ABSTENTION


def test_fact_state_claim_uses_its_linked_memory_for_supported_predicate(monkeypatch):
    evidence={"user_id":"alice","query_category":"FACTUAL","transitions":[],"causal_path":[],
        "memories":[{"id":"home-memory","user_id":"alice","memory":"User lives in Toronto."}],
        "states":[{"user_id":"alice","state_id":"home-state","state_type":"FACT",
            "state_value":"Toronto","source_memory_ids":["home-memory"]}]}
    spy=AnswerSpy();spy.structured_result={"claims":[{"text":"You live in Toronto.",
        "claim_type":"fact","evidence":[{"kind":"state","id":"home-state"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics("Where do I live?",evidence)

    assert answer=="You live in Toronto. [Sources: state:home-state]"
    assert diagnostics["accepted_claim_count"]==1
    assert diagnostics["rejected_claims"]==[]


@pytest.mark.parametrize(("state_sources","memory_text","claim_text"),[
    (["home-memory"],"User lives in Toronto.","You live in Vancouver."),
    ([],"User traveled through Toronto.","You live in Toronto."),
])
def test_fact_state_does_not_accept_unsupported_or_unlinked_location_claim(
        monkeypatch,state_sources,memory_text,claim_text):
    evidence={"user_id":"alice","query_category":"FACTUAL","transitions":[],"causal_path":[],
        "memories":[{"id":"home-memory","user_id":"alice","memory":memory_text}],
        "states":[{"user_id":"alice","state_id":"home-state","state_type":"FACT",
            "state_value":"Toronto","source_memory_ids":state_sources}]}
    spy=AnswerSpy();spy.structured_result={"claims":[{"text":claim_text,
        "claim_type":"fact","evidence":[{"kind":"state","id":"home-state"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics("Where do I live?",evidence)

    assert answer==answer_generator.ABSTENTION
    assert diagnostics["rejection_categories"]=={"claim_not_supported_by_source":1}
    assert diagnostics["rejected_claims"][0]["evidence_references"][0]["rejection_reason"]==(
        "fact_lexical_support_failed")


def test_predecessor_temporal_claim_can_use_unique_linked_memory_provenance(monkeypatch):
    evidence={"user_id":"alice","query_category":"HISTORY","retrieval_sources":["Timeline"],
        "memories":[{"id":"guitar-memory","user_id":"alice","memory":"User wants to learn guitar."}],
        "states":[{"user_id":"alice","state_id":"old","state_type":"GOAL",
            "state_value":"learn guitar","status":"INACTIVE","source_memory_ids":["guitar-memory"],
            "valid_from":"2025-01-01T00:00:00+00:00","valid_until":"2025-06-01T00:00:00+00:00"}],
        "transitions":[{"user_id":"alice","transition_id":"succession","from_state_id":"old",
            "to_state_id":"new","relation_kind":"TEMPORAL_SUCCESSION",
            "timestamp":"2025-06-01T00:00:00+00:00"}],"causal_path":[]}
    spy=AnswerSpy();spy.structured_result={"claims":[{"text":"My goal was to learn guitar before the change.",
        "claim_type":"temporal","evidence":[{"kind":"memory","id":"guitar-memory"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer=answer_generator.generate_answer("What goal did I have before I changed it?",evidence)

    assert answer=="My goal was to learn guitar before the change. [Sources: memory:guitar-memory]"


def test_asof_current_wording_can_be_supported_by_valid_state_only(monkeypatch):
    evidence={"user_id":"alice","query_category":"TEMPORAL",
        "requested_timestamp":"2025-06-02T00:00:00+00:00","memories":[],"transitions":[],
        "states":[{"user_id":"alice","state_id":"piano","state_type":"GOAL",
            "state_value":"learn piano","status":"ACTIVE",
            "valid_from":"2025-06-01T00:00:00+00:00","valid_until":None}]}
    spy=AnswerSpy();spy.structured_result={"claims":[{"text":"My current goal is to learn piano.",
        "claim_type":"fact","evidence":[{"kind":"state","id":"piano"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    assert answer_generator.generate_answer("What is my goal?",evidence)==(
        "My current goal is to learn piano. [Sources: state:piano]")


def test_current_state_fact_claim_with_now_wording_uses_current_interval(monkeypatch):
    evidence={"user_id":"alice","query_category":"STATE",
        "memories":[{"id":"rust-source","user_id":"alice",
                     "memory":"User prefers Rust for systems programming"}],
        "states":[{"user_id":"alice","state_id":"rust","state_type":"PREFERENCE",
            "state_value":"Rust","status":"ACTIVE",
            "valid_from":"2025-01-01T00:00:00+00:00","valid_until":None,
            "source_memory_ids":["rust-source"]}],
        "transitions":[],"causal_path":[]}
    spy=AnswerSpy();spy.structured_result={"claims":[{
        "text":"User currently prefers Rust for systems programming",
        "claim_type":"fact","evidence":[{"kind":"state","id":"rust"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer=answer_generator.generate_answer(
        "Which systems programming language do I prefer right now?",evidence)

    assert answer=="User currently prefers Rust for systems programming [Sources: state:rust]"


def test_current_activity_question_accepts_focused_state_with_query_predicate(
        tmp_path,monkeypatch):
    from datetime import datetime,timezone
    store=Database(str(tmp_path/"current_activity.sqlite"))
    running=State(user_id="alice",state_type="INTEREST",state_value="running",
                  status="INACTIVE",source_memory_ids=["running-memory"],
                  valid_from=datetime(2025,1,1,tzinfo=timezone.utc),
                  valid_until=datetime(2026,1,1,tzinfo=timezone.utc))
    swimming=State(user_id="alice",state_type="ACTIVITY",state_value="swimming",
                   source_memory_ids=["switch-memory"])
    store.save_state(running);store.save_state(swimming)
    memories=[
        {"id":"running-memory","user_id":"alice","memory":"User enjoys running"},
        {"id":"switch-memory","user_id":"alice",
         "memory":"User switched from running to swimming due to a knee injury"},
    ]
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _query,_user:memories)

    result=hybrid_retriever.retrieve(
        "What activity do I currently enjoy?","alice","PREFERENCE")

    assert result["query_category"]=="STATE"
    assert result["state_focus"]["candidate_state_ids"]==[swimming.state_id]
    assert [state["state_value"] for state in result["states"]]==["swimming"]

    spy=AnswerSpy();spy.structured_result={"claims":[{
        "text":"I currently enjoy swimming","claim_type":"fact",
        "evidence":[{"kind":"state","ref":"S1"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)
    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "What activity do I currently enjoy?",result)

    assert answer=="I currently enjoy swimming [Sources: state:"+swimming.state_id+"]"
    assert diagnostics["accepted_claim_count"]==1
    assert diagnostics["rejected_claim_count"]==0
    assert diagnostics["answer_reason"]=="validated_claims"
    assert diagnostics["citation_validation"][0]["reason_code"]=="citation_resolved"


def test_current_state_fact_claim_rejects_inactive_state_even_with_valid_interval(monkeypatch):
    evidence={"user_id":"alice","query_category":"STATE",
        "memories":[{"id":"rust-source","user_id":"alice",
                     "memory":"User prefers Rust for systems programming"}],
        "states":[{"user_id":"alice","state_id":"rust-old","state_type":"PREFERENCE",
            "state_value":"Rust","status":"INACTIVE",
            "valid_from":"2025-01-01T00:00:00+00:00","valid_until":None,
            "source_memory_ids":["rust-source"]}],
        "transitions":[],"causal_path":[]}
    spy=AnswerSpy();spy.structured_result={"claims":[{
        "text":"User currently prefers Rust for systems programming",
        "claim_type":"fact","evidence":[{"kind":"state","id":"rust-old"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "Which systems programming language do I prefer right now?",evidence)

    assert answer==answer_generator.ABSTENTION
    assert diagnostics["answer_reason"]=="no_validated_claims"
    assert diagnostics["rejected_claims"][0]["evidence_references"][0][
        "rejection_reason"]=="fact_lexical_support_failed"
    assert "state_not_active_at_current_timestamp" in diagnostics[
        "temporal_validation"][0]["reason_codes"]


@pytest.mark.parametrize("claim_type",["fact","temporal"])
@pytest.mark.parametrize("requested_timestamp",[
    "2025-06-01T00:00:00+00:00","2025-06-02T00:00:00+00:00"])
def test_asof_claim_can_use_same_user_linked_memory_for_state_summary_wording(
        monkeypatch,claim_type,requested_timestamp):
    evidence={"user_id":"alice","query_category":"TEMPORAL",
        "requested_timestamp":requested_timestamp,"transitions":[],
        "memories":[{"id":"piano-source","user_id":"alice",
                     "memory":"I want to practice piano."}],
        "states":[{"user_id":"alice","state_id":"piano","state_type":"GOAL",
            "state_value":"learn piano","source_memory_ids":["piano-source"],
            "valid_from":"2025-06-01T00:00:00+00:00","valid_until":None}]}
    spy=AnswerSpy();spy.structured_result={"claims":[{"text":"I want to practice piano.",
        "claim_type":claim_type,"evidence":[{"kind":"state","id":"piano"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer=answer_generator.generate_answer("What is my goal?",evidence)

    assert answer.endswith("[Sources: state:piano]")


def test_predecessor_claim_can_use_its_same_user_linked_source_memory(monkeypatch):
    evidence={"user_id":"alice","query_category":"HISTORY",
        "retrieval_sources":["Timeline"],
        "requested_timestamp":"2025-06-02T00:00:00+00:00",
        "memories":[{"id":"guitar-source","user_id":"alice",
                     "memory":"I wanted to practice guitar."}],
        "states":[
            {"user_id":"alice","state_id":"guitar","state_type":"GOAL",
             "state_value":"learn guitar","source_memory_ids":["guitar-source"],
             "status":"INACTIVE","valid_from":"2025-01-01T00:00:00+00:00",
             "valid_until":"2025-06-01T00:00:00+00:00"},
            {"user_id":"alice","state_id":"piano","state_type":"GOAL",
             "state_value":"learn piano","status":"ACTIVE",
             "valid_from":"2025-06-01T00:00:00+00:00","valid_until":None}],
        "transitions":[{"user_id":"alice","transition_id":"succession",
            "from_state_id":"guitar","to_state_id":"piano",
            "relation_kind":"TEMPORAL_SUCCESSION",
            "timestamp":"2025-06-01T00:00:00+00:00"}],"causal_path":[]}
    spy=AnswerSpy();spy.structured_result={"claims":[{
        "text":"I wanted to practice guitar before the change.",
        "claim_type":"temporal","evidence":[{"kind":"state","id":"guitar"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer=answer_generator.generate_answer(
        "What goal did I have before I changed it?",evidence)

    assert answer.endswith("[Sources: state:guitar]")


def test_asof_claim_does_not_use_unlinked_or_foreign_memory_for_state_support(monkeypatch):
    for memory in (
        {"id":"other-source","user_id":"alice","memory":"I want to practice piano."},
        {"id":"piano-source","user_id":"bob","memory":"I want to practice piano."},
    ):
        evidence={"user_id":"alice","query_category":"TEMPORAL",
            "requested_timestamp":"2025-06-02T00:00:00+00:00","transitions":[],
            "memories":[memory],"states":[{"user_id":"alice","state_id":"piano",
                "state_type":"GOAL","state_value":"learn piano",
                "source_memory_ids":["piano-source"],
                "valid_from":"2025-06-01T00:00:00+00:00","valid_until":None}]}
        spy=AnswerSpy();spy.structured_result={"claims":[{"text":"I want to practice piano.",
            "claim_type":"temporal","evidence":[{"kind":"state","id":"piano"}]}]}
        monkeypatch.setattr(answer_generator,"llm",spy)

        answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
            "What is my goal?",evidence)

        assert answer==answer_generator.ABSTENTION
        assert diagnostics["rejection_categories"]=={"claim_not_supported_by_source":1}
        assert diagnostics["temporal_validation"][0]["reason_codes"]==[
            "temporal_lexical_support_failed"]


def test_causal_claim_can_use_explicit_transition_endpoints_and_trigger_but_not_endpoints_alone(monkeypatch):
    evidence=_evidence(trigger="a knee injury made running painful",
        reason="I switched from running to swimming because a knee injury made running painful.")
    spy=AnswerSpy();spy.structured_result={"claims":[{"text":
        "I switched from running to swimming because a knee injury made running painful.",
        "claim_type":"causal","evidence":[{"kind":"transition","id":"edge-id"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)
    answer=answer_generator.generate_answer("Why did I switch from running to swimming?",evidence)
    assert "[Sources: transition:edge-id]" in answer

    spy.structured_result={"claims":[{"text":"I switched from running to swimming.",
        "claim_type":"causal","evidence":[{"kind":"transition","id":"edge-id"}]}]}
    assert answer_generator.generate_answer(
        "Why did I switch from running to swimming?",evidence)==answer_generator.ABSTENTION


def test_causal_claim_with_comma_separated_recorded_cause_and_change_is_supported(tmp_path,monkeypatch):
    """The exact recorded reason may place the cause and change in comma-separated clauses."""
    from database import Database

    store=Database(str(tmp_path/"causal-comma.sqlite3"))
    running=State(user_id="alice",state_type="ACTIVITY",state_value="running",
                  source_memory_ids=["running-memory"])
    swimming=State(user_id="alice",state_type="ACTIVITY",state_value="swimming",
                   source_memory_ids=["switch-memory"])
    store.save_state(running)
    store.save_state(swimming)
    edge=Transition(user_id="alice",from_state_id=running.state_id,to_state_id=swimming.state_id,
        relation_kind="EXPLICIT_CAUSE",trigger="my knee injury",
        reason="Due to my knee injury, I switched from running to swimming",
        source_memory_ids=["switch-memory"])
    store.save_transition(edge)
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(causal_graph,"db",store)
    monkeypatch.setattr("temporal_engine.db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _question,_user:[
        {"id":"switch-memory","user_id":"alice",
         "memory":"User switched from running to swimming due to a knee injury"},
        {"id":"running-memory","user_id":"alice","memory":"User enjoys running"},
    ])
    result=hybrid_retriever.retrieve("why i switched to swimming","alice","HISTORY")

    class StructuredAnswerer:
        def __init__(self): self.calls=[]
        def answer_with_provenance(self,question,evidence):
            self.calls.append((question,evidence))
            return {"claims":[{"text":"I switched from running to swimming due to my knee injury.",
                "claim_type":"causal","evidence":[{"kind":"transition","id":edge.transition_id}]}]}

    spy=StructuredAnswerer()
    monkeypatch.setattr(answer_generator,"llm",spy)
    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "why i switched to swimming",result)

    assert result["query_category"]=="HISTORY"
    assert result["transitions"]==[edge.model_dump(mode="json")]
    assert len(spy.calls)==1
    assert answer==("I switched from running to swimming due to my knee injury. "
                    "[Sources: transition:"+edge.transition_id+"]")
    assert diagnostics["causal_gate_passed"] is True
    assert diagnostics["accepted_claim_count"]==1
    assert diagnostics["rejection_categories"]=={}


@pytest.mark.parametrize(("reference_kind","accepted"),[("transition",True),("memory",False)])
def test_retrieved_cause_requires_model_to_cite_transition_not_memory(
        reference_kind,accepted,tmp_path,monkeypatch):
    """A cause-present retrieval only authorizes claims citing its transition record."""
    store=Database(str(tmp_path/"cause-citation-routing.sqlite3"))
    running=State(user_id="alice",state_type="ACTIVITY",state_value="running",
                  status="INACTIVE",source_memory_ids=["running-memory"])
    swimming=State(user_id="alice",state_type="ACTIVITY",state_value="swimming",
                   source_memory_ids=["switch-memory"])
    store.save_state(running); store.save_state(swimming)
    edge=Transition(user_id="alice",from_state_id=running.state_id,to_state_id=swimming.state_id,
        relation_kind="EXPLICIT_CAUSE",trigger="my knee injury",
        reason="Due to my knee injury, I switched from running to swimming",
        source_memory_ids=["switch-memory"])
    store.save_transition(edge)
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(causal_graph,"db",store)
    monkeypatch.setattr("temporal_engine.db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _question,_user:[
        {"id":"switch-memory","user_id":"alice",
         "memory":"User switched from running to swimming due to a knee injury"},
        {"id":"running-memory","user_id":"alice","memory":"User enjoys running"},
    ])
    question="Why did I switch to swimming?"
    evidence=hybrid_retriever.retrieve(question,"alice","HISTORY")
    assert evidence["transitions"]==[edge.model_dump(mode="json")]

    spy=AnswerSpy()
    spy.structured_result={"claims":[{
        "text":"I switched from running to swimming due to my knee injury.",
        "claim_type":"causal",
        "evidence":[{"kind":reference_kind,
                     "id":edge.transition_id if reference_kind=="transition" else "switch-memory"}],
    }]}
    monkeypatch.setattr(answer_generator,"llm",spy)
    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(question,evidence)

    assert diagnostics["causal_gate_passed"] is True
    assert diagnostics["accepted_claim_count"]==int(accepted)
    assert diagnostics["causal_transition_citation_accepted"] is accepted
    if accepted:
        assert "knee injury" in answer
        assert diagnostics["rejection_categories"]=={}
    else:
        assert answer==answer_generator.ABSTENTION
        assert diagnostics["rejection_categories"]=={"evidence_not_in_model_bundle":1}
        assert diagnostics["citation_validation"][0]["reason_code"]=="citation_not_in_model_bundle"
        assert diagnostics["rejected_claims"][0]["rejection_reasons"]==[
            "evidence_not_in_model_bundle"]


def test_causal_answer_routing_uses_transition_evidence_when_memories_compete(tmp_path,monkeypatch):
    """Reproduce a model choosing a personal-memory citation over an available cause edge."""
    store=Database(str(tmp_path/"causal-evidence-routing.sqlite3"))
    running=State(user_id="alice",state_type="ACTIVITY",state_value="running",
                  status="INACTIVE",source_memory_ids=["running-memory"])
    swimming=State(user_id="alice",state_type="ACTIVITY",state_value="swimming",
                   source_memory_ids=["switch-memory"])
    store.save_state(running); store.save_state(swimming)
    edge=Transition(user_id="alice",from_state_id=running.state_id,to_state_id=swimming.state_id,
        relation_kind="EXPLICIT_CAUSE",trigger="my knee injury",
        reason="Due to my knee injury, I switched from running to swimming",
        source_memory_ids=["switch-memory"])
    store.save_transition(edge)
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(causal_graph,"db",store)
    monkeypatch.setattr("temporal_engine.db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _question,_user:[
        {"id":"switch-memory","user_id":"alice",
         "memory":"User switched from running to swimming due to a knee injury"},
        {"id":"running-memory","user_id":"alice","memory":"User enjoys running"},
    ])
    question="Why did I switch to swimming?"
    evidence=hybrid_retriever.retrieve(question,"alice","HISTORY")

    class SourceChoosingModel:
        def __init__(self): self.calls=[]
        def answer_with_provenance(self,_question,model_evidence):
            self.calls.append(model_evidence)
            # Deterministic stand-in for the observed citation choice: when
            # general evidence competes, the model cites the matching memory;
            # with transition-only evidence, it cites the causal edge.
            if model_evidence.get("memories") or model_evidence.get("states"):
                reference={"kind":"memory","id":"switch-memory"}
            else:
                reference={"kind":"transition","id":edge.transition_id}
            return {"claims":[{
                "text":"I switched from running to swimming due to my knee injury.",
                "claim_type":"causal","evidence":[reference],
            }]}

    spy=SourceChoosingModel()
    monkeypatch.setattr(answer_generator,"llm",spy)
    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(question,evidence)

    assert len(spy.calls)==1
    assert spy.calls[0].get("memories")==[]
    assert spy.calls[0].get("states")==[]
    assert spy.calls[0]["transitions"][0]["transition_id"]==edge.transition_id
    assert diagnostics["causal_gate_passed"] is True
    assert diagnostics["causal_transition_citation_accepted"] is True
    assert diagnostics["accepted_claim_count"]==1
    assert "knee injury" in answer


def test_direct_change_and_typed_cause_questions_accept_transition_claims(monkeypatch):
    evidence={
        "user_id":"alice","query_category":"EVOLUTION",
        "memories":[],"causal_path":[],
        "states":[
            {"user_id":"alice","state_id":"running","state_type":"INTEREST",
             "state_value":"running","status":"ACTIVE"},
            {"user_id":"alice","state_id":"swimming","state_type":"ACTIVITY",
             "state_value":"swimming","status":"ACTIVE"},
        ],
        "transitions":[{
            "user_id":"alice","transition_id":"edge","from_state_id":"running",
            "to_state_id":"swimming","transition_type":"ACTIVITY_CHANGE",
            "relation_kind":"EXPLICIT_CAUSE","trigger":"my knee injury",
            "reason":"Due to my knee injury, I switched from running to swimming",
            "timestamp":"2025-01-01T00:00:00+00:00",
        }],
    }
    spy=AnswerSpy()
    spy.structured_result={"claims":[{"text":"You switched from running to swimming.",
        "claim_type":"temporal","evidence":[{"kind":"transition","id":"edge"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "What activity did I switch to?",evidence)

    assert answer=="You switched from running to swimming. [Sources: transition:edge]"
    assert diagnostics["accepted_claim_count"]==1

    spy.structured_result={"claims":[{
        "text":"A knee injury caused the switch from running to swimming.",
        "claim_type":"causal","evidence":[{"kind":"transition","id":"edge"}]}]}
    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "What caused my change in activity?",evidence)

    assert "knee injury" in answer
    assert diagnostics["causal_gate_passed"] is True
    assert diagnostics["accepted_claim_count"]==1


def test_live_swimming_transition_shape_accepts_matching_transition_citation(monkeypatch):
    evidence={
        "user_id":"default","query_category":"HISTORY",
        "memories":[{"id":"52449090-01db-4bf8-a1b1-ecf3c79371d8",
                     "user_id":"default",
                     "memory":"User switched from running to swimming due to a knee injury"}],
        "states":[
            {"user_id":"default","state_id":"e7ac2228-a4be-4625-bc69-5385610d46f9",
             "state_type":"INTEREST","state_value":"running",
             "source_memory_ids":["51767945-6c5b-4b89-b281-84ecfe8d49f0"]},
            {"user_id":"default","state_id":"cb369dfb-f9b2-4864-9d50-e7faca1ec8f7",
             "state_type":"ACTIVITY","state_value":"swimming",
             "source_memory_ids":["52449090-01db-4bf8-a1b1-ecf3c79371d8"]},
        ],
        "transitions":[{
            "transition_id":"9d9302ed-2d3e-43dc-b341-a55539d4342a",
            "user_id":"default",
            "from_state_id":"e7ac2228-a4be-4625-bc69-5385610d46f9",
            "to_state_id":"cb369dfb-f9b2-4864-9d50-e7faca1ec8f7",
            "relation_kind":"EXPLICIT_CAUSE","transition_type":"ACTIVITY_CHANGE",
            "trigger":"my knee injury",
            "reason":"Due to my knee injury, I switched from running to swimming",
            "source_memory_ids":["52449090-01db-4bf8-a1b1-ecf3c79371d8"],
            "timestamp":"2026-10-22T15:56:44.392351Z",
        }],
        "causal_path":[{
            "transition_id":"9d9302ed-2d3e-43dc-b341-a55539d4342a",
            "user_id":"default",
            "from_state_id":"e7ac2228-a4be-4625-bc69-5385610d46f9",
            "to_state_id":"cb369dfb-f9b2-4864-9d50-e7faca1ec8f7",
            "relation_kind":"EXPLICIT_CAUSE","from":"running","to":"swimming",
        }],
    }
    transition_id=evidence["transitions"][0]["transition_id"]
    spy=AnswerSpy(); spy.structured_result={"claims":[{
        "text":"Due to my knee injury, I switched from running to swimming",
        "claim_type":"causal","evidence":[{"kind":"transition","id":transition_id}],
    }]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "Why did I switch to swimming?",evidence)

    assert answer.endswith("[Sources: transition:"+transition_id+"]")
    assert diagnostics["causal_gate_passed"] is True
    assert diagnostics["causal_transition_citation_accepted"] is True
    assert diagnostics["answer_reason"]=="validated_claims"
    supplied=spy.calls[0][1]["transitions"][0]
    assert supplied["transition_id"]==transition_id
    assert supplied["from"]=="running" and supplied["to"]=="swimming"
    assert supplied["trigger"]=="my knee injury"
    assert supplied["reason"]==evidence["transitions"][0]["reason"]


def test_causal_prompt_example_wording_is_supported_by_explicit_cause(monkeypatch):
    evidence=_evidence(trigger="my knee injury",
        reason="Due to my knee injury, I switched from running to swimming")
    spy=AnswerSpy(); spy.structured_result={"claims":[{
        "text":"The recorded reason for the change was a knee injury",
        "claim_type":"causal","evidence":[{"kind":"transition","id":"edge-id"}],
    }]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "Why did I switch to swimming?",evidence)

    assert answer=="You switched from running to swimming because of my knee injury. [Sources: transition:edge-id]"
    assert diagnostics["causal_transition_citation_accepted"] is True
    assert diagnostics["answer_reason"]=="validated_claims"


def test_causal_transition_citation_id_mismatch_fails_closed_without_id_diagnostics(monkeypatch):
    evidence=_evidence(reason="Due to my knee injury, I switched from running to swimming")
    spy=AnswerSpy(); spy.structured_result={"claims":[{
        "text":"Due to my knee injury, I switched from running to swimming",
        "claim_type":"causal","evidence":[{"kind":"transition","id":"wrong-transition-id"}],
    }]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "Why did I switch to swimming?",evidence)
    summary=answer_generator.safe_answer_diagnostic_summary(diagnostics)

    assert answer==answer_generator.ABSTENTION
    assert diagnostics["rejection_categories"]=={"unknown_or_unowned_evidence_id":1}
    assert summary["required_transition_citation_accepted"] is False
    assert "wrong-transition-id" not in repr(summary)


def test_causal_transition_with_foreign_endpoint_is_not_owned_evidence(monkeypatch):
    evidence=_evidence(reason="Due to my knee injury, I switched from running to swimming")
    evidence["states"].append({"user_id":"bob","state_id":"bob-swimming",
                               "state_value":"swimming"})
    evidence["transitions"][0]["to_state_id"]="bob-swimming"
    # Another valid edge keeps the global causal gate open while the cited
    # edge fails endpoint ownership resolution.
    evidence["states"].extend([
        {"user_id":"alice","state_id":"other-running","state_value":"running"},
        {"user_id":"alice","state_id":"other-swimming","state_value":"swimming"},
    ])
    evidence["transitions"].append({
        "user_id":"alice","transition_id":"valid-gate-edge",
        "from_state_id":"other-running","to_state_id":"other-swimming",
        "relation_kind":"EXPLICIT_CAUSE","trigger":"knee injury",
        "reason":"knee injury explains running to swimming",
    })
    spy=AnswerSpy(); spy.structured_result={"claims":[{
        "text":"Due to my knee injury, I switched from running to swimming",
        "claim_type":"causal","evidence":[{"kind":"transition","id":"edge-id"}],
    }]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "Why did I switch to swimming?",evidence)
    summary=answer_generator.safe_answer_diagnostic_summary(diagnostics)

    assert answer==answer_generator.ABSTENTION
    assert diagnostics["causal_gate_passed"] is True
    assert diagnostics["rejection_categories"]=={"unknown_or_unowned_evidence_id":1}
    assert diagnostics["rejected_claims"][0]["rejection_reasons"]==[
        "unknown_or_unowned_evidence_id"]
    assert "bob-swimming" not in repr(summary)


@pytest.mark.parametrize(("relation","trigger","reason","expected_reason"),[
    ("TEMPORAL_SUCCESSION","knee injury","switched from running to swimming",
     "causal_relation_not_explicit"),
    ("EXPLICIT_CAUSE",None,None,"causal_recorded_cause_missing"),
    ("EXPLICIT_CAUSE","schedule changed","switched from running to swimming",
     "causal_claim_missing_cause_terms"),
    ("EXPLICIT_CAUSE","knee injury","switched from running to swimming",
     "causal_claim_lexical_support_failed"),
])
def test_causal_transition_rejection_reasons_identify_failed_support_predicate(
        relation,trigger,reason,expected_reason,monkeypatch):
    evidence=_evidence(relation=relation,trigger=trigger,reason=reason)
    # Keep the global gate satisfied for noncausal/empty candidate edges so
    # validation reaches and reports the specific cited-edge predicate.
    if relation!="EXPLICIT_CAUSE" or not (trigger or reason):
        evidence["states"].extend([
            {"user_id":"alice","state_id":"other-running","state_value":"running"},
            {"user_id":"alice","state_id":"other-swimming","state_value":"swimming"},
        ])
        evidence["transitions"].append({
            "user_id":"alice","transition_id":"valid-gate-edge",
            "from_state_id":"other-running","to_state_id":"other-swimming",
            "relation_kind":"EXPLICIT_CAUSE","trigger":"knee injury",
            "reason":"knee injury explains running to swimming",
        })
    cited_id="edge-id"
    spy=AnswerSpy(); spy.structured_result={"claims":[{
        "text":("I switched from running to swimming because of a knee injury"
                if expected_reason!="causal_claim_lexical_support_failed"
                else "I switched from running to swimming because my knee injury caused dizziness"),
        "claim_type":"causal","evidence":[{"kind":"transition","id":cited_id}],
    }]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "Why did running change to swimming?",evidence)

    assert answer==answer_generator.ABSTENTION
    transition_was_supplied=(relation=="EXPLICIT_CAUSE" and bool(trigger or reason))
    if transition_was_supplied:
        assert diagnostics["rejection_categories"]=={"causal_relation_or_support_rejected":1}
        assert diagnostics["rejected_claims"][0]["rejection_reasons"]==[expected_reason]
    else:
        assert diagnostics["rejection_categories"]=={"evidence_not_in_model_bundle":1}
        assert diagnostics["citation_validation"][0]["reason_code"]=="citation_not_in_model_bundle"
    # The model-facing projection properly omits this invalid/non-causal edge.
    # Exercise the validator predicate directly as well, to retain coverage of
    # the relation/cause checks when the cited edge is available to the model.
    rejection_counts={}; rejection_details={}; rejected_claims=[]
    validated=answer_generator._validate_claims(
        "Why did running change to swimming?",evidence,spy.structured_result,
        rejection_counts,rejection_details,rejected_claims,[],evidence,[])
    assert validated==[]
    if transition_was_supplied:
        assert rejection_counts=={"causal_relation_or_support_rejected":1}
        assert rejected_claims[0]["rejection_reasons"]==[expected_reason]
    else:
        assert rejection_counts=={"causal_relation_or_support_rejected":1}
        assert rejected_claims[0]["rejection_reasons"]==[expected_reason]
    safe=answer_generator.safe_answer_diagnostic_summary(diagnostics)
    expected_ui_reason=(expected_reason if transition_was_supplied else
                        "evidence_not_in_model_bundle")
    assert safe["rejection_reasons"]=={expected_ui_reason:1}
    assert "edge-id" not in repr(safe)
    assert "knee injury" not in repr(safe)


def test_causal_ui_diagnostics_report_gate_and_owned_transition_citation(monkeypatch):
    evidence=_evidence(reason="Due to my knee injury, I switched from running to swimming")
    spy=AnswerSpy()
    spy.structured_result={"claims":[{"text":"I switched from running to swimming due to my knee injury.",
        "claim_type":"causal","evidence":[{"kind":"transition","id":"edge-id"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "why i switched to swimming",evidence)
    summary=answer_generator.safe_answer_diagnostic_summary(diagnostics)

    assert "[Sources: transition:edge-id]" in answer
    assert diagnostics["causal_gate_passed"] is True
    assert diagnostics["causal_transition_citation_accepted"] is True
    assert summary["causal_gate_passed"] is True
    assert summary["required_transition_citation_accepted"] is True
    assert summary["answer_reason"]=="validated_claims"
    assert "edge-id" not in repr(summary)
    assert "knee injury" not in repr(summary)
    assert "swimming" not in repr(summary)


@pytest.mark.parametrize(("reference","expected_category"),[
    (None,"no_valid_evidence_reference"),
    ({"kind":"transition","id":"missing-edge"},"unknown_or_unowned_evidence_id"),
    ({"kind":"memory","id":"cause-memory"},"evidence_not_in_model_bundle"),
])
def test_causal_ui_diagnostics_reject_missing_invalid_or_memory_only_citation(
        reference,expected_category,monkeypatch):
    evidence=_evidence(reason="Due to my knee injury, I switched from running to swimming")
    evidence["memories"]=[{"id":"cause-memory","user_id":"alice",
                           "memory":"User switched from running to swimming due to a knee injury"}]
    refs=[] if reference is None else [reference]
    spy=AnswerSpy()
    spy.structured_result={"claims":[{"text":"I switched from running to swimming due to my knee injury.",
        "claim_type":"causal","evidence":refs}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "why i switched to swimming",evidence)
    summary=answer_generator.safe_answer_diagnostic_summary(diagnostics)

    assert answer==answer_generator.ABSTENTION
    assert diagnostics["causal_gate_passed"] is True
    assert diagnostics["causal_transition_citation_accepted"] is False
    assert diagnostics["rejection_categories"]=={expected_category:1}
    assert summary["required_transition_citation_accepted"] is False
    assert summary["rejection_categories"]=={expected_category:1}
    assert summary["rejection_reasons"]
    assert set(summary["rejection_reasons"]).issubset(
        answer_generator._UI_REJECTION_REASON_CODES)
    assert "missing-edge" not in repr(summary)
    assert "cause-memory" not in repr(summary)
    assert "knee injury" not in repr(summary)


def test_causal_ui_diagnostics_report_fail_closed_gate_without_claim_content(monkeypatch):
    evidence=_evidence(relation="TEMPORAL_SUCCESSION",trigger=None,reason=None)
    spy=AnswerSpy()
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "why i switched to swimming",evidence)
    summary=answer_generator.safe_answer_diagnostic_summary(diagnostics)

    assert answer==answer_generator.ABSTENTION
    assert diagnostics["causal_gate_passed"] is False
    assert diagnostics["causal_transition_citation_accepted"] is False
    assert diagnostics["model_response_status"]=="not_called"
    assert summary["answer_reason"]=="explicit_cause_gate_rejected"
    assert summary["required_transition_citation_accepted"] is False
    assert spy.calls==[]


def test_safe_answer_diagnostic_summary_allowlists_fields_and_reason_codes():
    summary=answer_generator.safe_answer_diagnostic_summary({
        "causal_gate_passed":True,"causal_transition_citation_accepted":False,
        "answer_reason":"validated_claims","model_response_status":"structured_response",
        "accepted_claim_count":1,"rejected_claim_count":1,
        "rejection_categories":{"causal_relation_or_support_rejected":1,
                               "private claim text":99},
        "rejection_details":{"causal_relation_or_support_rejected":1,
                             "private claim text":88},
        "rejected_claims":[{"claim_index":0,"claim_type":"causal",
            "rejection_reasons":["causal_relation_or_support_rejected","private claim text"],
            "claim_text":"private claim text","evidence_text":"private evidence",
            "prompt":"private prompt","memory_id":"private-id",
            "evidence_references":[{"id":"private-id","text":"private evidence"}]}],
        "claim_text":"private claim text","evidence_text":"private evidence",
        "prompt":"private prompt","raw_model_response":{"secret":"hidden"},
        "transition_id":"private-id"})

    assert summary=={
        "causal_gate_passed":True,"required_transition_citation_accepted":False,
        "answer_reason":"validated_claims","model_response_status":"structured_response",
        "accepted_claim_count":1,"rejected_claim_count":1,
        "rejection_categories":{"causal_relation_or_support_rejected":1},
        "rejection_reasons":{"causal_relation_or_support_rejected":1},
    }
    assert "private" not in repr(summary)


def test_safe_answer_diagnostic_summary_exposes_rejection_reason_codes_without_claim_data():
    summary=answer_generator.safe_answer_diagnostic_summary({
        "rejected_claim_count":1,
        "rejection_categories":{"claim_not_supported_by_source":1},
        "rejected_claims":[{
            "claim_index":0,"claim_type":"fact",
            "rejection_reasons":["fact_lexical_support_failed"],
            "text":"I live in Toronto","evidence_text":"User lives in Toronto",
            "prompt":"private prompt","raw_model_response":"private response",
            "user_id":"alice","memory_id":"private-memory-id",
            "transition_id":"private-transition-id",
        }],
    })

    assert summary["rejection_reasons"]=={"fact_lexical_support_failed":1}
    rendered=repr(summary)
    for sensitive in ("Toronto","private","alice","memory-id","transition-id"):
        assert sensitive not in rendered


@pytest.mark.parametrize("claim,reference,owner",[
    ("I switched from running to swimming due to my knee injury.",
     {"kind":"memory","id":"switch-memory"},"alice"),
    ("I switched from running to swimming due to a broken ankle.",
     {"kind":"transition","id":"edge-id"},"alice"),
    ("I switched from running to swimming due to my knee injury.",
     {"kind":"transition","id":"edge-id"},"bob"),
])
def test_causal_claim_rejects_memory_only_unsupported_or_foreign_transition(
        claim,reference,owner,monkeypatch):
    evidence=_evidence(reason="Due to my knee injury, I switched from running to swimming")
    evidence["memories"]=[{"id":"switch-memory","user_id":"alice",
                           "memory":"User switched from running to swimming due to a knee injury"}]
    evidence["transitions"][0]["user_id"]=owner
    spy=AnswerSpy()
    spy.structured_result={"claims":[{"text":claim,"claim_type":"causal","evidence":[reference]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer=answer_generator.generate_answer("why i switched to swimming",evidence)

    assert answer==answer_generator.ABSTENTION
    if owner!="alice":
        assert spy.calls==[]
    else:
        assert len(spy.calls)==1


def test_temporal_claim_does_not_accept_foreign_memory_link(monkeypatch):
    evidence={"user_id":"alice","query_category":"TEMPORAL",
        "requested_timestamp":"2025-06-02T00:00:00+00:00",
        "memories":[{"id":"piano-memory","user_id":"bob","memory":"User wants to learn piano."}],
        "states":[{"user_id":"alice","state_id":"piano","state_type":"GOAL",
            "state_value":"learn piano","source_memory_ids":["piano-memory"],
            "valid_from":"2025-06-01T00:00:00+00:00","valid_until":None}],
        "transitions":[],"causal_path":[]}
    spy=AnswerSpy();spy.structured_result={"claims":[{"text":"My goal is to learn piano.",
        "claim_type":"temporal","evidence":[{"kind":"memory","id":"piano-memory"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)
    assert answer_generator.generate_answer("What is my goal?",evidence)==answer_generator.ABSTENTION


def test_existing_empty_evidence_abstention_and_noncausal_answer_path(monkeypatch):
    spy=AnswerSpy();monkeypatch.setattr(answer_generator,"llm",spy)

    assert answer_generator.generate_answer("What do I enjoy?",{"memories":[],"states":[]})==answer_generator.ABSTENTION
    assert spy.calls==[]
    evidence={"user_id":"alice","query_category":"STATE","memories":[],
              "states":[{"user_id":"alice","state_id":"running-id","state_value":"running"}]}
    spy.result="running"
    assert answer_generator.generate_answer("What is my state?",evidence)=="running [Sources: state:running-id]"
    assert len(spy.calls)==1


def test_llm_abstention_does_not_invent_java_from_learning_query(monkeypatch):
    spy=AnswerSpy(answer_generator.ABSTENTION)
    monkeypatch.setattr(answer_generator,"llm",spy)
    evidence={
        "user_id":"alice","query_category":"PREFERENCE",
        "memories":[{"id":"spanish-memory","user_id":"alice","memory":"I prefer Spanish conversation."}],
        "states":[{"user_id":"alice","state_id":"spanish","state_type":"PREFERENCE",
                   "state_value":"Spanish conversation","status":"ACTIVE"}],
    }

    answer=answer_generator.generate_answer("What do I prefer learning?",evidence)

    assert answer==answer_generator.ABSTENTION
    assert spy.calls==[("What do I prefer learning?",evidence)]


def test_java_keyword_without_supporting_preference_still_abstains(monkeypatch):
    spy=AnswerSpy(answer_generator.ABSTENTION)
    monkeypatch.setattr(answer_generator,"llm",spy)
    evidence={"user_id":"alice","query_category":"PREFERENCE",
              "memories":[{"id":"question-memory","user_id":"alice",
                           "memory":"I asked how Java streams work."}],
              "states":[],"transitions":[],"causal_path":[]}

    answer=answer_generator.generate_answer("Do I prefer Java?",evidence)

    assert answer==answer_generator.ABSTENTION
    assert len(spy.calls)==1


def test_supported_java_preference_uses_normal_llm_answer_path(monkeypatch):
    expected="You prefer Java for backend projects."
    spy=AnswerSpy(expected)
    monkeypatch.setattr(answer_generator,"llm",spy)
    evidence={
        "user_id":"alice","query_category":"PREFERENCE",
        "memories":[{"id":"java-pref","user_id":"alice",
                     "memory":"I prefer Java for backend projects."}],
        "states":[{"user_id":"alice","state_id":"java-pref-state",
                   "state_type":"PREFERENCE","state_value":"Java for backend projects",
                   "status":"ACTIVE","source_memory_ids":["java-pref"]}],
    }

    answer=answer_generator.generate_answer("What language do I prefer?",evidence)
    assert answer==expected+" [Sources: memory:java-pref]"
    assert spy.calls==[("What language do I prefer?",evidence)]


def test_retrieved_mem0_citation_survives_retrieval_with_unlinked_other_state(tmp_path,monkeypatch):
    """A state without provenance must not make a separately owned Mem0 record uncitable."""
    store=Database(str(tmp_path/"favorite_language.sqlite3"))
    memory_id="3f1bc75f-95d4-432b-ac1f-8e69b43b33b3"
    state_id="14fa06cd-c72e-42d0-8125-6f23fd1d978f"
    store.save_state(State(user_id="default",state_id=state_id,state_type="OTHER",
        state_value="My favorite programming language is Java.",status="ACTIVE",
        source_memory_ids=[]))
    record={"id":memory_id,"memory":"User prefers Java","user_id":"default",
            "attributed_to":"user"}
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _query,_user:[record])
    evidence=hybrid_retriever.retrieve(
        "What is my favorite programming language?","default","FACTUAL")

    spy=AnswerSpy()
    spy.structured_result={"claims":[{"text":"You prefer Java","claim_type":"fact",
        "evidence":[{"kind":"memory","id":memory_id}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "What is my favorite programming language?",evidence)

    assert evidence["memories"]==[record]
    assert evidence["states"][0]["state_id"]==state_id
    assert evidence["states"][0]["source_memory_ids"]==[]
    assert "state_focus" not in evidence
    assert spy.calls[0][1]["memories"]==[record]
    assert answer==f"You prefer Java [Sources: memory:{memory_id}]"
    assert diagnostics["accepted_claim_count"]==1
    assert diagnostics["answer_reason"]=="validated_claims"


def test_short_citation_ref_resolves_long_owned_memory_for_favorite_language(monkeypatch):
    """The model may cite a short token instead of transcribing a long UUID."""
    memory_id="3f1bc75f-95d4-432b-ac1f-8e69b43b33b3"
    state_id="14fa06cd-c72e-42d0-8125-6f23fd1d978f"
    evidence={
        "user_id":"default","query_category":"FACTUAL",
        "memories":[{"id":memory_id,"user_id":"default","memory":"User prefers Java"}],
        "states":[{"user_id":"default","state_id":state_id,"state_type":"OTHER",
                   "state_value":"My favorite programming language is Java.",
                   "status":"ACTIVE","source_memory_ids":[]}],
        "transitions":[],"causal_path":[],
    }
    spy=AnswerSpy()
    spy.structured_result={"claims":[{"text":"You prefer Java","claim_type":"fact",
        "evidence":[{"kind":"memory","ref":"M1"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "What is my favorite programming language?",evidence)

    assert answer==f"You prefer Java [Sources: memory:{memory_id}]"
    assert diagnostics["accepted_claim_count"]==1
    assert diagnostics["rejection_categories"]=={}
    summary=answer_generator.safe_answer_diagnostic_summary(diagnostics)
    assert memory_id not in repr(summary)
    assert state_id not in repr(summary)


@pytest.mark.parametrize("question",[
    "What is my favorite food?",
    "What is my favorite movie?",
])
def test_favorite_focus_must_be_supported_by_cited_evidence_or_owned_bridge(
        monkeypatch,question):
    memory_id="java-memory"
    evidence={
        "user_id":"default","query_category":"HISTORY",
        "memories":[
            {"id":memory_id,"user_id":"default","memory":"User prefers Java"},
            {"id":"running-memory","user_id":"default","memory":"User enjoys running"},
            {"id":"switch-memory","user_id":"default",
             "memory":"User switched from running to swimming due to a knee injury"},
        ],
        "states":[{"user_id":"default","state_id":"java-state","state_type":"OTHER",
                   "state_value":"My favorite programming language is Java.",
                   "status":"ACTIVE","source_memory_ids":[]}],
        "transitions":[],"causal_path":[],
    }
    spy=AnswerSpy()
    spy.structured_result={"claims":[{"text":"User prefers Java","claim_type":"fact",
        "evidence":[{"kind":"memory","id":memory_id}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(question,evidence)

    assert answer==answer_generator.ABSTENTION
    assert diagnostics["rejection_categories"]=={"claim_not_supported_by_source":1}
    assert diagnostics["rejected_claims"][0]["rejection_reasons"]==[
        "claim_not_relevant_to_question"]


def test_short_citation_ref_resolves_long_state_id_without_provenance(monkeypatch):
    state_id="14fa06cd-c72e-42d0-8125-6f23fd1d978f"
    evidence={
        "user_id":"default","query_category":"FACTUAL",
        "memories":[],"transitions":[],"causal_path":[],
        "states":[{"user_id":"default","state_id":state_id,"state_type":"OTHER",
                   "state_value":"My favorite programming language is Java.",
                   "status":"ACTIVE","source_memory_ids":[]}],
    }
    spy=AnswerSpy()
    spy.structured_result={"claims":[{
        "text":"My favorite programming language is Java.","claim_type":"fact",
        "evidence":[{"kind":"state","ref":"S1"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "What is my favorite programming language?",evidence)

    assert answer==f"My favorite programming language is Java. [Sources: state:{state_id}]"
    assert diagnostics["accepted_claim_count"]==1
    assert diagnostics["rejection_categories"]=={}


@pytest.mark.parametrize(("reference","expected_detail"),[
    ({"kind":"state","ref":"S9"},"citation_ref_unknown"),
    ({"kind":"state","ref":"S1","id":"different-id"},"citation_ref_conflict"),
    ({"kind":"memory","ref":"S1"},"citation_ref_conflict"),
    ({"kind":"state","ref":""},"evidence_reference_id_missing"),
])
def test_short_citation_ref_unknown_conflicting_or_malformed_fails_closed(
        monkeypatch,reference,expected_detail):
    evidence={"user_id":"alice","query_category":"FACTUAL","transitions":[],
        "states":[{"user_id":"alice","state_id":"java-state","state_type":"PREFERENCE",
                   "state_value":"Java","status":"ACTIVE","source_memory_ids":[]}],
        "memories":[{"id":"java-memory","user_id":"alice","memory":"User prefers Java"}]}
    spy=AnswerSpy()
    spy.structured_result={"claims":[{"text":"You prefer Java","claim_type":"fact",
        "evidence":[reference]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "What language do I prefer?",evidence)

    assert answer==answer_generator.ABSTENTION
    assert diagnostics["rejection_categories"]==(
        {"malformed_evidence_reference":1}
        if expected_detail=="evidence_reference_id_missing"
        else {"unknown_or_unowned_evidence_id":1})
    assert diagnostics["rejection_details"]=={expected_detail:1}
    summary=answer_generator.safe_answer_diagnostic_summary(diagnostics)
    assert expected_detail in summary["rejection_reasons"]
    assert "S1" not in repr(summary) and "S9" not in repr(summary)


def test_short_citation_ref_cannot_select_evidence_omitted_from_model_bundle(monkeypatch):
    evidence={"user_id":"alice","query_category":"PREFERENCE","transitions":[],
        "states":[{"user_id":"alice","state_id":"preference-state",
                   "state_type":"PREFERENCE","state_value":"tea",
                   "source_memory_ids":["shown-memory"]}],
        "memories":[
            {"id":"shown-memory","user_id":"alice","memory":"I enjoy tea."},
            {"id":"omitted-memory","user_id":"alice","memory":"I prefer tea."},
        ],
        "state_focus":{"state_types":["PREFERENCE"],
                       "candidate_state_ids":["preference-state"],
                       "source_memory_ids":["shown-memory"]}}
    spy=AnswerSpy()
    spy.structured_result={"claims":[{"text":"I prefer tea","claim_type":"fact",
        "evidence":[{"kind":"memory","ref":"M2"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "What do I prefer?",evidence)

    assert [item["id"] for item in spy.calls[0][1]["memories"]]==["shown-memory"]
    assert answer==answer_generator.ABSTENTION
    assert diagnostics["rejection_categories"]=={"unknown_or_unowned_evidence_id":1}
    assert diagnostics["rejection_details"]=={"citation_ref_unknown":1}


def test_retrieved_fact_citation_rejects_unknown_and_foreign_memory_ids(monkeypatch):
    evidence={"user_id":"alice","query_category":"FACTUAL","states":[],"transitions":[],
        "memories":[{"id":"owned-java","user_id":"alice","memory":"User prefers Java"}]}
    spy=AnswerSpy()
    monkeypatch.setattr(answer_generator,"llm",spy)

    spy.structured_result={"claims":[{"text":"You prefer Java","claim_type":"fact",
        "evidence":[{"kind":"memory","id":"not-retrieved"}]}]}
    answer,unknown=answer_generator.generate_answer_with_diagnostics(
        "What language do I prefer?",evidence)
    assert answer==answer_generator.ABSTENTION
    assert unknown["rejection_categories"]=={"unknown_or_unowned_evidence_id":1}

    evidence["memories"].append({"id":"bob-java","user_id":"bob",
                                  "memory":"User prefers Java"})
    spy.structured_result={"claims":[{"text":"You prefer Java","claim_type":"fact",
        "evidence":[{"kind":"memory","id":"bob-java"}]}]}
    answer,foreign=answer_generator.generate_answer_with_diagnostics(
        "What language do I prefer?",evidence)
    assert answer==answer_generator.ABSTENTION
    assert foreign["rejection_categories"]=={"unknown_or_unowned_evidence_id":1}


@pytest.mark.parametrize(("reference","extra_memories","expected_category","expected_reason",
                          "resolved","kind_matches"),[
    ({"kind":"memory","id":"not-in-evidence"},[],
     "unknown_or_unowned_evidence_id","citation_id_unknown",False,None),
    ({"kind":"state","id":"java-memory"},[],
     "unknown_or_unowned_evidence_id","citation_kind_mismatch",True,False),
    ({"kind":"memory","id":"foreign-java"},
     [{"id":"foreign-java","user_id":"bob","memory":"User prefers Java"}],
     "unknown_or_unowned_evidence_id","citation_foreign_user",False,None),
    ({"kind":"memory"},[],
     "malformed_evidence_reference","citation_structure_malformed",False,None),
])
def test_citation_diagnostics_classify_unknown_mismatched_foreign_and_malformed(
        monkeypatch,reference,extra_memories,expected_category,expected_reason,
        resolved,kind_matches):
    evidence={"user_id":"alice","query_category":"FACTUAL","states":[],"transitions":[],
        "memories":[{"id":"java-memory","user_id":"alice","memory":"User prefers Java"},
                    *extra_memories]}
    spy=AnswerSpy()
    spy.structured_result={"claims":[{"text":"You prefer Java","claim_type":"fact",
        "evidence":[reference]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "What language do I prefer?",evidence)

    assert answer==answer_generator.ABSTENTION
    assert diagnostics["rejection_categories"]=={expected_category:1}
    assert diagnostics["citation_validation"]==[{
        "citation_kind":reference.get("kind") if reference.get("kind") in {
            "memory","state","transition"} else "invalid",
        "evidence_id_resolves_owned":resolved,
        "kind_matches_resolved_type":kind_matches,
        "reason_code":expected_reason,
    }]
    summary=answer_generator.safe_answer_diagnostic_summary(diagnostics)
    assert summary["citation_validation"]==diagnostics["citation_validation"]
    rendered=repr(summary)
    for sensitive in ("java-memory","foreign-java","User prefers Java","You prefer Java","alice","bob"):
        assert sensitive not in rendered


def test_citation_not_supplied_to_model_is_rejected_and_diagnosed(monkeypatch):
    """Validation must not accept an owned record omitted from the model's evidence view."""
    evidence={"user_id":"alice","query_category":"PREFERENCE","transitions":[],
        "states":[{"user_id":"alice","state_id":"preference-state",
                   "state_type":"PREFERENCE","state_value":"tea",
                   "source_memory_ids":["shown-memory"]}],
        "memories":[
            {"id":"shown-memory","user_id":"alice","memory":"I enjoy tea."},
            {"id":"omitted-memory","user_id":"alice","memory":"I prefer tea."},
        ],
        "state_focus":{"state_types":["PREFERENCE"],
                       "candidate_state_ids":["preference-state"],
                       "source_memory_ids":["shown-memory"]}}
    spy=AnswerSpy()
    spy.structured_result={"claims":[{"text":"I prefer tea","claim_type":"fact",
        "evidence":[{"kind":"memory","id":"omitted-memory"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "What do I prefer?",evidence)

    assert [item["id"] for item in spy.calls[0][1]["memories"]]==["shown-memory"]
    assert answer==answer_generator.ABSTENTION
    assert diagnostics["rejection_categories"]=={"evidence_not_in_model_bundle":1}
    assert diagnostics["citation_validation"]==[{
        "citation_kind":"memory","evidence_id_resolves_owned":True,
        "kind_matches_resolved_type":True,"reason_code":"citation_not_in_model_bundle"}]
    summary=answer_generator.safe_answer_diagnostic_summary(diagnostics)
    assert "omitted-memory" not in repr(summary)
    assert "I prefer tea" not in repr(summary)


def test_claim_provenance_cites_one_and_multiple_actual_memory_ids(monkeypatch):
    spy=AnswerSpy("I prefer tea")
    spy.structured_result={"claims":[{"text":"I prefer tea","claim_type":"fact",
        "evidence":[{"kind":"memory","id":"tea-1"},{"kind":"memory","id":"tea-2"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)
    evidence={"user_id":"alice","query_category":"PREFERENCE","states":[],"transitions":[],
        "memories":[{"id":"tea-1","user_id":"alice","memory":"I prefer tea."},
                    {"id":"tea-2","user_id":"alice","memory":"I prefer tea in the morning."}]}
    answer=answer_generator.generate_answer("What do I prefer?",evidence)
    assert answer=="I prefer tea [Sources: memory:tea-1, memory:tea-2]"


@pytest.mark.parametrize("claim",["I prefer coffee","I prefer tea because coffee is bad"])
def test_unsupported_or_mismatched_claim_is_not_rendered(monkeypatch,claim):
    spy=AnswerSpy()
    spy.structured_result={"claims":[{"text":claim,"claim_type":"fact",
        "evidence":[{"kind":"memory","id":"tea-id"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)
    evidence={"user_id":"alice","query_category":"PREFERENCE","states":[],"transitions":[],
        "memories":[{"id":"tea-id","user_id":"alice","memory":"I prefer tea."}]}
    assert answer_generator.generate_answer("What do I prefer?",evidence)==answer_generator.ABSTENTION


def test_ambiguous_claim_abstains_without_arbitrarily_choosing_source(monkeypatch):
    spy=AnswerSpy("I prefer tea")
    monkeypatch.setattr(answer_generator,"llm",spy)
    evidence={"user_id":"alice","query_category":"PREFERENCE","ambiguity":{"kind":"conflict"},
        "states":[],"transitions":[],"memories":[{"id":"tea","user_id":"alice","memory":"I prefer tea."},
        {"id":"coffee","user_id":"alice","memory":"I prefer coffee."}]}
    assert answer_generator.generate_answer("What do I prefer?",evidence)==answer_generator.ABSTENTION
    assert spy.calls==[]


def test_causal_claim_cites_only_explicit_cause_transition(monkeypatch):
    spy=AnswerSpy("A knee injury made running painful")
    spy.structured_result={"claims":[{"text":"A knee injury made running painful",
        "claim_type":"causal","evidence":[{"kind":"transition","id":"edge-id"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)
    evidence=_evidence(trigger="A knee injury made running painful")
    answer=answer_generator.generate_answer("Why did running change to swimming?",evidence)
    assert "[Sources: transition:edge-id]" in answer


def test_temporal_succession_cannot_support_cited_causal_claim(monkeypatch):
    spy=AnswerSpy("A knee injury made running painful")
    spy.structured_result={"claims":[{"text":"A knee injury made running painful",
        "claim_type":"causal","evidence":[{"kind":"transition","id":"edge-id"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)
    evidence=_evidence(relation="TEMPORAL_SUCCESSION",trigger="A knee injury made running painful")
    assert answer_generator.generate_answer("Why did running change to swimming?",evidence)==answer_generator.ABSTENTION


def test_foreign_or_missing_provenance_never_creates_citation(monkeypatch):
    spy=AnswerSpy("I prefer tea")
    spy.structured_result={"claims":[{"text":"I prefer tea","claim_type":"fact",
        "evidence":[{"kind":"memory","id":"bob-tea"},{"kind":"memory","id":"invented"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)
    evidence={"user_id":"alice","query_category":"PREFERENCE","states":[],"transitions":[],
        "memories":[{"id":"bob-tea","user_id":"bob","memory":"I prefer tea."},
                    {"memory":"I prefer tea.","user_id":"alice"}]}
    assert answer_generator.generate_answer("What do I prefer?",evidence)==answer_generator.ABSTENTION


@pytest.mark.parametrize(("old_value","new_value","old_text","new_text","query","classifier_category"),[
    ("Python","Java","User prefers Python for backend projects",
     "User now prefers Java for backend projects",
     "Which language do I currently prefer for backend projects?","HISTORY"),
    ("tea","coffee","User prefers tea over coffee",
     "User no longer prefers tea, now prefers coffee",
     "What do I currently prefer to drink?","PREFERENCE"),
])
def test_current_state_query_routes_to_active_state_and_filters_superseded_memory(
        tmp_path,monkeypatch,old_value,new_value,old_text,new_text,query,classifier_category):
    store=Database(str(tmp_path/f"current-{new_value}.sqlite"))
    old=State(user_id="alice",state_type="PREFERENCE",state_value=old_value,
              source_memory_ids=["old-memory"])
    new=State(user_id="alice",state_type="PREFERENCE",state_value=new_value,
              source_memory_ids=["new-memory"])
    store.save_state(old);store.save_state(new)
    store.save_transition(Transition(user_id="alice",from_state_id=old.state_id,to_state_id=new.state_id,
        relation_kind="TEMPORAL_SUCCESSION",source_memory_ids=["new-memory"]))
    store.close_state(old.state_id,store.transitions("alice")[0].timestamp.isoformat())
    memories=[{"id":"old-memory","memory":old_text,"user_id":"alice"},
              {"id":"new-memory","memory":new_text,"user_id":"alice"}]
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda question,user_id:memories)

    result=hybrid_retriever.retrieve(query,"alice",classifier_category)

    assert result["query_category"]=="STATE"
    assert [state["state_value"] for state in result["states"]]==[new_value]
    assert [memory["id"] for memory in result["memories"]]==["new-memory"]
    spy=AnswerSpy(new_value)
    monkeypatch.setattr(answer_generator,"llm",spy)
    answer=answer_generator.generate_answer(query,result)
    assert answer.startswith(new_value+" [Sources: ")
    assert f"state:{result['states'][0]['state_id']}" in answer
    assert spy.calls and spy.calls[0][1]["states"][0]["status"]=="ACTIVE"
    if old_value=="tea":
        spy.result=old_value
        assert answer_generator.generate_answer(query,result)==answer_generator.ABSTENTION


def test_specific_current_state_type_excludes_unrelated_active_types_and_users(tmp_path,monkeypatch):
    store=Database(str(tmp_path/"typed_current_state.sqlite"))
    guitar=State(user_id="alice",state_type="GOAL",state_value="learn guitar",
                 source_memory_ids=["alice-goal-memory"])
    swimming=State(user_id="alice",state_type="INTEREST",state_value="swimming",
                   source_memory_ids=["alice-interest-memory"])
    bob_goal=State(user_id="bob",state_type="GOAL",state_value="learn piano",
                   source_memory_ids=["bob-goal-memory"])
    for state in (guitar,swimming,bob_goal):
        store.save_state(state)
    records=[
        {"id":"alice-goal-memory","user_id":"alice","memory":"I want to learn guitar."},
        {"id":"alice-interest-memory","user_id":"alice","memory":"I am interested in swimming."},
    ]
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _query,_user:records)

    result=hybrid_retriever.retrieve("What is my current goal?","alice","HISTORY")

    assert {state["state_value"] for state in result["states"]}=={"learn guitar","swimming"}
    assert result["state_focus"]["candidate_state_ids"]==[guitar.state_id]
    assert [memory["id"] for memory in result["memories"]]==["alice-goal-memory"]
    assert result["user_id"]=="alice"
    spy=AnswerSpy("learn guitar")
    monkeypatch.setattr(answer_generator,"llm",spy)
    assert answer_generator.generate_answer("What is my current goal?",result)==f"learn guitar [Sources: memory:alice-goal-memory, state:{guitar.state_id}]"
    assert [state["state_value"] for state in spy.calls[0][1]["states"]]==["learn guitar"]


def test_broad_history_keeps_multiple_relevant_historical_states(tmp_path,monkeypatch):
    import temporal_engine
    store=Database(str(tmp_path/"broad_history.sqlite"))
    states=[
        State(user_id="alice",state_type="GOAL",state_value="learn guitar",status="INACTIVE",
              source_memory_ids=["guitar-memory"]),
        State(user_id="alice",state_type="GOAL",state_value="learn piano",
              source_memory_ids=["piano-memory"]),
        State(user_id="alice",state_type="INTEREST",state_value="vegetarian cooking",
              source_memory_ids=["cooking-memory"]),
    ]
    for state in states:
        store.save_state(state)
    records=[{"id":memory_id,"user_id":"alice","memory":value}
             for memory_id,value in (("guitar-memory","learn guitar"),
                                     ("piano-memory","learn piano"),
                                     ("cooking-memory","vegetarian cooking"))]
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(temporal_engine,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _query,_user:records)

    result=hybrid_retriever.retrieve("What changed in my history?","alice","HISTORY")

    assert {state["state_value"] for state in result["states"]}=={
        "learn guitar","learn piano","vegetarian cooking"}
    assert {memory["id"] for memory in result["memories"]}=={
        "guitar-memory","piano-memory","cooking-memory"}


def test_conflicting_distinct_preference_records_are_preserved_and_marked_ambiguous(tmp_path,monkeypatch):
    store=Database(str(tmp_path/"preference_ambiguity.sqlite"))
    tea=State(user_id="alice",state_type="PREFERENCE",state_value="tea",
              source_memory_ids=["tea-memory"])
    coffee=State(user_id="alice",state_type="PREFERENCE",state_value="coffee",
                 source_memory_ids=["coffee-memory"])
    store.save_state(tea);store.save_state(coffee)
    records=[
        {"id":"tea-memory","user_id":"alice","memory":"I prefer tea."},
        {"id":"coffee-memory","user_id":"alice","memory":"I prefer coffee."},
    ]
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _query,_user:records)

    result=hybrid_retriever.retrieve("What do I currently prefer?","alice","PREFERENCE")

    assert {state["state_value"] for state in result["states"]}=={"tea","coffee"}
    assert {memory["id"] for memory in result["memories"]}=={"tea-memory","coffee-memory"}
    assert result["ambiguity"]["kind"]=="multiple_state_candidates"


def test_equivalent_duplicate_current_states_do_not_create_ambiguity(tmp_path,monkeypatch):
    from datetime import datetime, timedelta, timezone
    store=Database(str(tmp_path/"equivalent_preferences.sqlite"))
    now=datetime.now(timezone.utc)
    older=State(user_id="alice",state_type="PREFERENCE",state_value="Python",
                valid_from=now-timedelta(minutes=1),
                source_memory_ids=["older-python-memory"],
                metadata={"preference_key":"favorite:programming language"})
    newer=State(user_id="alice",state_type="PREFERENCE",state_value="Python",
                valid_from=now,
                source_memory_ids=["newer-python-memory"],
                metadata={"preference_key":"favorite:programming language"})
    store.save_state(older);store.save_state(newer)
    records=[
        {"id":"older-python-memory","user_id":"alice","memory":"I now prefer Python."},
        {"id":"newer-python-memory","user_id":"alice","memory":"I now prefer Python."},
    ]
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _query,_user:records)

    result=hybrid_retriever.retrieve(
        "What is my current programming language preference?","alice","PREFERENCE")

    assert [state["state_value"] for state in result["states"]]==["Python"]
    assert result["state_focus"]["candidate_state_ids"]==[newer.state_id]
    assert "ambiguity" not in result


def test_predecessor_without_recorded_transition_reports_missing_link(
        tmp_path,monkeypatch):
    store=Database(str(tmp_path/"missing_predecessor.sqlite"))
    python=State(user_id="alice",state_type="PREFERENCE",state_value="Python",
                 source_memory_ids=["python-memory"])
    store.save_state(python)
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _query,_user:[])
    spy=AnswerSpy()
    monkeypatch.setattr(answer_generator,"llm",spy)

    result=hybrid_retriever.retrieve(
        "What did I prefer before Python?","alice","PREFERENCE")
    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "What did I prefer before Python?",result)

    assert result["query_category"]=="HISTORY"
    assert result["states"]==[] and result["transitions"]==[]
    assert result["retrieval_reason"]=="missing_predecessor_transition"
    assert answer==answer_generator.ABSTENTION
    assert diagnostics["answer_reason"]=="missing_predecessor_transition"
    assert diagnostics["model_response_status"]=="not_called"
    assert spy.calls==[]


def test_causal_question_without_explicit_cause_evidence_reports_missing_edge(monkeypatch):
    evidence={"user_id":"alice","query_category":"HISTORY",
        "memories":[{"id":"coffee-memory","user_id":"alice",
                     "memory":"User now prefers coffee"}],
        "states":[],"transitions":[],"causal_path":[]}
    spy=AnswerSpy()
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "Why did I change my drink preference?",evidence)

    assert answer==answer_generator.ABSTENTION
    assert diagnostics["answer_reason"]=="missing_explicit_cause_transition"
    assert diagnostics["model_response_status"]=="not_called"
    assert spy.calls==[]


def test_multiple_plausible_causal_transitions_are_preserved_but_answer_abstains(tmp_path,monkeypatch):
    store=Database(str(tmp_path/"causal_ambiguity.sqlite"))
    running=State(user_id="alice",state_type="ACTIVITY",state_value="running")
    swimming=State(user_id="alice",state_type="ACTIVITY",state_value="swimming")
    cycling=State(user_id="alice",state_type="ACTIVITY",state_value="cycling")
    for state in (running,swimming,cycling):
        store.save_state(state)
    transitions=[
        Transition(user_id="alice",from_state_id=running.state_id,to_state_id=swimming.state_id,
                   relation_kind="EXPLICIT_CAUSE",trigger="knee injury"),
        Transition(user_id="alice",from_state_id=running.state_id,to_state_id=cycling.state_id,
                   relation_kind="EXPLICIT_CAUSE",trigger="pool closure"),
    ]
    for transition in transitions:
        store.save_transition(transition)
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(causal_graph,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _query,_user:[])
    spy=AnswerSpy("The injury caused it.")
    monkeypatch.setattr(answer_generator,"llm",spy)

    result=hybrid_retriever.retrieve("Why did I switch from running?","alice","CAUSAL")

    assert {edge["transition_id"] for edge in result["transitions"]}=={
        edge.transition_id for edge in transitions}
    assert result["ambiguity"]["kind"]=="multiple_causal_candidates"
    assert answer_generator.generate_answer("Why did I switch from running?",result)==answer_generator.ABSTENTION
    assert spy.calls==[]


def test_answer_diagnostics_distinguish_causal_gate_from_model_abstention(monkeypatch):
    spy=AnswerSpy()
    monkeypatch.setattr(answer_generator,"llm",spy)
    evidence=_evidence(relation="TEMPORAL_SUCCESSION",trigger=None,reason=None)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "Why did running change to swimming?",evidence)

    assert answer==answer_generator.ABSTENTION
    assert diagnostics["answer_reason"]=="explicit_cause_gate_rejected"
    assert diagnostics["causal_gate_passed"] is False
    assert diagnostics["model_response_status"]=="not_called"
    assert spy.calls==[]


def test_answer_diagnostics_distinguish_proposals_from_validation_rejections(monkeypatch):
    evidence={"user_id":"alice","query_category":"FACTUAL",
        "memories":[{"id":"m-toronto","user_id":"alice","memory":"User lives in Toronto"}],
        "states":[],"transitions":[],"causal_path":[]}
    spy=AnswerSpy();spy.structured_result={"claims":[]}
    monkeypatch.setattr(answer_generator,"llm",spy)
    answer,diagnostics=answer_generator.generate_answer_with_diagnostics("Where do I live?",evidence)
    assert answer==answer_generator.ABSTENTION
    assert diagnostics["model_response_status"]=="structured_response"
    assert diagnostics["proposed_claim_count"]==0
    assert diagnostics["accepted_claim_count"]==0
    assert diagnostics["answer_reason"]=="no_validated_claims"

    spy.structured_result={"claims":[{"text":"You live in Toronto","claim_type":"fact",
        "evidence":[{"kind":"memory","id":"not-a-retrieved-id"}]}]}
    answer,diagnostics=answer_generator.generate_answer_with_diagnostics("Where do I live?",evidence)
    assert answer==answer_generator.ABSTENTION
    assert diagnostics["proposed_claim_count"]==1
    assert diagnostics["accepted_claim_count"]==0
    assert diagnostics["rejected_claim_count"]==1
    assert diagnostics["rejection_categories"]=={"unknown_or_unowned_evidence_id":1}
    assert diagnostics["rejection_details"]=={}


def test_rejected_fact_diagnostics_identify_lexical_support_without_content(monkeypatch):
    evidence={"user_id":"alice","query_category":"FACTUAL","states":[],"transitions":[],
        "memories":[{"id":"m-toronto","user_id":"alice","memory":"User lives in Toronto"}]}
    claim_text="You live in Vancouver"
    spy=AnswerSpy();spy.structured_result={"claims":[{"text":claim_text,"claim_type":"fact",
        "evidence":[{"kind":"memory","id":"m-toronto"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics("Where do I live?",evidence)

    assert answer==answer_generator.ABSTENTION
    assert diagnostics["rejection_categories"]=={"claim_not_supported_by_source":1}
    assert diagnostics["rejected_claims"]==[{
        "claim_index":0,"claim_type":"fact","rejection_reasons":["fact_lexical_support_failed"],
        "evidence_references":[{"kind":"memory","evidence_id_resolved":True,
                                 "rejection_reason":"fact_lexical_support_failed"}],
    }]
    serialized=str(diagnostics)
    assert claim_text not in serialized
    assert "User lives in Toronto" not in serialized
    assert "m-toronto" not in serialized


def test_rejected_temporal_claim_diagnostics_identify_validity_failure(monkeypatch):
    evidence={"user_id":"alice","query_category":"TEMPORAL",
        "requested_timestamp":"2025-01-01T00:00:00+00:00","memories":[],"transitions":[],
        "states":[{"user_id":"alice","state_id":"goal-1","state_type":"GOAL",
            "state_value":"learn piano","valid_from":"2025-02-01T00:00:00+00:00",
            "valid_until":None}]}
    spy=AnswerSpy();spy.structured_result={"claims":[{"text":"My goal was to learn piano",
        "claim_type":"temporal","evidence":[{"kind":"state","id":"goal-1"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics("What was my goal?",evidence)

    assert answer==answer_generator.ABSTENTION
    assert diagnostics["rejection_categories"]=={"claim_not_supported_by_source":1}
    assert diagnostics["rejected_claims"]==[{
        "claim_index":0,"claim_type":"temporal","rejection_reasons":["temporal_validity_failed"],
        "evidence_references":[{"kind":"state","evidence_id_resolved":True,
                                 "rejection_reason":"temporal_validity_failed"}],
    }]


def test_temporal_validation_diagnostics_separate_interval_and_lexical_failures(monkeypatch):
    instant="2025-06-01T00:00:00+00:00"
    state={"user_id":"alice","state_id":"goal-1","state_type":"GOAL",
        "state_value":"learn piano","valid_from":"2025-01-01T00:00:00+00:00",
        "valid_until":instant}
    evidence={"user_id":"alice","query_category":"TEMPORAL",
        "requested_timestamp":instant,"memories":[],"states":[state],
        "transitions":[],"causal_path":[]}
    spy=AnswerSpy();spy.structured_result={"claims":[{"text":"My goal was to learn piano",
        "claim_type":"temporal","evidence":[{"kind":"state","id":"goal-1"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,interval_diagnostics=answer_generator.generate_answer_with_diagnostics(
        "What was my goal?",evidence)

    assert answer==answer_generator.ABSTENTION
    assert interval_diagnostics["rejection_categories"]=={"claim_not_supported_by_source":1}
    assert interval_diagnostics["temporal_validation"]==[{
        "requested_timestamp_normalized":True,
        "state_reference_resolved":True,
        "state_validity_timestamps_normalized":True,
        "half_open_interval_passed":False,
        "predecessor_boundary_passed":False,
        "lexical_support_passed":True,
        "state_label_value_lexical_support_passed":True,
        "linked_memory_lexical_support_passed":False,
        "combined_lexical_support_passed":True,
        "both_sources_required_for_lexical_support":False,
        "neither_source_supports_lexically":False,
        "final_temporal_validation_passed":False,
        "reason_codes":["state_interval_not_valid_at_requested_timestamp"],
    }]

    state["valid_until"]=None
    spy.structured_result={"claims":[{"text":"My goal was to learn violin",
        "claim_type":"temporal","evidence":[{"kind":"state","id":"goal-1"}]}]}
    answer,lexical_diagnostics=answer_generator.generate_answer_with_diagnostics(
        "What was my goal?",evidence)

    assert answer==answer_generator.ABSTENTION
    assert lexical_diagnostics["rejection_categories"]=={"claim_not_supported_by_source":1}
    assert lexical_diagnostics["temporal_validation"]==[{
        "requested_timestamp_normalized":True,
        "state_reference_resolved":True,
        "state_validity_timestamps_normalized":True,
        "half_open_interval_passed":True,
        "predecessor_boundary_passed":False,
        "lexical_support_passed":False,
        "state_label_value_lexical_support_passed":False,
        "linked_memory_lexical_support_passed":False,
        "combined_lexical_support_passed":False,
        "both_sources_required_for_lexical_support":False,
        "neither_source_supports_lexically":True,
        "final_temporal_validation_passed":False,
        "reason_codes":["temporal_lexical_support_failed"],
    }]
    assert "learn violin" not in repr(lexical_diagnostics["temporal_validation"])
    assert "goal-1" not in repr(lexical_diagnostics["temporal_validation"])


@pytest.mark.parametrize("claim_text",[
    "My preference on Oct 24, 2026 at 19:50 UTC was Python",
    "My preference on 2026-10-24T19:50:00Z was Python",
])
def test_temporal_claim_date_framing_does_not_require_state_text_support(
        monkeypatch,claim_text):
    state_id="python-state"
    evidence={"user_id":"alice","query_category":"TEMPORAL",
        "requested_timestamp":"2026-10-24T19:50:00Z","memories":[],
        "states":[{"user_id":"alice","state_id":state_id,"state_type":"PREFERENCE",
            "state_value":"Python","valid_from":"2026-10-24T14:15:45Z",
            "valid_until":None}],"transitions":[],"causal_path":[]}
    spy=AnswerSpy();spy.structured_result={"claims":[{"text":claim_text,
        "claim_type":"temporal","evidence":[{"kind":"state","id":state_id}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "What was my preference on Oct 24, 2026 at 19:50 UTC?",evidence)

    assert answer==f"{claim_text} [Sources: state:{state_id}]"
    assert diagnostics["answer_reason"]=="validated_claims"
    assert diagnostics["accepted_claim_count"]==1
    assert diagnostics["temporal_validation"][0]["lexical_support_passed"] is True


def test_temporal_validation_diagnostics_report_valid_boundary_and_foreign_state(monkeypatch):
    evidence={"user_id":"alice","query_category":"TEMPORAL",
        "requested_timestamp":"2025-06-01T00:00:00Z","memories":[],
        "states":[{"user_id":"alice","state_id":"piano","state_type":"GOAL",
            "state_value":"learn piano","valid_from":"2025-06-01T00:00:00+00:00",
            "valid_until":None}],"transitions":[],"causal_path":[]}
    spy=AnswerSpy();spy.structured_result={"claims":[{"text":"My goal was to learn piano",
        "claim_type":"temporal","evidence":[{"kind":"state","id":"piano"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics("What was my goal?",evidence)

    assert answer=="My goal was to learn piano [Sources: state:piano]"
    assert diagnostics["temporal_validation"]==[{
        "requested_timestamp_normalized":True,
        "state_reference_resolved":True,
        "state_validity_timestamps_normalized":True,
        "half_open_interval_passed":True,
        "predecessor_boundary_passed":False,
        "lexical_support_passed":True,
        "state_label_value_lexical_support_passed":True,
        "linked_memory_lexical_support_passed":False,
        "combined_lexical_support_passed":True,
        "both_sources_required_for_lexical_support":False,
        "neither_source_supports_lexically":False,
        "final_temporal_validation_passed":True,
        "reason_codes":["temporal_validation_passed"],
    }]
    assert "learn piano" not in repr(diagnostics["temporal_validation"])
    assert "piano" not in repr(diagnostics["temporal_validation"])

    evidence["states"][0]["user_id"]="bob"
    answer,foreign_diagnostics=answer_generator.generate_answer_with_diagnostics(
        "What was my goal?",evidence)

    assert answer==answer_generator.ABSTENTION
    assert foreign_diagnostics["rejection_categories"]=={"unknown_or_unowned_evidence_id":1}
    assert foreign_diagnostics["temporal_validation"]==[{
        "requested_timestamp_normalized":True,
        "state_reference_resolved":False,
        "state_validity_timestamps_normalized":False,
        "half_open_interval_passed":False,
        "predecessor_boundary_passed":False,
        "lexical_support_passed":False,
        "state_label_value_lexical_support_passed":False,
        "linked_memory_lexical_support_passed":False,
        "combined_lexical_support_passed":False,
        "both_sources_required_for_lexical_support":False,
        "neither_source_supports_lexically":True,
        "final_temporal_validation_passed":False,
        "reason_codes":["state_reference_unresolved_or_unowned"],
    }]


def test_temporal_validation_diagnostics_report_timestamp_normalization_failure(monkeypatch):
    evidence={"user_id":"alice","query_category":"TEMPORAL",
        "requested_timestamp":"2025-06-01T00:00:00","memories":[],"transitions":[],
        "states":[{"user_id":"alice","state_id":"piano","state_type":"GOAL",
            "state_value":"learn piano","valid_from":"2025-01-01T00:00:00+00:00",
            "valid_until":None}]}
    spy=AnswerSpy();spy.structured_result={"claims":[{"text":"My goal was to learn piano",
        "claim_type":"temporal","evidence":[{"kind":"state","id":"piano"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics("What was my goal?",evidence)

    assert answer==answer_generator.ABSTENTION
    details=diagnostics["temporal_validation"][0]
    assert details["requested_timestamp_normalized"] is False
    assert details["state_reference_resolved"] is True
    assert details["half_open_interval_passed"] is False
    assert details["lexical_support_passed"] is True
    assert details["final_temporal_validation_passed"] is False
    assert details["reason_codes"]==["requested_timestamp_invalid"]


@pytest.mark.parametrize(("claim_text","state_value","linked_text","expected"),[
    ("My goal was to learn piano","learn piano",None,
     (True,False,True,False,False)),
    ("My target is violin","piano","My target is violin",
     (False,True,True,False,False)),
    ("My goal was to pursue piano","piano","pursue",
     (False,False,True,True,False)),
    ("My goal was to pursue violin","piano","pursue piano",
     (False,False,False,False,True)),
])
def test_temporal_lexical_diagnostics_split_state_and_linked_source_support(
        monkeypatch,claim_text,state_value,linked_text,expected):
    memory_id="linked-source" if linked_text else None
    state={"user_id":"alice","state_id":"state-1","state_type":"GOAL",
        "state_value":state_value,"valid_from":"2025-01-01T00:00:00Z",
        "valid_until":None,"source_memory_ids":[memory_id] if memory_id else []}
    evidence={"user_id":"alice","query_category":"TEMPORAL",
        "requested_timestamp":"2025-06-01T00:00:00Z","states":[state],
        "memories":([{"id":memory_id,"user_id":"alice","memory":linked_text}]
                    if linked_text else []),"transitions":[],"causal_path":[]}
    spy=AnswerSpy();spy.structured_result={"claims":[{"text":claim_text,
        "claim_type":"temporal","evidence":[{"kind":"state","id":"state-1"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics("What was my goal?",evidence)

    details=diagnostics["temporal_validation"][0]
    assert (details["state_label_value_lexical_support_passed"],
            details["linked_memory_lexical_support_passed"],
            details["combined_lexical_support_passed"],
            details["both_sources_required_for_lexical_support"],
            details["neither_source_supports_lexically"])==expected
    assert details["lexical_support_passed"]==expected[2]
    assert details["final_temporal_validation_passed"]==expected[2]
    assert claim_text not in repr(details)
    if linked_text:
        assert linked_text not in repr(details)
    assert "linked-source" not in repr(details)
    assert (answer!=answer_generator.ABSTENTION)==expected[2]


def test_foreign_linked_memory_does_not_count_in_temporal_support_diagnostics(monkeypatch):
    evidence={"user_id":"alice","query_category":"TEMPORAL",
        "requested_timestamp":"2025-06-01T00:00:00Z","states":[{
            "user_id":"alice","state_id":"state-1","state_type":"GOAL",
            "state_value":"piano","valid_from":"2025-01-01T00:00:00Z",
            "valid_until":None,"source_memory_ids":["foreign-memory"]}],
        "memories":[{"id":"foreign-memory","user_id":"bob","memory":"target violin"}],
        "transitions":[],"causal_path":[]}
    spy=AnswerSpy();spy.structured_result={"claims":[{"text":"target violin",
        "claim_type":"temporal","evidence":[{"kind":"state","id":"state-1"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics("What was my goal?",evidence)

    details=diagnostics["temporal_validation"][0]
    assert answer==answer_generator.ABSTENTION
    assert details["state_label_value_lexical_support_passed"] is False
    assert details["linked_memory_lexical_support_passed"] is False
    assert details["combined_lexical_support_passed"] is False
    assert details["neither_source_supports_lexically"] is True
    assert "target violin" not in repr(details)
    assert "foreign-memory" not in repr(details)


@pytest.mark.parametrize(("reference","expected_kind","expected_reason"),[
    ({"kind":"memory","id":"bob-only-memory"},"memory","unknown_or_unowned_evidence_id"),
    ({"kind":"foreign-citation","id":"m-toronto"},None,"evidence_reference_kind_invalid"),
])
def test_rejected_reference_diagnostics_report_resolution_without_ids(
        monkeypatch,reference,expected_kind,expected_reason):
    evidence={"user_id":"alice","query_category":"FACTUAL","states":[],"transitions":[],
        "memories":[{"id":"m-toronto","user_id":"alice","memory":"User lives in Toronto"}]}
    spy=AnswerSpy();spy.structured_result={"claims":[{"text":"You live in Toronto",
        "claim_type":"fact","evidence":[reference]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics("Where do I live?",evidence)

    assert answer==answer_generator.ABSTENTION
    rejected=diagnostics["rejected_claims"][0]
    assert rejected["claim_index"]==0
    assert rejected["claim_type"]=="fact"
    assert rejected["rejection_reasons"]==[expected_reason]
    ref_result=rejected["evidence_references"][0]
    assert ref_result["evidence_id_resolved"] is False
    assert ref_result["rejection_reason"]==expected_reason
    assert ref_result.get("kind")==expected_kind
    assert "bob-only-memory" not in str(diagnostics)
    assert "m-toronto" not in str(diagnostics)
    assert "You live in Toronto" not in str(diagnostics)


def test_successful_claim_diagnostics_do_not_expose_claim_or_source_content(monkeypatch):
    evidence={"user_id":"alice","query_category":"FACTUAL","states":[],"transitions":[],
        "memories":[{"id":"m-toronto","user_id":"alice","memory":"User lives in Toronto"}]}
    claim_text="You live in Toronto"
    spy=AnswerSpy();spy.structured_result={"claims":[{"text":claim_text,"claim_type":"fact",
        "evidence":[{"kind":"memory","id":"m-toronto"}]}]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics("Where do I live?",evidence)

    assert answer==f"{claim_text} [Sources: memory:m-toronto]"
    assert diagnostics["rejected_claims"]==[]
    assert claim_text not in str(diagnostics)
    assert "User lives in Toronto" not in str(diagnostics)
    assert "m-toronto" not in str(diagnostics)

    spy.structured_result={"claims":[{"text":"Toronto","claim_type":"fact",
        "evidence":[{"kind":"memory","id":"m-toronto"}]}]}
    answer,diagnostics=answer_generator.generate_answer_with_diagnostics("Where do I live?",evidence)
    assert answer=="Toronto [Sources: memory:m-toronto]"
    assert diagnostics["proposed_claim_count"]==diagnostics["accepted_claim_count"]==1
    assert diagnostics["rejected_claim_count"]==0
    assert diagnostics["answer_reason"]=="validated_claims"
    assert diagnostics["rejection_details"]=={}


@pytest.mark.parametrize(("claim","expected_detail","expected_category"),[
    ({"claim_type":"fact","evidence":[{"kind":"memory","id":"m-toronto"}]},
     "claim_text_missing","malformed_claim"),
    ({"text":7,"claim_type":"fact","evidence":[{"kind":"memory","id":"m-toronto"}]},
     "claim_text_not_string","malformed_claim"),
    ({"text":"   ","claim_type":"fact","evidence":[{"kind":"memory","id":"m-toronto"}]},
     "claim_text_empty","malformed_claim"),
    ({"text":"Toronto","claim_type":"unsupported","evidence":[{"kind":"memory","id":"m-toronto"}]},
     "claim_type_invalid","malformed_claim"),
    ({"text":"Toronto","claim_type":[],"evidence":[{"kind":"memory","id":"m-toronto"}]},
     "claim_type_invalid","malformed_claim"),
    ({"text":"Toronto","claim_type":"fact","evidence":"memory:m-toronto"},
     "claim_evidence_not_list","malformed_claim"),
    ({"text":"Toronto","claim_type":"fact","evidence":["memory:m-toronto"]},
     "evidence_reference_not_object","malformed_evidence_reference"),
    ({"text":"Toronto","claim_type":"fact","evidence":[{"id":"m-toronto"}]},
     "evidence_reference_kind_missing","malformed_evidence_reference"),
    ({"text":"Toronto","claim_type":"fact","evidence":[{"kind":"citation","id":"m-toronto"}]},
     "evidence_reference_kind_invalid","malformed_evidence_reference"),
    ({"text":"Toronto","claim_type":"fact","evidence":[{"kind":[],"id":"m-toronto"}]},
     "evidence_reference_kind_invalid","malformed_evidence_reference"),
    ({"text":"Toronto","claim_type":"fact","evidence":[{"kind":"memory"}]},
     "evidence_reference_id_missing","malformed_evidence_reference"),
    ({"text":"Toronto","claim_type":"fact","evidence":[{"kind":"memory","id":4}]},
     "evidence_reference_id_not_string","malformed_evidence_reference"),
    ({"text":"Toronto","claim_type":"fact","evidence":[{"kind":"memory","id":"  "}]},
     "evidence_reference_id_empty","malformed_evidence_reference"),
])
def test_answer_diagnostics_report_malformed_claim_fields_without_content(
        monkeypatch,claim,expected_detail,expected_category):
    evidence={"user_id":"alice","query_category":"FACTUAL",
        "memories":[{"id":"m-toronto","user_id":"alice","memory":"User lives in Toronto"}],
        "states":[],"transitions":[],"causal_path":[]}
    spy=AnswerSpy();spy.structured_result={"claims":[claim]}
    monkeypatch.setattr(answer_generator,"llm",spy)

    answer,diagnostics=answer_generator.generate_answer_with_diagnostics("Where do I live?",evidence)

    assert answer==answer_generator.ABSTENTION
    assert diagnostics["rejection_categories"]=={expected_category:1}
    assert diagnostics["rejection_details"]=={expected_detail:1}
    assert diagnostics["accepted_claim_count"]==0

def test_favorite_color_retrieves_current_interest_and_ignores_unrelated_legacy_edge(
        tmp_path,monkeypatch):
    from datetime import datetime,timedelta,timezone
    store=Database(str(tmp_path/"favorite_color.sqlite"))
    start=datetime(2026,10,22,4,38,tzinfo=timezone.utc)
    boundary=datetime(2026,10,22,15,56,44,373584,tzinfo=timezone.utc)
    running=State(user_id="alice",state_type="INTEREST",state_value="running",
                  status="INACTIVE",valid_from=start,valid_until=boundary)
    swimming=State(user_id="alice",state_type="ACTIVITY",state_value="swimming",
                   valid_from=boundary)
    blue=State(user_id="alice",state_type="INTEREST",state_value="blue",
               source_memory_ids=["blue-memory"])
    java=State(user_id="alice",state_type="OTHER",
               state_value="My favorite programming language is Java.")
    for state in (running,swimming,blue,java): store.save_state(state)
    store.save_transition(Transition(
        user_id="alice",from_state_id=running.state_id,to_state_id=swimming.state_id,
        relation_kind="EXPLICIT_CAUSE",trigger="knee injury",
        timestamp=boundary+timedelta(milliseconds=18)))
    records=[
        {"id":"blue-memory","user_id":"alice","memory":"User's favorite color is blue"},
        {"id":"java-memory","user_id":"alice","memory":"User prefers Java"},
    ]
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _q,_u:records)

    result=hybrid_retriever.retrieve("What is my favorite color?","alice","HISTORY")

    assert result["query_category"]=="STATE"
    assert "ambiguity" not in result
    assert blue.state_id in {state["state_id"] for state in result["states"]}
    assert "blue-memory" in {memory["id"] for memory in result["memories"]}
    assert result["state_focus"]=={
        "state_types":["INTEREST"],
        "candidate_state_ids":[blue.state_id],
        "source_memory_ids":["blue-memory"],
    }

    spy=AnswerSpy("blue");monkeypatch.setattr(answer_generator,"llm",spy)
    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "What is my favorite color?",result)
    assert answer.startswith("blue [Sources: ")
    assert f"state:{blue.state_id}" in answer
    assert diagnostics["answer_reason"]=="validated_claims"
    focused_evidence=spy.calls[0][1]
    assert [state["state_id"] for state in focused_evidence["states"]]==[blue.state_id]
    assert [memory["id"] for memory in focused_evidence["memories"]]==["blue-memory"]


def test_activity_predecessor_follows_unique_transition_and_validates_interval(
        tmp_path,monkeypatch):
    from datetime import datetime,timezone
    store=Database(str(tmp_path/"activity_predecessor.sqlite"))
    start=datetime(2026,10,22,4,38,tzinfo=timezone.utc)
    boundary=datetime(2026,10,22,15,56,44,373584,tzinfo=timezone.utc)
    running=State(user_id="alice",state_type="INTEREST",state_value="running",
                  status="INACTIVE",valid_from=start,valid_until=boundary)
    swimming=State(user_id="alice",state_type="ACTIVITY",state_value="swimming",
                   valid_from=boundary)
    store.save_state(running);store.save_state(swimming)
    edge=Transition(user_id="alice",from_state_id=running.state_id,
                    to_state_id=swimming.state_id,relation_kind="EXPLICIT_CAUSE",
                    trigger="knee injury",timestamp=boundary)
    store.save_transition(edge)
    import temporal_engine
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(temporal_engine,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _q,_u:[])

    result=hybrid_retriever.retrieve(
        "What activity did I do before swimming?","alice","FACTUAL")

    assert result["query_category"]=="HISTORY"
    assert [state["state_value"] for state in result["states"]]==["running"]
    assert [item["transition_id"] for item in result["transitions"]]==[edge.transition_id]
    spy=AnswerSpy("running");monkeypatch.setattr(answer_generator,"llm",spy)
    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "What activity did I do before swimming?",result)
    assert answer.startswith("running [Sources: ")
    assert f"state:{running.state_id}" in answer
    assert diagnostics["answer_reason"]=="validated_claims"


def test_activity_predecessor_without_transition_abstains_safely(tmp_path,monkeypatch):
    from datetime import datetime,timezone
    store=Database(str(tmp_path/"missing_activity_predecessor.sqlite"))
    swimming=State(user_id="alice",state_type="ACTIVITY",state_value="swimming",
                   valid_from=datetime(2026,10,22,tzinfo=timezone.utc))
    store.save_state(swimming)
    import temporal_engine
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(temporal_engine,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _q,_u:[])

    result=hybrid_retriever.retrieve(
        "What activity did I do before swimming?","alice","FACTUAL")
    spy=AnswerSpy();monkeypatch.setattr(answer_generator,"llm",spy)
    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "What activity did I do before swimming?",result)

    assert result["states"]==[] and result["transitions"]==[]
    assert result["retrieval_reason"]=="missing_predecessor_transition"
    assert answer==answer_generator.ABSTENTION
    assert diagnostics["answer_reason"]=="missing_predecessor_transition"
    assert spy.calls==[]


def test_java_to_python_without_recorded_transition_does_not_fabricate_cause(
        tmp_path,monkeypatch):
    store=Database(str(tmp_path/"java_python_no_edge.sqlite"))
    java=State(user_id="alice",state_type="INTEREST",state_value="Java")
    python=State(user_id="alice",state_type="PREFERENCE",state_value="Python")
    store.save_state(java);store.save_state(python)
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _q,_u:[])

    result=hybrid_retriever.retrieve(
        "Why did I switch from Java to Python?","alice","HISTORY")
    spy=AnswerSpy();monkeypatch.setattr(answer_generator,"llm",spy)
    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "Why did I switch from Java to Python?",result)

    assert result["transitions"]==[] and result["causal_path"]==[]
    assert result["states"]==[]
    assert answer==answer_generator.ABSTENTION
    assert diagnostics["answer_reason"]=="missing_explicit_cause_transition"
    assert spy.calls==[]


def test_asof_missing_activity_evidence_abstains_instead_of_operation_failure(
        tmp_path,monkeypatch):
    store=Database(str(tmp_path/"asof_missing_activity.sqlite"))
    import temporal_engine
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(temporal_engine,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _q,_u:[])

    question="What activity did I do on October 24, 2026 at 19:50 UTC?"
    result=hybrid_retriever.retrieve(question,"alice","HISTORY")
    spy=AnswerSpy();monkeypatch.setattr(answer_generator,"llm",spy)
    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(question,result)

    assert result["states"]==[]
    assert answer==answer_generator.ABSTENTION
    assert diagnostics["answer_reason"]=="empty_evidence"
    assert spy.calls==[]
