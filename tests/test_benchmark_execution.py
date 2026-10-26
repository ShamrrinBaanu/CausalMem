import subprocess
import io
import json
from pathlib import Path

import pytest

import benchmark


PROJECT=Path(__file__).resolve().parents[1]
PYTHON=PROJECT/".venv"/"Scripts"/"python.exe"


def test_transition_provenance_fixture_keys_require_same_user_endpoints():
    evidence={"states":[
        {"state_id":"old","user_id":"alice"},
        {"state_id":"new","user_id":"alice"},
        {"state_id":"foreign","user_id":"bob"}],
        "transitions":[
            {"user_id":"alice","from_state_id":"old","to_state_id":"new",
             "source_memory_ids":["replacement-message"]},
            {"user_id":"alice","from_state_id":"old","to_state_id":"foreign",
             "source_memory_ids":["foreign-message"]},
            {"user_id":"bob","from_state_id":"old","to_state_id":"new",
             "source_memory_ids":["bob-message"]}]}
    mapping={"replacement-message":{"goal-piano"},"foreign-message":{"foreign"},
             "bob-message":{"bob"}}

    owners={"replacement-message":{"alice"},"foreign-message":{"alice"},
            "bob-message":{"alice"}}
    assert benchmark._transition_provenance_fixture_keys(
        evidence,mapping,owners,"alice")==["goal-piano"]
    owners["foreign-message"]={"bob"}
    owners["bob-message"]={"bob"}
    assert benchmark._transition_provenance_fixture_keys(
        evidence,mapping,owners,"alice")==["goal-piano"]


def test_transition_provenance_can_validate_omitted_endpoint_from_user_scoped_history():
    evidence={
        "states":[{"state_id":"old","user_id":"alice"}],
        "transitions":[{"user_id":"alice","from_state_id":"old",
                         "to_state_id":"new","source_memory_ids":["replacement-message"]}],
    }
    fixtures={"replacement-message":{"goal-piano"}}
    memory_owners={"replacement-message":{"alice"}}

    # The destination is not answer evidence for a predecessor question, but
    # its owner was verified from the isolated user's full SQLite history.
    assert benchmark._transition_provenance_fixture_keys(
        evidence,fixtures,memory_owners,"alice",
        {"old":"alice","new":"alice"})==["goal-piano"]
    assert benchmark._transition_provenance_fixture_keys(
        evidence,fixtures,memory_owners,"alice",
        {"old":"alice","new":"bob"})==[]


def test_benchmark_dataset_validates_transition_provenance_fixture_keys(monkeypatch):
    case={"case_id":"history","expected_behavior":"answer",
        "expected_evidence_keys":["old"],"required_answer_facts":[],
        "forbidden_answer_facts":[],"expected_transition_provenance_keys":["replacement"],
        "conversation":[{"fixture_key":"old","text":"old state"},
                       {"fixture_key":"replacement","text":"replacement"}]}
    monkeypatch.setattr(Path,"read_text",lambda self,encoding=None:json.dumps([case]))

    assert benchmark._read_dataset(Path("cases.json"))==[case]

    case["expected_transition_provenance_keys"]=["not-a-fixture"]
    with pytest.raises(ValueError,match="expected_transition_provenance_keys"):
        benchmark._read_dataset(Path("cases.json"))


def _python(code,*args):
    return subprocess.run([str(PYTHON),"-c",code,*map(str,args)],cwd=PROJECT,capture_output=True,text=True)


def test_full_mode_uses_actual_application_pipeline_on_one_paired_fixture(tmp_path):
    script=r"""
import sys
from types import SimpleNamespace
import benchmark
paths=benchmark._configure_isolation(sys.argv[1])
runtime=benchmark._load_runtime(paths)
import hybrid_retriever,answer_generator
import state_extractor
assert runtime.retrieve is hybrid_retriever.retrieve
assert runtime.generate_answer is answer_generator.generate_answer
assert runtime.generate_answer_with_diagnostics is answer_generator.generate_answer_with_diagnostics
assert runtime.ingest_message is state_extractor.ingest_message

fixtures={}; added=[]; ingested=[]; search_calls=[]; llm_calls=[]
def add_memory(message,user_id):
    added.append((message,user_id)); fixtures.setdefault(user_id,[]).append(message)
def get_memories(user_id):
    return [{"id":f"{user_id}-{i}","memory":text,"user_id":user_id}
            for i,text in enumerate(fixtures.get(user_id,[]))]
def ingest(message,user_id,ids,memories):
    ingested.append((message,user_id))
def search(query,user_id):
    search_calls.append((query,user_id))
    return get_memories(user_id)
class State:
    def __init__(self,state_id,user_id,label):
        self.state_id=state_id; self.user_id=user_id; self.state_value=label
    def model_dump(self,mode="json"):
        index=0 if self.state_id.endswith("running") or self.state_id.endswith("cycling") else 1
        return {"state_id":self.state_id,"user_id":self.user_id,"state_value":self.state_value,
                "status":"ACTIVE","valid_from":"2026-01-01T00:00:00+00:00","valid_until":None,
                "source_memory_ids":[f"{self.user_id}-{index}"]}
class FakeDB:
    def states(self,user_id,include_inactive=True):
        labels=("cycling","walking") if "-case-b-" in user_id else ("running","swimming")
        return [State(f"{user_id}-{labels[0]}",user_id,labels[0]),State(f"{user_id}-{labels[1]}",user_id,labels[1])]
allow_cause=[True]
def transitions(user_id,query,causal_only=None):
    if not allow_cause[0]: return []
    labels=("cycling","walking") if "-case-b-" in user_id else ("running","swimming")
    reason="weather" if "-case-b-" in user_id else "knee injury"
    return [{"transition_id":f"{user_id}-edge","user_id":user_id,
        "from_state_id":f"{user_id}-{labels[0]}","to_state_id":f"{user_id}-{labels[1]}",
        "relation_kind":"EXPLICIT_CAUSE","trigger":reason,"reason":f"User stated {reason} caused the switch",
        "timestamp":"2026-01-01T00:00:00+00:00","source_memory_ids":[f"{user_id}-1"]}]
hybrid_retriever.search_memory=search
hybrid_retriever.db=FakeDB()
hybrid_retriever.relevant_transitions=transitions
runtime.add_memory=add_memory; runtime.get_memories=get_memories; runtime.ingest_message=ingest
runtime.search_memory=search
runtime.classify=lambda query: SimpleNamespace(category="CAUSAL")
runtime.llm.answer_with_provenance=lambda query,evidence: (llm_calls.append((query,evidence)) or ({"claims":[{"text":evidence["transitions"][0]["trigger"],"claim_type":"causal","evidence":[{"kind":"transition","id":evidence["transitions"][0]["transition_id"]}]}]} if evidence.get("transitions") else {"claims":[]}))

cases=[
 {"case_id":"case-a","fixture_user":"alice","conversation":[{"fixture_key":"running","text":"I like running."},{"fixture_key":"swimming","text":"I switched to swimming because of a knee injury."}],
  "distractor_users":[{"fixture_user":"bob","conversation":[{"fixture_key":"bob-running","text":"I like running."}]}],
  "question":"Why did running change to swimming?","category":"CAUSAL","expected_behavior":"answer","expected_evidence_keys":["running","swimming"],"required_answer_facts":["swimming"],"forbidden_answer_facts":[]},
 {"case_id":"case-b","fixture_user":"alice","conversation":[{"fixture_key":"cycling","text":"I enjoy cycling."},{"fixture_key":"walking","text":"I switched to walking because of the weather."}],
  "question":"Why did cycling change to walking?","category":"CAUSAL","expected_behavior":"answer","expected_evidence_keys":["cycling","walking"],"required_answer_facts":["walking"],"forbidden_answer_facts":[]},
]
rows=benchmark._run_dataset(runtime,cases,["mem0","full"])
from evaluator import evaluate
paired_metrics=evaluate(rows)["metrics"]
assert paired_metrics["mem0"]["evidence_retrieval"]["f1"]==1
assert paired_metrics["full"]["evidence_retrieval"]["f1"]==1
assert len(added)==5 and len(ingested)==5, (added,ingested)
assert len(rows)==4
for case_id in ("case-a","case-b"):
    paired=[row for row in rows if row["case_id"]==case_id]
    assert {row["mode"] for row in paired}=={"mem0","full"}
    assert paired[0]["question"]==paired[1]["question"]
    assert paired[0]["evaluation"]==paired[1]["evaluation"]
assert len({uid for _,uid in added})==3
assert len({uid for _,uid in ingested})==3
assert all(uid in {x[1] for x in added} for _,uid in search_calls)
assert search_calls[0][0]==search_calls[1][0]
assert search_calls[0][1]==search_calls[1][1]
mem0_b=next(row for row in rows if row["case_id"]=="case-b" and row["mode"]=="mem0")
assert "cycling" in mem0_b["evidence_text"] and "running" not in mem0_b["evidence_text"]
full_a=next(row for row in rows if row["case_id"]=="case-a" and row["mode"]=="full")
assert "EXPLICIT_CAUSE" in full_a["evidence_text"]
assert "knee injury" in full_a["evidence_text"]
assert "running" in full_a["evidence_text"] and "swimming" in full_a["evidence_text"]
assert set(full_a["evidence_fixture_keys"])=={"running","swimming"}
assert full_a["answer_diagnostics"]["accepted_claim_count"]==1
assert full_a["answer_diagnostics"]["proposed_claim_count"]==1
assert "bob-running" not in full_a["evidence_fixture_keys"]
assert all("evaluation" not in evidence for _,evidence in llm_calls)
full_b=next(row for row in rows if row["case_id"]=="case-b" and row["mode"]=="full")
assert "cycling" in full_b["evidence_text"] and "walking" in full_b["evidence_text"]
assert "running" not in full_b["evidence_text"] and "swimming" not in full_b["evidence_text"]

# The same real answer gate must abstain on a causal query without eligible evidence.
before=len(llm_calls); allow_cause[0]=False
blocked=benchmark._run_dataset(runtime,[{"case_id":"no-cause","conversation":["I enjoy hiking."],
  "question":"Why did running change to swimming?","category":"CAUSAL","expected_behavior":"abstain","expected_evidence_keys":[],"required_answer_facts":[],"forbidden_answer_facts":[]}],["full"])
assert blocked[0]["answer"]==answer_generator.ABSTENTION
assert len(llm_calls)==before
print('actual retrieval and answer gate used with paired case fixtures')
"""
    result=_python(script,tmp_path/"paired")
    assert result.returncode==0, result.stderr
    assert "actual retrieval and answer gate used" in result.stdout


def test_full_benchmark_routes_annotation_timestamp_without_prompting_it(tmp_path):
    class Runtime:
        def __init__(self): self.calls=[]; self.prompts=[]
        def get_memories(self,*args): return []
        def add_memory(self,*args): return {}
        def ingest_message(self,*args): pass
        def classify(self,query): return type("Analysis",(),{"category":"HISTORY"})()
        def retrieve(self,*args):
            self.calls.append(args)
            return {"user_id":args[1],"query_category":args[2],
                    "memories":[],"states":[],"transitions":[],"causal_path":[]}
        def generate_answer(self,query,evidence):
            self.prompts.append((query,evidence))
            return "historical answer"
        def has_supported_cause(self,*args): return False

    runtime=Runtime()
    timestamp="2025-06-01T00:00:00Z"
    rows=benchmark._run_dataset(runtime,[{"case_id":"asof-route","conversation":[],
        "question":"What was true?","requested_timestamp":timestamp}], ["full"])

    assert len(runtime.calls)==1 and runtime.calls[0][3]==timestamp
    assert runtime.prompts[0][0]=="What was true?"
    assert timestamp not in json.dumps(runtime.prompts[0][1])
    assert rows[0]["answer"]=="historical answer"


def test_transition_relative_timestamp_reaches_actual_full_retriever():
    from datetime import datetime, timezone
    from types import SimpleNamespace

    edge=SimpleNamespace(timestamp=datetime(2025,6,1,tzinfo=timezone.utc),source_memory_ids=["memory-piano"])
    class Store:
        def transitions(self,user_id):
            return [edge]
    class Runtime:
        def __init__(self): self.db=Store(); self.calls=[]
        def get_memories(self,user_id): return []
        def add_memory(self,text,user_id):
            memory_id="memory-guitar" if "guitar" in text else "memory-piano"
            return {"results":[{"id":memory_id}]}
        def ingest_message(self,*_args): pass
        def classify(self,_query): return SimpleNamespace(category="TEMPORAL")
        def retrieve(self,*args):
            self.calls.append(args)
            return {"user_id":args[1],"query_category":args[2],"memories":[],"states":[],
                    "transitions":[],"causal_path":[]}
        def generate_answer(self,*_args): return "answer"
        def has_supported_cause(self,*_args): return False
    runtime=Runtime()
    case={"case_id":"relative","fixture_user":"alice",
        "conversation":[{"fixture_key":"goal-guitar","text":"My goal is guitar."},
                        {"fixture_key":"goal-piano","text":"My goal is piano instead."}],
        "question":"What was my goal?","requested_timestamp_from_transition":{
            "source_fixture_key":"goal-piano","offset_microseconds":-1},
        "expected_behavior":"answer","expected_evidence_keys":[],"required_answer_facts":[],
        "forbidden_answer_facts":[],"expected_temporal_state_keys":["goal-guitar"],
        "temporal_metric_scope":"end_to_end"}
    row=benchmark._run_dataset(runtime,[case],["full"])[0]
    assert runtime.calls[0][3]=="2025-05-31T23:59:59.999999+00:00"
    assert row["evaluation"]["requested_timestamp"]==runtime.calls[0][3]
    assert "requested_timestamp_from_transition" not in row["evaluation"]
    assert row["source_id_fixture_keys"]=={"memory-guitar":"goal-guitar","memory-piano":"goal-piano"}


def test_synthetic_temporal_dataset_scores_actual_retriever_output(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from database import Database
    import answer_generator, causal_graph, hybrid_retriever, temporal_engine
    from evaluator import evaluate

    dataset=benchmark._read_dataset(PROJECT/"evaluation"/"temporal_end_to_end.jsonl")
    store=Database(str(tmp_path/"temporal-benchmark.sqlite"))
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(causal_graph,"db",store)
    monkeypatch.setattr(temporal_engine,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda query,user_id:[])

    class AnswerSpy:
        calls=[]
        def answer_with_provenance(self,question,evidence):
            self.calls.append((question,evidence))
            if evidence.get("transitions"):
                edge=evidence["transitions"][0]
                return {"claims":[{"text":edge.get("trigger") or "recorded change",
                    "claim_type":"causal","evidence":[{"kind":"transition","id":edge["transition_id"]}]}]}
            if evidence.get("states"):
                state=evidence["states"][0]
                return {"claims":[{"text":state["state_value"],"claim_type":"fact",
                    "evidence":[{"kind":"state","id":state["state_id"]}]}]}
            return {"claims":[]}
    spy=AnswerSpy()
    monkeypatch.setattr(answer_generator,"llm",spy)
    calls=[]
    def actual_retrieve(*args):
        calls.append(args)
        return hybrid_retriever.retrieve(*args)
    runtime=SimpleNamespace(
        db=store,get_memories=lambda user_id:[],add_memory=lambda *args:{},
        ingest_message=lambda *args:None,
        classify=lambda query:SimpleNamespace(category="CAUSAL" if query.startswith("Why") else "TEMPORAL"),
        retrieve=actual_retrieve,generate_answer=answer_generator.generate_answer,
        has_supported_cause=answer_generator._has_supported_cause,llm=spy,
    )

    rows=benchmark._run_dataset(runtime,dataset,["full"])
    metrics=evaluate(rows)["metrics"]["full"]
    temporal=metrics["end_to_end_temporal_evidence"]
    assert len(calls)==len(dataset)==6
    assert [call[3] for call in calls]==[case["requested_timestamp"] for case in dataset]
    assert temporal["cases_scored"]==6
    assert temporal["precision"]==temporal["recall"]==temporal["f1"]==1
    assert temporal["exact_match_accuracy"]==1
    assert all(item["exact_match"] for item in temporal["per_case"])
    boundary=next(row for row in rows if row["case_id"]=="e2e-pref-at-boundary")
    assert boundary["evidence_fixture_keys"]==["pref-java"]
    assert "bob-java" not in boundary["evidence_fixture_keys"]
    future=next(row for row in rows if row["case_id"]=="e2e-future-evidence-excluded")
    assert future["evidence_fixture_keys"]==["activity-running"]
    assert future["causal_evidence_fixture_keys"]==[]
    succession=next(row for row in rows if row["case_id"]=="e2e-succession-not-cause")
    assert succession["answer"]==answer_generator.ABSTENTION
    explicit=next(row for row in rows if row["case_id"]=="e2e-explicit-cause")
    assert explicit["causal_evidence_fixture_keys"]==["switch-cause"]
    assert all("temporal_fixture" not in json.dumps(evidence)
               and "expected_temporal_state_keys" not in json.dumps(evidence)
               for _,evidence in spy.calls)


@pytest.mark.parametrize("failure_stage",["retrieve","generate"])
def test_pipeline_failures_are_reported_with_case_and_condition(failure_stage):
    class Runtime:
        def add_memory(self,*args): pass
        def get_memories(self,*args): return []
        def ingest_message(self,*args): pass
        def search_memory(self,*args): return []
        def classify(self,*args): return type("Analysis",(),{"category":"FACTUAL"})()
        class llm:
            @staticmethod
            def answer(*args): return "baseline"
        def retrieve(self,*args):
            if failure_stage=="retrieve": raise OSError("retrieval fixture failed")
            return {"user_id":args[1],"query_category":args[2],"memories":[{"memory":"x"}],"states":[],"transitions":[],"causal_path":[]}
        def generate_answer(self,*args):
            if failure_stage=="generate": raise OSError("generation fixture failed")
            return "answer"
        def has_supported_cause(self,*args): return False
    with pytest.raises(benchmark.BenchmarkExecutionError) as caught:
        benchmark._run_dataset(Runtime(),[{"case_id":"failure-case","conversation":["fixture"],
            "question":"What do I enjoy?","category":"FACTUAL","expected_any":[]}],["full"])
    message=str(caught.value)
    assert "failure-case" in message and "full" in message
    failed_operation="retrieval" if failure_stage=="retrieve" else "generation"
    assert f"{failed_operation} fixture failed" in message


def test_succession_fixture_ingestion_fallback_preserves_fixture_attribution(tmp_path):
    script=r"""
import json,sys
from collections import defaultdict
import benchmark
paths=benchmark._configure_isolation(sys.argv[1])
runtime=benchmark._load_runtime(paths)
dataset=benchmark._read_dataset(benchmark._default_dataset_path())
case=next(item for item in dataset if item["case_id"]=="succession-not-cause")
user_calls=defaultdict(int); ingested=[]; added=[]
stable_id="runtime-coding-record"
def get_memories(user_id):
    user_calls[user_id]+=1
    call=user_calls[user_id]
    if call in (1,2): return []
    if call==3: return [{"id":stable_id,"memory":"earlier merged record"}]
    if call==4: return [{"id":stable_id,"memory":"Later, I started learning programming."}]
    return [{"id":stable_id,"memory":"Later, I started learning programming."}]
runtime.get_memories=get_memories
runtime.add_memory=lambda text,user_id: (added.append((text,user_id)) or {})
runtime.ingest_message=lambda text,user_id,memory_ids,memories: ingested.append((text,list(memory_ids)))
runtime.search_memory=lambda query,user_id: get_memories(user_id)
runtime.llm.answer=lambda query,evidence:"I don't have enough information in your memory to determine that."
rows=benchmark._run_dataset(runtime,[case],["mem0"])
assert len(added)==2 and len(ingested)==2
assert ingested[0][1]==[]
assert ingested[1][1]==[stable_id]
assert rows[0]["evidence_fixture_keys"]==["coding-later"]
assert rows[0]["fixture_key_mapping_ambiguities"]==0
print("succession fixture attribution preserved")
"""
    result=_python(script,tmp_path/"succession-fixture")
    assert result.returncode==0, result.stderr
    assert "succession fixture attribution preserved" in result.stdout


def test_benchmark_error_report_keeps_traceback_frames_without_exception_text():
    error=benchmark.BenchmarkExecutionError(
        "Benchmark fixture ingestion failed for case 'succession-not-cause': private detail"
    )
    try:
        raise NameError("private provider detail")
    except NameError as cause:
        error.__cause__=cause

    stream=io.StringIO()
    benchmark._print_error_traceback(error,stream)
    summary=benchmark._concise_error_summary(error)

    assert "NameError" in summary
    assert "succession-not-cause" in summary
    assert "private" not in summary
    assert "Underlying traceback" in stream.getvalue()
    assert "NameError" not in stream.getvalue()
    assert "private" not in stream.getvalue()


def test_fixture_ingestion_error_keeps_case_context_and_original_cause():
    class Runtime:
        def get_memories(self,*args): return []
        def add_memory(self,*args): raise NameError("record_id is not defined")

    case={"case_id":"succession-not-cause","conversation":[
        {"fixture_key":"coding-later","text":"Later, I started learning programming."}],
        "question":"What caused me to start learning programming?"}
    with pytest.raises(benchmark.BenchmarkExecutionError) as caught:
        benchmark._run_dataset(Runtime(),[case],["mem0"])

    assert "succession-not-cause" in str(caught.value)
    assert "record_id is not defined" in str(caught.value)
    assert isinstance(caught.value.__cause__,NameError)


def test_progress_records_are_flushed_and_do_not_include_fixture_text():
    class FlushTrackingStream(io.StringIO):
        flushed=False
        def flush(self):
            self.flushed=True
            super().flush()

    stream=FlushTrackingStream()
    secret_text="private conversation payload"
    result=benchmark._run_operation("mem0_add",lambda:secret_text,case_id="case-1",
                                    fixture_key="fixture-1",mode="mem0")
    assert result==secret_text
    benchmark._progress("start",case_id="case-1",fixture_key="fixture-1",mode="mem0",
                        operation="mem0_add",stream=stream)
    record=json.loads(stream.getvalue())
    assert stream.flushed
    assert record["timestamp"]
    assert record["case_id"]=="case-1"
    assert record["fixture_key"]=="fixture-1"
    assert record["mode"]=="mem0"
    assert secret_text not in stream.getvalue()


def test_operation_failure_logs_type_not_exception_contents(capsys):
    with pytest.raises(RuntimeError,match="private failure detail"):
        benchmark._run_operation(
            "ollama_call",lambda:(_ for _ in ()).throw(RuntimeError("private failure detail")),
            case_id="case-1",mode="full")

    output=capsys.readouterr().out
    records=[json.loads(line) for line in output.splitlines()]
    assert [record["event"] for record in records]==["start","error"]
    assert records[-1]["error_type"]=="RuntimeError"
    assert records[-1]["elapsed_seconds"]>=0
    assert "private failure detail" not in output


def test_mem0_stage_timing_progress_is_content_free(capsys):
    class Runtime:
        def add_memory(self,_message,_user_id,diagnostics_callback=None):
            diagnostics_callback({"stage":"llm_memory_extraction","duration_seconds":2.5,
                                 "success":False,"error_category":"timeout"})
            return {"results":[]}

    benchmark._add_memory_with_diagnostics(Runtime(),"private fixture text","private-user")
    record=json.loads(capsys.readouterr().out.splitlines()[-1])
    assert record["event"]=="stage"
    assert record["operation"]=="mem0_add:llm_memory_extraction"
    assert record["elapsed_seconds"]==2.5
    assert record["diagnostics"]=={"success":False,"error_category":"timeout"}
    assert "private" not in repr(record)


def test_mem0_unobserved_stage_status_survives_progress_serialization(capsys):
    class Runtime:
        def add_memory(self,_message,_user_id,diagnostics_callback=None):
            diagnostics_callback({"stage":"vector_store_write","duration_seconds":None,
                                 "status":"not_observed","error_category":"stage_not_invoked"})
            return {"results":[]}

    benchmark._add_memory_with_diagnostics(Runtime(),"private fixture text","private-user")
    record=json.loads(capsys.readouterr().out.splitlines()[-1])
    assert record["operation"]=="mem0_add:vector_store_write"
    assert "elapsed_seconds" not in record
    assert record["diagnostics"]=={"status":"not_observed","error_category":"stage_not_invoked"}
    assert "private" not in repr(record)
