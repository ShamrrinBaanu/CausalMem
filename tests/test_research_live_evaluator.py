import json
import os
import subprocess
from pathlib import Path

import pytest

import research_evaluator
import research_live_evaluator as live


PROJECT=Path(__file__).resolve().parents[1]
PYTHON=PROJECT/".venv"/"Scripts"/"python.exe"


def test_live_dataset_is_versioned_unique_and_covers_required_scenarios():
    cases=live.load_dataset()
    ids=[case["case_id"] for case in cases]
    assert len(ids)==len(set(ids))==12
    assert {case["dataset_version"] for case in cases}=={"1.0"}
    assert {"direct-fact","updated-current-goal","historical-predecessor",
            "asof-before-replacement","asof-at-replacement","asof-after-replacement",
            "explicit-cause","succession-not-cause","ambiguous-preference",
            "unsupported-question","foreign-user-isolation","retrieval-failure-mock-only"}==set(ids)


def test_live_case_subset_preserves_dataset_order_and_rejects_invalid_ids():
    cases=live.load_dataset()
    requested=["direct-fact","explicit-cause","historical-predecessor"]
    selected=live.select_cases(cases,requested)
    assert [case["case_id"] for case in selected]==[
        "direct-fact","historical-predecessor","explicit-cause"]
    assert live.select_cases(cases,None)==cases
    with pytest.raises(ValueError,match="unique"):
        live.select_cases(cases,["direct-fact","direct-fact"])
    with pytest.raises(ValueError,match="unknown"):
        live.select_cases(cases,["not-a-case"])


def test_live_dataset_rejects_duplicate_ids(tmp_path):
    case={"dataset_version":"1.0","case_id":"x","category":"FACTUAL","fixture_user":"u",
          "conversation":[],"question":"Q?","expected_behavior":"abstain","expected_evidence_keys":[],
          "required_answer_facts":[],"forbidden_answer_facts":[]}
    path=tmp_path/"duplicate.jsonl"
    path.write_text(json.dumps(case)+"\n"+json.dumps(case),encoding="utf-8")
    with pytest.raises(ValueError,match="unique"):
        live.load_dataset(path)


def test_preflight_checks_installed_models_without_live_generation():
    class Response:
        def __enter__(self): return self
        def __exit__(self,*_args): pass
        def read(self): return b'{"models":[{"name":"llama3.2:3b"},{"name":"nomic-embed-text:latest"}]}'
    calls=[]
    settings=type("Settings",(),{"provider":"ollama","model":"llama3.2:3b",
        "ollama_server_url":"http://localhost:11434"})()
    result=live.local_ollama_preflight(settings,lambda url,timeout:(calls.append((url,timeout)) or Response()))
    assert result=={"available":True,"reason_code":None}
    assert calls==[("http://localhost:11434/api/tags",4)]
    settings.provider="openai"
    assert live.local_ollama_preflight(settings)["reason_code"]=="provider_not_ollama"


def _prediction(case_id="case",answer="A fact [Sources: memory:m1]",expected="answer"):
    return {"case_id":case_id,"mode":"full","category":"FACTUAL","answer":answer,
        "eligible_causal_evidence":False,"evidence_fixture_keys":["fact-a"],
        "state_evidence":[],"causal_evidence_fixture_keys":[],"latency_seconds":2.0,
        "retrieval_latency_seconds":1.0,"answer_latency_seconds":1.0,
        "source_id_fixture_keys":{"m1":"fact-a"},
        "evidence_text":json.dumps({"memories":[{"id":"m1","user_id":"u"}],"states":[],"transitions":[]}),
        "evaluation":{"expected_behavior":expected,"expected_evidence_keys":["fact-a"],
            "required_answer_facts":["fact"] if expected=="answer" else [],
            "forbidden_answer_facts":[],"expected_claim_source_keys":[["fact-a"]] if expected=="answer" else []}}


def test_live_claim_citation_groups_map_only_real_evidence_ids():
    row=_prediction()
    row["answer"]="Fact [Sources: memory:m1] Unsupported [Sources: memory:not-present]"
    metric=live._claim_group_metrics([row])
    assert metric["agreements"]==1
    assert metric["groups"]==2
    assert metric["agreement"]==0.5
    row["answer"]="Fact [Sources: memory:invented]"
    assert live._claim_group_metrics([row])["agreement"]==0.0


def test_explicit_cause_annotation_matches_one_compound_claim_citation_group():
    case=next(item for item in live.load_dataset() if item["case_id"]=="explicit-cause")
    assert case["expected_claim_source_keys"]==[["switch-cause"]]
    row=_prediction("explicit-cause")
    row["evaluation"]={"expected_claim_source_keys":case["expected_claim_source_keys"]}
    row["answer"]="Supported causal claim [Sources: transition:t1]"
    row["source_id_fixture_keys"]={"m-cause":"switch-cause"}
    row["evidence_text"]=json.dumps({"memories":[],"states":[],"transitions":[{
        "transition_id":"t1","source_memory_ids":["m-cause"]}]})

    metric=live._claim_group_metrics([row])

    assert metric["agreements"]==metric["groups"]==1
    assert metric["agreement"]==1.0


def test_live_dataset_transition_provenance_annotation_is_validated():
    case=next(item for item in live.load_dataset() if item["case_id"]=="historical-predecessor")
    assert case["expected_evidence_keys"]==["history-guitar"]
    assert case["expected_historical_state_keys"]==["history-guitar"]
    assert case["expected_transition_provenance_keys"]==["history-piano"]


def test_case_runner_labels_completed_failed_skipped_and_interrupted(tmp_path):
    cases=[{"case_id":"ok"},{"case_id":"bad"},{"case_id":"diagnostic","run_scope":"mock_only"}]
    def run(case):
        if case["case_id"]=="bad": raise RuntimeError("private detail must not be saved")
        return [_prediction(case["case_id"])]
    progress=tmp_path/"progress.jsonl"
    records,status=live.execute_cases(cases,run,progress)
    assert [record["status"] for record in records]==["completed","failed","skipped"]
    assert status=="completed_with_failures"
    encoded=progress.read_text(encoding="utf-8")
    assert "private detail" not in encoded
    assert [json.loads(line)["status"] for line in encoded.splitlines()]==["completed","failed","skipped"]
    assert records[1]["failure_stage"]=="case_execution"
    assert records[1]["traceback_frames"]

    def missing_transition(_case):
        raise ValueError("timestamp selector did not identify a transition")
    temporal,temporal_status=live.execute_cases([{"case_id":"asof"}],missing_transition)
    assert temporal_status=="completed_with_failures"
    assert temporal[0]["failure_stage"]=="temporal_fixture_resolution"

    interrupted,interrupt_status=live.execute_cases([{"case_id":"stop"},{"case_id":"later"}],
        lambda _case:(_ for _ in ()).throw(KeyboardInterrupt()))
    assert interrupt_status=="interrupted"
    assert [record["status"] for record in interrupted]==["interrupted","skipped"]


def test_persisted_progress_redacts_predictions_but_keeps_safe_status_and_timing(tmp_path):
    secrets=["PRIVATE_QUESTION_MARKER","PRIVATE_ANSWER_MARKER","PRIVATE_MEMORY_MARKER",
             "PRIVATE_PROMPT_MARKER","PRIVATE_RESPONSE_MARKER","PRIVATE_USER_MARKER",
             "PRIVATE_EVIDENCE_ID_MARKER","PRIVATE_TOKEN_MARKER","PRIVATE_ENV_MARKER"]
    prediction={
        "case_id":"synthetic-case","mode":"full","category":"FACTUAL",
        "answer_behavior":"answer","latency_seconds":2.5,
        "retrieval_latency_seconds":1.0,"answer_latency_seconds":1.5,
        "question":secrets[0],"answer":secrets[1],"prompt":secrets[3],
        "raw_model_response":secrets[4],"user_id":secrets[5],
        "evidence_text":json.dumps({"memory":secrets[2],"memory_id":secrets[6],
                                     "credentials":secrets[7],"environment":secrets[8]}),
        "evidence_count":2,"accepted_claim_count":1,"rejected_claim_count":1,
        "eligible_causal_evidence":True,
        "answer_diagnostics":{
            "answer_reason":"no_validated_claims",
            "model_response_status":"structured_response",
            "causal_gate_passed":False,
            "rejection_categories":{"claim_not_supported_by_source":1},
            "rejected_claims":[{"claim_text":secrets[1],"evidence_id":secrets[6]}],
        },
        "evaluation":{"expected_answer":secrets[1]},
    }
    progress=tmp_path/"progress.jsonl"
    callbacks=[]
    records,status=live.execute_cases(
        [{"case_id":"synthetic-case"}],lambda _case:[prediction],progress,
        progress_callback=callbacks.append)

    persisted=json.loads(progress.read_text(encoding="utf-8").splitlines()[0])
    serialized=progress.read_text(encoding="utf-8")
    assert status=="completed"
    assert records[0]["prediction"] is prediction
    assert all(secret not in serialized for secret in secrets)
    assert all(secret not in repr(callbacks) for secret in secrets)
    assert persisted["case_id"]=="synthetic-case"
    assert persisted["status"]=="completed"
    assert isinstance(persisted["elapsed_seconds"],float)
    assert persisted["prediction_summary"]=={
        "mode":"full","category":"FACTUAL","answer_behavior":"answer",
        "latency_seconds":2.5,"retrieval_latency_seconds":1.0,"answer_latency_seconds":1.5,
          "evidence_count":2,"accepted_claim_count":1,"rejected_claim_count":1,
          "eligible_causal_evidence":True,
          "answer_diagnostics":{
              "answer_reason":"no_validated_claims",
              "model_response_status":"structured_response",
              "causal_gate_passed":False,
              "rejection_categories":{"claim_not_supported_by_source":1},
          },
    }
    assert callbacks==[persisted]
    assert "question" not in persisted and "answer" not in persisted and "evidence_text" not in persisted


def test_live_report_has_versioned_metrics_and_explicit_undefined_values():
    row=_prediction()
    report=live.build_report([
        {"case_id":"case","status":"completed","prediction":row},
        {"case_id":"failed","status":"failed","error_type":"ConnectionError"}],
        [{"case_id":"case","run_scope":"live"},{"case_id":"failed","run_scope":"live"}],
        {"available":True},{"provider":"ollama","components":{"ollama_llm":"live"}},"completed_with_failures")
    assert report["schema_version"]==report["dataset_version"]=="1.0"
    assert report["evaluation_type"]=="live_pipeline"
    assert report["run_counts"]=={"cases_in_dataset":2,"attempted":2,"completed":1,"failed":1,"skipped":0}
    assert report["metrics"]["answer_behavior"]["answer_or_abstain_accuracy"]==1.0
    evidence=report["metrics"]["evidence_retrieval"]
    assert (evidence["true_positive_keys"],evidence["false_positive_keys"],evidence["false_negative_keys"])==(1,0,0)
    assert evidence["precision"]==evidence["recall"]==evidence["f1"]==1.0
    assert report["metrics"]["claim_citation_group_agreement"]["agreement"]==1.0
    assert report["metrics"]["metric_denominators"]["answerability"]=={
        "correct":1,"denominator":1,"abstention_true_positive":0,
        "predicted_abstentions":0,"expected_abstentions":0}
    assert report["metrics"]["causal_transition_id_accuracy"]["accuracy"] is None
    assert json.loads(json.dumps(report,sort_keys=True))["evaluation_type"]=="live_pipeline"
    empty=live.build_report([],[],{"available":False},{},"unavailable")
    assert empty["metrics"] is None
    assert empty["run_counts"]["completed"]==0
    assert empty["retrieval_failure_detection"]["metric"] is None


def test_deterministic_phase_51_results_remain_separate_and_reproducible():
    first=research_evaluator.evaluate_dataset()
    second=research_evaluator.evaluate_dataset()
    assert first==second
    assert first["evaluation_type"]=="deterministic_policy_fixture"
    report=live.build_report([],[],{"available":False},{},"unavailable")
    assert report["evaluation_type"]=="live_pipeline"


def test_unavailable_provider_still_uses_fresh_isolation_and_writes_skips(tmp_path):
    root=tmp_path/"fresh-live-run"
    output=root/"results"/"live.json"
    env=os.environ.copy()
    env.update({"LLM_PROVIDER":"openai","CAUSALMEM_DB":str(tmp_path/"normal.sqlite"),
                "MEM0_PATH":str(tmp_path/"normal-mem0"),"MEM0_DIR":str(tmp_path/"normal-config")})
    result=subprocess.run([str(PYTHON),str(PROJECT/"research_live_evaluator.py"),
        "--isolation-dir",str(root),"--dataset",str(live.DEFAULT_DATASET),"--output",str(output)],
        cwd=PROJECT,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    report=json.loads(output.read_text(encoding="utf-8"))
    assert report["evaluation_type"]=="live_pipeline"
    assert report["status"]=="unavailable"
    assert report["run_counts"]["completed"]==0
    assert report["run_counts"]["skipped"]==12
    assert (root/"storage"/"causalmem.sqlite3").exists() is False
    assert report["components"]["components"]["qdrant_mem0"]=="not_initialized"
    assert Path(report["storage_paths"]["sqlite"]).is_relative_to(root.resolve())
    assert Path(report["storage_paths"]["qdrant_mem0"]).is_relative_to(root.resolve())
    assert output.is_relative_to(root)
    assert output.with_name("live.progress.jsonl").is_file()
