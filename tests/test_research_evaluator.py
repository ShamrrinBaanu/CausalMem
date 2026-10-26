import json
from pathlib import Path

import pytest

import research_evaluator as research


def test_versioned_research_dataset_has_unique_ids_and_required_scenarios():
    cases=research.load_dataset()
    ids=[case["case_id"] for case in cases]
    assert len(ids)==len(set(ids))==10
    assert {case["dataset_version"] for case in cases}=={"1.0"}
    assert {"fact-direct","updated-current-fact","historical-asof-fact",
            "explicit-cause-supported","succession-is-not-cause","ambiguous-conflict",
            "unsupported-question-abstains","foreign-evidence-rejected",
            "claim-provenance-multiple-sources","sqlite-retrieval-failure-fails-closed"}==set(ids)


def test_dataset_rejects_duplicate_case_ids(tmp_path):
    path=tmp_path/"duplicate.jsonl"
    case={"dataset_version":"1.0","case_id":"same","category":"FACTUAL",
          "question":"Q?","user_id":"alice","evidence":{"user_id":"alice"},
          "model_response":{"claims":[]},"expected_behavior":"abstain","expected_source_refs":[],
          "expected_claim_source_refs":[]}
    path.write_text(json.dumps(case)+"\n"+json.dumps(case),encoding="utf-8")
    with pytest.raises(ValueError,match="unique"):
        research.load_dataset(path)


def test_known_metrics_include_denominators_and_expected_behavior():
    rows=[
        {"case_id":"a","category":"FACTUAL","expected_behavior":"answer","answer":"fact",
         "expected_source_refs":[{"kind":"memory","id":"m1"}],
         "expected_claim_source_refs":[[{"kind":"memory","id":"m1"}]],
         "source_refs":[{"kind":"memory","id":"m1"}],"retrieval_status":"complete"},
        {"case_id":"b","category":"FACTUAL","expected_behavior":"abstain",
         "answer":"I don't have enough information in your memory to determine that.",
         "expected_source_refs":[],"source_refs":[],"retrieval_status":"complete"},
    ]
    metrics=research.score_predictions(rows)
    assert metrics["answerability"]=={
        "n":2,"correct":2,"accuracy":1.0,"abstention_true_positive":1,
        "predicted_abstentions":1,"expected_abstentions":1,
        "abstention_precision":1.0,"abstention_recall":1.0}
    source=metrics["source_id_match_proxy"]
    assert (source["true_positive"],source["false_positive"],source["false_negative"])==(1,0,0)
    assert source["precision"]==source["recall"]==source["f1"]==1.0


def test_claim_support_metric_detects_citations_attached_to_wrong_claim():
    row={"case_id":"claim-map","category":"FACTUAL","expected_behavior":"answer",
        "answer":"Tea [Sources: memory:tea-id] Coffee [Sources: memory:coffee-id]",
        "expected_source_refs":[{"kind":"memory","id":"tea-id"},{"kind":"memory","id":"coffee-id"}],
        "expected_claim_source_refs":[[{"kind":"memory","id":"coffee-id"}],
                                       [{"kind":"memory","id":"tea-id"}]],
        "source_refs":[{"kind":"memory","id":"tea-id"},{"kind":"memory","id":"coffee-id"}],
        "retrieval_status":"complete"}
    metric=research.score_predictions([row])["claim_source_support_proxy"]
    assert metric["claim_groups"]==2
    assert metric["supported_claim_groups"]==0
    assert metric["rate"]==0


def test_empty_metrics_use_explicit_zero_denominators():
    metrics=research.score_predictions([])
    assert metrics["answerability"]["n"]==0
    assert metrics["answerability"]["accuracy"] is None
    assert metrics["source_id_match_proxy"]["precision"] is None
    assert metrics["source_id_match_proxy"]["recall"] is None
    assert metrics["claim_source_support_proxy"]["rate"] is None
    assert metrics["temporal_state_id_match"]["exact_match_accuracy"] is None
    assert metrics["retrieval_failure_detection"]["recall"] is None


def test_fixed_dataset_runs_deterministically_and_emits_versioned_json():
    cases=research.load_dataset()
    first=research.evaluate_dataset()
    second=research.evaluate_dataset()
    assert first==second
    assert first["schema_version"]==first["dataset_version"]=="1.0"
    assert first["evaluation_type"]=="deterministic_policy_fixture"
    encoded=json.dumps(first,sort_keys=True)
    assert json.loads(encoded)==first
    assert first["n_cases"]==10
    assert first["metrics"]["retrieval_failure_detection"]["detected"]==1
    assert first["metrics"]["temporal_state_id_match"]["exact_match_accuracy"]==1
    assert first["metrics"]["causal_transition_id_match"]["true_positive"]==1


def test_policy_cases_cover_expected_abstention_provenance_and_failure():
    cases=research.load_dataset()
    rows=research.run_cases(cases)
    by_id={row["case_id"]:row for row in rows}
    assert by_id["succession-is-not-cause"]["answer"]=="I don't have enough information in your memory to determine that."
    assert by_id["ambiguous-conflict"]["source_refs"]==[]
    assert by_id["foreign-evidence-rejected"]["source_refs"]==[]
    assert by_id["claim-provenance-multiple-sources"]["source_refs"]==[
        {"kind":"memory","id":"tea-a"},{"kind":"memory","id":"tea-b"}]
    assert by_id["sqlite-retrieval-failure-fails-closed"]["failure_detected"] is True
    assert by_id["sqlite-retrieval-failure-fails-closed"]["failure_source"]=="SQLite states"


def test_stale_state_supplement_is_versioned_and_abstains_on_ambiguous_evidence():
    path=Path(__file__).resolve().parents[1]/"evaluation"/"research_stale_state_v1.jsonl"
    cases=research.load_dataset(path)
    assert len(cases)==1
    assert cases[0]["fixture_set_version"]=="5.3-stale-state-v1"
    assert cases[0]["case_id"]=="stale-active-predecessor-current-abstains"
    report=research.evaluate_dataset(path)
    row=report["predictions"][0]
    assert row["answer"]=="I don't have enough information in your memory to determine that."
    assert row["source_refs"]==[]
    assert report["metrics"]["answerability"]=={
        "n":1,"correct":1,"accuracy":1.0,"abstention_true_positive":1,
        "predicted_abstentions":1,"expected_abstentions":1,
        "abstention_precision":1.0,"abstention_recall":1.0}
