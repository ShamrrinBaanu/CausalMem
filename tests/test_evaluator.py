from pathlib import Path

import pytest

from benchmark import _default_dataset_path, _fixture_keys_for_ids, _read_dataset
from evaluator import ABSTENTION, evaluate, state_valid_at


PROJECT=Path(__file__).resolve().parents[1]
DATASET=PROJECT/"evaluation"/"ground_truth.jsonl"


def _annotation(case_id,behavior="answer",**extra):
    return {"case_id":case_id,"expected_behavior":behavior,"expected_evidence_keys":[],
            "required_answer_facts":[],"forbidden_answer_facts":[],**extra}


def test_ground_truth_jsonl_has_stable_keys_and_manually_checkable_coverage():
    assert _default_dataset_path()==DATASET.resolve()
    cases=_read_dataset(DATASET)
    assert 12<=len(cases)<=20
    assert len({case["case_id"] for case in cases})==len(cases)
    required_topics={"cause-running-swimming","cause-missing","succession-not-cause","ambiguous-cause",
        "current-preference","historical-preference","asof-before-start","asof-at-start",
        "asof-before-end","asof-at-end","asof-after-end","conflicting-superseded",
        "relevant-with-distractor","empty-history","cross-user-distractor",
        "supported-noncausal-answer","no-support-abstain","cause-anchor-unrelated"}
    assert {case["case_id"] for case in cases}==required_topics
    for case in cases:
        fixture_keys={entry["fixture_key"] for entry in case.get("conversation",[])}
        for distractor in case.get("distractor_users",[]):
            fixture_keys.update(entry["fixture_key"] for entry in distractor["conversation"])
        assert set(case["expected_evidence_keys"])<=fixture_keys
        assert set(case.get("expected_causal_evidence_keys",[]))<=fixture_keys
        assert set(case.get("expected_transition_provenance_keys",[]))<=fixture_keys
        assert case["expected_behavior"] in {"answer","abstain"}
    assert len([case for case in cases if case.get("temporal_oracle")])==5


def test_fixture_key_mapping_is_stable_and_omits_ambiguous_ids():
    mapping={"mem-a":{"running"},"mem-merged":{"first","second"}}
    assert _fixture_keys_for_ids(["mem-a","mem-merged","unmapped"],mapping)==["running"]


def test_temporal_oracle_boundaries_follow_half_open_recorded_intervals():
    cases=_read_dataset(DATASET)
    for case in cases:
        oracle=case.get("temporal_oracle")
        if not oracle: continue
        actual=sorted(state["fixture_key"] for state in oracle["states"]
                      if state_valid_at(state,case["requested_timestamp"]))
        assert actual==sorted(oracle["expected_state_keys"]),case["case_id"]
    old={"valid_from":"2025-01-01T00:00:00Z","valid_until":"2025-06-01T00:00:00Z"}
    assert not state_valid_at(old,"2024-12-31T23:59:59Z")
    assert state_valid_at(old,"2025-01-01T00:00:00Z")
    assert not state_valid_at(old,"2025-06-01T00:00:00Z")
    assert state_valid_at({"valid_from":"2025-01-01T00:00:00Z","valid_until":None},"2026-01-01T00:00:00Z")
    assert not state_valid_at({"valid_from":None,"valid_until":None},"2026-01-01T00:00:00Z")
    assert not state_valid_at({"valid_from":"bad-date","valid_until":None},"2026-01-01T00:00:00Z")
    assert state_valid_at({"valid_from":"2025-01-01T00:00:00","valid_until":None},"2026-01-01T00:00:00Z")
    assert not state_valid_at({"valid_from":"2025-01-01T00:00:00Z","valid_until":None},"2026-01-01T00:00:00")


def test_evidence_and_answer_metrics_cover_partial_empty_and_irrelevant_cases():
    rows=[
        {"mode":"hybrid","case_id":"partial","answer":"Fact A because invented","answer_behavior":"answer",
         "evaluation":_annotation("partial","answer",expected_evidence_keys=["a","b"],required_answer_facts=["Fact A"],forbidden_answer_facts=["invented"],expected_causal_evidence_keys=["cause"]),
         "evidence_fixture_keys":["a","noise"],"causal_evidence_fixture_keys":[],"eligible_causal_evidence":False,"latency_seconds":1.0},
        {"mode":"hybrid","case_id":"missed","answer":ABSTENTION,"answer_behavior":"abstain",
         "evaluation":_annotation("missed","answer",expected_evidence_keys=["b"],required_answer_facts=["Fact B"]),
         "evidence_fixture_keys":[],"causal_evidence_fixture_keys":[],"eligible_causal_evidence":False,"latency_seconds":3.0},
        {"mode":"hybrid","case_id":"empty-correct","answer":ABSTENTION,"answer_behavior":"abstain",
         "evaluation":_annotation("empty-correct","abstain"),"evidence_fixture_keys":[],"causal_evidence_fixture_keys":[],
         "eligible_causal_evidence":False,"latency_seconds":2.0},
        {"mode":"hybrid","case_id":"empty-noisy","answer":"Noise is present","answer_behavior":"answer",
         "evaluation":_annotation("empty-noisy","abstain"),"evidence_fixture_keys":["noise"],
         "causal_evidence_fixture_keys":[],"eligible_causal_evidence":False},
    ]
    metrics=evaluate(rows)["metrics"]["hybrid"]
    evidence=metrics["evidence_retrieval"]
    assert evidence["precision"]==pytest.approx(1/3)
    assert evidence["recall"]==pytest.approx(1/3)
    assert evidence["f1"]==pytest.approx(1/3)
    assert evidence["cases_with_no_expected_relevant_evidence"]==2
    behavior=metrics["answer_behavior"]
    assert behavior["answer_or_abstain_accuracy"]==.5
    assert behavior["correct_answer_coverage"]==.5
    assert behavior["incorrect_answer_rate_among_answered"]==1
    assert behavior["abstention_precision"]==.5 and behavior["abstention_recall"]==.5
    facts=metrics["answer_facts_proxy"]
    assert facts["required_fact_recall_proxy"]==.5
    assert facts["forbidden_fact_violation_rate_proxy"]==1
    assert facts["manual_review_required"] is True
    causal=metrics["causal_claims_proxy"]
    assert causal["unsupported_causal_assertion_count_proxy"]==1
    assert causal["causal_evidence_recall"]==0
    assert causal["manual_review_required_count"]==1


def test_transition_provenance_is_scored_separately_without_relaxing_exact_sets():
    row={"mode":"full","case_id":"historical-predecessor","answer":"Learn guitar",
        "evaluation":_annotation("historical-predecessor",
            expected_evidence_keys=["history-guitar"],
            expected_historical_state_keys=["history-guitar"],
            expected_transition_provenance_keys=["history-piano"]),
        "evidence_fixture_keys":["history-guitar","history-piano"],
        "transition_provenance_fixture_keys":["history-piano"],
        "state_evidence":[{"status":"INACTIVE","source_fixture_keys":["history-guitar"]}],
        "eligible_causal_evidence":False}

    metrics=evaluate([row])["metrics"]["full"]

    # Evidence scoring recognizes the explicitly annotated transition
    # provenance, while historical state scoring still expects only the source.
    assert metrics["evidence_retrieval"]["precision"]==1
    assert metrics["evidence_retrieval"]["recall"]==1
    assert metrics["temporal_evidence"]["evidence_consistency_accuracy"]==1
    assert metrics["transition_provenance"]=={
        "precision":1.0,"recall":1.0,"f1":1.0,"true_positive_keys":1,
        "false_positive_keys":0,"false_negative_keys":0,"cases_scored":1,
        "interpretation":"Exact fixture-key agreement for explicitly annotated same-user transition source provenance."}

    row["evidence_fixture_keys"].append("unrelated")
    row["transition_provenance_fixture_keys"].append("unrelated")
    metrics=evaluate([row])["metrics"]["full"]
    assert metrics["evidence_retrieval"]["precision"]==pytest.approx(2/3)
    assert metrics["evidence_retrieval"]["recall"]==1
    assert metrics["transition_provenance"]["precision"]==pytest.approx(1/2)


def test_temporal_evidence_metrics_check_current_and_asof_states():
    rows=[
        {"mode":"full","case_id":"current","answer":"Java","evaluation":_annotation("current",expected_current_state_keys=["java"]),
         "evidence_fixture_keys":["python","java"],"eligible_causal_evidence":False,
         "state_evidence":[{"status":"INACTIVE","source_fixture_keys":["python"]},{"status":"ACTIVE","source_fixture_keys":["java"]}]},
        {"mode":"full","case_id":"asof","answer":"Java","evaluation":_annotation("asof",expected_temporal_state_keys=["java"],requested_timestamp="2025-06-01T00:00:00Z"),
         "evidence_fixture_keys":["python","java"],"eligible_causal_evidence":False,
         "state_evidence":[{"source_fixture_keys":["python"],"valid_from":"2025-01-01T00:00:00Z","valid_until":"2025-06-01T00:00:00Z"},
                           {"source_fixture_keys":["java"],"valid_from":"2025-06-01T00:00:00Z","valid_until":None}]},
        {"mode":"mem0","case_id":"not-applicable","answer":"Java","evaluation":_annotation("not-applicable",expected_current_state_keys=["java"]),
         "evidence_fixture_keys":[],"eligible_causal_evidence":False,"state_evidence":None},
    ]
    temporal=evaluate(rows)["metrics"]["full"]["temporal_evidence"]
    assert temporal["evidence_consistency_accuracy"]==1
    assert temporal["cases_scored"]==2
    mem0=evaluate(rows)["metrics"]["mem0"]["temporal_evidence"]
    assert mem0["evidence_consistency_accuracy"] is None and mem0["cases_not_applicable"]==1


def test_end_to_end_temporal_metric_scores_returned_keys_without_oracle_filtering():
    rows=[
        {"mode":"full","case_id":"retrieval-returned","answer":"Java",
         "evaluation":_annotation("retrieval-returned",expected_temporal_state_keys=["java"],
             requested_timestamp="2025-06-01T00:00:00Z",temporal_metric_scope="end_to_end"),
         "evidence_fixture_keys":["java"],"eligible_causal_evidence":False,
         # Deliberately inconsistent interval: this metric must report what
         # retrieval returned, not re-filter it with oracle timestamps.
         "state_evidence":[{"source_fixture_keys":["java"],
             "valid_from":"2025-06-02T00:00:00Z","valid_until":None}]},
        {"mode":"full","case_id":"oracle-excluded","answer":"Python",
         "evaluation":_annotation("oracle-excluded",expected_temporal_state_keys=["python"],
             requested_timestamp="2025-01-01T00:00:00Z",temporal_metric_scope="oracle_only"),
         "evidence_fixture_keys":["python"],"eligible_causal_evidence":False,
         "state_evidence":[{"source_fixture_keys":["python"],
             "valid_from":"2025-01-01T00:00:00Z","valid_until":None}]},
    ]
    metrics=evaluate(rows)["metrics"]["full"]
    e2e=metrics["end_to_end_temporal_evidence"]
    assert e2e["precision"]==e2e["recall"]==e2e["f1"]==1
    assert e2e["per_case"][0]["retrieved_state_keys"]==["java"]
    assert metrics["temporal_evidence"]["cases_scored"]==0
    assert metrics["temporal_evidence"]["cases_not_applicable"]==1


def test_latency_reports_mean_median_p95_and_missing_timing_without_fabrication():
    rows=[{"mode":"x","answer":"a","latency_seconds":value,"retrieval_latency_seconds":value/2,
           "answer_latency_seconds":value/2} for value in range(1,21)]
    rows.append({"mode":"x","answer":"missing"})
    latency=evaluate(rows)["metrics"]["x"]["latency"]
    assert latency["total"]["n"]==20
    assert latency["total"]["mean_seconds"]==10.5
    assert latency["total"]["median_seconds"]==10.5
    assert latency["total"]["p95_seconds"]==19
    assert latency["retrieval"]["n"]==20
    assert latency["answer_generation"]["n"]==20


def test_zero_denominators_empty_rows_and_malformed_annotations_are_explicit(tmp_path):
    assert evaluate([])=={"metrics":{},"predictions":[]}
    perfect_empty={"mode":"x","case_id":"empty","answer":ABSTENTION,
        "evaluation":_annotation("empty","abstain"),"evidence_fixture_keys":[],"eligible_causal_evidence":False,"state_evidence":None}
    report=evaluate([perfect_empty])["metrics"]["x"]
    assert report["evidence_retrieval"]["precision"]==1
    assert report["evidence_retrieval"]["recall"] is None
    malformed={**perfect_empty,"evaluation":{**perfect_empty["evaluation"],"expected_evidence_keys":"not-a-list"}}
    with pytest.raises(ValueError,match="expected_evidence_keys"):
        evaluate([malformed])
    bad_jsonl=tmp_path/"invalid.jsonl"
    bad_jsonl.write_text('{"case_id":',encoding="utf-8")
    with pytest.raises(ValueError,match="Invalid JSONL"):
        _read_dataset(bad_jsonl)
