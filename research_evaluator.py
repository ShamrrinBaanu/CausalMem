"""Deterministic policy evaluation for CausalMem's evidence/answer boundary.

This is not a live model-quality benchmark. Each case supplies a fixed evidence
bundle and a fixed structured model response so scoring is reproducible.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

SCHEMA_VERSION = "1.0"
DEFAULT_DATASET = Path(__file__).resolve().parent / "evaluation" / "research_v1.jsonl"
_SOURCE_BLOCK = re.compile(r"\[Sources:\s*([^\]]+)\]")
_SOURCE_REF = re.compile(r"^(memory|state|transition):(.+)$")


def load_dataset(path=DEFAULT_DATASET):
    cases=[]
    seen=set()
    for line_number,line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():
            continue
        try:
            case=json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid research dataset JSONL at line {line_number}") from exc
        if not isinstance(case,dict) or case.get("dataset_version")!=SCHEMA_VERSION:
            raise ValueError(f"Line {line_number}: unsupported dataset_version")
        case_id=case.get("case_id")
        if not isinstance(case_id,str) or not case_id.strip() or case_id in seen:
            raise ValueError(f"Line {line_number}: case_id must be unique and non-empty")
        seen.add(case_id)
        for key in ("category","question","user_id"):
            if not isinstance(case.get(key),str) or not case[key].strip():
                raise ValueError(f"Case {case_id!r}: {key} must be a non-empty string")
        if case.get("expected_behavior") not in {"answer","abstain","retrieval_error"}:
            raise ValueError(f"Case {case_id!r}: invalid expected_behavior")
        refs=case.get("expected_source_refs")
        if not isinstance(refs,list) or any(not _valid_ref(ref) for ref in refs):
            raise ValueError(f"Case {case_id!r}: expected_source_refs must contain kind/id objects")
        claim_refs=case.get("expected_claim_source_refs")
        if (not isinstance(claim_refs,list) or any(not isinstance(group,list)
                or any(not _valid_ref(ref) for ref in group) for group in claim_refs)):
            raise ValueError(f"Case {case_id!r}: expected_claim_source_refs must be a list of source-ref groups")
        if case["expected_behavior"]=="retrieval_error":
            if not isinstance(case.get("retrieval_failure_source"),str):
                raise ValueError(f"Case {case_id!r}: retrieval_failure_source is required")
        else:
            if not isinstance(case.get("evidence"),dict) or not isinstance(case.get("model_response"),dict):
                raise ValueError(f"Case {case_id!r}: evidence and model_response objects are required")
            if case["evidence"].get("user_id")!=case["user_id"]:
                raise ValueError(f"Case {case_id!r}: evidence user_id must match case user_id")
        for field in ("expected_temporal_state_ids","expected_causal_transition_ids"):
            if field in case and (not isinstance(case[field],list)
                                  or any(not isinstance(value,str) or not value for value in case[field])):
                raise ValueError(f"Case {case_id!r}: {field} must be a list of IDs")
        cases.append(case)
    return cases


def _valid_ref(value):
    return (isinstance(value,dict) and value.get("kind") in {"memory","state","transition"}
            and isinstance(value.get("id"),str) and bool(value["id"].strip()))


def _refs(answer):
    result=[]
    for block in _SOURCE_BLOCK.findall(answer or ""):
        for raw in block.split(","):
            match=_SOURCE_REF.fullmatch(raw.strip())
            if match:
                item={"kind":match.group(1),"id":match.group(2)}
                if item not in result:
                    result.append(item)
    return result


def _source_blocks(answer):
    blocks=[]
    for match in _SOURCE_BLOCK.finditer(answer or ""):
        refs=[]
        for raw in match.group(1).split(","):
            parsed=_SOURCE_REF.fullmatch(raw.strip())
            if parsed:
                refs.append({"kind":parsed.group(1),"id":parsed.group(2)})
        blocks.append(refs)
    return blocks


def _run_retrieval_failure(case):
    import hybrid_retriever

    source=case["retrieval_failure_source"]
    class BrokenStateStore:
        def states(self,*_args,**_kwargs):
            raise OSError("isolated deterministic store failure")

    with patch.object(hybrid_retriever,"search_memory",return_value=[]), \
         patch.object(hybrid_retriever,"db",BrokenStateStore()):
        try:
            hybrid_retriever.retrieve(case["question"],case["user_id"],case["category"])
        except hybrid_retriever.RetrievalFailure as exc:
            return {"retrieval_status":"failed","failure_source":exc.source,
                    "failure_detected":exc.source==source,"answer":None,"source_refs":[]}
    return {"retrieval_status":"complete","failure_source":None,
            "failure_detected":False,"answer":None,"source_refs":[]}


def run_cases(cases):
    """Run fixed model responses through the current deterministic answer gate."""
    import answer_generator

    predictions=[]
    for case in cases:
        if case["expected_behavior"]=="retrieval_error":
            result=_run_retrieval_failure(case)
        else:
            class FixedResponse:
                def answer_with_provenance(self,_question,_evidence):
                    return case["model_response"]
            with patch.object(answer_generator,"llm",FixedResponse()):
                answer=answer_generator.generate_answer(case["question"],case["evidence"])
            if "expected_temporal_state_ids" in case:
                from evaluator import state_valid_at
                requested=case.get("requested_timestamp")
                temporal_valid_ids=[]
                for state in case["evidence"].get("states",[]):
                    is_valid=(state_valid_at(state,requested) if requested is not None
                              else state.get("status")=="ACTIVE")
                    if is_valid:
                        temporal_valid_ids.append(state.get("state_id"))
            else:
                temporal_valid_ids=None
            result={"retrieval_status":"complete","failure_source":None,
                    "failure_detected":False,"answer":answer,"source_refs":_refs(answer),
                    "temporal_valid_state_ids":temporal_valid_ids}
        predictions.append({
            "case_id":case["case_id"],"category":case["category"],
            "expected_behavior":case["expected_behavior"],"expected_source_refs":case["expected_source_refs"],
            "expected_claim_source_refs":case["expected_claim_source_refs"],
            "expected_temporal_state_ids":case.get("expected_temporal_state_ids"),
            "expected_causal_transition_ids":case.get("expected_causal_transition_ids"),
            **result,
        })
    return predictions


def score_predictions(predictions):
    """Calculate count-backed policy metrics; zero denominators are null."""
    if not predictions:
        return {"answerability":{"n":0,"correct":0,"accuracy":None,
                    "abstention_true_positive":0,"predicted_abstentions":0,"expected_abstentions":0,
                    "abstention_precision":None,"abstention_recall":None},
                "source_id_match_proxy":{"true_positive":0,"false_positive":0,"false_negative":0,
                    "precision":None,"recall":None,"f1":None,
                    "interpretation":"Exact expected source-ID agreement; does not establish semantic entailment."},
                "claim_source_support_proxy":{"supported_claim_groups":0,"claim_groups":0,"rate":None,
                    "interpretation":"Rendered citation groups exactly matching their case-specific expected source refs."},
                "temporal_state_id_match":{"exact_cases":0,"cases":0,"exact_match_accuracy":None,
                    "interpretation":"Exact cited state-ID match for temporal annotations; no event-time inference."},
                "causal_transition_id_match":{"true_positive":0,"false_positive":0,"false_negative":0,"cases":0,
                    "precision":None,"recall":None,
                    "interpretation":"Exact cited transition-ID agreement; relation correctness is separately asserted by deterministic cases."},
                "retrieval_failure_detection":{"detected":0,"expected_failures":0,"true_positive":0,
                    "predicted_failures":0,"precision":None,"recall":None},"by_category":{}}
    eligible=[row for row in predictions if row["expected_behavior"]!="retrieval_error"]
    answer_correct=abstain_tp=predicted_abstain=expected_abstain=0
    source_tp=source_fp=source_fn=0
    support_groups=supported_groups=0
    temporal_cases=temporal_exact=0
    causal_tp=causal_fp=causal_fn=causal_cases=0
    failure_expected=failure_detected=failure_predicted=0
    categories=defaultdict(lambda:{"n":0,"correct":0})
    for row in predictions:
        failure_predicted+=row.get("retrieval_status")=="failed"
        if row["expected_behavior"]=="retrieval_error":
            failure_expected+=1
            failure_detected+=bool(row.get("failure_detected"))
            continue
        category=categories[row["category"]]
        category["n"]+=1
        expected=row["expected_behavior"]
        is_abstain=row.get("answer")=="I don't have enough information in your memory to determine that."
        actual="abstain" if is_abstain else "answer"
        correct=actual==expected
        answer_correct+=correct
        category["correct"]+=correct
        predicted_abstain+=is_abstain
        expected_abstain+=expected=="abstain"
        abstain_tp+=is_abstain and expected=="abstain"

        expected_refs={ (ref["kind"],ref["id"]) for ref in row["expected_source_refs"] }
        actual_refs={ (ref["kind"],ref["id"]) for ref in row.get("source_refs",[]) }
        source_tp+=len(expected_refs & actual_refs)
        source_fp+=len(actual_refs-expected_refs)
        source_fn+=len(expected_refs-actual_refs)
        actual_groups=_source_blocks(row.get("answer") or "")
        expected_groups=row.get("expected_claim_source_refs",[])
        support_groups+=max(len(actual_groups),len(expected_groups))
        for index in range(max(len(actual_groups),len(expected_groups))):
            actual_group={(ref["kind"],ref["id"]) for ref in actual_groups[index]} if index<len(actual_groups) else set()
            expected_group={(ref["kind"],ref["id"]) for ref in expected_groups[index]} if index<len(expected_groups) else set()
            supported_groups+=bool(actual_group) and actual_group==expected_group

        expected_states=row.get("expected_temporal_state_ids")
        if expected_states is not None:
            temporal_cases+=1
            actual_states={ref["id"] for ref in row.get("source_refs",[]) if ref["kind"]=="state"}
            valid_states=set(row.get("temporal_valid_state_ids") or [])
            temporal_exact+=actual_states==set(expected_states) and actual_states.issubset(valid_states)
        expected_edges=row.get("expected_causal_transition_ids")
        if expected_edges is not None:
            causal_cases+=1
            actual_edges={ref["id"] for ref in row.get("source_refs",[]) if ref["kind"]=="transition"}
            causal_tp+=len(actual_edges & set(expected_edges))
            causal_fp+=len(actual_edges-set(expected_edges))
            causal_fn+=len(set(expected_edges)-actual_edges)

    predicted=source_tp+source_fp; relevant=source_tp+source_fn
    p=source_tp/predicted if predicted else (1.0 if relevant==0 else None)
    r=source_tp/relevant if relevant else None
    source_f1=(2*p*r/(p+r) if p is not None and r is not None and p+r else
               1.0 if p==1 and r is None else 0.0 if predicted or relevant else None)
    return {
        "answerability":{"n":len(eligible),"correct":answer_correct,
            "accuracy":answer_correct/len(eligible) if eligible else None,
            "abstention_true_positive":abstain_tp,"predicted_abstentions":predicted_abstain,
            "expected_abstentions":expected_abstain,
            "abstention_precision":abstain_tp/predicted_abstain if predicted_abstain else None,
            "abstention_recall":abstain_tp/expected_abstain if expected_abstain else None},
        "source_id_match_proxy":{"true_positive":source_tp,"false_positive":source_fp,
            "false_negative":source_fn,"precision":p,"recall":r,"f1":source_f1,
            "interpretation":"Exact expected source-ID agreement; does not establish semantic entailment."},
        "claim_source_support_proxy":{"supported_claim_groups":supported_groups,
            "claim_groups":support_groups,
            "rate":supported_groups/support_groups if support_groups else None,
            "interpretation":"Rendered citation groups exactly matching their case-specific expected source refs."},
        "temporal_state_id_match":{"exact_cases":temporal_exact,"cases":temporal_cases,
            "exact_match_accuracy":temporal_exact/temporal_cases if temporal_cases else None,
            "interpretation":"Exact cited state-ID match for temporal annotations; no event-time inference."},
        "causal_transition_id_match":{"true_positive":causal_tp,"false_positive":causal_fp,
            "false_negative":causal_fn,"cases":causal_cases,
            "precision":causal_tp/(causal_tp+causal_fp) if causal_tp+causal_fp else (1.0 if causal_tp+causal_fn==0 and causal_cases else None),
            "recall":causal_tp/(causal_tp+causal_fn) if causal_tp+causal_fn else None,
            "interpretation":"Exact cited transition-ID agreement; relation correctness is separately asserted by deterministic cases."},
        "retrieval_failure_detection":{"detected":failure_detected,"expected_failures":failure_expected,
            "true_positive":failure_detected,"predicted_failures":failure_predicted,
            "precision":failure_detected/failure_predicted if failure_predicted else None,
            "recall":failure_detected/failure_expected if failure_expected else None},
        "by_category":{key:{"n":value["n"],"correct":value["correct"],
            "accuracy":value["correct"]/value["n"] if value["n"] else None}
            for key,value in sorted(categories.items())},
    }


def evaluate_dataset(path=DEFAULT_DATASET):
    cases=load_dataset(path)
    predictions=run_cases(cases)
    return {"schema_version":SCHEMA_VERSION,"dataset_version":SCHEMA_VERSION,
            "evaluation_type":"deterministic_policy_fixture","n_cases":len(cases),
            "metrics":score_predictions(predictions),"predictions":predictions,
            "limitations":["Fixed mocked claim responses evaluate policy and citation validation, not live model quality.",
                           "Lexical support and exact ID agreement are proxies, not semantic entailment.",
                           "Latency is not measured in this deterministic harness."]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolation-dir",required=True)
    parser.add_argument("--dataset",default=str(DEFAULT_DATASET))
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    # Reuse the benchmark's established fail-closed path checks. Import the
    # storage-aware application only after overrides have been installed.
    import benchmark
    isolation=benchmark._configure_isolation(args.isolation_dir)
    output=Path(args.output).expanduser().resolve()
    paths=benchmark._resolve_evaluation_paths(
        isolation,output.parent,args.dataset,output,False)
    report=evaluate_dataset(paths.dataset_path)
    with paths.result_path.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False)
    print(f"Saved deterministic evaluation to {paths.result_path}",flush=True)


if __name__=="__main__":
    main()
