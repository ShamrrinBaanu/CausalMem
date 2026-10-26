"""Deterministic benchmark metrics over explicit fixture-key annotations.

Fact matching is a normalized, case-insensitive substring proxy, not semantic
entailment. Causal-language detection is also a lexical proxy and requires
manual review for claims that matter.
"""
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ABSTENTION="I don't have enough information in your memory to determine that."
_CAUSAL_LANGUAGE=re.compile(r"\b(?:because|because of|due to|caused by|cause was|the cause|as a result of|led to|resulted in)\b",re.I)


def _require_string_list(value,name,case_id):
    if not isinstance(value,list) or any(not isinstance(item,str) or not item.strip() for item in value):
        raise ValueError(f"Case {case_id!r}: {name} must be a list of non-empty strings")
    return value


def _fact_present(answer,fact):
    normalized=" ".join(str(answer).casefold().split())
    needle=" ".join(str(fact).casefold().split())
    return bool(needle) and needle in normalized


def _is_abstention(answer):
    return isinstance(answer,str) and " ".join(answer.casefold().split()) == " ".join(ABSTENTION.casefold().split())


def _aware_datetime(value):
    if not isinstance(value,str):
        return None
    try:
        result=datetime.fromisoformat(value.replace("Z","+00:00"))
    except ValueError:
        return None
    if result.tzinfo is None or result.utcoffset() is None:
        return None
    return result.astimezone(timezone.utc)


def _stored_datetime(value):
    """SQLite julianday treats naive stored ISO timestamps as UTC."""
    if not isinstance(value,str):
        return None
    try:
        result=datetime.fromisoformat(value.replace("Z","+00:00"))
    except ValueError:
        return None
    if result.tzinfo is None or result.utcoffset() is None:
        result=result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def state_valid_at(state,requested_timestamp):
    """Apply recorded-time half-open validity; requested time must be aware.

    Stored naive ISO timestamps follow SQLite julianday's UTC interpretation;
    null/malformed valid_from and malformed valid_until fail closed.
    """
    if not isinstance(state,dict):
        return False
    requested=_aware_datetime(requested_timestamp)
    valid_from=_stored_datetime(state.get("valid_from"))
    if requested is None or valid_from is None or valid_from>requested:
        return False
    valid_until_value=state.get("valid_until")
    if valid_until_value is None:
        return True
    valid_until=_stored_datetime(valid_until_value)
    return valid_until is not None and requested<valid_until


def _fact_metrics(rows):
    required_total=required_found=forbidden_total=forbidden_found=0
    cases_with_required=required_cases_passed=0
    for row in rows:
        annotation=row["evaluation"]
        required=_require_string_list(annotation.get("required_answer_facts",[]),"required_answer_facts",row["case_id"])
        forbidden=_require_string_list(annotation.get("forbidden_answer_facts",[]),"forbidden_answer_facts",row["case_id"])
        answer=row.get("answer","")
        required_matches=sum(_fact_present(answer,fact) for fact in required)
        forbidden_matches=sum(_fact_present(answer,fact) for fact in forbidden)
        required_total+=len(required); required_found+=required_matches
        forbidden_total+=len(forbidden); forbidden_found+=forbidden_matches
        if required:
            cases_with_required+=1
            if required_matches==len(required) and forbidden_matches==0:
                required_cases_passed+=1
    return {
        "required_fact_recall_proxy": required_found/required_total if required_total else None,
        "forbidden_fact_violation_rate_proxy": forbidden_found/forbidden_total if forbidden_total else None,
        "all_required_and_no_forbidden_case_rate_proxy": required_cases_passed/cases_with_required if cases_with_required else None,
        "manual_review_required": True,
        "matching_rule":"normalized, case-insensitive substring; not semantic entailment",
    }


def _evidence_metrics(rows):
    tp=fp=fn=0; no_expected=0; scored=0
    for row in rows:
        annotation=row["evaluation"]
        expected=set(_require_string_list(annotation.get("expected_evidence_keys",[]),"expected_evidence_keys",row["case_id"]))
        # Only cases that explicitly annotate transition provenance include
        # those keys in their exact evidence-set target.
        expected.update(_require_string_list(
            annotation.get("expected_transition_provenance_keys",[]),
            "expected_transition_provenance_keys",row["case_id"]))
        retrieved=set(_require_string_list(row.get("evidence_fixture_keys",[]),"evidence_fixture_keys",row["case_id"]))
        if not expected: no_expected+=1
        tp+=len(expected & retrieved); fp+=len(retrieved-expected); fn+=len(expected-retrieved); scored+=1
    predicted=tp+fp; relevant=tp+fn
    precision=tp/predicted if predicted else (1.0 if relevant==0 else None)
    recall=tp/relevant if relevant else None
    f1=(2*precision*recall/(precision+recall)) if precision is not None and recall is not None and precision+recall else (1.0 if precision==1 and recall is None else 0.0 if predicted or relevant else 1.0)
    return {"precision":precision,"recall":recall,"f1":f1,"true_positive_keys":tp,
            "false_positive_keys":fp,"false_negative_keys":fn,"cases_scored":scored,
            "fixture_key_mapping_ambiguities":sum(int(row.get("fixture_key_mapping_ambiguities",0) or 0) for row in rows),
            "cases_with_no_expected_relevant_evidence":no_expected,
            "empty_rule":"no retrieved and no expected keys => precision/F1 1, recall N/A; no retrieval when relevant keys exist => precision N/A, recall/F1 0"}


def _transition_provenance_metrics(rows):
    tp=fp=fn=cases=0
    for row in rows:
        annotation=row.get("evaluation",{})
        if "expected_transition_provenance_keys" not in annotation:
            continue
        expected=set(_require_string_list(annotation["expected_transition_provenance_keys"],
                                          "expected_transition_provenance_keys",row["case_id"]))
        actual=set(_require_string_list(row.get("transition_provenance_fixture_keys",[]),
                                        "transition_provenance_fixture_keys",row["case_id"]))
        tp+=len(expected&actual); fp+=len(actual-expected); fn+=len(expected-actual); cases+=1
    precision=tp/(tp+fp) if tp+fp else (1.0 if tp+fn==0 and cases else None)
    recall=tp/(tp+fn) if tp+fn else (1.0 if cases else None)
    f1=(2*precision*recall/(precision+recall) if precision is not None and recall is not None
        and precision+recall else (1.0 if precision==1 and recall==1 else 0.0 if cases else None))
    return {"precision":precision,"recall":recall,"f1":f1,"true_positive_keys":tp,
            "false_positive_keys":fp,"false_negative_keys":fn,"cases_scored":cases,
            "interpretation":"Exact fixture-key agreement for explicitly annotated same-user transition source provenance."}


def _answer_behavior_metrics(rows):
    evaluated=[row for row in rows if "evaluation" in row]
    if not evaluated:
        return {"n":0,"accuracy":None,"correct_answer_coverage":None,"incorrect_answer_rate_among_answered":None,
                "abstention_precision":None,"abstention_recall":None}
    correct=should_answer=answered_expected=incorrect_answered=factually_correct_answered=0
    true_abstain=predicted_abstain=expected_abstain=0
    for row in evaluated:
        annotation=row["evaluation"]; expected=annotation.get("expected_behavior")
        if expected not in {"answer","abstain"}:
            raise ValueError(f"Case {row.get('case_id')!r}: expected_behavior must be 'answer' or 'abstain'")
        abstained=_is_abstention(row.get("answer"))
        actual="abstain" if abstained else "answer"
        correct+=actual==expected; predicted_abstain+=abstained; expected_abstain+=expected=="abstain"
        true_abstain+=(abstained and expected=="abstain")
        if expected=="answer":
            should_answer+=1
            if not abstained:
                answered_expected+=1
                required=annotation.get("required_answer_facts",[])
                forbidden=annotation.get("forbidden_answer_facts",[])
                good=all(_fact_present(row.get("answer",""),fact) for fact in required)
                good=good and not any(_fact_present(row.get("answer",""),fact) for fact in forbidden)
                incorrect_answered+=not good
                factually_correct_answered+=good
    abstention_precision=true_abstain/predicted_abstain if predicted_abstain else None
    abstention_recall=true_abstain/expected_abstain if expected_abstain else None
    if abstention_precision is None or abstention_recall is None:
        abstention_f1=None
    elif abstention_precision+abstention_recall:
        abstention_f1=2*abstention_precision*abstention_recall/(abstention_precision+abstention_recall)
    else:
        abstention_f1=0.0
    return {"n":len(evaluated),"answer_or_abstain_accuracy":correct/len(evaluated),
            "correct_answer_coverage":answered_expected/should_answer if should_answer else None,
            "incorrect_answer_rate_among_answered":incorrect_answered/answered_expected if answered_expected else None,
            "incorrect_or_abstained_rate_among_should_answer":1-factually_correct_answered/should_answer if should_answer else None,
            "abstention_precision":abstention_precision,"abstention_recall":abstention_recall,
            "abstention_f1":abstention_f1}


def _causal_metrics(rows):
    claims=unsupported=manual=unverified=0; causal_tp=causal_fp=causal_fn=0; causal_cases=0
    support_methods=defaultdict(int)
    for row in rows:
        annotation=row.get("evaluation",{})
        if isinstance(annotation,dict) and "expected_causal_evidence_keys" in annotation:
            expected=set(_require_string_list(annotation["expected_causal_evidence_keys"],"expected_causal_evidence_keys",row.get("case_id")))
            retrieved=set(_require_string_list(row.get("causal_evidence_fixture_keys",[]),"causal_evidence_fixture_keys",row.get("case_id")))
            causal_tp+=len(expected & retrieved); causal_fp+=len(retrieved-expected); causal_fn+=len(expected-retrieved); causal_cases+=1
        answer=row.get("answer","")
        has_claim=isinstance(answer,str) and bool(_CAUSAL_LANGUAGE.search(answer)) and not _is_abstention(answer)
        if not has_claim: continue
        claims+=1; manual+=1
        eligible=row.get("eligible_causal_evidence")
        if not isinstance(eligible,bool):
            unverified+=1
            continue
        support_methods[row.get("causal_support_method","unspecified")]+=1
        unsupported+=not eligible
    causal_precision=causal_tp/(causal_tp+causal_fp) if causal_tp+causal_fp else (1.0 if causal_tp+causal_fn==0 else None)
    causal_recall=causal_tp/(causal_tp+causal_fn) if causal_tp+causal_fn else None
    if causal_precision is not None and causal_recall is not None:
        causal_f1=(2*causal_precision*causal_recall/(causal_precision+causal_recall)
                   if causal_precision+causal_recall else 0.0)
    elif causal_cases and causal_tp+causal_fp+causal_fn==0:
        causal_f1=1.0
    else:
        causal_f1=0.0 if causal_tp+causal_fp+causal_fn else None
    return {"causal_evidence_precision":causal_precision,"causal_evidence_recall":causal_recall,
            "causal_evidence_f1":causal_f1,
            "causal_evidence_cases_scored":causal_cases,
            "causal_assertion_count_proxy":claims,
            "unsupported_causal_assertion_count_proxy":unsupported,
            "unsupported_causal_assertion_rate_proxy":unsupported/(claims-unverified) if claims>unverified else None,
            "causal_claims_without_support_check":unverified,
            "manual_review_required_count":manual,
            "causal_support_method_counts":dict(support_methods),
            "claim_detection_rule":"lexical causal markers only; bare causal nouns and implicit claims may be missed"}


def _temporal_metrics(rows):
    scored=[]; not_applicable=0
    for row in rows:
        annotation=row.get("evaluation",{})
        current=annotation.get("expected_current_state_keys")
        historical=annotation.get("expected_historical_state_keys")
        at_time=annotation.get("expected_temporal_state_keys")
        if current is None and historical is None and at_time is None:
            continue
        if annotation.get("temporal_metric_scope")=="end_to_end":
            # End-to-end scores compare directly against states actually
            # returned by retrieval; do not apply ground-truth intervals here.
            continue
        if annotation.get("temporal_metric_scope")=="oracle_only" or row.get("state_evidence") is None:
            not_applicable+=1
            continue
        states=row["state_evidence"]
        if not isinstance(states,list) or any(not isinstance(state,dict) for state in states):
            raise ValueError(f"Case {row.get('case_id')!r}: state_evidence must be a list of state objects or null")
        for state in states:
            keys=state.get("source_fixture_keys",[])
            if not isinstance(keys,list) or any(not isinstance(key,str) or not key.strip() for key in keys):
                raise ValueError(f"Case {row.get('case_id')!r}: source_fixture_keys must be a list of non-empty strings")
        available=set()
        expected=set()
        if current is not None:
            expected.update(_require_string_list(current,"expected_current_state_keys",row["case_id"]))
            available.update(key for state in states if state.get("status")=="ACTIVE" for key in state.get("source_fixture_keys",[]))
        if historical is not None:
            expected.update(_require_string_list(historical,"expected_historical_state_keys",row["case_id"]))
            available.update(key for state in states for key in state.get("source_fixture_keys",[]))
        if at_time is not None:
            expected.update(_require_string_list(at_time,"expected_temporal_state_keys",row["case_id"]))
            timestamp=annotation.get("requested_timestamp")
            if _aware_datetime(timestamp) is None:
                raise ValueError(f"Case {row['case_id']!r}: requested_timestamp must be timezone-aware ISO-8601")
            available.update(key for state in states if state_valid_at(state,timestamp) for key in state.get("source_fixture_keys",[]))
        # Empty expected set is a true empty-history assertion; any qualifying state fails it.
        scored.append(not bool(available ^ expected))
    return {"evidence_consistency_accuracy":sum(scored)/len(scored) if scored else None,
            "cases_scored":len(scored),"cases_not_applicable":not_applicable,
            "limitation":"As-of score filters returned state evidence by recorded-time intervals; it does not imply the application's retriever routed the requested timestamp."}


def _end_to_end_temporal_metrics(rows):
    """Score as-of state keys returned by retrieval, without oracle filtering."""
    tp=fp=fn=0
    per_case=[]
    for row in rows:
        annotation=row.get("evaluation",{})
        if annotation.get("temporal_metric_scope")!="end_to_end":
            continue
        expected=set(_require_string_list(annotation.get("expected_temporal_state_keys"),
                                          "expected_temporal_state_keys",row.get("case_id")))
        timestamp=annotation.get("requested_timestamp")
        if _aware_datetime(timestamp) is None:
            raise ValueError(f"Case {row.get('case_id')!r}: requested_timestamp must be timezone-aware ISO-8601")
        states=row.get("state_evidence")
        if not isinstance(states,list) or any(not isinstance(state,dict) for state in states):
            raise ValueError(f"Case {row.get('case_id')!r}: end-to-end temporal scoring requires returned state_evidence")
        actual=set()
        for state in states:
            keys=_require_string_list(state.get("source_fixture_keys",[]),"source_fixture_keys",row.get("case_id"))
            actual.update(keys)
        case_tp=len(expected&actual); case_fp=len(actual-expected); case_fn=len(expected-actual)
        tp+=case_tp; fp+=case_fp; fn+=case_fn
        per_case.append({"case_id":row.get("case_id"),"requested_timestamp":timestamp,
                         "expected_state_keys":sorted(expected),"retrieved_state_keys":sorted(actual),
                         "true_positive_keys":sorted(expected&actual),"false_positive_keys":sorted(actual-expected),
                         "false_negative_keys":sorted(expected-actual),"exact_match":expected==actual})
    precision=tp/(tp+fp) if tp+fp else (1.0 if tp+fn==0 and per_case else None)
    recall=tp/(tp+fn) if tp+fn else None
    f1=(2*precision*recall/(precision+recall) if precision is not None and recall is not None
        and precision+recall else (1.0 if precision==1 and recall is None else 0.0 if tp+fp+fn else None))
    return {"precision":precision,"recall":recall,"f1":f1,"true_positive_keys":tp,
            "false_positive_keys":fp,"false_negative_keys":fn,"cases_scored":len(per_case),
            "exact_match_accuracy":sum(case["exact_match"] for case in per_case)/len(per_case) if per_case else None,
            "per_case":per_case,
            "scoring_rule":"fixture keys on states returned by the actual retrieval call; no ground-truth interval filtering"}


def _latency_metrics(rows):
    def summarize(field):
        values=[]
        for row in rows:
            value=row.get(field)
            if isinstance(value,(int,float)) and math.isfinite(value) and value>=0:
                values.append(float(value))
        values.sort()
        return {"n":len(values),"mean_seconds":statistics.mean(values) if values else None,
                "median_seconds":statistics.median(values) if values else None,
                "p95_seconds":values[max(0,math.ceil(.95*len(values))-1)] if len(values)>=20 else None,
                "p95_minimum_observations":20}
    return {"total":summarize("latency_seconds"),"retrieval":summarize("retrieval_latency_seconds"),
            "answer_generation":summarize("answer_latency_seconds")}


def evaluate(rows):
    """Score saved benchmark rows; expected annotations are never model inputs."""
    if not isinstance(rows,(list,tuple)):
        raise TypeError("rows must be a list or tuple of prediction objects")
    grouped=defaultdict(list)
    for index,row in enumerate(rows):
        if not isinstance(row,dict):
            raise ValueError(f"Prediction row {index} must be an object")
        annotation=row.get("evaluation")
        if annotation is not None:
            if not isinstance(annotation,dict):
                raise ValueError(f"Prediction row {index}: evaluation must be an object")
            case_id=row.get("case_id")
            if not isinstance(case_id,str) or not case_id.strip():
                raise ValueError(f"Prediction row {index}: case_id is required for annotated metrics")
            if annotation.get("expected_behavior") not in {"answer","abstain"}:
                raise ValueError(f"Case {case_id!r}: expected_behavior must be 'answer' or 'abstain'")
            for field in ("expected_evidence_keys","required_answer_facts","forbidden_answer_facts"):
                _require_string_list(annotation.get(field),field,case_id)
            if "expected_transition_provenance_keys" in annotation:
                _require_string_list(annotation["expected_transition_provenance_keys"],
                                     "expected_transition_provenance_keys",case_id)
                _require_string_list(row.get("transition_provenance_fixture_keys"),
                                     "transition_provenance_fixture_keys",case_id)
            if not isinstance(row.get("answer"),str):
                raise ValueError(f"Case {case_id!r}: answer must be a string")
            _require_string_list(row.get("evidence_fixture_keys"),"evidence_fixture_keys",case_id)
            if not isinstance(row.get("eligible_causal_evidence"),bool):
                raise ValueError(f"Case {case_id!r}: eligible_causal_evidence must be boolean")
            if "expected_causal_evidence_keys" in annotation:
                _require_string_list(row.get("causal_evidence_fixture_keys"),"causal_evidence_fixture_keys",case_id)
            if any(field in annotation for field in ("expected_current_state_keys","expected_historical_state_keys","expected_temporal_state_keys")):
                if "state_evidence" not in row:
                    raise ValueError(f"Case {case_id!r}: state_evidence is required for temporal scoring")
        grouped[row.get("mode","unknown")].append(row)
    report={}
    for mode,items in grouped.items():
        annotated=[row for row in items if isinstance(row.get("evaluation"),dict)]
        legacy=[]
        for row in items:
            if "expected_any" not in row: continue
            expected=row.get("expected_any")
            if not isinstance(expected,list) or any(not isinstance(x,str) for x in expected):
                raise ValueError(f"Case {row.get('case_id')!r}: expected_any must be a list of strings")
            answer=str(row.get("answer","")); evidence=str(row.get("evidence_text",""))
            legacy.append({"correct":bool(expected) and any(x.casefold() in answer.casefold() for x in expected),
                           "evidence_supported":bool(expected) and any(x.casefold() in evidence.casefold() for x in expected)})
        legacy_by_category=defaultdict(list)
        for row,result in zip([r for r in items if "expected_any" in r],legacy):
            legacy_by_category[row.get("category","OTHER")].append(result["correct"])
        latency=_latency_metrics(items)
        report[mode]={
            # Retain explicitly named legacy lexical proxies for old result files.
            "accuracy":sum(x["correct"] for x in legacy)/len(legacy) if legacy else None,
            "evidence_support_rate":sum(x["evidence_supported"] for x in legacy)/len(legacy) if legacy else None,
            "legacy_keyword_proxy_n":len(legacy),
            "by_category":{category:sum(values)/len(values) for category,values in legacy_by_category.items()},
            "evidence_retrieval":_evidence_metrics(annotated),
            "transition_provenance":_transition_provenance_metrics(annotated),
            "answer_behavior":_answer_behavior_metrics(annotated),
            "answer_facts_proxy":_fact_metrics(annotated),
            "causal_claims_proxy":_causal_metrics(items),
            "temporal_evidence":_temporal_metrics(annotated),
            "end_to_end_temporal_evidence":_end_to_end_temporal_metrics(annotated),
            "latency":latency,
            "mean_latency_seconds":latency["total"]["mean_seconds"],
            "n":len(items),
        }
    return {"metrics":report,"predictions":list(rows)}


def save(rows,path="outputs/benchmark_results.json"):
    result=evaluate(rows); target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False)
    return target
