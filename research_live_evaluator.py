"""Isolated live-pipeline evaluation, separate from deterministic policy tests.

The runner uses benchmark.py's real Mem0 ingestion and full retrieval/answer
path. It never substitutes mocked results and labels unavailable components as
skipped. Expected annotations are attached after generation by benchmark.py.
"""
import argparse
import json
import math
import random
import re
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

SCHEMA_VERSION = "1.0"
DATASET_VERSION = "1.0"
DEFAULT_DATASET = Path(__file__).resolve().parent / "evaluation" / "live_research_v1.jsonl"
ABSTENTION = "I don't have enough information in your memory to determine that."
_SOURCE_BLOCK = re.compile(r"\[Sources:\s*([^\]]+)\]")
_SOURCE_REF = re.compile(r"^(memory|state|transition):(.+)$")


def load_dataset(path=DEFAULT_DATASET):
    cases=[]; seen=set()
    for line_number,line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():
            continue
        try:
            case=json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid live research JSONL at line {line_number}") from exc
        if not isinstance(case,dict) or case.get("dataset_version")!=DATASET_VERSION:
            raise ValueError(f"Line {line_number}: unsupported live dataset_version")
        case_id=case.get("case_id")
        if not isinstance(case_id,str) or not case_id.strip() or case_id in seen:
            raise ValueError(f"Line {line_number}: case_id must be unique and non-empty")
        seen.add(case_id)
        for key in ("category","fixture_user","question"):
            if not isinstance(case.get(key),str) or not case[key].strip():
                raise ValueError(f"Case {case_id!r}: {key} must be a non-empty string")
        if case.get("expected_behavior") not in {"answer","abstain","retrieval_error"}:
            raise ValueError(f"Case {case_id!r}: invalid expected_behavior")
        for key in ("expected_evidence_keys","required_answer_facts","forbidden_answer_facts"):
            if not isinstance(case.get(key),list) or any(not isinstance(value,str) or not value.strip() for value in case[key]):
                raise ValueError(f"Case {case_id!r}: {key} must be a list of non-empty strings")
        if not isinstance(case.get("conversation"),list):
            raise ValueError(f"Case {case_id!r}: conversation must be a list")
        fixture_keys=set()
        identities=[(case["fixture_user"],case["conversation"])]
        distractors=case.get("distractor_users",[])
        if not isinstance(distractors,list):
            raise ValueError(f"Case {case_id!r}: distractor_users must be a list")
        for distractor in distractors:
            if (not isinstance(distractor,dict) or not isinstance(distractor.get("fixture_user"),str)
                    or not isinstance(distractor.get("conversation",[]),list)):
                raise ValueError(f"Case {case_id!r}: malformed distractor user")
            identities.append((distractor["fixture_user"],distractor.get("conversation",[])))
        if len({identity for identity,_ in identities})!=len(identities):
            raise ValueError(f"Case {case_id!r}: fixture users must be unique")
        for identity,messages in identities:
            if not identity.strip():
                raise ValueError(f"Case {case_id!r}: fixture user must be non-empty")
            for index,message in enumerate(messages):
                if isinstance(message,str):
                    key=f"{identity}-message-{index}"; text=message
                elif isinstance(message,dict):
                    key=message.get("fixture_key"); text=message.get("text")
                else:
                    raise ValueError(f"Case {case_id!r}: malformed conversation entry")
                if (not isinstance(key,str) or not key.strip() or not isinstance(text,str)
                        or not text.strip() or key in fixture_keys):
                    raise ValueError(f"Case {case_id!r}: invalid or duplicate fixture entry")
                fixture_keys.add(key)
        for key in ("expected_causal_evidence_keys","expected_current_state_keys",
                    "expected_transition_provenance_keys","expected_historical_state_keys",
                    "expected_temporal_state_keys"):
            if key in case:
                values=case[key]
                if not isinstance(values,list) or any(not isinstance(value,str) or not value.strip() for value in values):
                    raise ValueError(f"Case {case_id!r}: {key} must be a list of non-empty fixture keys")
                if not set(values)<=fixture_keys:
                    raise ValueError(f"Case {case_id!r}: {key} contains an unknown fixture key")
        claim_groups=case.get("expected_claim_source_keys")
        if claim_groups is not None and (not isinstance(claim_groups,list)
                or any(not isinstance(group,list) or any(not isinstance(key,str) or key not in fixture_keys for key in group)
                       for group in claim_groups)):
            raise ValueError(f"Case {case_id!r}: expected_claim_source_keys contains invalid fixture references")
        selector=case.get("requested_timestamp_from_transition")
        if selector is not None and (not isinstance(selector,dict)
                or not isinstance(selector.get("source_fixture_key"),str)
                or not isinstance(selector.get("offset_microseconds",0),int)
                or selector.get("source_fixture_key") not in fixture_keys):
            raise ValueError(f"Case {case_id!r}: invalid transition-relative timestamp selector")
        if "requested_timestamp" in case:
            value=case["requested_timestamp"]
            try:
                parsed=datetime.fromisoformat(value.replace("Z","+00:00"))
            except (AttributeError,ValueError) as exc:
                raise ValueError(f"Case {case_id!r}: requested_timestamp must be aware ISO-8601") from exc
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError(f"Case {case_id!r}: requested_timestamp must be aware ISO-8601")
        if case.get("run_scope","live") not in {"live","mock_only"}:
            raise ValueError(f"Case {case_id!r}: unsupported run_scope")
        if case.get("run_scope","live")=="mock_only" and case["expected_behavior"]!="retrieval_error":
            raise ValueError(f"Case {case_id!r}: mock_only cases must be diagnostic retrieval_error cases")
        cases.append(case)
    return cases


def select_cases(cases,case_ids=None):
    """Select a validated dataset subset without rewriting its source file."""
    if case_ids is None:
        return list(cases)
    if not isinstance(case_ids,(list,tuple)) or any(not isinstance(item,str) or not item.strip()
                                                     for item in case_ids):
        raise ValueError("case IDs must be non-empty strings")
    if len(case_ids)!=len(set(case_ids)):
        raise ValueError("case IDs must be unique")
    wanted=set(case_ids)
    available={case.get("case_id") for case in cases}
    unknown=wanted-available
    if unknown:
        raise ValueError("unknown case IDs were requested")
    return [case for case in cases if case.get("case_id") in wanted]


def local_ollama_preflight(settings, opener=urlopen):
    """Check local service and already-installed models; never downloads models."""
    if settings.provider != "ollama":
        return {"available":False,"reason_code":"provider_not_ollama"}
    try:
        with opener(settings.ollama_server_url.rstrip("/")+"/api/tags",timeout=4) as response:
            payload=json.loads(response.read().decode("utf-8"))
        names={item.get("name") for item in payload.get("models",[]) if isinstance(item,dict)}
    except Exception as exc:
        return {"available":False,"reason_code":"ollama_service_unavailable",
                "error_type":type(exc).__name__}
    def present(model):
        return any(name==model or name==model+":latest" for name in names)
    missing=[]
    if not present(settings.model): missing.append("configured_llm_model")
    if not present("nomic-embed-text"): missing.append("mem0_embedding_model")
    if missing:
        return {"available":False,"reason_code":"required_model_not_installed","missing_components":missing}
    return {"available":True,"reason_code":None}


def _log(event,**fields):
    record={"timestamp":datetime.now(timezone.utc).isoformat(),"event":event,**fields}
    print(json.dumps(record,separators=(",",":")),flush=True)


def _write_progress(path,record,first=False):
    mode="x" if first else "a"
    with Path(path).open(mode,encoding="utf-8") as stream:
        stream.write(json.dumps(_safe_progress_record(record),ensure_ascii=False)+"\n")
        stream.flush()


_PROGRESS_TIMINGS=("elapsed_seconds","latency_seconds","retrieval_latency_seconds","answer_latency_seconds")


def _safe_progress_code(value):
    if not isinstance(value,str) or len(value)>100 or not re.fullmatch(r"[A-Za-z0-9_.-]+",value):
        return None
    return value


def _safe_number(value):
    return value if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) else None


def _safe_progress_record(record):
    """Allowlist compact progress metadata; never persist a prediction/evidence bundle."""
    if not isinstance(record,dict):
        return {"status":"failed","error_type":"invalid_progress_record"}
    safe={}
    for key in ("case_id","status","skip_reason","failure_stage","error_type","reason","mode","category","answer_behavior"):
        value=_safe_progress_code(record.get(key))
        if value is not None:
            safe[key]=value
    for key in _PROGRESS_TIMINGS:
        value=_safe_number(record.get(key))
        if value is not None:
            safe[key]=value
    prediction=record.get("prediction")
    if isinstance(prediction,dict):
        summary={}
        for key in ("mode","category","answer_behavior"):
            value=_safe_progress_code(prediction.get(key))
            if value is not None:
                summary[key]=value
        for key in _PROGRESS_TIMINGS:
            value=_safe_number(prediction.get(key))
            if value is not None:
                summary[key]=value
        for key in ("evidence_count","accepted_claim_count","rejected_claim_count"):
            value=prediction.get(key)
            if isinstance(value,int) and not isinstance(value,bool) and value>=0:
                summary[key]=value
        if isinstance(prediction.get("eligible_causal_evidence"),bool):
            summary["eligible_causal_evidence"]=prediction["eligible_causal_evidence"]
        diagnostics=prediction.get("answer_diagnostics")
        if isinstance(diagnostics,dict):
            diagnostic_summary={}
            for key in ("answer_reason","model_response_status"):
                value=_safe_progress_code(diagnostics.get(key))
                if value is not None:
                    diagnostic_summary[key]=value
            for key in ("causal_gate_passed",):
                value=diagnostics.get(key)
                if isinstance(value,bool):
                    diagnostic_summary[key]=value
            rejection_counts=diagnostics.get("rejection_categories")
            if isinstance(rejection_counts,dict):
                safe_counts={code:count for raw_code,count in rejection_counts.items()
                             if (code:=_safe_progress_code(raw_code)) is not None
                             and isinstance(count,int) and not isinstance(count,bool) and count>=0}
                if safe_counts:
                    diagnostic_summary["rejection_categories"]=safe_counts
            if diagnostic_summary:
                summary["answer_diagnostics"]=diagnostic_summary
        safe["prediction_summary"]=summary
    return safe


def _failure_frames(exc):
    import traceback
    cause=exc.__cause__ if exc.__cause__ is not None else exc
    if cause.__traceback__ is None:
        return []
    return [{"file":Path(frame.filename).name,"line":frame.lineno,"function":frame.name}
            for frame in traceback.extract_tb(cause.__traceback__)]


def execute_cases(cases,case_runner,progress_path=None,progress_callback=None):
    """Execute each case independently so completed rows survive later errors."""
    records=[]; interrupted=False
    for index,case in enumerate(cases):
        case_id=case["case_id"]
        if case.get("run_scope","live")=="mock_only":
            record={"case_id":case_id,"status":"skipped","skip_reason":"mock_only_diagnostic"}
        else:
            started=time.perf_counter()
            try:
                rows=case_runner(case)
                if not isinstance(rows,list) or len(rows)!=1 or rows[0].get("case_id")!=case_id:
                    raise ValueError("live case runner returned an invalid row set")
                prediction=rows[0]
                record={"case_id":case_id,"status":"completed","prediction":prediction}
            except KeyboardInterrupt:
                interrupted=True
                record={"case_id":case_id,"status":"interrupted","error_type":"KeyboardInterrupt"}
            except Exception as exc:
                message=str(exc).casefold()
                if "timestamp selector" in message:
                    stage="temporal_fixture_resolution"
                elif "fixture ingestion" in message:
                    stage="fixture_ingestion"
                elif "retrieval or generation" in message:
                    stage="retrieval_or_generation"
                else:
                    stage="case_execution"
                record={"case_id":case_id,"status":"failed","error_type":type(exc.__cause__).__name__ if exc.__cause__ else type(exc).__name__,
                        "failure_stage":stage,"traceback_frames":_failure_frames(exc)}
                try:
                    import benchmark
                    benchmark._print_error_traceback(exc)
                except Exception:
                    pass
            if record["status"]=="completed":
                record["elapsed_seconds"]=time.perf_counter()-started
        records.append(record)
        if progress_path is not None:
            _write_progress(progress_path,record,first=index==0)
        if progress_callback is not None:
            progress_callback(_safe_progress_record(record))
        if interrupted:
            for remaining in cases[index+1:]:
                skipped={"case_id":remaining["case_id"],"status":"skipped","skip_reason":"run_interrupted_before_case"}
                records.append(skipped)
                if progress_path is not None:
                    _write_progress(progress_path,skipped)
                if progress_callback is not None:
                    progress_callback(_safe_progress_record(skipped))
            break
    return records,"interrupted" if interrupted else ("completed_with_failures" if any(r["status"]=="failed" for r in records) else "completed")


def _citation_groups(prediction):
    try:
        evidence=json.loads(prediction.get("evidence_text","{}"))
    except (TypeError,ValueError):
        return []
    memory_to_keys=prediction.get("source_id_fixture_keys",{})
    state_to_keys={state.get("state_id"):[key for memory_id in state.get("source_memory_ids",[])
                                             for key in [memory_to_keys.get(memory_id)] if key]
                   for state in evidence.get("states",[]) if isinstance(state,dict)}
    transition_to_keys={edge.get("transition_id"):[key for memory_id in edge.get("source_memory_ids",[])
                                                     for key in [memory_to_keys.get(memory_id)] if key]
                        for edge in evidence.get("transitions",[]) if isinstance(edge,dict)}
    groups=[]
    for block in _SOURCE_BLOCK.findall(prediction.get("answer") or ""):
        group=set()
        for raw in block.split(","):
            match=_SOURCE_REF.fullmatch(raw.strip())
            if not match:
                continue
            kind,source_id=match.groups()
            if kind=="memory": group.update([memory_to_keys[source_id]] if source_id in memory_to_keys else [])
            elif kind=="state": group.update(state_to_keys.get(source_id,[]))
            else: group.update(transition_to_keys.get(source_id,[]))
        groups.append(group)
    return groups


def _claim_group_metrics(predictions):
    total=agree=0
    per_case=[]
    for prediction in predictions:
        annotation=prediction.get("evaluation",{})
        expected=annotation.get("expected_claim_source_keys")
        if expected is None:
            continue
        actual=_citation_groups(prediction)
        n=max(len(expected),len(actual))
        matched=0
        for index in range(n):
            wanted=set(expected[index]) if index<len(expected) else set()
            got=actual[index] if index<len(actual) else set()
            matched+=bool(got) and wanted==got
        total+=n; agree+=matched
        per_case.append({"case_id":prediction["case_id"],"agree":matched,"groups":n})
    return {"agreement":agree/total if total else None,"agreements":agree,"groups":total,
            "per_case":per_case,
            "interpretation":"Exact fixture-key mapping of rendered citation groups; it is not semantic entailment."}


def _metric_denominators(predictions):
    from evaluator import _fact_present, _is_abstention
    answers=[row for row in predictions if isinstance(row.get("evaluation"),dict)]
    answer_correct=abstain_tp=predicted_abstain=expected_abstain=0
    required_total=required_found=forbidden_total=forbidden_found=0
    categories={}
    for row in answers:
        annotation=row["evaluation"]
        expected=annotation["expected_behavior"]
        abstained=_is_abstention(row.get("answer"))
        predicted="abstain" if abstained else "answer"
        answer_correct+=predicted==expected
        predicted_abstain+=abstained
        expected_abstain+=expected=="abstain"
        abstain_tp+=abstained and expected=="abstain"
        required=annotation.get("required_answer_facts",[])
        forbidden=annotation.get("forbidden_answer_facts",[])
        required_total+=len(required)
        required_found+=sum(_fact_present(row.get("answer",""),item) for item in required)
        forbidden_total+=len(forbidden)
        forbidden_found+=sum(_fact_present(row.get("answer",""),item) for item in forbidden)
        category=categories.setdefault(row.get("category","OTHER"),{"n":0,"correct":0})
        category["n"]+=1; category["correct"]+=predicted==expected
    return {"answerability":{"correct":answer_correct,"denominator":len(answers),
                "abstention_true_positive":abstain_tp,"predicted_abstentions":predicted_abstain,
                "expected_abstentions":expected_abstain},
            "answer_fact_proxy":{"required_facts_found":required_found,"required_facts_total":required_total,
                "forbidden_facts_found":forbidden_found,"forbidden_facts_total":forbidden_total},
            "answerability_by_category":{key:{**value,"accuracy":value["correct"]/value["n"] if value["n"] else None}
                for key,value in sorted(categories.items())}}


def build_report(records,dataset,preflight,config_metadata,status,storage_paths=None):
    predictions=[record["prediction"] for record in records if record["status"]=="completed"]
    failures=[record for record in records if record["status"]=="failed"]
    skipped=[record for record in records if record["status"]=="skipped"]
    attempted=sum(record["status"] in {"completed","failed","interrupted"} for record in records)
    metrics=None
    if predictions:
        from evaluator import evaluate
        metrics=evaluate(predictions)["metrics"].get("full")
        metrics["claim_citation_group_agreement"]=_claim_group_metrics(predictions)
        metrics["metric_denominators"]=_metric_denominators(predictions)
        metrics["temporal_state_fixture_key_match"]={
            "exact_match_cases":metrics["end_to_end_temporal_evidence"]["per_case"],
            "case_count":metrics["end_to_end_temporal_evidence"]["cases_scored"],
            "interpretation":"End-to-end expected versus retrieved state fixture-key match; raw runtime state IDs are generated per run."}
        metrics["causal_transition_id_accuracy"]={"correct":None,"denominator":0,"accuracy":None,
            "reason":"The fixed dataset labels causal source fixture keys, not run-specific transition IDs; causal_evidence_precision/recall is the available fixture-key proxy."}
    return {"schema_version":SCHEMA_VERSION,"dataset_version":DATASET_VERSION,
            "evaluator_version":SCHEMA_VERSION,"random_seed":0,
            "evaluation_type":"live_pipeline","status":status,
            "run_counts":{"cases_in_dataset":len(dataset),"attempted":attempted,
                          "completed":len(predictions),"failed":len(failures),"skipped":len(skipped)},
            "components":config_metadata,"preflight":preflight,
            "storage_paths":storage_paths,
            "metrics":metrics,
            "retrieval_failure_detection":{"live_failures_observed":len(failures),
                "mock_only_cases_skipped":sum(case.get("run_scope")=="mock_only" for case in dataset),
                "metric":None,"reason":"No retrieval source was intentionally fault-injected in the live run."},
            "latency_boundary":"Per-case full retrieval (including query classification) plus answer generation; excludes fixture ingestion, preflight, and result writing.",
            "failures":failures,"skipped_cases":skipped,"case_results":records,
            "limitations":["Evidence and citation agreement are fixture-ID proxies, not proof of semantic entailment.",
                           "Model answers are live outputs; a small synthetic dataset does not establish general reliability.",
                           "Mock-only retrieval-source failure diagnostics are explicitly skipped in this live mode."]}


def _versions():
    result={}
    for package in ("mem0ai","qdrant-client","openai","streamlit"):
        try: result[package]=metadata.version(package)
        except metadata.PackageNotFoundError: result[package]=None
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolation-dir",required=True)
    parser.add_argument("--dataset",default=str(DEFAULT_DATASET))
    parser.add_argument("--case-ids",nargs="+",help="Optional case IDs to run from the dataset")
    parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    cases=select_cases(load_dataset(args.dataset),args.case_ids)
    if not cases:
        parser.error("the selected dataset subset is empty")
    import benchmark
    isolation=benchmark._configure_isolation(args.isolation_dir)
    output=Path(args.output).expanduser().resolve()
    paths=benchmark._resolve_evaluation_paths(isolation,output.parent,args.dataset,output,False)
    progress_path=output.with_name(output.stem+".progress.jsonl")
    if progress_path.exists() or progress_path==output:
        raise benchmark.IsolationError("Live evaluation progress output already exists; choose a fresh output path.")
    random.seed(0)
    _log("setup",operation="isolation_configured")
    import config
    settings=config.settings
    preflight=local_ollama_preflight(settings)
    config_metadata={"provider":settings.provider,"model":settings.model,
                     "embedding_model":"nomic-embed-text" if settings.provider=="ollama" else "configured_mem0_embedding",
                     "python_version":sys.version.split()[0],"packages":_versions(),
                     "random_seed":0,
                     "components":{"sqlite":"isolated","qdrant_mem0":"not_initialized",
                                   "ollama_llm":"not_started","hybrid_retrieval":"not_started",
                                   "answer_generation":"not_started"}}
    storage_paths={"isolation_root":str(isolation.root),"sqlite":str(isolation.db_path),
                   "qdrant_mem0":str(isolation.mem0_ollama_path if settings.provider=="ollama" else isolation.mem0_path),
                   "mem0_config":str(isolation.mem0_config),"output":str(paths.result_path),
                   "progress":str(progress_path)}
    if not preflight["available"]:
        records=[{"case_id":case["case_id"],"status":"skipped","skip_reason":preflight["reason_code"]}
                 for case in cases if case.get("run_scope","live")=="live"]
        records.extend({"case_id":case["case_id"],"status":"skipped","skip_reason":"mock_only_diagnostic"}
                       for case in cases if case.get("run_scope")=="mock_only")
        _write_progress(progress_path,{"status":"started","reason":"preflight_unavailable"},first=True)
        for record in records: _write_progress(progress_path,record)
        report=build_report(records,cases,preflight,config_metadata,"unavailable",storage_paths)
    else:
        try:
            runtime=benchmark._load_runtime(isolation)
        except Exception as exc:
            config_metadata["components"]["qdrant_mem0"]="failed_isolation_verification"
            records=[{"case_id":case["case_id"],"status":"skipped","skip_reason":"storage_runtime_verification_failed"}
                     for case in cases]
            _write_progress(progress_path,{"status":"setup_failed","error_type":type(exc).__name__,
                "traceback_frames":_failure_frames(exc)},first=True)
            for record in records: _write_progress(progress_path,record)
            try: benchmark._print_error_traceback(exc)
            except Exception: pass
            report=build_report(records,cases,preflight,config_metadata,"setup_failed",storage_paths)
            report["setup_failure"]={"error_type":type(exc).__name__,"traceback_frames":_failure_frames(exc)}
            with paths.result_path.open("x",encoding="utf-8") as stream:
                json.dump(report,stream,indent=2,ensure_ascii=False)
            print(f"Saved failed live-pipeline setup report to {paths.result_path}",flush=True)
            raise SystemExit(2)
        config_metadata["components"].update({"qdrant_mem0":"live","ollama_llm":"live",
            "hybrid_retrieval":"live","answer_generation":"live"})
        _log("preflight_passed",operation="live_components_ready")
        def run_case(case):
            return benchmark._run_dataset(runtime,[case],["full"])
        records,status=execute_cases(cases,run_case,progress_path,
            progress_callback=lambda record:_log("case_"+record["status"],case_id=record["case_id"]))
        report=build_report(records,cases,preflight,config_metadata,status,storage_paths)
    with paths.result_path.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2,ensure_ascii=False)
    print(f"Saved live-pipeline evaluation to {paths.result_path}",flush=True)


if __name__=="__main__":
    main()
