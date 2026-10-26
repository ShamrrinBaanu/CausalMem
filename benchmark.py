"""Run benchmark modes with SQLite and Mem0 confined to a fresh test root.

No application storage modules are imported until ``_configure_isolation`` has
set and checked all persistent paths. Existing stores are never reused or
cleared; choose a new --isolation-dir for each run.
"""
import argparse
import importlib
import inspect
import json
import os
import re
import sys
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

MODES=("mem0","state","temporal","graph","full")
_STORAGE_MODULES={"config","database","mem0_engine","hybrid_retriever","state_extractor","causal_graph","temporal_engine","answer_generator","llm_service","query_router"}


class IsolationError(RuntimeError):
    """Benchmark storage could not be proven separate from application data."""


class BenchmarkExecutionError(RuntimeError):
    """A benchmark case or condition failed and must not count as an answer."""


def _concise_error_summary(exc):
    """Keep CLI failures useful without echoing provider exception contents."""
    if isinstance(exc,BenchmarkExecutionError):
        message=str(exc)
        context=message.split(": ",1)[0]
        cause_type=type(exc.__cause__).__name__ if exc.__cause__ is not None else type(exc).__name__
        return f"{context} (underlying error type: {cause_type})"
    return str(exc)


def _print_error_traceback(exc,stream=None):
    """Print stack frames only; exception messages can contain sensitive details."""
    cause=exc.__cause__ if exc.__cause__ is not None else exc
    tb=cause.__traceback__
    if tb is not None:
        output=stream or sys.stderr
        print("Underlying traceback frames (exception details omitted):",file=output,flush=True)
        for frame in traceback.extract_tb(tb):
            print(f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}',file=output,flush=True)


def _progress(event,*,case_id=None,fixture_key=None,mode=None,operation=None,elapsed=None,error_type=None,diagnostics=None,stream=None):
    """Emit flushed, content-free progress records suitable for live diagnosis."""
    record={"timestamp":datetime.now(timezone.utc).isoformat(),"event":event}
    for key,value in (("case_id",case_id),("fixture_key",fixture_key),("mode",mode),
                      ("operation",operation),("elapsed_seconds",elapsed),("error_type",error_type)):
        if value is not None:
            record[key]=round(elapsed,3) if key=="elapsed_seconds" else value
    if isinstance(diagnostics,dict):
        record["diagnostics"]=diagnostics
    print(json.dumps(record,separators=(",",":")),file=stream or sys.stdout,flush=True)


def _run_operation(operation_name,callback,*,case_id,fixture_key=None,mode=None):
    """Run one opaque operation with timing logs; never log arguments or error text."""
    context={"case_id":case_id,"fixture_key":fixture_key,"mode":mode,"operation":operation_name}
    _progress("start",**context)
    started=time.perf_counter()
    try:
        result=callback()
    except Exception as exc:
        _progress("error",**context,elapsed=time.perf_counter()-started,error_type=type(exc).__name__)
        raise
    _progress("end",**context,elapsed=time.perf_counter()-started)
    return result


def _supports_diagnostics_callback(function):
    try:
        parameters=inspect.signature(function).parameters.values()
    except (TypeError,ValueError):
        return False
    return any(parameter.name=="diagnostics_callback" or
               parameter.kind==inspect.Parameter.VAR_KEYWORD for parameter in parameters)


def _add_memory_with_diagnostics(runtime,message,user_id):
    add=runtime.add_memory
    if not _supports_diagnostics_callback(add):
        return add(message,user_id)

    def report(record):
        stage=record.get("stage")
        duration=record.get("duration_seconds")
        if not isinstance(stage,str):
            return
        diagnostics={}
        if isinstance(record.get("success"),bool):
            diagnostics["success"]=record["success"]
        if isinstance(record.get("status"),str):
            diagnostics["status"]=record["status"]
        if isinstance(record.get("error_category"),str):
            diagnostics["error_category"]=record["error_category"]
        elapsed=duration if isinstance(duration,(int,float)) else None
        _progress("stage",operation=f"mem0_add:{stage}",elapsed=elapsed,diagnostics=diagnostics)

    return add(message,user_id,diagnostics_callback=report)


def _read_dataset(path):
    text=Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.casefold()==".jsonl":
        try:
            cases=[json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL dataset at line {exc.lineno}: {exc.msg}") from exc
    else:
        cases=json.loads(text)
    if not isinstance(cases,list) or any(not isinstance(case,dict) for case in cases):
        raise ValueError("Evaluation dataset must contain a list of case objects")
    seen=set()
    for index,case in enumerate(cases):
        case_id=case.get("case_id")
        if not isinstance(case_id,str) or not case_id.strip() or case_id in seen:
            raise ValueError(f"Dataset case {index} has a missing or duplicate case_id")
        seen.add(case_id)
        if "expected_behavior" not in case:
            continue  # Older generated fixtures remain runnable as legacy data.
        if case["expected_behavior"] not in {"answer","abstain"}:
            raise ValueError(f"Case {case_id!r}: expected_behavior must be 'answer' or 'abstain'")
        for field in ("expected_evidence_keys","required_answer_facts","forbidden_answer_facts"):
            values=case.get(field)
            if not isinstance(values,list) or any(not isinstance(value,str) or not value.strip() for value in values):
                raise ValueError(f"Case {case_id!r}: {field} must be a list of non-empty strings")
        for field in ("expected_causal_evidence_keys","expected_transition_provenance_keys","expected_current_state_keys",
                      "expected_historical_state_keys","expected_temporal_state_keys"):
            if field in case and (not isinstance(case[field],list) or any(not isinstance(v,str) or not v.strip() for v in case[field])):
                raise ValueError(f"Case {case_id!r}: {field} must be a list of non-empty strings")
        if "requested_timestamp" in case and not isinstance(case["requested_timestamp"],str):
            raise ValueError(f"Case {case_id!r}: requested_timestamp must be an ISO-8601 string")
        if "requested_timestamp" in case:
            try:
                timestamp=datetime.fromisoformat(case["requested_timestamp"].replace("Z","+00:00"))
            except ValueError as exc:
                raise ValueError(f"Case {case_id!r}: requested_timestamp is not valid ISO-8601") from exc
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError(f"Case {case_id!r}: requested_timestamp must include a timezone")
        known_keys=set()
        if not isinstance(case.get("distractor_users",[]),list):
            raise ValueError(f"Case {case_id!r}: distractor_users must be a list")
        identities=[(case.get("fixture_user","user"),case.get("conversation",[]))]
        identities.extend((item.get("fixture_user"),item.get("conversation",[]))
                          for item in case.get("distractor_users",[]) if isinstance(item,dict))
        for identity,messages in identities:
            if not isinstance(identity,str) or not identity.strip():
                raise ValueError(f"Case {case_id!r}: fixture_user must be a non-empty string")
            known_keys.update(entry["fixture_key"] for entry in _message_entries(messages,identity))
        temporal_fixture=case.get("temporal_fixture")
        if temporal_fixture is not None:
            if not isinstance(temporal_fixture,dict) or not isinstance(temporal_fixture.get("states"),list):
                raise ValueError(f"Case {case_id!r}: temporal_fixture must contain a states list")
            if any(messages for _,messages in identities):
                raise ValueError(f"Case {case_id!r}: temporal_fixture cases cannot also ingest conversations")
            state_keys=set()
            for state in temporal_fixture["states"]:
                if not isinstance(state,dict):
                    raise ValueError(f"Case {case_id!r}: temporal_fixture states must be objects")
                key=state.get("fixture_key")
                if not isinstance(key,str) or not key.strip() or key in known_keys or key in state_keys:
                    raise ValueError(f"Case {case_id!r}: temporal state fixture_key must be unique and non-empty")
                if not isinstance(state.get("state_value"),str) or not state["state_value"].strip():
                    raise ValueError(f"Case {case_id!r}: temporal states need a non-empty state_value")
                if state.get("owner",case.get("fixture_user","user")) not in {identity for identity,_ in identities}:
                    raise ValueError(f"Case {case_id!r}: temporal state owner must be a declared fixture user")
                for field in ("valid_from",):
                    value=state.get(field)
                    if not isinstance(value,str):
                        raise ValueError(f"Case {case_id!r}: temporal state {field} must be timezone-aware ISO-8601")
                    try: parsed=datetime.fromisoformat(value.replace("Z","+00:00"))
                    except ValueError as exc: raise ValueError(f"Case {case_id!r}: invalid temporal state {field}") from exc
                    if parsed.tzinfo is None or parsed.utcoffset() is None:
                        raise ValueError(f"Case {case_id!r}: temporal state {field} must include a timezone")
                valid_until=state.get("valid_until")
                if valid_until is not None:
                    try: parsed=datetime.fromisoformat(valid_until.replace("Z","+00:00"))
                    except (ValueError,AttributeError) as exc: raise ValueError(f"Case {case_id!r}: invalid temporal state valid_until") from exc
                    if parsed.tzinfo is None or parsed.utcoffset() is None:
                        raise ValueError(f"Case {case_id!r}: temporal state valid_until must include a timezone")
                state_keys.add(key)
            known_keys.update(state_keys)
            transitions=temporal_fixture.get("transitions",[])
            if not isinstance(transitions,list):
                raise ValueError(f"Case {case_id!r}: temporal_fixture transitions must be a list")
            for edge in transitions:
                if not isinstance(edge,dict) or edge.get("from_fixture_key") not in state_keys or edge.get("to_fixture_key") not in state_keys:
                    raise ValueError(f"Case {case_id!r}: temporal transition endpoints must reference fixture states")
                if edge.get("owner",case.get("fixture_user","user")) not in {identity for identity,_ in identities}:
                    raise ValueError(f"Case {case_id!r}: temporal transition owner must be a declared fixture user")
                timestamp=edge.get("timestamp")
                try: parsed=datetime.fromisoformat(timestamp.replace("Z","+00:00"))
                except (ValueError,AttributeError) as exc: raise ValueError(f"Case {case_id!r}: temporal transition timestamp must be ISO-8601") from exc
                if parsed.tzinfo is None or parsed.utcoffset() is None:
                    raise ValueError(f"Case {case_id!r}: temporal transition timestamp must include a timezone")
                source_key=edge.get("source_fixture_key")
                if source_key is not None and source_key not in state_keys:
                    # A transition may have a dedicated evidence key, which is
                    # accepted as a new fixture key below.
                    if not isinstance(source_key,str) or not source_key.strip() or source_key in known_keys:
                        raise ValueError(f"Case {case_id!r}: invalid temporal transition source_fixture_key")
                    known_keys.add(source_key)
        for field in ("expected_evidence_keys","expected_causal_evidence_keys","expected_transition_provenance_keys","expected_current_state_keys",
                      "expected_historical_state_keys","expected_temporal_state_keys"):
            unknown=set(case.get(field,[]))-known_keys
            if unknown:
                raise ValueError(f"Case {case_id!r}: {field} references unknown fixture keys {sorted(unknown)}")
        oracle=case.get("temporal_oracle")
        if oracle is not None:
            if not isinstance(oracle,dict) or not isinstance(oracle.get("states"),list):
                raise ValueError(f"Case {case_id!r}: temporal_oracle must contain a states list")
            for state in oracle["states"]:
                if not isinstance(state,dict) or state.get("fixture_key") not in known_keys:
                    raise ValueError(f"Case {case_id!r}: temporal_oracle contains an unknown state fixture_key")
            if not isinstance(oracle.get("expected_state_keys"),list) or not set(oracle["expected_state_keys"])<=known_keys:
                raise ValueError(f"Case {case_id!r}: temporal_oracle expected_state_keys must use known fixture keys")
    return cases


def _default_dataset_path():
    """Prefer the checked-in annotated set; synthetic generator is fallback only."""
    candidate=Path(__file__).resolve().parent/"evaluation"/"ground_truth.jsonl"
    return candidate if candidate.is_file() else None


def _message_entries(messages,identity):
    if not isinstance(messages,list):
        raise ValueError(f"Conversation for fixture user {identity!r} must be a list")
    result=[]
    for index,item in enumerate(messages):
        if isinstance(item,str):
            result.append({"fixture_key":f"{identity}-message-{index}","text":item})
        elif isinstance(item,dict):
            key=item.get("fixture_key"); text=item.get("text")
            if not isinstance(key,str) or not key.strip() or not isinstance(text,str) or not text.strip():
                raise ValueError(f"Conversation item {index} for {identity!r} needs non-empty fixture_key and text")
            result.append({"fixture_key":key,"text":text})
        else:
            raise ValueError(f"Conversation item {index} for {identity!r} must be a string or object")
    return result


def _record_ids(records):
    result=set()
    for record in records or []:
        if isinstance(record,dict):
            value=record.get("id") or record.get("memory_id")
            if isinstance(value,str) and value.strip(): result.add(value)
    return result


def _added_ids(result):
    if not isinstance(result,dict): return set()
    values=[]
    for key in ("id","memory_id"):
        if isinstance(result.get(key),str): values.append(result[key])
    for key in ("results","memories"):
        if isinstance(result.get(key),list): values.extend(result[key])
    return _record_ids(values)


def _fixture_keys_for_ids(memory_ids,id_to_fixture):
    # A merged or reused Mem0 ID cannot safely be attributed to one source
    # fixture; omit it from automatic scoring instead of crediting several.
    if not isinstance(memory_ids,(list,tuple,set)):
        return []
    return sorted({fixture_key for memory_id in memory_ids
                   for fixture_key in id_to_fixture.get(memory_id,())
                   if len(id_to_fixture.get(memory_id,()))==1})


def _bundle_fixture_keys(bundle,id_to_fixture,sections=("memories","states","transitions","causal_path")):
    ids=set()
    for section in sections:
        for record in bundle.get(section,[]) or []:
            if not isinstance(record,dict): continue
            ids.update(_record_ids([record]))
            source_ids=record.get("source_memory_ids",[])
            if isinstance(source_ids,list): ids.update(x for x in source_ids if isinstance(x,str))
    return _fixture_keys_for_ids(ids,id_to_fixture)


def _transition_provenance_fixture_keys(bundle,id_to_fixture,id_to_user,user_id,
                                        verified_state_owners=None):
    """Map transition provenance after validating both endpoint owners.

    A historical predecessor query may return its transition but intentionally
    omit the successor state from answer evidence. In that case callers may
    supply endpoint owners read from the user-filtered SQLite state history.
    """
    if verified_state_owners is None:
        state_owners={state.get("state_id"):state.get("user_id")
                      for state in bundle.get("states",[]) or []
                      if isinstance(state,dict) and isinstance(state.get("state_id"),str)}
    else:
        state_owners=dict(verified_state_owners)
    ids=set()
    for transition in bundle.get("transitions",[]) or []:
        if (not isinstance(transition,dict) or transition.get("user_id")!=user_id
                or state_owners.get(transition.get("from_state_id"))!=user_id
                or state_owners.get(transition.get("to_state_id"))!=user_id):
            continue
        source_ids=transition.get("source_memory_ids",[])
        if isinstance(source_ids,list):
            ids.update(memory_id for memory_id in source_ids if isinstance(memory_id,str)
                       and id_to_user.get(memory_id)=={user_id})
    return _fixture_keys_for_ids(ids,id_to_fixture)


def _seed_temporal_fixture(runtime,case,user_ids,id_to_fixture,id_to_user,fixture_keys):
    """Insert deterministic synthetic recorded-time evidence into isolated SQLite."""
    from models import State, Transition

    fixture=case["temporal_fixture"]
    state_by_key={}
    memory_by_key={}
    default_owner=case.get("fixture_user","user")
    states=[]
    for item in fixture["states"]:
        key=item["fixture_key"]
        if key in fixture_keys:
            raise ValueError(f"duplicate fixture_key {key!r}")
        fixture_keys.add(key)
        owner=item.get("owner",default_owner)
        memory_id=f"synthetic:{case['case_id']}:{key}"
        memory_by_key[key]=memory_id
        id_to_fixture[memory_id]={key}
        id_to_user[memory_id]={user_ids[owner]}
        state=State(user_id=user_ids[owner],state_type=item.get("state_type","OTHER"),
                    state_value=item["state_value"],status=item.get("status","ACTIVE"),
                    valid_from=item["valid_from"],valid_until=item.get("valid_until"),
                    source_memory_ids=[memory_id])
        state_by_key[key]=state
        states.append(state)
    edges=[]
    for item in fixture.get("transitions",[]):
        owner=item.get("owner",default_owner)
        source_key=item.get("source_fixture_key")
        source_ids=[]
        if source_key:
            if source_key not in memory_by_key:
                memory_id=f"synthetic:{case['case_id']}:{source_key}"
                memory_by_key[source_key]=memory_id
                id_to_fixture[memory_id]={source_key}
                id_to_user[memory_id]={user_ids[owner]}
                fixture_keys.add(source_key)
            source_ids=[memory_by_key[source_key]]
        edges.append(Transition(user_id=user_ids[owner],from_state_id=state_by_key[item["from_fixture_key"]].state_id,
            to_state_id=state_by_key[item["to_fixture_key"]].state_id,
            relation_kind=item.get("relation_kind","TEMPORAL_SUCCESSION"),
            transition_type=item.get("relation_kind","TEMPORAL_SUCCESSION"),
            trigger=item.get("trigger"),reason=item.get("reason"),timestamp=item["timestamp"],
            source_memory_ids=source_ids))
    with runtime.db.connect() as connection:
        for state in states:
            runtime.db.save_state(state,connection=connection)
        for edge in edges:
            runtime.db.save_transition(edge,connection=connection)


def _mem0_causal_support_proxy(question,memories,user_id=None):
    """Lexical indication of explicit causal wording in a retrieved Mem0 record."""
    causal=re.compile(r"\b(?:because|because of|due to|caused by|as a result of)\b",re.I)
    generic={"why","what","cause","causes","caused","change","changed","state","this","that","did","the","my"}
    anchors={word.casefold() for word in re.findall(r"[\w'-]+",question) if len(word)>2 and word.casefold() not in generic}
    for record in memories or []:
        if not isinstance(record,dict): continue
        record_user=record.get("user_id")
        if user_id is not None and record_user is not None and record_user!=user_id:
            continue
        text=" ".join(str(record.get(key,"")) for key in ("memory","text","content"))
        terms={word.casefold() for word in re.findall(r"[\w'-]+",text) if len(word)>2}
        if anchors & terms and causal.search(text): return True
    return False


def _configured_path(name,default):
    """Read a storage setting without importing application config or secrets."""
    value=os.environ.get(name)
    if not value:
        # Parse only the requested path without importing config.py, which
        # creates storage directories as an import side effect.
        from dotenv import dotenv_values
        value=dotenv_values(Path.cwd()/".env").get(name)
    return Path(value or default).expanduser().resolve()


def _overlaps(left,right):
    return left==right or left in right.parents or right in left.parents


def _configure_isolation(isolation_dir):
    """Set isolated paths before any config/database/Mem0 module is imported."""
    loaded=sorted(_STORAGE_MODULES.intersection(sys.modules))
    if loaded:
        raise IsolationError("Storage modules were already imported; restart benchmark in a fresh Python process.")

    root=Path(isolation_dir).expanduser().resolve()
    normal_db=_configured_path("CAUSALMEM_DB","data/causalmem.sqlite3")
    normal_mem0=_configured_path("MEM0_PATH","data/mem0")
    normal_mem0_config=_configured_path("MEM0_DIR",str(normal_mem0.parent/"mem0-config"))
    db_path=(root/"storage"/"causalmem.sqlite3").resolve()
    mem0_path=(root/"storage"/"mem0").resolve()
    # mem0_engine selects this sibling path when Ollama is configured.
    mem0_ollama_path=mem0_path.with_name(f"{mem0_path.name}-ollama")
    mem0_config=(root/"storage"/"mem0-config").resolve()
    isolated_stores=(db_path,mem0_path,mem0_ollama_path,mem0_config)
    configured_stores=(normal_db,normal_mem0,normal_mem0_config)
    if any(_overlaps(isolated,configured)
           for isolated in isolated_stores for configured in configured_stores):
        raise IsolationError("The requested benchmark store overlaps a configured application store.")

    # Never reuse existing data, even if it is under the designated test root.
    existing=[p for p in isolated_stores if p.exists()]
    if existing:
        raise IsolationError("Benchmark storage target already exists; select a fresh isolation directory.")

    root.mkdir(parents=True,exist_ok=True)
    (root/"storage").mkdir(parents=True,exist_ok=True)
    os.environ["CAUSALMEM_DB"]=str(db_path)
    os.environ["MEM0_PATH"]=str(mem0_path)
    os.environ["MEM0_DIR"]=str(mem0_config)
    return SimpleNamespace(root=root,db_path=db_path,mem0_path=mem0_path,
                           mem0_ollama_path=mem0_ollama_path,mem0_config=mem0_config)


def _resolve_evaluation_paths(isolation,output_dir=None,dataset=None,output_file=None,generate=False):
    root=isolation.root
    output_root=Path(output_dir).expanduser().resolve() if output_dir else (root/"outputs").resolve()
    if output_root!=root and root not in output_root.parents:
        raise IsolationError("--output-dir must be inside --isolation-dir.")
    storage_paths=(isolation.db_path,isolation.mem0_path,isolation.mem0_ollama_path,isolation.mem0_config)
    if any(_overlaps(output_root,p) for p in storage_paths):
        raise IsolationError("Evaluation output must be separate from benchmark storage.")
    output_root.mkdir(parents=True,exist_ok=True)

    result_path=Path(output_file).expanduser().resolve() if output_file else output_root/"benchmark_results.json"
    if result_path!=root and root not in result_path.parents:
        raise IsolationError("--output must be inside --isolation-dir.")
    if any(_overlaps(result_path,p) for p in storage_paths):
        raise IsolationError("Benchmark results must be separate from benchmark storage.")
    if result_path.exists():
        raise IsolationError("Benchmark output already exists; select a new output file or isolation directory.")

    dataset_path=Path(dataset).expanduser().resolve() if dataset else output_root/"evaluation_questions.json"
    if dataset_path==result_path:
        raise IsolationError("Dataset and benchmark result paths must be different.")
    if any(_overlaps(dataset_path,p) for p in storage_paths):
        raise IsolationError("Dataset path must be separate from benchmark storage.")
    if (generate or not dataset_path.exists()) and not (dataset_path==root or root in dataset_path.parents):
        raise IsolationError("Generated datasets must be inside --isolation-dir.")
    if generate and dataset_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {dataset_path}")
    return SimpleNamespace(output_dir=output_root,result_path=result_path,dataset_path=dataset_path)


def _load_runtime(isolation):
    """Import application modules only after path overrides are established."""
    config=importlib.import_module("config")
    settings=config.settings
    if Path(settings.database_path).expanduser().resolve()!=isolation.db_path:
        raise IsolationError("Loaded SQLite configuration did not retain the isolated path.")
    if Path(settings.mem0_path).expanduser().resolve()!=isolation.mem0_path:
        raise IsolationError("Loaded Mem0 configuration did not retain the isolated path.")
    backend_path=Path(settings.mem0_path)
    if settings.provider=="ollama":
        backend_path=backend_path.with_name(f"{backend_path.name}-ollama")
    backend_path=backend_path.expanduser().resolve()
    expected_backend=isolation.mem0_ollama_path if settings.provider=="ollama" else isolation.mem0_path
    if backend_path!=expected_backend:
        raise IsolationError("Loaded Mem0 vector-store backend path is not isolated.")
    if Path(os.environ.get("MEM0_DIR","")).expanduser().resolve()!=isolation.mem0_config:
        raise IsolationError("Mem0 configuration directory is not isolated.")

    database=importlib.import_module("database")
    if Path(database.db.path).expanduser().resolve()!=isolation.db_path:
        raise IsolationError("Database client opened a non-isolated SQLite path.")
    mem0_engine=importlib.import_module("mem0_engine")
    # Current Mem0 setup persists vectors in local Qdrant. Fail closed if that
    # backend stops using the configured local path or the source is unverifiable.
    try:
        source="".join(inspect.getsource(mem0_engine._memory).split())
    except (OSError,TypeError) as exc:
        raise IsolationError("Cannot verify the configured Mem0 persistence backend.") from exc
    if ('"provider":"qdrant"' not in source or "settings.mem0_path" not in source or
            "store_path.resolve()" not in source or '"path"' not in source):
        raise IsolationError("Mem0 is not verifiably configured for isolated local Qdrant storage.")

    return SimpleNamespace(
        db=database.db,
        mem0_backend_path=backend_path,
        search_memory=mem0_engine.search_memory,
        add_memory=mem0_engine.add_memory,
        get_memories=mem0_engine.get_memories,
        ingest_message=importlib.import_module("state_extractor").ingest_message,
        timeline=importlib.import_module("temporal_engine").timeline,
        relevant_path=importlib.import_module("causal_graph").relevant_path,
        retrieve=importlib.import_module("hybrid_retriever").retrieve,
        generate_answer=importlib.import_module("answer_generator").generate_answer,
        generate_answer_with_diagnostics=importlib.import_module("answer_generator").generate_answer_with_diagnostics,
        has_supported_cause=importlib.import_module("answer_generator")._has_supported_cause,
        llm=importlib.import_module("llm_service").llm,
        classify=importlib.import_module("query_router").classify_query,
        save=importlib.import_module("evaluator").save,
    )


def _evidence(runtime,question,user_id,category,mode,*,case_id=None,progress_mode=None):
    memories=_run_operation("mem0_search",lambda:runtime.search_memory(question,user_id),
                            case_id=case_id,mode=progress_mode)
    states=[]; transitions=[]; path=[]; sources=["Mem0"]
    if mode in {"state","temporal","graph","full"}:
        states=[s.model_dump(mode="json") for s in _run_operation("sqlite_current_states",lambda:runtime.db.states(user_id,include_inactive=False),case_id=case_id,mode=progress_mode)]; sources.append("State")
    if mode in {"temporal","full"}:
        states=[s.model_dump(mode="json") for s in _run_operation("temporal_timeline",lambda:runtime.timeline(user_id,question),case_id=case_id,mode=progress_mode)]; sources.append("Timeline")
    if mode in {"graph","full"}:
        transitions=[t.model_dump(mode="json") for t in _run_operation("sqlite_transitions",lambda:runtime.db.transitions(user_id),case_id=case_id,mode=progress_mode)]; path=_run_operation("causal_graph_path",lambda:runtime.relevant_path(user_id,question),case_id=case_id,mode=progress_mode); sources.append("Graph")
    return {"memories":memories,"states":states,"transitions":transitions,"causal_path":path,"retrieval_sources":sources}


def _run_dataset(runtime,dataset,modes):
    """Build each case once, then run every condition against that same fixture."""
    rows=[]
    run_id=uuid.uuid4().hex
    for case_index,case in enumerate(dataset):
        case_id=case.get("case_id")
        if not isinstance(case_id,str) or not case_id.strip():
            raise ValueError(f"Dataset case {case_index} requires a non-empty case_id")
        if not isinstance(case.get("question"),str) or not case["question"].strip():
            raise ValueError(f"Case {case_id!r} requires a non-empty question")
        case_started=time.perf_counter()
        _progress("start",case_id=case_id,operation="case")
        target_identity=case.get("fixture_user","user")
        user_specs=[(target_identity,case.get("conversation",[]))]
        distractors=case.get("distractor_users",[])
        if not isinstance(distractors,list):
            raise ValueError(f"Case {case_id!r}: distractor_users must be a list")
        for distractor in distractors:
            if not isinstance(distractor,dict) or not isinstance(distractor.get("fixture_user"),str):
                raise ValueError(f"Case {case_id!r}: each distractor needs a fixture_user and conversation")
            user_specs.append((distractor["fixture_user"],distractor.get("conversation",[])))
        identities=[identity for identity,_ in user_specs]
        if any(not isinstance(identity,str) or not identity.strip() for identity in identities) or len(set(identities))!=len(identities):
            raise ValueError(f"Case {case_id!r}: fixture_user identities must be unique non-empty strings")
        user_ids={identity:f"bench-{run_id}-{case_index}-{case_id}-{identity}" for identity in identities}
        user_id=user_ids[target_identity]
        id_to_fixture={}; id_to_user={}; fixture_keys=set(); fixture_ingestion=[]
        try:
            if case.get("temporal_fixture") is not None:
                _seed_temporal_fixture(runtime,case,user_ids,id_to_fixture,id_to_user,fixture_keys)
            for identity,messages in user_specs:
                actual_user_id=user_ids[identity]
                entries=_message_entries(messages,identity)
                for entry in entries:
                    fixture_key=entry["fixture_key"]
                    if fixture_key in fixture_keys:
                        raise ValueError(f"duplicate fixture_key {fixture_key!r}")
                    fixture_keys.add(fixture_key)
                    fixture_started=time.perf_counter()
                    _progress("start",case_id=case_id,fixture_key=fixture_key,operation="fixture_ingestion")
                    before=_run_operation("mem0_get_before",lambda:runtime.get_memories(actual_user_id),
                                          case_id=case_id,fixture_key=fixture_key)
                    before_ids=_record_ids(before)
                    add_result=_run_operation("mem0_add",lambda:_add_memory_with_diagnostics(
                                                  runtime,entry["text"],actual_user_id),
                                              case_id=case_id,fixture_key=fixture_key)
                    after=_run_operation("mem0_get_after",lambda:runtime.get_memories(actual_user_id),
                                         case_id=case_id,fixture_key=fixture_key)
                    ids=_added_ids(add_result)
                    if not ids:
                        ids=_record_ids(after)-before_ids
                    if not ids:
                        ids={record_id for record_id in _record_ids(after)
                             if any(isinstance(record,dict) and record.get("id")==record_id and
                                    record.get("memory")==entry["text"] for record in after)}
                    for memory_id in ids:
                        id_to_fixture.setdefault(memory_id,set()).add(fixture_key)
                        id_to_user.setdefault(memory_id,set()).add(actual_user_id)
                    ingestion_result=_run_operation(
                        "state_ingestion",
                        lambda:runtime.ingest_message(entry["text"],actual_user_id,sorted(ids),after),
                        case_id=case_id,fixture_key=fixture_key)
                    created_states,created_edges=(ingestion_result if isinstance(ingestion_result,tuple)
                                                   and len(ingestion_result)==2 else ([],[]))
                    fixture_ingestion.append({
                        "fixture_key":fixture_key,
                        "created_states":[{"state_id":state.state_id,"state_type":state.state_type,
                            "state_value":state.state_value,
                            "source_fixture_keys":_fixture_keys_for_ids(state.source_memory_ids,id_to_fixture)}
                            for state in created_states],
                        "created_transitions":[{"transition_id":edge.transition_id,
                            "relation_kind":edge.relation_kind,"from_state_id":edge.from_state_id,
                            "to_state_id":edge.to_state_id,
                            "timestamp":edge.timestamp.isoformat() if hasattr(edge.timestamp,"isoformat") else str(edge.timestamp),
                            "source_fixture_keys":_fixture_keys_for_ids(edge.source_memory_ids,id_to_fixture),
                            "has_trigger":bool(edge.trigger),"has_reason":bool(edge.reason)}
                            for edge in created_edges],
                    })
                    _progress("end",case_id=case_id,fixture_key=fixture_key,operation="fixture_ingestion",
                              elapsed=time.perf_counter()-fixture_started,
                              diagnostics={"created_state_count":len(created_states),
                                  "created_state_types":sorted({state.state_type for state in created_states}),
                                  "created_transition_count":len(created_edges),
                                  "created_transition_relations":sorted({edge.relation_kind for edge in created_edges})})
        except Exception as exc:
            _progress("error",case_id=case_id,operation="fixture_ingestion",error_type=type(exc).__name__)
            raise BenchmarkExecutionError(
                f"Benchmark fixture ingestion failed for case {case_id!r}: {exc}"
            ) from exc

        requested_timestamp=case.get("requested_timestamp")
        timestamp_selector=case.get("requested_timestamp_from_transition")
        if timestamp_selector is not None:
            if not isinstance(timestamp_selector,dict):
                raise ValueError(f"Case {case_id!r}: requested_timestamp_from_transition must be an object")
            fixture_key=timestamp_selector.get("source_fixture_key")
            offset=timestamp_selector.get("offset_microseconds",0)
            if not isinstance(fixture_key,str) or not fixture_key.strip() or not isinstance(offset,int):
                raise ValueError(f"Case {case_id!r}: timestamp selector needs source_fixture_key and integer offset_microseconds")
            matching_memory_ids={memory_id for memory_id,keys in id_to_fixture.items()
                                 if keys=={fixture_key}}
            transitions=runtime.db.transitions(user_id)
            matching_edges=[edge for edge in transitions
                            if matching_memory_ids.intersection(edge.source_memory_ids)]
            if len(matching_edges)!=1:
                raise ValueError(f"Case {case_id!r}: timestamp selector did not identify exactly one recorded transition")
            from temporal_engine import parse_requested_timestamp
            resolved=parse_requested_timestamp(matching_edges[0].timestamp)+timedelta(microseconds=offset)
            requested_timestamp=resolved.isoformat()

        for mode in modes:
            started=time.perf_counter(); retrieval_latency=None; answer_latency=None
            answer_diagnostics=None
            _progress("start",case_id=case_id,mode=mode,operation="condition")
            try:
                query=case["question"]
                query_category=None
                if mode=="mem0":
                    retrieval_started=time.perf_counter()
                    hits=_run_operation("mem0_search",lambda:runtime.search_memory(query,user_id),case_id=case_id,mode=mode)
                    retrieval_latency=time.perf_counter()-retrieval_started
                    answer_started=time.perf_counter()
                    answer=_run_operation("answer_generation",lambda:runtime.llm.answer(query,{"memories":hits,"states":[],"transitions":[],"causal_path":[]}),case_id=case_id,mode=mode)
                    answer_latency=time.perf_counter()-answer_started
                    evidence={"user_id":user_id,"memories":hits,"states":[],"transitions":[],"causal_path":[]}
                elif mode=="full":
                    retrieval_started=time.perf_counter()
                    analysis=_run_operation("query_classification",lambda:runtime.classify(query),case_id=case_id,mode=mode)
                    query_category=analysis.category
                    evidence=_run_operation(
                        "hybrid_retrieval",
                        lambda:(runtime.retrieve(query,user_id,query_category,requested_timestamp)
                                if requested_timestamp is not None
                                else runtime.retrieve(query,user_id,query_category)),
                        case_id=case_id,mode=mode)
                    retrieval_latency=time.perf_counter()-retrieval_started
                    answer_started=time.perf_counter()
                    diagnostic_answerer=getattr(runtime,"generate_answer_with_diagnostics",None)
                    if callable(diagnostic_answerer):
                        answer_result=_run_operation(
                            "answer_generation",lambda:diagnostic_answerer(query,evidence),
                            case_id=case_id,mode=mode)
                        if (not isinstance(answer_result,tuple) or len(answer_result)!=2
                                or not isinstance(answer_result[0],str)
                                or not isinstance(answer_result[1],dict)):
                            raise ValueError("diagnostic answerer returned an invalid result")
                        answer,answer_diagnostics=answer_result
                    else:
                        answer=_run_operation("answer_generation",lambda:runtime.generate_answer(query,evidence),case_id=case_id,mode=mode)
                    answer_latency=time.perf_counter()-answer_started
                else:
                    retrieval_started=time.perf_counter()
                    analysis=_run_operation("query_classification",lambda:runtime.classify(query),case_id=case_id,mode=mode)
                    query_category=analysis.category
                    evidence=_run_operation("evidence_retrieval",lambda:_evidence(runtime,query,user_id,query_category,mode,case_id=case_id,progress_mode=mode),case_id=case_id,mode=mode)
                    retrieval_latency=time.perf_counter()-retrieval_started
                    answer_started=time.perf_counter()
                    answer=_run_operation("answer_generation",lambda:runtime.generate_answer(query,evidence),case_id=case_id,mode=mode)
                    answer_latency=time.perf_counter()-answer_started
            except Exception as exc:
                _progress("error",case_id=case_id,mode=mode,operation="condition",error_type=type(exc).__name__)
                raise BenchmarkExecutionError(
                    f"Benchmark retrieval or generation failed for case {case_id!r}, "
                    f"condition {mode!r}: {exc}"
                ) from exc
            states=evidence.get("states",[]) if mode!="mem0" else None
            state_evidence=None if states is None else [
                {**state,"source_fixture_keys":_fixture_keys_for_ids(state.get("source_memory_ids",[]),id_to_fixture)}
                for state in states if isinstance(state,dict)
            ]
            eligible_cause=(runtime.has_supported_cause(query,evidence) if mode!="mem0"
                            else _mem0_causal_support_proxy(query,evidence.get("memories",[]),user_id))
            causal_keys=[]
            state_owners={state.get("state_id"):state.get("user_id") for state in evidence.get("states",[])
                          if isinstance(state,dict) and state.get("state_id")}
            path_edges={(path.get("from_state_id"),path.get("to_state_id"),path.get("user_id"))
                        for path in evidence.get("causal_path",[]) if isinstance(path,dict)}
            for transition in evidence.get("transitions",[]) or []:
                if (isinstance(transition,dict) and transition.get("user_id")==user_id
                        and transition.get("relation_kind")=="EXPLICIT_CAUSE"
                        and any(isinstance(transition.get(k),str) and transition[k].strip() for k in ("trigger","reason"))
                        and transition.get("from_state_id") and transition.get("to_state_id")
                        and (state_owners.get(transition["from_state_id"])==user_id
                             or (transition["from_state_id"],transition["to_state_id"],user_id) in path_edges)
                        and (state_owners.get(transition["to_state_id"])==user_id
                             or (transition["from_state_id"],transition["to_state_id"],user_id) in path_edges)):
                    causal_keys.extend(_fixture_keys_for_ids(transition.get("source_memory_ids",[]),id_to_fixture))
            if mode=="mem0":
                causal_marker=re.compile(r"\b(?:because|because of|due to|caused by|as a result of)\b",re.I)
                for memory in evidence.get("memories",[]) or []:
                    if (isinstance(memory,dict) and (not memory.get("user_id") or memory.get("user_id")==user_id)
                            and causal_marker.search(" ".join(str(memory.get(k,"")) for k in ("memory","text","content")))):
                        causal_keys.extend(_fixture_keys_for_ids(_record_ids([memory]),id_to_fixture))
            evaluation={key:value for key,value in case.items()
                        if key not in {"conversation","distractor_users","temporal_fixture"}}
            if timestamp_selector is not None:
                evaluation.pop("requested_timestamp_from_transition",None)
                evaluation["requested_timestamp"]=requested_timestamp
            verified_state_owners=None
            if case.get("expected_transition_provenance_keys") is not None:
                # This ownership lookup is user-scoped and uses the isolated
                # SQLite database. It verifies endpoints without adding states
                # that were not selected as answer evidence.
                verified_state_owners={state.state_id:state.user_id
                    for state in runtime.db.states(user_id,include_inactive=True)}
            row={"case_id":case_id,"mode":mode,"category":case.get("category","OTHER"),"question":case["question"],
                 "answer":answer,"answer_behavior":"abstain" if str(answer).strip().casefold()=="i don't have enough information in your memory to determine that." else "answer",
                 "evidence_count":sum(len(evidence.get(k,[]) or []) for k in ("memories","states","transitions")),
                 "evidence_text":json.dumps(evidence,ensure_ascii=False),
                 "evidence_fixture_keys":_bundle_fixture_keys(evidence,id_to_fixture),
                 "transition_provenance_fixture_keys":_transition_provenance_fixture_keys(
                     evidence,id_to_fixture,id_to_user,user_id,verified_state_owners),
                 "source_id_fixture_keys":{memory_id:next(iter(keys)) for memory_id,keys in sorted(id_to_fixture.items())
                                           if len(keys)==1},
                 "state_evidence":state_evidence,"causal_evidence_fixture_keys":sorted(set(causal_keys)),
                 "fixture_key_mapping_ambiguities":sum(len(keys)>1 for keys in id_to_fixture.values()),
                 "eligible_causal_evidence":bool(eligible_cause),
                 "causal_support_method":"retrieved_memory_causal_phrase_proxy" if mode=="mem0" else "current_explicit_cause_gate",
                 "latency_seconds":time.perf_counter()-started,
                 "retrieval_latency_seconds":retrieval_latency,"answer_latency_seconds":answer_latency,
                 "fixture_ingestion":fixture_ingestion}
            if isinstance(answer_diagnostics,dict):
                row["answer_diagnostics"]=answer_diagnostics
            if "expected_behavior" in case:
                row["evaluation"]=evaluation
            if "expected_any" in case:
                row["expected_any"]=case["expected_any"]
            if query_category is not None:
                row["query_category"]=query_category
            rows.append(row)
            _progress("end",case_id=case_id,mode=mode,operation="condition",elapsed=time.perf_counter()-started)
        _progress("end",case_id=case_id,operation="case",elapsed=time.perf_counter()-case_started)
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--isolation-dir",required=True,help="Fresh directory for all benchmark stores and outputs")
    parser.add_argument("--output-dir",help="Results/dataset directory, required to be inside --isolation-dir")
    parser.add_argument("--output",help="Optional results filename, required to be inside --isolation-dir")
    parser.add_argument("--dataset",help="Read-only dataset (defaults to evaluation/ground_truth.jsonl when present)")
    parser.add_argument("--generate",action="store_true",help="Generate the dataset at the isolated dataset path; refuses overwrite")
    parser.add_argument("--modes",nargs="+",choices=MODES,default=list(MODES))
    args=parser.parse_args(argv)
    try:
        _progress("start",operation="benchmark_setup")
        isolation=_configure_isolation(args.isolation_dir)
        dataset_path=args.dataset
        if dataset_path is None and not args.generate:
            dataset_path=_default_dataset_path()
        paths=_resolve_evaluation_paths(isolation,args.output_dir,dataset_path,args.output,args.generate)
        if args.generate or not paths.dataset_path.exists():
            if paths.dataset_path.exists():
                raise FileExistsError(f"Refusing to overwrite existing dataset: {paths.dataset_path}")
            generator=importlib.import_module("dataset_generator")
            generator.generate(paths.dataset_path)
        dataset=_read_dataset(paths.dataset_path)
        runtime=_load_runtime(isolation)
        _progress("end",operation="benchmark_setup")
        _progress("start",operation="dataset_execution")
        rows=_run_dataset(runtime,dataset,args.modes)
        _progress("end",operation="dataset_execution")
        _progress("start",operation="result_save")
        runtime.save(rows,str(paths.result_path))
        _progress("end",operation="result_save")
        print(f"Saved measured predictions and metrics to {paths.result_path}",flush=True)
    except (IsolationError,BenchmarkExecutionError,FileExistsError,ValueError) as exc:
        if isinstance(exc,BenchmarkExecutionError):
            _print_error_traceback(exc)
        parser.error(_concise_error_summary(exc))


if __name__=="__main__":
    main()
