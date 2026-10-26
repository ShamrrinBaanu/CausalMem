"""Adapter for the actual mem0ai OSS Memory API."""
import atexit
from contextlib import ExitStack
import time
from functools import lru_cache
from pathlib import Path
from config import settings

@lru_cache(maxsize=1)
def _memory():
    from mem0 import Memory
    # OpenAI's default embedder and Ollama's nomic embedder have different
    # vector dimensions. Keep provider stores separate to avoid invalid Qdrant
    # dimension comparisons when a project switches LLM providers.
    store_path=Path(settings.mem0_path)
    collection="causalmem"
    dimensions=1536
    if settings.provider=="ollama":
        store_path=store_path.with_name(f"{store_path.name}-ollama")
        collection="causalmem_ollama"
        dimensions=768
    cfg={"vector_store":{"provider":"qdrant","config":{"collection_name":collection,"path":str(store_path.resolve()),"embedding_model_dims":dimensions}},"llm":{"provider":"openai","config":{"model":settings.model,"api_key":settings.api_key or ""}}}
    if settings.provider=="ollama":
        cfg["llm"]={"provider":"ollama","config":{"model":settings.model,"ollama_base_url":settings.ollama_server_url}}
        cfg["embedder"]={"provider":"ollama","config":{"model":"nomic-embed-text","ollama_base_url":settings.ollama_server_url}}
    return Memory.from_config(cfg)
def _close_memory_resources():
    """Close cached Mem0 resources safely at interpreter shutdown."""
    if _memory.cache_info().currsize == 0:
        return

    try:
        memory = _memory()
    except Exception:
        return

    client = getattr(
        getattr(memory, "vector_store", None),
        "client",
        None,
    )

    try:
        memory.close()
    except Exception:
        pass

    if client is not None:
        try:
            client.close()
        except Exception:
            pass

    _memory.cache_clear()


atexit.register(_close_memory_resources)

def _safe_error_category(exc):
    """Map exceptions to text-free, low-cardinality diagnostic categories."""
    name=type(exc).__name__.casefold()
    if isinstance(exc,TimeoutError) or "timeout" in name:
        return "timeout"
    if isinstance(exc,ConnectionError) or "connection" in name:
        return "connection"
    return "operation_error"


def _emit_timing(callback,stage,started,success=None,error_category=None,status=None):
    if callback is None:
        return
    record={"stage":stage,"duration_seconds":None if status else time.perf_counter()-started}
    if success is not None:
        record["success"]=success
    if status is not None:
        record["status"]=status
    if error_category is not None:
        record["error_category"]=error_category
    try:
        callback(record)
    except BaseException:
        # Diagnostic sinks must not change memory-write behavior.
        pass


def _instrument_method(stack,target,method_name,stage,callback,observed):
    """Temporarily time one Mem0 component method without logging its arguments."""
    original=getattr(target,method_name,None)
    if not callable(original):
        return False

    def timed(*args,**kwargs):
        started=time.perf_counter()
        observed.add(stage)
        try:
            result=original(*args,**kwargs)
        except BaseException as exc:
            _emit_timing(callback,stage,started,False,_safe_error_category(exc))
            raise
        _emit_timing(callback,stage,started,True)
        return result

    try:
        setattr(target,method_name,timed)
    except (AttributeError,TypeError):
        return False
    stack.callback(setattr,target,method_name,original)
    return True


def _emit_stage_gaps(callback,availability,observed):
    """Mark stages that could not be wrapped or did not run on this add path."""
    for stage,available in availability.items():
        if not available:
            _emit_timing(callback,stage,0,status="unavailable",
                         error_category="instrumentation_unavailable")
        elif stage not in observed:
            _emit_timing(callback,stage,0,status="not_observed",
                         error_category="stage_not_invoked")


def _timed_mem0_add(memory,callback):
    """Patch only the duration of a diagnostic-enabled add call."""
    stack=ExitStack()
    observed=set()
    availability={}
    stages=(
        (getattr(memory,"embedding_model",None),"embed","embedding_generation"),
        (getattr(memory,"embedding_model",None),"embed_batch","batch_embedding"),
        (getattr(memory,"vector_store",None),"search","vector_search"),
        (getattr(memory,"llm",None),"generate_response","llm_memory_extraction"),
        (getattr(memory,"vector_store",None),"insert","vector_store_write"),
    )
    for target,method_name,stage in stages:
        availability[stage]=target is not None and _instrument_method(
            stack,target,method_name,stage,callback,observed)
    stack.callback(_emit_stage_gaps,callback,availability,observed)
    return stack


def add_memory(message,user_id,diagnostics_callback=None,infer=True):
    started=time.perf_counter()
    try:
        memory=_memory()
        if diagnostics_callback is None:
            result=memory.add([{"role":"user","content":message}],user_id=user_id,
                              infer=infer)
        else:
            with _timed_mem0_add(memory,diagnostics_callback):
                result=memory.add([{"role":"user","content":message}],user_id=user_id,
                                  infer=infer)
    except BaseException as exc:
        _emit_timing(diagnostics_callback,"total_mem0_add",started,False,_safe_error_category(exc))
        raise
    _emit_timing(diagnostics_callback,"total_mem0_add",started,True)
    # Mem0 OSS 2.2.1 returns a list of {"id", "memory", "event"} records.
    # Normalize that response to the adapter's stable dict shape without
    # discarding the exact IDs returned by the successful write.
    if isinstance(result,list): return {"results":result}
    return result if isinstance(result,dict) else {"results":[]}

def memory_ids_from_add_response(result):
    """Return only usable stable IDs explicitly returned by Mem0.add."""
    from collections.abc import Mapping

    records=result if isinstance(result,list) else result.get("results",[]) if isinstance(result,Mapping) else []
    if isinstance(records,Mapping): records=[records]
    if not isinstance(records,(list,tuple)): return []
    ids=[]
    for record in records:
        memory_id=record.get("id") if isinstance(record,Mapping) else None
        if isinstance(memory_id,str) and memory_id.strip() and memory_id not in ids:
            ids.append(memory_id)
    return ids

def _belongs_to_user(record, user_id):
    """Accept Mem0 records only when their returned ownership is explicit."""
    from collections.abc import Mapping

    if not isinstance(record, Mapping) or not isinstance(user_id, str) or not user_id:
        return False
    owners = []
    if "user_id" in record:
        owners.append(record["user_id"])
    metadata = record.get("metadata")
    if isinstance(metadata, Mapping) and "user_id" in metadata:
        owners.append(metadata["user_id"])
    return bool(owners) and all(isinstance(owner, str) and owner == user_id for owner in owners)

def _read_results(result,user_id,operation):
    if not isinstance(result,dict) or "results" not in result or not isinstance(result["results"],list):
        raise ValueError(f"Mem0 {operation} returned a malformed result envelope")
    records=result["results"]
    if any(not _belongs_to_user(record,user_id) for record in records):
        raise ValueError(f"Mem0 {operation} returned a record with unverified user ownership")
    return records

def search_memory(query,user_id,limit=8):
    # Current Mem0 OSS scopes reads through filters and uses top_k.
    result=_memory().search(query,filters={"user_id":user_id},top_k=limit)
    # An empty result list is a valid search result. A malformed envelope or
    # any record with missing/conflicting/foreign ownership is a failed search,
    # since silently filtering it would make partial evidence look complete.
    return _read_results(result,user_id,"search")
def get_memories(user_id):
    result=_memory().get_all(filters={"user_id":user_id})
    return _read_results(result,user_id,"get_all")
def update_memory(memory_id,data): return _memory().update(memory_id=memory_id,data=data)
def delete_memory(memory_id): return _memory().delete(memory_id=memory_id)
