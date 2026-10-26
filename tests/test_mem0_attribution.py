from database import Database
from mem0_engine import add_memory, get_memories, memory_ids_from_add_response, search_memory
from models import Extraction
import state_extractor


class FakeMem0:
    def __init__(self, response):
        self.response=response

    def add(self,*_args,**_kwargs):
        return self.response


class RunningExtractor:
    def extract(self,_message,_memories,_states):
        return Extraction(states=[{
            "state_type":"ACTIVITY","state_value":"running","confidence":.9,
        }])


def test_add_memory_preserves_exact_ids_from_installed_mem0_list_shape(monkeypatch):
    monkeypatch.setattr("mem0_engine._memory",lambda:FakeMem0([
        {"id":"new-memory-id","memory":"enjoys running","event":"ADD"},
    ]))

    response=add_memory("I enjoy running.","alice")

    assert response["results"][0]["id"]=="new-memory-id"
    assert memory_ids_from_add_response(response)==["new-memory-id"]


def test_add_memory_forwards_infer_choice_to_mem0(monkeypatch):
    seen = {}

    class CapturingMem0:
        def add(self, *_args, **kwargs):
            seen.update(kwargs)
            return [{"id": "id"}]

    monkeypatch.setattr("mem0_engine._memory", lambda: CapturingMem0())

    add_memory("I live in Chennai.", "alice", infer=False)
    assert seen["infer"] is False
    assert seen["user_id"] == "alice"

    add_memory("I love running.", "alice")
    assert seen["infer"] is True


def test_add_memory_default_path_does_not_install_timing_wrappers(monkeypatch):
    monkeypatch.setattr("mem0_engine._memory",lambda:FakeMem0([{"id":"id"}]))
    def unexpected(*_args,**_kwargs): raise AssertionError("timing instrumentation should be off")
    monkeypatch.setattr("mem0_engine._timed_mem0_add",unexpected)
    assert add_memory("message","alice")["results"][0]["id"]=="id"


def test_malformed_or_missing_add_ids_are_not_guessed_from_memories(tmp_path,monkeypatch):
    store=Database(str(tmp_path/"attribution.sqlite"))
    monkeypatch.setattr(state_extractor,"db",store)
    monkeypatch.setattr(state_extractor,"llm",RunningExtractor())
   # This models get_all results with a new-looking record first and an older
# record last. Mem0's current get_all is bounded and does not promise order.
    memories=[
        {"id":"candidate-memory","memory":"enjoys running","user_id":"alice"},
        {"id":"older-last","memory":"unrelated older memory","user_id":"alice"},
    ]
    for response in (
        {"results":[{"memory":"enjoys running","event":"ADD"}]},
        {"results":[{"id":"","memory":"enjoys running"},{"id":123}]},
        {"results":[]},
    ):
        monkeypatch.setattr("mem0_engine._memory",lambda response=response:FakeMem0(response))
        add_result=add_memory("I enjoy running.","alice")
        ids=memory_ids_from_add_response(add_result)
        assert ids==[]

        created,_=state_extractor.ingest_message("I enjoy running.","alice",ids,memories)
        persisted=store.states("alice")
        assert created
        assert all(state.source_memory_ids==[] for state in persisted)
        assert "older-last" not in {
            memory_id for state in persisted for memory_id in state.source_memory_ids
        }
        store.clear_user("alice")


def test_add_memory_keeps_legacy_dict_response_ids(monkeypatch):
    monkeypatch.setattr("mem0_engine._memory",lambda:FakeMem0({
        "results":[{"id":"legacy-wrapper-id","memory":"enjoys running","event":"ADD"}],
    }))

    response=add_memory("I enjoy running.","alice")

    assert memory_ids_from_add_response(response)==["legacy-wrapper-id"]


def test_mem0_reads_preserve_only_explicitly_owned_records(monkeypatch):
    local = {"id": "alice-memory", "memory": "enjoys running", "user_id": "alice"}
    nested_local = {
        "id": "alice-nested", "memory": "likes trails", "metadata": {"user_id": "alice"},
    }
    records = [
        local,
        nested_local,
        {"id": "bob-memory", "memory": "likes cycling", "user_id": "bob"},
        {"id": "unknown-owner", "memory": "unattributed"},
        {"id": "conflicting-owner", "user_id": "alice", "metadata": {"user_id": "bob"}},
    ]

    class ReadMem0:
        def get_all(self, **kwargs):
            assert kwargs["filters"] == {"user_id": "alice"}
            return {"results": records}

        def search(self, _query, **kwargs):
            assert kwargs["filters"] == {"user_id": "alice"}
            return {"results": records}

    monkeypatch.setattr("mem0_engine._memory", lambda: ReadMem0())

    import pytest
    with pytest.raises(ValueError,match="unverified user ownership"):
        get_memories("alice")
    with pytest.raises(ValueError,match="unverified user ownership"):
        search_memory("running", "alice")


def test_mem0_reads_return_explicitly_owned_records_when_complete(monkeypatch):
    records=[{"id":"alice-memory","memory":"running","user_id":"alice"},
             {"id":"nested","memory":"cycling","metadata":{"user_id":"alice"}}]
    class ReadMem0:
        def get_all(self,**kwargs): return {"results":records}
        def search(self,*_args,**_kwargs): return {"results":records}
    monkeypatch.setattr("mem0_engine._memory",lambda:ReadMem0())

    assert get_memories("alice")==records
    assert search_memory("running","alice")==records


def test_mem0_search_accepts_empty_results_but_rejects_malformed_envelopes(monkeypatch):
    import pytest

    class ReadMem0:
        def __init__(self,response): self.response=response
        def search(self,*_args,**_kwargs): return self.response

    monkeypatch.setattr("mem0_engine._memory",lambda:ReadMem0({"results":[]}))
    assert search_memory("anything","alice")==[]
    for response in (None,{}, {"results":None}, {"results":{"id":"mem"}}):
        monkeypatch.setattr("mem0_engine._memory",lambda response=response:ReadMem0(response))
        with pytest.raises(ValueError,match="malformed result envelope"):
            search_memory("anything","alice")


def test_add_memory_provider_failure_still_propagates(monkeypatch):
    class FailingMem0:
        def add(self,*_args,**_kwargs):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr("mem0_engine._memory",lambda:FailingMem0())

    import pytest
    with pytest.raises(RuntimeError,match="provider unavailable"):
        add_memory("I enjoy running.","alice")


def test_add_memory_emits_content_free_stage_timings(monkeypatch):
    records=[]

    class Component:
        def embed(self,*_args,**_kwargs): return [0.1]
        def embed_batch(self,*_args,**_kwargs): return [[0.2]]
        def search(self,*_args,**_kwargs): return []
        def generate_response(self,*_args,**_kwargs): return '{"memory":[]}'
        def insert(self,*_args,**_kwargs): return None

    class InstrumentedMem0:
        embedding_model=Component()
        vector_store=Component()
        llm=Component()

        def add(self,*_args,**_kwargs):
            self.embedding_model.embed("private input","search")
            self.vector_store.search(query="private input")
            self.llm.generate_response(messages=[{"content":"private prompt"}])
            self.embedding_model.embed_batch(["private extracted memory"])
            self.vector_store.insert(payloads=[{"data":"private extracted memory"}])
            return [{"id":"created-id","memory":"private extracted memory"}]

    monkeypatch.setattr("mem0_engine._memory",lambda:InstrumentedMem0())
    result=add_memory("private input","alice",diagnostics_callback=records.append)

    assert result["results"][0]["id"]=="created-id"
    assert [record["stage"] for record in records]==[
        "embedding_generation","vector_search","llm_memory_extraction",
        "batch_embedding","vector_store_write","total_mem0_add",
    ]
    assert all(record["success"] is True for record in records)
    assert all(isinstance(record["duration_seconds"],float) for record in records)
    serialized=repr(records)
    assert "private" not in serialized
    assert "alice" not in serialized
    assert "created-id" not in serialized


def test_add_memory_stage_failure_is_timed_and_propagated_without_error_text(monkeypatch):
    records=[]
    failure_text="private provider response"

    class Component:
        def embed(self,*_args,**_kwargs): return [0.1]
        def embed_batch(self,*_args,**_kwargs): return [[0.1]]
        def search(self,*_args,**_kwargs): raise ConnectionError(failure_text)
        def insert(self,*_args,**_kwargs): return None
        def generate_response(self,*_args,**_kwargs): return "{}"

    class FailingMem0:
        embedding_model=Component()
        vector_store=Component()
        llm=Component()
        def add(self,*_args,**_kwargs):
            self.embedding_model.embed("private input","search")
            self.vector_store.search(query="private input")

    monkeypatch.setattr("mem0_engine._memory",lambda:FailingMem0())

    import pytest
    with pytest.raises(ConnectionError,match=failure_text):
        add_memory("private input","alice",diagnostics_callback=records.append)

    assert [(record["stage"],record.get("success"),record.get("status")) for record in records]==[
        ("embedding_generation",True,None),
        ("vector_search",False,None),
        ("batch_embedding",None,"not_observed"),
        ("llm_memory_extraction",None,"not_observed"),
        ("vector_store_write",None,"not_observed"),
        ("total_mem0_add",False,None),
    ]
    assert records[1]["error_category"]=="connection"
    assert all(record["error_category"]=="stage_not_invoked" for record in records[2:5])
    assert records[5]["error_category"]=="connection"
    assert failure_text not in repr(records)


def test_unwrappable_mem0_stages_are_reported_as_unavailable():
    from types import SimpleNamespace

    class ReadOnlyComponent:
        __slots__=()
        def embed(self,*_args,**_kwargs): return [0.1]
        def embed_batch(self,*_args,**_kwargs): return [[0.1]]
        def search(self,*_args,**_kwargs): return []
        def insert(self,*_args,**_kwargs): return None
        def generate_response(self,*_args,**_kwargs): return "{}"

    records=[]
    memory=SimpleNamespace(embedding_model=ReadOnlyComponent(),
                           vector_store=ReadOnlyComponent(),llm=ReadOnlyComponent())
    with __import__("mem0_engine")._timed_mem0_add(memory,records.append):
        pass
    assert {record["stage"]:record.get("status") for record in records}=={
        "embedding_generation":"unavailable",
        "batch_embedding":"unavailable",
        "vector_search":"unavailable",
        "llm_memory_extraction":"unavailable",
        "vector_store_write":"unavailable",
    }
    assert all(record["duration_seconds"] is None for record in records)
    assert all(record["error_category"]=="instrumentation_unavailable" for record in records)


def test_timing_callback_failure_does_not_mask_mem0_failure(monkeypatch):
    original=ConnectionError("provider failure details")
    class FailingMem0:
        def add(self,*_args,**_kwargs): raise original
    monkeypatch.setattr("mem0_engine._memory",lambda:FailingMem0())

    def broken_callback(_record): raise KeyboardInterrupt("diagnostic sink failure")

    import pytest
    with pytest.raises(ConnectionError) as caught:
        add_memory("private input","alice",diagnostics_callback=broken_callback)
    assert caught.value is original
