from database import Database
from models import State, Transition
import causal_graph

def test_user_isolation(tmp_path):
    store=Database(str(tmp_path/"isolation.sqlite"))
    store.save_state(State(user_id="alice",state_type="SKILL",state_value="Python"))
    store.save_state(State(user_id="bob",state_type="SKILL",state_value="Java"))
    assert [s.state_value for s in store.states("alice")] == ["Python"]
    assert store.states("unknown") == []

def test_persistent_graph_and_evidence(tmp_path,monkeypatch):
    store=Database(str(tmp_path/"graph.sqlite"))
    a=State(user_id="u",state_type="SKILL",state_value="Python",source_memory_ids=["mem-a"])
    b=State(user_id="u",state_type="SKILL",state_value="Java",source_memory_ids=["mem-b"])
    store.save_state(a);store.save_state(b)
    edge=Transition(user_id="u",from_state_id=a.state_id,to_state_id=b.state_id,relation_kind="TEMPORAL_SUCCESSION",source_memory_ids=["mem-b"])
    store.save_transition(edge);monkeypatch.setattr(causal_graph,"db",store)
    graph=causal_graph.build_graph("u")
    assert graph.has_edge(a.state_id,b.state_id)
    assert graph.edges[a.state_id,b.state_id]["source_memory_ids"] == ["mem-b"]

def test_transition_requires_existing_same_user_endpoints(tmp_path):
    import pytest
    store=Database(str(tmp_path/"transition_ownership.sqlite"))
    alice=State(user_id="alice",state_type="ACTIVITY",state_value="running")
    bob=State(user_id="bob",state_type="ACTIVITY",state_value="swimming")
    store.save_state(alice);store.save_state(bob)
    valid=Transition(user_id="alice",from_state_id=alice.state_id,to_state_id=alice.state_id)
    store.save_transition(valid)
    assert store.transitions("alice") == [valid]
    before_states={u:[s.state_id for s in store.states(u)] for u in ("alice","bob")}
    with pytest.raises(ValueError,match="endpoints"):
        store.save_transition(Transition(user_id="alice",from_state_id=alice.state_id,to_state_id=bob.state_id))
    with pytest.raises(ValueError,match="endpoints"):
        store.save_transition(Transition(user_id="alice",from_state_id=alice.state_id,to_state_id="missing-state"))
    assert {u:[s.state_id for s in store.states(u)] for u in ("alice","bob")} == before_states
    assert store.transitions("alice") == [valid]
    assert store.transitions("bob") == []

def test_graph_isolated_by_user_with_only_owned_edges(tmp_path,monkeypatch):
    store=Database(str(tmp_path/"graph_users.sqlite"))
    alice_a=State(user_id="alice",state_type="ACTIVITY",state_value="running")
    alice_b=State(user_id="alice",state_type="ACTIVITY",state_value="swimming")
    bob_a=State(user_id="bob",state_type="ACTIVITY",state_value="cycling")
    bob_b=State(user_id="bob",state_type="ACTIVITY",state_value="hiking")
    for state in (alice_a,alice_b,bob_a,bob_b): store.save_state(state)
    store.save_transition(Transition(user_id="alice",from_state_id=alice_a.state_id,to_state_id=alice_b.state_id))
    store.save_transition(Transition(user_id="bob",from_state_id=bob_a.state_id,to_state_id=bob_b.state_id))
    monkeypatch.setattr(causal_graph,"db",store)
    graph_alice=causal_graph.build_graph("alice");graph_bob=causal_graph.build_graph("bob")
    assert {d["state_value"] for _,d in graph_alice.nodes(data=True)} == {"running","swimming"}
    assert {d["state_value"] for _,d in graph_bob.nodes(data=True)} == {"cycling","hiking"}
    assert graph_alice.number_of_edges()==graph_bob.number_of_edges()==1
    assert all(graph_alice.nodes[n]["user_id"]=="alice" for n in graph_alice)
    assert all(graph_bob.nodes[n]["user_id"]=="bob" for n in graph_bob)

def test_timeline_and_graph_paths_remain_user_scoped(tmp_path,monkeypatch):
    from temporal_engine import timeline
    from causal_graph import relevant_path
    store=Database(str(tmp_path/"temporal_users.sqlite"))
    alice_old=State(user_id="alice",state_type="ACTIVITY",state_value="running",status="INACTIVE")
    alice_new=State(user_id="alice",state_type="ACTIVITY",state_value="swimming")
    bob_old=State(user_id="bob",state_type="ACTIVITY",state_value="cycling",status="INACTIVE")
    bob_new=State(user_id="bob",state_type="ACTIVITY",state_value="hiking")
    for state in (alice_old,alice_new,bob_old,bob_new): store.save_state(state)
    store.save_transition(Transition(user_id="alice",from_state_id=alice_old.state_id,to_state_id=alice_new.state_id,relation_kind="EXPLICIT_CAUSE",trigger="injury"))
    store.save_transition(Transition(user_id="bob",from_state_id=bob_old.state_id,to_state_id=bob_new.state_id,relation_kind="TEMPORAL_SUCCESSION"))
    monkeypatch.setattr("temporal_engine.db",store);monkeypatch.setattr("causal_graph.db",store)
    assert {s.state_value for s in timeline("alice")}=={"running","swimming"}
    assert {s.state_value for s in timeline("bob")}=={"cycling","hiking"}
    assert relevant_path("alice","running swimming")[0]["from"]=="running"
    assert relevant_path("bob","cycling hiking")[0]["from"]=="cycling"

def test_states_at_uses_half_open_intervals_and_deterministic_order(tmp_path):
    from datetime import datetime, timezone
    store=Database(str(tmp_path/"as_of.sqlite"))
    start=datetime(2024,1,1,tzinfo=timezone.utc); end=datetime(2024,2,1,tzinfo=timezone.utc)
    a=State(state_id="a",user_id="alice",state_value="first",valid_from=start,valid_until=end)
    b=State(state_id="b",user_id="alice",state_value="second",valid_from=start)
    future=State(state_id="c",user_id="alice",state_value="future",valid_from=end)
    other=State(state_id="d",user_id="bob",state_value="bob",valid_from=start)
    for state in (a,b,future,other): store.save_state(state)
    assert store.states_at("alice",datetime(2023,12,31,tzinfo=timezone.utc))==[]
    assert [s.state_id for s in store.states_at("alice",start)]==["a","b"]
    assert [s.state_id for s in store.states_at("alice",end)]==["b","c"]
    assert [s.state_id for s in store.states_at("bob",start)]==["d"]
    assert store.states_at("nobody",start)==[]

def test_states_at_excludes_unknown_stored_timestamps_and_requires_aware_input(tmp_path):
    import pytest
    from datetime import datetime, timezone
    store=Database(str(tmp_path/"as_of_invalid.sqlite"))
    with store.connect() as c:
        c.execute("INSERT INTO states VALUES(?,?,?,?,?,?,?,?,?,?)",("null-start","alice","ACTIVITY","null start","ACTIVE",None,None,.5,"[]","{}"))
        c.execute("INSERT INTO states VALUES(?,?,?,?,?,?,?,?,?,?)",("bad-start","alice","ACTIVITY","bad start","ACTIVE","not-a-date",None,.5,"[]","{}"))
        c.execute("INSERT INTO states VALUES(?,?,?,?,?,?,?,?,?,?)",("bad-end","alice","ACTIVITY","bad end","INACTIVE","2024-01-01T00:00:00+00:00","not-a-date",.5,"[]","{}"))
    at=datetime(2024,1,2,tzinfo=timezone.utc)
    assert store.states_at("alice",at)==[]
    assert store.states_at("alice","2024-01-02T00:00:00-05:00")==[]
    with pytest.raises(ValueError,match="timezone"):
        store.states_at("alice",datetime(2024,1,2))

def test_states_at_preserves_microsecond_transition_boundaries_and_timezone_offsets(tmp_path):
    import pytest
    from datetime import datetime, timedelta, timezone
    store=Database(str(tmp_path/"as_of_microseconds.sqlite"))
    boundary=datetime(2024,6,1,12,0,0,123456,tzinfo=timezone.utc)
    source=State(state_id="source",user_id="alice",state_value="old goal",
                 valid_from=boundary-timedelta(seconds=1),valid_until=boundary,status="INACTIVE")
    destination=State(state_id="destination",user_id="alice",state_value="new goal",
                      valid_from=boundary)
    open_state=State(state_id="open",user_id="alice",state_value="ongoing",valid_from=boundary)
    other_user=State(state_id="bob",user_id="bob",state_value="private",valid_from=boundary)
    for state in (source,destination,open_state,other_user):
        store.save_state(state)

    before=boundary-timedelta(microseconds=1)
    after=boundary+timedelta(microseconds=1)
    assert [s.state_id for s in store.states_at("alice",before)]==["source"]
    assert [s.state_id for s in store.states_at("alice",boundary)]==["destination","open"]
    assert [s.state_id for s in store.states_at("alice",after)]==["destination","open"]
    # The same instant expressed with a different offset must select the same states.
    equivalent=boundary.astimezone(timezone(timedelta(hours=5,minutes=30)))
    assert [s.state_id for s in store.states_at("alice",equivalent)]==["destination","open"]
    assert [s.state_id for s in store.states_at("bob",boundary)]==["bob"]
    assert store.states_at("nobody",boundary)==[]

    with pytest.raises(ValueError,match="ISO-8601"):
        store.states_at("alice","not-a-timestamp")
    with pytest.raises(ValueError,match="timezone"):
        store.states_at("alice","2024-06-01T12:00:00.123456")

def test_temporal_wrapper_preserves_user_scope(tmp_path,monkeypatch):
    from datetime import datetime, timezone
    from temporal_engine import states_at
    import temporal_engine
    store=Database(str(tmp_path/"as_of_wrapper.sqlite"))
    stamp=datetime(2024,1,1,tzinfo=timezone.utc)
    store.save_state(State(user_id="alice",state_value="alice state",valid_from=stamp))
    store.save_state(State(user_id="bob",state_value="bob state",valid_from=stamp))
    monkeypatch.setattr(temporal_engine,"db",store)
    assert [s.state_value for s in states_at("alice",stamp)]==["alice state"]


def test_natural_utc_date_is_end_of_day_and_unsupported_dates_fail_closed(tmp_path,monkeypatch):
    from datetime import datetime, timezone
    import pytest
    import hybrid_retriever, temporal_engine
    store=Database(str(tmp_path/"natural_date.sqlite"))
    boundary=datetime(2025,10,1,12,tzinfo=timezone.utc)
    old=State(user_id="alice",state_type="GOAL",state_value="learn guitar",
              valid_from=datetime(2025,1,1,tzinfo=timezone.utc),valid_until=boundary,status="INACTIVE")
    new=State(user_id="alice",state_type="GOAL",state_value="learn piano",valid_from=boundary)
    store.save_state(old);store.save_state(new)
    store.save_transition(Transition(user_id="alice",from_state_id=old.state_id,to_state_id=new.state_id,
                                     relation_kind="TEMPORAL_SUCCESSION",timestamp=boundary))
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(temporal_engine,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _q,_u:[])

    result=hybrid_retriever.retrieve("What was my goal on October 1, 2025 UTC?","alice","HISTORY")
    assert result["query_category"]=="TEMPORAL"
    assert [state["state_value"] for state in result["states"]]==["learn piano"]

    for question in (
        "What was my goal on October 1?",
        "What was my goal on October 1, 2025?",
        "What was my goal in October?",
        "What was my goal last year?",
        "What was my goal on October 1, 2025 UTC as of 2025-10-02T00:00:00Z?",
    ):
        with pytest.raises(ValueError):
            hybrid_retriever.retrieve(question,"alice","HISTORY")


def test_natural_utc_datetime_is_exact_and_preserves_half_open_boundaries(tmp_path,monkeypatch):
    from datetime import datetime, timedelta, timezone
    import pytest
    import hybrid_retriever, temporal_engine
    from temporal_engine import parse_requested_timestamp, resolve_requested_timestamp

    boundary=datetime(2026,10,24,19,50,tzinfo=timezone.utc)
    assert resolve_requested_timestamp(
        "What activity did I do on October 24, 2026 at 19:50 UTC?")==boundary
    assert resolve_requested_timestamp(
        "What activity did I do on October 24, 2026 at 20:00 UTC?")==boundary+timedelta(minutes=10)
    assert resolve_requested_timestamp(
        "What activity did I do on Oct 24, 2026 at 19:50 UTC?")==boundary
    assert resolve_requested_timestamp(
        "What activity did I do on Sept 24, 2026 at 19:50 UTC?")==datetime(
            2026,9,24,19,50,tzinfo=timezone.utc)
    assert parse_requested_timestamp("2026-10-24T21:50:00+02:00")==boundary
    with pytest.raises(ValueError):
        resolve_requested_timestamp(
            "What activity did I do on October 24, 2026 at 19:50?")

    store=Database(str(tmp_path/"natural_datetime.sqlite"))
    running=State(user_id="alice",state_type="ACTIVITY",state_value="running",
                  status="INACTIVE",valid_from=datetime(2026,10,22,15,56,tzinfo=timezone.utc),
                  valid_until=boundary)
    swimming=State(user_id="alice",state_type="ACTIVITY",state_value="swimming",
                   valid_from=boundary)
    store.save_state(running);store.save_state(swimming)
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(temporal_engine,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _q,_u:[])

    at_boundary=hybrid_retriever.retrieve(
        "What activity did I do on October 24, 2026 at 19:50 UTC?","alice","HISTORY")
    assert at_boundary["requested_timestamp"]==boundary.isoformat()
    assert [state["state_value"] for state in at_boundary["states"]]==["swimming"]

    abbreviated=hybrid_retriever.retrieve(
        "What activity did I do on Oct 24, 2026 at 19:50 UTC?","alice","HISTORY")
    assert abbreviated["requested_timestamp"]==boundary.isoformat()
    assert [state["state_value"] for state in abbreviated["states"]]==["swimming"]

    before_boundary=hybrid_retriever.retrieve(
        "What activity did I do on October 24, 2026 at 19:49 UTC?","alice","HISTORY")
    assert [state["state_value"] for state in before_boundary["states"]]==["running"]

    later=hybrid_retriever.retrieve(
        "What activity did I do on October 24, 2026 at 20:00 UTC?","alice","HISTORY")
    assert [state["state_value"] for state in later["states"]]==["swimming"]

    offset=hybrid_retriever.retrieve(
        "What activity did I do?","alice","HISTORY","2026-10-24T21:50:00+02:00")
    assert offset["requested_timestamp"]==boundary.isoformat()
    assert [state["state_value"] for state in offset["states"]]==["swimming"]


def test_explicit_requested_timestamp_takes_precedence_over_unsupported_query_date(tmp_path,monkeypatch):
    from datetime import datetime, timezone
    import pytest
    import hybrid_retriever, temporal_engine

    store=Database(str(tmp_path/"explicit_timestamp_precedence.sqlite"))
    old=State(user_id="alice",state_type="PREFERENCE",state_value="Python",
              valid_from=datetime(2025,1,1,tzinfo=timezone.utc),
              valid_until=datetime(2025,6,1,tzinfo=timezone.utc),status="INACTIVE")
    new=State(user_id="alice",state_type="PREFERENCE",state_value="Java",
              valid_from=datetime(2025,6,1,tzinfo=timezone.utc))
    store.save_state(old);store.save_state(new)
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(temporal_engine,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _query,_user:[])

    question="Which language was active before June 1, 2025?"
    instant="2025-05-31T23:59:59Z"
    result=hybrid_retriever.retrieve(question,"alice","TEMPORAL",instant)
    assert [state["state_value"] for state in result["states"]]==["Python"]
    assert result["query_category"]=="TEMPORAL"
    with pytest.raises(ValueError):
        hybrid_retriever.retrieve(question,"alice","TEMPORAL")


def test_predecessor_query_selects_one_owned_transition_and_abstains_when_ambiguous(tmp_path,monkeypatch):
    from datetime import datetime, timedelta, timezone
    import hybrid_retriever, temporal_engine, causal_graph, answer_generator
    store=Database(str(tmp_path/"predecessor.sqlite"))
    start=datetime(2025,1,1,tzinfo=timezone.utc)
    boundary=start+timedelta(days=5)
    later=boundary+timedelta(days=5)
    guitar=State(user_id="alice",state_type="GOAL",state_value="learn guitar",
                 status="INACTIVE",valid_from=start,valid_until=boundary,
                 source_memory_ids=["guitar-memory"])
    piano=State(user_id="alice",state_type="GOAL",state_value="learn piano",
                valid_from=boundary,source_memory_ids=["piano-memory"])
    unchanged=State(user_id="alice",state_type="FACT",state_value="vegetarian",
                    valid_from=start,source_memory_ids=["diet-memory"])
    bob_old=State(user_id="bob",state_type="GOAL",state_value="learn drums",
                  status="INACTIVE",valid_from=start,valid_until=boundary)
    bob_new=State(user_id="bob",state_type="GOAL",state_value="learn bass",valid_from=boundary)
    for state in (guitar,piano,unchanged,bob_old,bob_new): store.save_state(state)
    alice_edge=Transition(user_id="alice",from_state_id=guitar.state_id,to_state_id=piano.state_id,
                          relation_kind="TEMPORAL_SUCCESSION",timestamp=boundary,
                          source_memory_ids=["piano-memory"])
    bob_edge=Transition(user_id="bob",from_state_id=bob_old.state_id,to_state_id=bob_new.state_id,
                        relation_kind="TEMPORAL_SUCCESSION",timestamp=boundary)
    store.save_transition(alice_edge);store.save_transition(bob_edge)
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(temporal_engine,"db",store)
    monkeypatch.setattr(causal_graph,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _q,_u:[
        {"id":"guitar-memory","memory":"learn guitar","user_id":"alice",
         "created_at":start.isoformat()},
        {"id":"piano-memory","memory":"learn piano","user_id":"alice",
         "created_at":boundary.isoformat()},
    ])

    previous=hybrid_retriever.retrieve("What was my goal before I changed it?","alice","HISTORY")
    assert previous["query_category"]=="HISTORY"
    assert [state["state_value"] for state in previous["states"]]==["learn guitar"]
    assert previous["transitions"]==[alice_edge.model_dump(mode="json")]
    assert [memory["id"] for memory in previous["memories"]]==["guitar-memory"]
    assert all(state["user_id"]=="alice" for state in previous["states"])

    # "Have" is an auxiliary in this wording, not a state subject anchor.
    predecessor=hybrid_retriever.retrieve(
        "What goal did I have before I changed it?","alice","HISTORY")
    assert [state["state_value"] for state in predecessor["states"]]==["learn guitar"]
    assert predecessor["transitions"]==[alice_edge.model_dump(mode="json")]

    named_successor=hybrid_retriever.retrieve(
        "What was my goal before learning piano?","alice","HISTORY")
    assert [state["state_value"] for state in named_successor["states"]]==[
        "learn guitar"]
    assert named_successor["transitions"]==[alice_edge.model_dump(mode="json")]

    after=hybrid_retriever.retrieve("What was true on October 2, 2025 UTC?","alice","HISTORY")
    assert {state["state_value"] for state in after["states"]}=={"learn piano","vegetarian"}
    assert all(state["user_id"]=="alice" for state in after["states"])

    spanish=State(user_id="alice",state_type="GOAL",state_value="learn Spanish",
                  valid_from=later,source_memory_ids=["spanish-memory"])
    store.save_state(spanish)
    store.close_state(piano.state_id,later.isoformat())
    second_edge=Transition(user_id="alice",from_state_id=piano.state_id,to_state_id=spanish.state_id,
                           relation_kind="TEMPORAL_SUCCESSION",timestamp=later)
    store.save_transition(second_edge)
    ambiguous=hybrid_retriever.retrieve("What was I before I changed it?","alice","HISTORY")
    assert ambiguous["states"]==[] and ambiguous["transitions"]==[] and ambiguous["memories"]==[]
    assert answer_generator.generate_answer("What was I before I changed it?",ambiguous)==answer_generator.ABSTENTION


def test_current_wording_routes_to_active_snapshot_and_keeps_unchanged_facts(tmp_path,monkeypatch):
    from datetime import datetime, timedelta, timezone
    import hybrid_retriever
    store=Database(str(tmp_path/"current_wording.sqlite"))
    boundary=datetime(2025,4,1,tzinfo=timezone.utc)
    old=State(user_id="alice",state_type="GOAL",state_value="learn guitar",status="INACTIVE",
              valid_from=boundary-timedelta(days=1),valid_until=boundary)
    new=State(user_id="alice",state_type="GOAL",state_value="learn piano",valid_from=boundary)
    fact=State(user_id="alice",state_type="FACT",state_value="vegetarian",valid_from=boundary-timedelta(days=2))
    for state in (old,new,fact): store.save_state(state)
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _q,_u:[])
    result=hybrid_retriever.retrieve("What is my current goal?","alice","HISTORY")
    assert result["query_category"]=="STATE"
    assert {state["state_value"] for state in result["states"]}=={"learn piano","vegetarian"}
    assert all(state["status"]=="ACTIVE" for state in result["states"])


def test_current_and_asof_retrieval_handle_replaced_predecessor_and_user_scope(tmp_path,monkeypatch):
    """A valid inactive predecessor is historical evidence, never current evidence."""
    from datetime import datetime, timedelta, timezone
    import hybrid_retriever, temporal_engine, causal_graph

    store=Database(str(tmp_path/"replacement_current_asof.sqlite"))
    start=datetime(2025,1,1,tzinfo=timezone.utc)
    boundary=start+timedelta(days=10)
    old=State(state_id="alice-old-goal",user_id="alice",state_type="GOAL",
              state_value="learn guitar",status="INACTIVE",valid_from=start,
              valid_until=boundary,source_memory_ids=["alice-old-memory"])
    new=State(state_id="alice-new-goal",user_id="alice",state_type="GOAL",
              state_value="learn piano",status="ACTIVE",valid_from=boundary,
              source_memory_ids=["alice-new-memory"])
    independent=State(state_id="alice-independent-fact",user_id="alice",state_type="FACT",
                      state_value="vegetarian",status="ACTIVE",valid_from=start,
                      source_memory_ids=["alice-fact-memory"])
    bob_old=State(state_id="bob-old-goal",user_id="bob",state_type="GOAL",
                  state_value="learn drums",status="INACTIVE",valid_from=start,
                  valid_until=boundary,source_memory_ids=["bob-old-memory"])
    bob_new=State(state_id="bob-new-goal",user_id="bob",state_type="GOAL",
                  state_value="learn bass",status="ACTIVE",valid_from=boundary)
    for state in (old,new,independent,bob_old,bob_new):
        store.save_state(state)
    edge=Transition(user_id="alice",from_state_id=old.state_id,to_state_id=new.state_id,
                    relation_kind="TEMPORAL_SUCCESSION",timestamp=boundary,
                    source_memory_ids=["alice-new-memory"])
    store.save_transition(edge)
    store.close_state(old.state_id,boundary.isoformat())

    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(temporal_engine,"db",store)
    monkeypatch.setattr(causal_graph,"db",store)
    searched=[]
    memories=[
        {"id":"alice-old-memory","memory":"learn guitar","user_id":"alice","created_at":start.isoformat()},
        {"id":"alice-new-memory","memory":"learn piano","user_id":"alice","created_at":boundary.isoformat()},
        {"id":"alice-fact-memory","memory":"vegetarian","user_id":"alice","created_at":start.isoformat()},
    ]
    monkeypatch.setattr(hybrid_retriever,"search_memory",
                        lambda query,user_id: searched.append(user_id) or memories)

    current=hybrid_retriever.retrieve("What is my current goal?","alice","HISTORY")
    assert current["query_category"]=="STATE"
    assert {state["state_id"] for state in current["states"]}=={
        new.state_id,independent.state_id}
    assert old.state_id not in {state["state_id"] for state in current["states"]}
    # The independent FACT remains in the full state snapshot, while its
    # linked memory is removed from this goal-focused answer bundle.
    assert {record["id"] for record in current["memories"]}=={"alice-new-memory"}
    assert current["state_focus"]["state_types"]==["GOAL"]
    assert current["state_focus"]["candidate_state_ids"]==[new.state_id]
    assert current.get("ambiguity") is None

    before=hybrid_retriever.retrieve(
        "What was my goal?","alice","HISTORY",(boundary-timedelta(microseconds=1)).isoformat())
    assert {state["state_id"] for state in before["states"]}=={
        old.state_id,independent.state_id}
    assert before["state_focus"]["candidate_state_ids"]==[old.state_id]
    assert new.state_id not in {state["state_id"] for state in before["states"]}
    assert all(state["user_id"]=="alice" for state in before["states"])
    assert [record["id"] for record in before["memories"]]==["alice-old-memory"]
    assert before["transitions"]==[]
    assert searched==["alice","alice"]


def test_stale_active_predecessor_makes_current_typed_query_ambiguous(tmp_path,monkeypatch):
    """Contradictory ACTIVE status plus a valid successor edge must not pick a winner."""
    from datetime import datetime, timedelta, timezone
    import answer_generator, hybrid_retriever, temporal_engine, causal_graph

    store=Database(str(tmp_path/"stale_active_predecessor.sqlite"))
    start=datetime(2025,1,1,tzinfo=timezone.utc)
    boundary=start+timedelta(days=5)
    stale=State(state_id="stale-goal",user_id="alice",state_type="GOAL",
                state_value="learn guitar",status="ACTIVE",valid_from=start,
                valid_until=None,source_memory_ids=["old-memory"])
    successor=State(state_id="successor-goal",user_id="alice",state_type="GOAL",
                    state_value="learn piano",status="ACTIVE",valid_from=boundary,
                    source_memory_ids=["new-memory"])
    independent=State(state_id="independent-interest",user_id="alice",state_type="INTEREST",
                      state_value="swimming",status="ACTIVE",valid_from=start)
    unrelated=State(state_id="bob-goal",user_id="bob",state_type="GOAL",
                     state_value="learn drums",status="ACTIVE",valid_from=start)
    for state in (stale,successor,independent,unrelated):
        store.save_state(state)
    edge=Transition(user_id="alice",from_state_id=stale.state_id,to_state_id=successor.state_id,
                    relation_kind="TEMPORAL_SUCCESSION",timestamp=boundary,
                    source_memory_ids=["new-memory"])
    store.save_transition(edge)
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(temporal_engine,"db",store)
    monkeypatch.setattr(causal_graph,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _query,_user:[])

    result=hybrid_retriever.retrieve("What is my current goal?","alice","STATE")
    assert {state["state_id"] for state in result["states"]}=={
        stale.state_id,successor.state_id,independent.state_id}
    assert result["ambiguity"]=={"kind":"inconsistent_state_history"}
    assert result["transitions"]==[]  # current-state routing does not relabel succession as cause

    # A broad current query must also be fail-closed; no state-type keyword is
    # required to detect that the source status contradicts its replacement.
    broad=hybrid_retriever.retrieve("What is current?","alice","STATE")
    assert broad["ambiguity"]=={"kind":"inconsistent_state_history"}
    state_query=hybrid_retriever.retrieve("What is my goal?","alice","STATE")
    assert state_query["ambiguity"]=={"kind":"inconsistent_state_history"}

    unrelated_type=hybrid_retriever.retrieve(
        "What is my current interest?","alice","STATE")
    assert unrelated_type.get("ambiguity") is None
    assert unrelated_type["state_focus"]["candidate_state_ids"]==[independent.state_id]

    class MustNotChoose:
        calls=0
        def answer_with_provenance(self,*_args):
            self.calls+=1
            return {"claims":[{"text":"learn guitar","claim_type":"fact",
                                "evidence":[{"kind":"state","id":stale.state_id}]}]}
    model=MustNotChoose()
    monkeypatch.setattr(answer_generator,"llm",model)
    answer,diagnostics=answer_generator.generate_answer_with_diagnostics(
        "What is my current goal?",result)
    assert answer==answer_generator.ABSTENTION
    assert diagnostics["answer_reason"]=="ambiguous_evidence"
    assert model.calls==0


def test_independent_active_types_remain_retrievable_without_cross_user_leakage(tmp_path,monkeypatch):
    from datetime import datetime, timezone
    import hybrid_retriever, temporal_engine, causal_graph

    store=Database(str(tmp_path/"independent_active_types.sqlite"))
    instant=datetime(2025,1,1,tzinfo=timezone.utc)
    alice_goal=State(state_id="alice-goal",user_id="alice",state_type="GOAL",
                     state_value="learn piano",status="ACTIVE",valid_from=instant)
    alice_interest=State(state_id="alice-interest",user_id="alice",state_type="INTEREST",
                         state_value="swimming",status="ACTIVE",valid_from=instant)
    alice_activity=State(state_id="alice-activity",user_id="alice",state_type="ACTIVITY",
                         state_value="walking",status="ACTIVE",valid_from=instant)
    bob_goal=State(state_id="bob-goal",user_id="bob",state_type="GOAL",
                   state_value="learn drums",status="ACTIVE",valid_from=instant)
    for state in (alice_goal,alice_interest,alice_activity,bob_goal):
        store.save_state(state)
    # A non-replacement graph relation is not lifecycle evidence.
    store.save_transition(Transition(user_id="alice",from_state_id=alice_interest.state_id,
                                     to_state_id=alice_activity.state_id,
                                     relation_kind="UNKNOWN",timestamp=instant))
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(temporal_engine,"db",store)
    monkeypatch.setattr(causal_graph,"db",store)
    searched=[]
    monkeypatch.setattr(hybrid_retriever,"search_memory",
                        lambda _query,user_id: searched.append(user_id) or [])

    result=hybrid_retriever.retrieve("What is my current goal?","alice","STATE")
    assert {state["state_id"] for state in result["states"]}=={
        alice_goal.state_id,alice_interest.state_id,alice_activity.state_id}
    assert result["state_focus"]["candidate_state_ids"]==[alice_goal.state_id]
    assert result.get("ambiguity") is None
    assert all(state["user_id"]=="alice" for state in result["states"])
    assert searched==["alice"]


def test_causal_retrieval_selects_relevant_explicit_evidence_and_preserves_fields(tmp_path,monkeypatch):
    from causal_graph import relevant_transitions
    store=Database(str(tmp_path/"causal_retrieval.sqlite"))
    running=State(user_id="alice",state_type="ACTIVITY",state_value="running")
    swimming=State(user_id="alice",state_type="ACTIVITY",state_value="swimming")
    cycling=State(user_id="alice",state_type="ACTIVITY",state_value="cycling")
    hiking=State(user_id="alice",state_type="ACTIVITY",state_value="hiking")
    for s in (running,swimming,cycling,hiking): store.save_state(s)
    relevant=Transition(user_id="alice",from_state_id=running.state_id,to_state_id=swimming.state_id,
        relation_kind="EXPLICIT_CAUSE",trigger="knee injury",reason="switched due to knee injury",
        source_memory_ids=["memory-17"])
    store.save_transition(relevant)
    store.save_transition(Transition(user_id="alice",from_state_id=cycling.state_id,to_state_id=hiking.state_id,
        relation_kind="EXPLICIT_CAUSE",trigger="weather",reason="rain caused the change",source_memory_ids=["memory-22"]))
    monkeypatch.setattr(causal_graph,"db",store)
    result=relevant_transitions("alice","Why did running change to swimming?")
    assert result==[relevant.model_dump(mode="json")]
    assert {k:result[0][k] for k in ("from_state_id","to_state_id","relation_kind","trigger","reason","timestamp","source_memory_ids")} == {
        "from_state_id":running.state_id,"to_state_id":swimming.state_id,"relation_kind":"EXPLICIT_CAUSE",
        "trigger":"knee injury","reason":"switched due to knee injury","timestamp":relevant.model_dump(mode="json")["timestamp"],
        "source_memory_ids":["memory-17"]}


def test_causal_retrieval_excludes_noncausal_relations_and_unrecorded_causes(tmp_path,monkeypatch):
    from causal_graph import relevant_transitions
    store=Database(str(tmp_path/"causal_relation_kinds.sqlite"))
    a=State(user_id="alice",state_value="running"); b=State(user_id="alice",state_value="swimming")
    store.save_state(a); store.save_state(b)
    for kind,trigger,reason in (("TEMPORAL_SUCCESSION","injury","changed after injury"),
                                ("UNKNOWN","injury","changed after injury"),
                                ("INFERRED_RELATIONSHIP","injury","changed after injury"),
                                ("EXPLICIT_CAUSE",None,None)):
        store.save_transition(Transition(user_id="alice",from_state_id=a.state_id,to_state_id=b.state_id,
            relation_kind=kind,trigger=trigger,reason=reason))
    monkeypatch.setattr(causal_graph,"db",store)
    assert relevant_transitions("alice","What caused running to change to swimming?")==[]


def test_causal_retrieval_no_match_empty_history_and_generic_question(tmp_path,monkeypatch):
    from causal_graph import relevant_transitions
    store=Database(str(tmp_path/"causal_no_match.sqlite"))
    a=State(user_id="alice",state_value="running"); b=State(user_id="alice",state_value="swimming")
    store.save_state(a);store.save_state(b)
    store.save_transition(Transition(user_id="alice",from_state_id=a.state_id,to_state_id=b.state_id,
        relation_kind="EXPLICIT_CAUSE",trigger="injury"))
    monkeypatch.setattr(causal_graph,"db",store)
    assert relevant_transitions("alice","Why did this change?")==[]
    assert relevant_transitions("alice","Why did cycling change to hiking?")==[]
    assert relevant_transitions("nobody","Why did running change to swimming?")==[]


def test_hybrid_causal_retrieval_is_user_scoped_and_path_is_causal_evidence(tmp_path,monkeypatch):
    import hybrid_retriever
    store=Database(str(tmp_path/"hybrid_causal_users.sqlite"))
    alice_a=State(user_id="alice",state_value="running");alice_b=State(user_id="alice",state_value="swimming")
    bob_a=State(user_id="bob",state_value="running");bob_b=State(user_id="bob",state_value="swimming")
    for s in (alice_a,alice_b,bob_a,bob_b):store.save_state(s)
    alice_edge=Transition(user_id="alice",from_state_id=alice_a.state_id,to_state_id=alice_b.state_id,
        relation_kind="EXPLICIT_CAUSE",trigger="knee injury",reason="due to a knee injury",source_memory_ids=["alice-mem"])
    store.save_transition(alice_edge)
    store.save_transition(Transition(user_id="bob",from_state_id=bob_a.state_id,to_state_id=bob_b.state_id,
        relation_kind="EXPLICIT_CAUSE",trigger="schedule",source_memory_ids=["bob-mem"]))
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(causal_graph,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda q,u: [])
    result=hybrid_retriever.retrieve("Why did running change to swimming?","alice","CAUSAL")
    assert result["transitions"]==[alice_edge.model_dump(mode="json")]
    assert len(result["causal_path"])==1
    assert result["causal_path"][0]["source_memory_ids"]==["alice-mem"]
    generic=hybrid_retriever.retrieve("Why did this change?","alice","CAUSAL")
    assert generic["transitions"]==[] and generic["causal_path"]==[]


def test_hybrid_causal_retrieval_includes_only_scoped_transition_endpoints(tmp_path,monkeypatch):
    import hybrid_retriever, temporal_engine
    from datetime import datetime, timedelta, timezone
    store=Database(str(tmp_path/"causal_endpoints.sqlite"))
    start=datetime(2025,1,1,tzinfo=timezone.utc)
    source=State(user_id="alice",state_type="ACTIVITY",state_value="running",
                 source_memory_ids=["running-memory"],valid_from=start)
    target=State(user_id="alice",state_type="ACTIVITY",state_value="swimming",
                 source_memory_ids=["switch-memory"],valid_from=start+timedelta(days=1))
    unrelated=State(user_id="alice",state_type="ACTIVITY",state_value="cycling",
                    valid_from=start+timedelta(days=3))
    bob_source=State(user_id="bob",state_type="ACTIVITY",state_value="running",valid_from=start)
    bob_target=State(user_id="bob",state_type="ACTIVITY",state_value="swimming",valid_from=start)
    for state in (source,target,unrelated,bob_source,bob_target): store.save_state(state)
    edge=Transition(user_id="alice",from_state_id=source.state_id,to_state_id=target.state_id,
        relation_kind="EXPLICIT_CAUSE",trigger="knee injury",reason="switched because of a knee injury",
        source_memory_ids=["switch-memory"],timestamp=start+timedelta(days=2))
    store.save_transition(edge)
    store.close_state(source.state_id,edge.timestamp.isoformat())
    store.save_transition(Transition(user_id="bob",from_state_id=bob_source.state_id,
        to_state_id=bob_target.state_id,relation_kind="EXPLICIT_CAUSE",trigger="schedule"))
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(causal_graph,"db",store)
    monkeypatch.setattr(temporal_engine,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda query,user_id:[])

    result=hybrid_retriever.retrieve("Why did I switch from running to swimming?","alice","HISTORY")

    states={state["state_id"]:state for state in result["states"]}
    assert set(states)=={source.state_id,target.state_id}
    assert states[source.state_id]["status"]=="INACTIVE"
    assert states[target.state_id]["status"]=="ACTIVE"
    assert states[source.state_id]["source_memory_ids"]==["running-memory"]
    assert states[target.state_id]["source_memory_ids"]==["switch-memory"]
    assert result["transitions"]==[edge.model_dump(mode="json")]
    assert result["causal_path"][0]["to"]=="swimming"
    causal_result=hybrid_retriever.retrieve("Why did I switch from running to swimming?","alice","CAUSAL")
    assert {state["state_id"] for state in causal_result["states"]}=={source.state_id,target.state_id}
    assert all(state["user_id"]=="alice" for state in causal_result["states"])
    # Recorded-time boundaries remain half-open; endpoint context in a causal
    # bundle does not change which state was valid at a requested instant.
    assert store.states_at("alice",source.valid_from-timedelta(microseconds=1))==[]
    assert [state.state_id for state in store.states_at("alice",source.valid_from)]==[source.state_id]
    assert [state.state_id for state in store.states_at("alice",edge.timestamp)]==[target.state_id]
    assert {state.state_id for state in store.states_at("bob",source.valid_from)}=={bob_source.state_id,bob_target.state_id}


def test_direct_change_questions_select_cross_type_transition_without_current_snapshot_ambiguity(
        tmp_path,monkeypatch):
    from datetime import datetime, timedelta, timezone
    import hybrid_retriever, temporal_engine, answer_generator

    store=Database(str(tmp_path/"direct_change.sqlite"))
    start=datetime(2025,1,1,tzinfo=timezone.utc)
    running=State(user_id="alice",state_type="INTEREST",state_value="running",
                  status="ACTIVE",valid_from=start,source_memory_ids=["running-memory"])
    swimming=State(user_id="alice",state_type="ACTIVITY",state_value="swimming",
                   status="ACTIVE",valid_from=start+timedelta(days=1),
                   source_memory_ids=["switch-memory"])
    java=State(user_id="alice",state_type="OTHER",
               state_value="My favorite programming language is Java.",
               status="ACTIVE",valid_from=start)
    for state in (running,swimming,java): store.save_state(state)
    edge=Transition(user_id="alice",from_state_id=running.state_id,to_state_id=swimming.state_id,
        transition_type="ACTIVITY_CHANGE",relation_kind="EXPLICIT_CAUSE",
        trigger="my knee injury",reason="Due to my knee injury, I switched from running to swimming",
        source_memory_ids=["switch-memory"],timestamp=start+timedelta(days=1,hours=1))
    store.save_transition(edge)
    store.save_transition(Transition(user_id="alice",from_state_id=swimming.state_id,
        to_state_id=running.state_id,relation_kind="UNKNOWN",
        timestamp=start+timedelta(days=2)))
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(temporal_engine,"db",store)
    monkeypatch.setattr(causal_graph,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _q,_u:[
        {"id":"running-memory","user_id":"alice","memory":"User enjoys running"},
        {"id":"switch-memory","user_id":"alice",
         "memory":"User switched from running to swimming due to a knee injury"},
    ])

    for question in ("What activity did I switch to?","What activity did I switch from?"):
        result=hybrid_retriever.retrieve(question,"alice","STATE")
        assert result.get("ambiguity") is None
        assert result["transitions"]==[edge.model_dump(mode="json")]
        assert {state["state_id"] for state in result["states"]}=={
            running.state_id,swimming.state_id}

    caused=hybrid_retriever.retrieve("What caused my change in activity?","alice","FACTUAL")
    assert causal_graph.is_causal_question("What caused my change in activity?")
    assert caused["transitions"]==[edge.model_dump(mode="json")]
    assert answer_generator._has_supported_cause("What caused my change in activity?",caused)

    changes=hybrid_retriever.retrieve(
        "What changes are recorded in my activity history?","alice","FACTUAL")
    assert changes["transitions"]==[edge.model_dump(mode="json")]

    explicit=hybrid_retriever.retrieve(
        "What explicitly caused the switch to swimming?","alice","HISTORY")
    assert causal_graph.is_causal_question("What explicitly caused the switch to swimming?")
    assert explicit["transitions"]==[edge.model_dump(mode="json")]

    before=hybrid_retriever.retrieve(
        "What activity did I do before switching?","alice","HISTORY")
    assert before["states"]==[running.model_dump(mode="json")]
    assert before["transitions"]==[edge.model_dump(mode="json")]


def test_hybrid_as_of_retrieval_routes_interval_and_transition_evidence(tmp_path,monkeypatch):
    import hybrid_retriever, temporal_engine
    from datetime import datetime, timedelta, timezone
    store=Database(str(tmp_path/"end_to_end_asof.sqlite"))
    start=datetime(2025,1,1,tzinfo=timezone.utc)
    replaced=start+timedelta(days=10)
    old=State(state_id="alice-guitar",user_id="alice",state_type="GOAL",state_value="learn guitar",
              valid_from=start,source_memory_ids=["guitar-memory"])
    new=State(state_id="alice-piano",user_id="alice",state_type="GOAL",state_value="learn piano",
              valid_from=replaced,source_memory_ids=["piano-memory"])
    bob=State(state_id="bob-goal",user_id="bob",state_type="GOAL",state_value="learn drums",
              valid_from=start)
    for state in (old,new,bob): store.save_state(state)
    edge=Transition(user_id="alice",from_state_id=old.state_id,to_state_id=new.state_id,
                    relation_kind="TEMPORAL_SUCCESSION",timestamp=replaced,
                    source_memory_ids=["piano-memory"])
    store.save_transition(edge)
    store.close_state(old.state_id,replaced.isoformat())
    future_old=Transition(user_id="bob",from_state_id=bob.state_id,to_state_id=bob.state_id,
                          relation_kind="EXPLICIT_CAUSE",trigger="later",timestamp=replaced+timedelta(days=2))
    store.save_transition(future_old)
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(temporal_engine,"db",store)
    monkeypatch.setattr(causal_graph,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda q,u:[
        {"id":"guitar-memory","memory":"goal guitar","user_id":u,"created_at":start.isoformat()},
        {"id":"piano-memory","memory":"goal piano","user_id":u,"created_at":replaced.isoformat()},
    ])

    before=hybrid_retriever.retrieve(
        "What was my guitar goal as of 2025-01-05T00:00:00Z?","alice","HISTORY")
    assert [state["state_id"] for state in before["states"]]==[old.state_id]
    assert [memory["id"] for memory in before["memories"]]==["guitar-memory"]
    assert before["transitions"]==[]

    at_boundary=hybrid_retriever.retrieve(
        "What was my piano goal?","alice","HISTORY",replaced.astimezone(
            timezone(timedelta(hours=5))).isoformat())
    assert [state["state_id"] for state in at_boundary["states"]]==[new.state_id]
    assert {memory["id"] for memory in at_boundary["memories"]}=={"piano-memory"}
    assert at_boundary["query_category"]=="TEMPORAL"

    after=hybrid_retriever.retrieve("What was my piano goal?","alice","HISTORY",
                                    (replaced+timedelta(days=1)).isoformat())
    assert [state["state_id"] for state in after["states"]]==[new.state_id]
    assert all(state["user_id"]=="alice" for state in after["states"])
    assert all(memory.get("user_id")!="bob" for memory in after["memories"])


def test_as_of_timestamp_parsing_rejects_naive_and_malformed_inputs(tmp_path,monkeypatch):
    import pytest, hybrid_retriever
    store=Database(str(tmp_path/"asof_parse.sqlite"))
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda q,u:[])
    for bad in ("2025-06-01T00:00:00", "not-a-timestamp", "2025-99-99T00:00:00Z"):
        with pytest.raises((ValueError,TypeError)):
            hybrid_retriever.retrieve("What was true?","alice","HISTORY",bad)
    with pytest.raises(ValueError,match="timezone"):
        hybrid_retriever.retrieve("What was true as of 2025-06-01?","alice","HISTORY")
    with pytest.raises(ValueError,match="ISO-8601"):
        hybrid_retriever.retrieve("What was true as of last year?","alice","HISTORY")


def test_as_of_causal_evidence_has_no_future_transition_and_preserves_relation_kind(tmp_path,monkeypatch):
    import hybrid_retriever, temporal_engine
    from datetime import datetime, timedelta, timezone
    store=Database(str(tmp_path/"asof_causal.sqlite"))
    start=datetime(2025,3,1,tzinfo=timezone.utc); changed=start+timedelta(days=1)
    source=State(user_id="alice",state_value="running",valid_from=start,valid_until=changed,status="INACTIVE")
    target=State(user_id="alice",state_value="swimming",valid_from=changed)
    for state in (source,target): store.save_state(state)
    explicit=Transition(user_id="alice",from_state_id=source.state_id,to_state_id=target.state_id,
        relation_kind="EXPLICIT_CAUSE",trigger="knee injury",timestamp=changed)
    succession=Transition(user_id="alice",from_state_id=source.state_id,to_state_id=target.state_id,
        relation_kind="TEMPORAL_SUCCESSION",timestamp=changed+timedelta(days=1))
    store.save_transition(explicit);store.save_transition(succession)
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(temporal_engine,"db",store)
    monkeypatch.setattr(causal_graph,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda q,u:[])

    before=hybrid_retriever.retrieve("Why did running change to swimming?","alice","CAUSAL",start)
    assert [state["state_value"] for state in before["states"]]==["running"]
    assert before["transitions"]==[] and before["causal_path"]==[]
    at_change=hybrid_retriever.retrieve("Why did running change to swimming?","alice","CAUSAL",changed)
    assert [state["state_value"] for state in at_change["states"]]==["swimming"]
    assert [edge["relation_kind"] for edge in at_change["transitions"]]==["EXPLICIT_CAUSE"]
    assert at_change["causal_path"][0]["trigger"]=="knee injury"

    # Even if a corrupt/early edge exists, a destination whose valid-time starts
    # after the requested instant cannot leak through causal evidence.
    future=State(user_id="alice",state_value="kayaking",valid_from=changed+timedelta(days=1))
    store.save_state(future)
    store.save_transition(Transition(user_id="alice",from_state_id=target.state_id,to_state_id=future.state_id,
        relation_kind="EXPLICIT_CAUSE",trigger="schedule",timestamp=changed))
    no_future_endpoint=hybrid_retriever.retrieve(
        "Why did swimming change to kayaking?","alice","CAUSAL",changed)
    assert all(edge["to_state_id"]!=future.state_id for edge in no_future_endpoint["transitions"])
    assert all(path["to_state_id"]!=future.state_id for path in no_future_endpoint["causal_path"])

    # Ordinary succession never becomes causal evidence, even after its time.
    succession_only=Database(str(tmp_path/"asof_succession.sqlite"))
    succession_only.save_state(source);succession_only.save_state(target)
    succession_only.save_transition(succession)
    monkeypatch.setattr(hybrid_retriever,"db",succession_only)
    monkeypatch.setattr(temporal_engine,"db",succession_only)
    monkeypatch.setattr(causal_graph,"db",succession_only)
    result=hybrid_retriever.retrieve("Why did running change to swimming?","alice","CAUSAL",
                                    changed+timedelta(days=2))
    assert result["transitions"]==[] and result["causal_path"]==[]

def test_quarantine_filters_default_reads_and_preserves_forensic_rows(tmp_path,monkeypatch):
    from datetime import datetime, timedelta, timezone
    import hybrid_retriever, temporal_engine
    store=Database(str(tmp_path/"quarantine.sqlite"))
    start=datetime(2026,10,22,15,56,44,373584,tzinfo=timezone.utc)
    running=State(user_id="alice",state_type="INTEREST",state_value="running",
                  valid_from=start,source_memory_ids=["running-memory"])
    swimming=State(user_id="alice",state_type="ACTIVITY",state_value="swimming",
                   valid_from=start+timedelta(hours=2),source_memory_ids=["swimming-memory"])
    junk=State(user_id="alice",state_type="OTHER",state_value="why i switched to swimmimg",
               valid_from=start+timedelta(hours=1),source_memory_ids=["junk-memory"],
               metadata={"source":"grounded_fallback","review_status":"quarantined",
                         "review_reason":"question_text_ingestion_artifact"})
    for state in (running,swimming,junk): store.save_state(state)
    explicit=Transition(user_id="alice",from_state_id=running.state_id,to_state_id=swimming.state_id,
        transition_type="ACTIVITY_CHANGE",relation_kind="EXPLICIT_CAUSE",trigger="knee injury",
        timestamp=start+timedelta(hours=2),source_memory_ids=["swimming-memory"])
    unknown=Transition(user_id="alice",from_state_id=swimming.state_id,to_state_id=running.state_id,
        relation_kind="UNKNOWN",timestamp=start+timedelta(hours=3),
        metadata={"review_status":"quarantined","review_reason":"derived_unknown_reverse_edge"})
    store.save_transition(explicit);store.save_transition(unknown)

    assert [state.state_value for state in store.states("alice")]==["running","swimming"]
    assert [state.state_value for state in store.states("alice",include_quarantined=True)]==[
        "running","why i switched to swimmimg","swimming"]
    assert store.states("alice",include_inactive=False)[0].state_value=="running"
    assert [edge.transition_id for edge in store.transitions("alice")]==[explicit.transition_id]
    forensic_edges=store.transitions("alice",include_quarantined=True)
    assert [edge.transition_id for edge in forensic_edges]==[
        explicit.transition_id,unknown.transition_id]
    assert forensic_edges[1].relation_kind=="UNKNOWN"
    assert forensic_edges[1].trigger is None and forensic_edges[1].reason is None

    junk_time=start+timedelta(hours=1)
    assert junk.state_value not in {state.state_value for state in store.states_at(
        "alice",junk_time)}
    assert junk.state_value in {state.state_value for state in store.states_at(
        "alice",junk_time,include_quarantined=True)}

    monkeypatch.setattr(temporal_engine,"db",store)
    monkeypatch.setattr(hybrid_retriever,"db",store)
    monkeypatch.setattr(causal_graph,"db",store)
    monkeypatch.setattr(hybrid_retriever,"search_memory",lambda _q,_u:[])
    assert junk.state_value not in {
        state.state_value for state in temporal_engine.timeline("alice")}
    assert junk.state_value in {state.state_value for state in temporal_engine.timeline(
        "alice",include_quarantined=True)}
    result=hybrid_retriever.retrieve("What activity am I doing right now?","alice","STATE")
    assert junk.state_value not in {state["state_value"] for state in result["states"]}
    assert unknown.transition_id not in {edge["transition_id"]
                                         for edge in result["transitions"]}


def test_zero_length_interval_is_never_valid(tmp_path):
    from datetime import datetime, timedelta, timezone
    store=Database(str(tmp_path/"zero_length.sqlite"))
    stamp=datetime(2026,10,22,16,5,14,779410,tzinfo=timezone.utc)
    state=State(user_id="alice",state_type="OTHER",state_value="question artifact",
                status="INACTIVE",valid_from=stamp,valid_until=stamp)
    store.save_state(state)
    assert store.states_at("alice",stamp-timedelta(microseconds=1))==[]
    assert store.states_at("alice",stamp)==[]
    assert store.states_at("alice",stamp+timedelta(microseconds=1))==[]


def test_sqlite_backup_retains_rows_before_a_metadata_mutation(tmp_path):
    import json, sqlite3
    store=Database(str(tmp_path/"backup_source.sqlite"))
    state=State(user_id="alice",state_value="quarantined candidate",
                metadata={"source":"grounded_fallback"})
    store.save_state(state)
    backup_path=str(tmp_path/"backup_copy.sqlite")
    with sqlite3.connect(store.path) as source, sqlite3.connect(backup_path) as target:
        source.backup(target)
    with store.connect() as connection:
        connection.execute("UPDATE states SET metadata=? WHERE state_id=?",
                           (json.dumps({"source":"grounded_fallback",
                                        "review_status":"quarantined"}),state.state_id))

    def metadata(path):
        with sqlite3.connect(path) as connection:
            return json.loads(connection.execute(
                "SELECT metadata FROM states WHERE state_id=?",(state.state_id,)).fetchone()[0])

    assert metadata(backup_path)=={"source":"grounded_fallback"}
    assert metadata(store.path)["review_status"]=="quarantined"
