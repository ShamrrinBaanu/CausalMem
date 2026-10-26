"""Select Mem0, timeline, and graph evidence according to classified intent."""
import re
import time
from datetime import datetime, timezone

from mem0_engine import search_memory
from temporal_engine import (
    is_predecessor_query, predecessor_state, timeline,
    resolve_requested_timestamp, parse_requested_timestamp,
)
from causal_graph import (
    relevant_path, relevant_transitions, is_causal_question,
    is_transition_question, transition_direction,
)
from database import db

TEMPORAL={"TEMPORAL","HISTORY","EVOLUTION","MULTI_HOP","COMPARISON","STATE"}
GRAPH={"CAUSAL","EVOLUTION","MULTI_HOP","COMPARISON"}
_CURRENT_STATE_QUERY=re.compile(r"\b(?:current(?:ly)?|right\s+now|at\s+present|today|now)\b",re.I)
_CURRENT_FAVORITE_QUERY=re.compile(
    r"\b(?:what(?:'s|\s+is|\s+are)|which\s+(?:is|are))\s+my\s+favou?rite\b",re.I)
_PRESENT_TYPED_QUERY=re.compile(
    r"^(?:what|which|where|who)\b.*\b(?:do|does|is|are|am)\b"
    r"|^(?:what|which)\s+(?:i|we)\s+(?:prefer|like|love|want|need)\b",re.I)
_PRESENT_YES_NO_QUERY=re.compile(
    r"^(?:do|does|am|is|are)\s+(?:i|we)\b",re.I)
_COMPLETION_QUERY=re.compile(
    r"\b(?:what|which)\b[^?]*\b(?:did|have|had)\s+(?:i|we)\s+"
    r"(?:complete|finish|achieve)\w*\b",re.I)
_STORED_PREFERENCES_QUERY=re.compile(
    r"\b(?:stored|recorded|saved|mentioned)\b[^?]*\bpreferences?\b|"
    r"\bpreferences?\b[^?]*\b(?:stored|recorded|saved)\b",re.I)
_PROGRAMMING_PREFERENCE_QUERY=re.compile(r"\b(?:programming|coding)\b",re.I)
_MOST_RECENT_PREFERENCE_QUERY=re.compile(
    r"\bmost\s+recently\s+(?:say|state|mention|tell)\b|"
    r"\bmost\s+recent(?:ly)?\s+(?:stated|mentioned)\b[^?]*"
    r"\b(?:preferences?|prefer(?:red|s)?)\b",
    re.I)
_GOAL_PURPOSE_QUERY=re.compile(
    r"\b(?:trying|planning|working|aiming)\s+to\s+"
    r"(?:learn|study|complete|finish|build|improve|achieve|become|develop)\b",re.I)
_FAVORITE_QUERY=re.compile(
    r"\bfavou?rite\s+([a-z][\w'-]*(?:\s+[a-z][\w'-]*){0,2})",re.I)
_FAVORITE_STOP={
    "on","at","in","for","of","the","my","is","are","was","were","do",
    "did","does","to","with","and","or","current","currently","now","today",
    "history","before","after",
}
_STATE_TYPE_TERMS={
    "goal":"GOAL", "goals":"GOAL",
    "preference":"PREFERENCE", "preferences":"PREFERENCE", "prefer":"PREFERENCE",
    "interest":"INTEREST", "interests":"INTEREST", "interested":"INTEREST",
    "activity":"ACTIVITY", "activities":"ACTIVITY",
    "skill":"SKILL", "skills":"SKILL",
    "plan":"PLAN", "plans":"PLAN",
    "fact":"FACT", "facts":"FACT",
    "relationship":"RELATIONSHIP", "relationships":"RELATIONSHIP",
    "location":"LOCATION", "locations":"LOCATION", "live":"LOCATION",
    "lives":"LOCATION", "living":"LOCATION", "reside":"LOCATION",
    "resides":"LOCATION", "residence":"LOCATION", "based":"LOCATION",
    "located":"LOCATION", "move":"LOCATION", "moved":"LOCATION",
    "moves":"LOCATION", "moving":"LOCATION",
    "status":"STATUS", "statuses":"STATUS",
    "life-event":"LIFE_EVENT", "life-events":"LIFE_EVENT",
    "complete":"LIFE_EVENT", "completed":"LIFE_EVENT",
    "completion":"LIFE_EVENT", "finish":"LIFE_EVENT",
    "finished":"LIFE_EVENT", "achieve":"LIFE_EVENT",
    "achieved":"LIFE_EVENT", "achievement":"LIFE_EVENT",
    "achievements":"LIFE_EVENT",
    "emotion":"EMOTION", "emotions":"EMOTION",
    "project":"PROJECT", "projects":"PROJECT",
    "sport":"ACTIVITY", "sports":"ACTIVITY", "exercise":"ACTIVITY",
    "exercises":"ACTIVITY",
    "city":"LOCATION", "cities":"LOCATION", "place":"LOCATION",
    "places":"LOCATION", "relocation":"LOCATION", "relocations":"LOCATION",
    "relocate":"LOCATION", "relocated":"LOCATION",
}
_MEMORY_RELEVANCE_STOP={
    "what","which","where","who","when","how","why","did","do","does",
    "is","are","am","was","were","have","has","had","i","me","my",
    "we","us","our","the","a","an","to","in","on","for","of","from",
    "before","after","now","right","currently","current","previous","previously",
    "stored","recorded","saved","mentioned","trying","working","aiming",
}


def _requested_state_types(question):
    tokens={token.casefold() for token in re.findall(r"[\w-]+",question or "")}
    types={_STATE_TYPE_TERMS[token] for token in tokens if token in _STATE_TYPE_TERMS}
    if (_programming_preference_query(question)
            and re.search(r"\b(?:preference|prefer(?:red|s)?)\b",question or "",re.I)):
        types.add("PREFERENCE")
    if _GOAL_PURPOSE_QUERY.search(question or ""):
        types.add("GOAL")
    if "LIFE_EVENT" in types and _COMPLETION_QUERY.search(question or ""):
        types.discard("PROJECT")
    return types


def _stored_preferences_query(question):
    return bool(_STORED_PREFERENCES_QUERY.search(question or ""))


def _programming_preference_query(question):
    text=question or ""
    return (bool(_PROGRAMMING_PREFERENCE_QUERY.search(text))
            and bool(re.search(r"\b(?:prefer(?:ence|ences|red|s)?|favou?rite)\b",
                               text,re.I)))


def _most_recent_preference_query(question):
    return bool(_MOST_RECENT_PREFERENCE_QUERY.search(question or ""))


def _canonical_preference_key(value):
    if not isinstance(value, str) or not value.strip():
        return None
    key = re.sub(r"\s+", " ", value.strip().casefold())
    domain = key.split(":", 1)[-1] if ":" in key else key
    if domain in {"programming", "programming language", "coding", "language"}:
        return "favorite:programming language"
    return key


def _is_programming_preference_state(state):
    return (state.state_type == "PREFERENCE"
            and _canonical_preference_key(
                (state.metadata or {}).get("preference_key"))
            == "favorite:programming language")


def _preference_conflict(question, states, memories, user_id):
    """Describe conflicting stored values only when same-domain evidence proves it."""
    if not _stored_preferences_query(question):
        return None
    memory_by_id={}
    for record in memories:
        if not isinstance(record,dict):
            continue
        identifier=record.get("id") or record.get("memory_id")
        text=record.get("memory") or record.get("text") or record.get("content")
        if (isinstance(identifier,str) and isinstance(text,str)
                and (record.get("user_id")==user_id
                     or (isinstance(record.get("metadata"),dict)
                         and record["metadata"].get("user_id")==user_id))):
            memory_by_id[identifier]=text
    groups={}
    for state in states:
        if (state.user_id!=user_id or state.state_type!="PREFERENCE"
                or state.status!="ACTIVE"):
            continue
        key=(state.metadata or {}).get("preference_key")
        if not isinstance(key,str) or not key.strip():
            linked=" ".join(memory_by_id.get(identifier,"")
                             for identifier in state.source_memory_ids)
            match=re.search(
                r"\b(?:prefer|like|love)\s+.+?\s+(?:for|as)\s+([^,.;!?]+)",
                linked,re.I)
            key=("domain:"+re.sub(r"\s+"," ",match.group(1)).strip().casefold()
                 if match else None)
        key=_canonical_preference_key(key)
        if key:
            groups.setdefault(key,[]).append(state)
    question_text=(question or "").casefold()
    programming_query=bool(re.search(r"\b(?:programming|languages?)\b",question_text))
    question_terms={word.casefold() for word in re.findall(r"[\w'-]+",question_text)
                    if len(word)>3 and word.casefold() not in {
                        "what","which","stored","recorded","saved","mentioned",
                        "preferences","preference","programming","language",
                        "languages","coding"}}
    conflicts={}
    for key,candidates in groups.items():
        domain=key.split(":",1)[-1]
        if domain=="programming language":
            domain="programming"
        if programming_query and "program" not in domain and "language" not in domain:
            continue
        if question_terms and not any(term in domain for term in question_terms):
            # A domain-specific request must not combine unrelated preference
            # records simply because they share the PREFERENCE type.
            continue
        group=conflicts.setdefault(domain,{"states":[],"memories":[]})
        group["states"].extend(candidates)

    # Legacy and review-era records may be typed OTHER/INTEREST even though
    # their owned Mem0 source is an explicit preference statement. Use that
    # source text as evidence without changing the persisted state type.
    memory_groups={}
    preference_statement=re.compile(
        r"\b(?:i|user)\s+(?:now\s+)?prefers?\s+"
        r"(?P<value>[\w+#.-]+(?:\s+[\w+#.-]+){0,2})\s+"
        r"(?:for|as)\s+(?P<domain>[^,.;!?]+)",re.I)
    for memory_id,text in memory_by_id.items():
        match=preference_statement.search(text)
        if not match:
            continue
        domain=re.sub(r"\s+"," ",match.group("domain")).strip().casefold()
        domain=_canonical_preference_key("domain:"+domain).split(":",1)[-1]
        if domain=="programming language":
            domain="programming"
        value=re.sub(r"\s+"," ",match.group("value")).strip()
        if programming_query and "program" not in domain and "language" not in domain:
            continue
        if question_terms and not any(term in domain for term in question_terms):
            continue
        group=memory_groups.setdefault(domain,{})
        group.setdefault(value.casefold(),{"value":value,"ids":[]})["ids"].append(memory_id)
    for domain,values in memory_groups.items():
        if len(values)>1:
            group=conflicts.setdefault(domain,{"states":[],"memories":[]})
            group["memories"]=[(memory_id,item["value"])
                                for item in values.values()
                                for memory_id in item["ids"]]

    conflicts={domain:group for domain,group in conflicts.items()
               if (len({state.state_value.strip().casefold() for state in group["states"]}
                       | {value.casefold() for _,value in group["memories"]})>1)}
    if len(conflicts)!=1:
        return None
    domain,group=next(iter(conflicts.items()))
    state_ids=list(dict.fromkeys(state.state_id for state in group["states"]))
    memory_values={}
    for memory_id,value in group["memories"]:
        memory_values[memory_id]=value
    values=list(dict.fromkeys(
        [state.state_value for state in group["states"]]
        +list(memory_values.values())))
    return {
        "domain":domain,
        "state_ids":state_ids,
        "memory_ids":list(memory_values),
        "memory_values":memory_values,
        "values":values,
    }


def _memory_fallback_ids(question,memories):
    """Keep query-matched owned memories when typed state extraction missed a fact."""
    terms={token.casefold() for token in re.findall(r"[\w'-]+",question or "")
           if len(token)>2 and token.casefold() not in _MEMORY_RELEVANCE_STOP}
    matched=[]
    for record in memories:
        if not isinstance(record,dict):
            continue
        identifier=record.get("id") or record.get("memory_id")
        text=record.get("memory") or record.get("text") or record.get("content")
        if not isinstance(identifier,str) or not isinstance(text,str):
            continue
        record_terms={token.casefold() for token in re.findall(r"[\w'-]+",text)}
        if terms.intersection(record_terms):
            matched.append(identifier)
    if matched:
        return list(dict.fromkeys(matched))
    return [record.get("id") or record.get("memory_id") for record in memories
            if isinstance(record,dict)
            and isinstance(record.get("id") or record.get("memory_id"),str)]


def _favorite_focus_terms(question):
    """Extract the object of a "favorite X" question without guessing a value."""
    match=_FAVORITE_QUERY.search(question or "")
    if match is None:
        return set()
    terms=[]
    for word in re.findall(r"[\w'-]+",match.group(1)):
        folded=word.casefold()
        if len(word)<=2 or folded in _FAVORITE_STOP:
            break
        terms.append(folded)
        if len(terms)>=3:
            break
    return set(terms)


def _favorite_state_focus(question,states,memories):
    """Focus a favorite question on states whose value or linked memory names it.

    The favorite object must be present in the state's own text or in a Mem0
    record linked by that state. Unlinked states are not used to narrow the
    bundle, so unrelated records keep their existing behavior.
    """
    focus=_favorite_focus_terms(question)
    if not focus:
        return None
    memory_text={}
    for record in memories:
        if not isinstance(record,dict):
            continue
        memory_id=record.get("id") or record.get("memory_id")
        text=record.get("memory") or record.get("text") or record.get("content")
        if isinstance(memory_id,str) and isinstance(text,str):
            memory_text[memory_id]=text
    candidates=[]
    for state in states:
        linked=[memory_id for memory_id in state.source_memory_ids
                if memory_id in memory_text]
        text=(state.state_value or "")+" "+" ".join(memory_text[memory_id]
                                                   for memory_id in linked)
        folded=text.casefold()
        if all(term in folded for term in focus):
            candidates.append((state,linked))
    candidates=[item for item in candidates if item[1]]
    if not candidates:
        return None
    return {
        "state_types":sorted({state.state_type for state,_ in candidates}),
        "candidate_state_ids":[state.state_id for state,_ in candidates],
        "source_memory_ids":sorted({memory_id for _,linked in candidates
                                    for memory_id in linked}),
    }


def _has_inconsistent_replacement_history(states,transitions,user_id,requested_state_types=None):
    """Detect lifecycle rows that contradict a recorded replacement edge.

    Only relations written by the replacement paths are considered:
    EXPLICIT_CAUSE changes close their source across state types, while
    TEMPORAL_SUCCESSION is a replacement only for GOAL-to-GOAL edges. Other
    outgoing edges do not imply that an independent state was superseded.
    """
    from temporal_engine import parse_requested_timestamp

    by_id={state.state_id:state for state in states if state.user_id==user_id}
    for edge in transitions:
        if edge.user_id!=user_id:
            continue
        source=by_id.get(edge.from_state_id)
        destination=by_id.get(edge.to_state_id)
        if source is None or destination is None:
            continue
        if (requested_state_types
                and source.state_type not in requested_state_types
                and destination.state_type not in requested_state_types):
            continue
        replacement=(edge.relation_kind=="EXPLICIT_CAUSE"
                     or (edge.relation_kind=="TEMPORAL_SUCCESSION"
                         and (source.state_type==destination.state_type=="GOAL"
                              or source.state_type==destination.state_type=="PREFERENCE"
                              or source.state_type==destination.state_type=="LOCATION"
                              or edge.transition_type in {"PREFERENCE_CHANGE",
                                                          "LOCATION_CHANGE",
                                                          "GOAL_COMPLETION"})))
        if not replacement:
            continue
        strict_boundary=(edge.relation_kind=="EXPLICIT_CAUSE"
                         or source.state_type==destination.state_type=="GOAL"
                         or edge.transition_type=="GOAL_COMPLETION")
        if not strict_boundary:
            # Preference/location history may come from older ingestion that
            # recorded slightly different boundary timestamps. The actionable
            # contradiction is a source still marked ACTIVE despite its edge.
            if source.status!="INACTIVE":
                return True
            continue
        try:
            edge_time=parse_requested_timestamp(edge.timestamp)
            source_start=parse_requested_timestamp(source.valid_from)
            destination_start=parse_requested_timestamp(destination.valid_from)
            source_end=(parse_requested_timestamp(source.valid_until)
                        if source.valid_until is not None else None)
        except (TypeError,ValueError):
            return True
        # The two intervals must share one boundary. Legacy ingestion can
        # record the transition row a few milliseconds after closing the
        # source, so the edge timestamp may be at or after that boundary; it
        # may not precede it. A stale ACTIVE source still fails closed.
        if (source.status!="INACTIVE" or source_end is None
                or source_start>source_end or source_end>edge_time
                or destination_start!=source_end):
            return True
    return False


class RetrievalFailure(RuntimeError):
    """A retrieval source failed; callers must not treat its evidence as complete."""

    def __init__(self,source,cause_type=None):
        self.source=source
        self.cause_type=cause_type
        super().__init__(f"Retrieval failed for {source}")


def _from_source(source,operation):
    try:
        return operation()
    except RetrievalFailure:
        raise
    except Exception as exc:
        raise RetrievalFailure(source,type(exc).__name__) from None

def _normalized_category(category):
    text = str(category or "OTHER").strip().upper()
    return text or "OTHER"


def retrieve(question,user_id,category,requested_timestamp=None,timings=None):
    started=time.perf_counter()
    timings={} if timings is None else timings
    def finish_timing():
        timings["total_ms"]=round((time.perf_counter()-started)*1000,1)
        return timings
    category = _normalized_category(category)
    as_of = resolve_requested_timestamp(question, requested_timestamp)
    predecessor_intent = is_predecessor_query(question)
    most_recent_preference_intent = (
        _most_recent_preference_query(question)
        and _programming_preference_query(question)
    )
    if predecessor_intent and as_of is not None:
        raise ValueError("A predecessor query cannot also specify an as-of timestamp")
    causal_intent=category=="CAUSAL" or is_causal_question(question)
    transition_intent=is_transition_question(question)
    requested_state_types=_requested_state_types(question)
    typed_current_intent=(as_of is None and not predecessor_intent
                          and not transition_intent and not causal_intent
                          and bool(requested_state_types)
                          and bool(_PRESENT_TYPED_QUERY.search(question)))
    completion_current_intent=(as_of is None and not predecessor_intent
                               and not transition_intent and not causal_intent
                               and "LIFE_EVENT" in requested_state_types
                               and bool(_COMPLETION_QUERY.search(question)))
    current_intent=(as_of is None and not predecessor_intent and not transition_intent
                    and bool(_CURRENT_STATE_QUERY.search(question)
                             or _CURRENT_FAVORITE_QUERY.search(question)
                             or _PRESENT_YES_NO_QUERY.search(question)
                             or typed_current_intent
                             or completion_current_intent)
                    and not causal_intent)
    current_snapshot=(as_of is None and not predecessor_intent and not transition_intent
                     and not causal_intent and (current_intent or category=="STATE"))
    routed_category=("HISTORY" if predecessor_intent
                     else "TEMPORAL" if as_of is not None and not causal_intent
                     else "STATE" if current_intent else category)
    use_time=routed_category in TEMPORAL
    use_graph=routed_category in GRAPH or causal_intent or transition_intent
    mem0_started=time.perf_counter()
    raw_memories=_from_source("Mem0",lambda:search_memory(question,user_id))
    timings["mem0_search_ms"]=round((time.perf_counter()-mem0_started)*1000,1)
    if not isinstance(raw_memories,(list,tuple)):
        raise RetrievalFailure("Mem0")
    if any(not _memory_belongs_to_user(record,user_id) for record in raw_memories):
        raise RetrievalFailure("Mem0 ownership validation")
    memories=_deduplicate_memories(raw_memories)
    if predecessor_intent:
        previous, edge, ambiguous = _from_source(
            "SQLite timeline",lambda:predecessor_state(user_id, question))
        if previous is None or ambiguous:
            # Empty the whole evidence bundle so the existing answer path
            # abstains instead of falling back to unrelated Mem0 text.
            finish_timing()
            return {"user_id":user_id,"query_category":"HISTORY","memories":[],
                    "states":[],"transitions":[],"causal_path":[],
                    "retrieval_sources":["Timeline"],
                    "retrieval_reason":("ambiguous_predecessor_transition"
                                        if ambiguous else
                                        "missing_predecessor_transition")}
        source_ids=set(previous.source_memory_ids)
        memories=[record for record in memories
                  if (record.get("id") or record.get("memory_id")) in source_ids]
        finish_timing()
        return {"user_id":user_id,"query_category":"HISTORY","memories":memories,
                "states":[previous.model_dump(mode="json")],
                "transitions":[edge.model_dump(mode="json")],"causal_path":[],
                "retrieval_sources":["Timeline"]}
    if as_of is not None:
        # Mem0 results without a trustworthy recorded timestamp cannot be
        # established as available at the requested instant. Mark that source
        # incomplete instead of silently returning a partial bundle.
        filtered=[]
        for record in memories:
            if not isinstance(record,dict):
                raise RetrievalFailure("Mem0 temporal metadata")
            created_at=record.get("created_at")
            if created_at is None:
                raise RetrievalFailure("Mem0 temporal metadata")
            try:
                if parse_requested_timestamp(created_at) <= as_of:
                    filtered.append(record)
            except (TypeError,ValueError):
                raise RetrievalFailure("Mem0 temporal metadata") from None
        memories=filtered
    if current_snapshot:
        history=_from_source("SQLite states",lambda:db.states(user_id))
        # Status is a lifecycle label, not a temporal proof. A row left ACTIVE
        # after its valid_until has elapsed must not be returned as current.
        states=_deduplicate_equivalent_states([
            state for state in _from_source(
                "SQLite states",
                lambda:db.states_at(user_id,datetime.now(timezone.utc)),
            )
            if state.status=="ACTIVE"
        ])
        # Quarantined rows are hidden from normal state retrieval, so their
        # linked memories must be hidden too unless another clean state links
        # the same memory.
        forensic_history=_from_source(
            "SQLite state review history",
            lambda:db.states(user_id,include_quarantined=True),
        )
        inactive_memory_ids={memory_id for state in forensic_history
                             if state.status!="ACTIVE"
                             or state.metadata.get("review_status")=="quarantined"
                             for memory_id in state.source_memory_ids}
        active_memory_ids={memory_id for state in states for memory_id in state.source_memory_ids}
        if inactive_memory_ids:
            memories=[record for record in memories
                      if not isinstance(record,dict)
                      or not isinstance(record.get("id") or record.get("memory_id"),str)
                      or (record.get("id") or record.get("memory_id")) not in inactive_memory_ids
                      or (record.get("id") or record.get("memory_id")) in active_memory_ids]
        current_transitions=_from_source("SQLite transitions",lambda:db.transitions(user_id))
        inconsistent_history=_has_inconsistent_replacement_history(
            history,current_transitions,user_id,requested_state_types)
    else:
        states=None
        inconsistent_history=False
    # STATE asks for the current snapshot; historical context belongs to the
    # timeline categories, which may include superseded transition endpoints.
    if states is None:
        if most_recent_preference_intent or (
                _programming_preference_query(question)
                and use_time and routed_category.upper()!="STATE"):
            # Timeline's lexical narrowing may select a legacy untyped
            # predecessor while omitting the typed programming preference.
            # These preference intents are resolved from the user's full
            # structured history, then scoped below by the domain key.
            states=_from_source("SQLite preference history",
                                lambda:db.states(user_id))
        elif as_of is not None or (use_time and routed_category.upper()!="STATE"):
            states=_from_source("SQLite timeline",lambda:timeline(user_id,question,as_of))
        else:
            states=_from_source("SQLite states",lambda:db.states(user_id,include_inactive=False))
    if _programming_preference_query(question):
        states=[state for state in states
                if state.state_type!="PREFERENCE"
                or _is_programming_preference_state(state)]
        all_user_states=_from_source("SQLite preference history",
                                     lambda:db.states(user_id))
        programming_memory_ids={memory_id for state in states
                                if _is_programming_preference_state(state)
                                for memory_id in state.source_memory_ids}
        unrelated_preference_memory_ids={
            memory_id for state in all_user_states
            if state.state_type=="PREFERENCE"
            and not _is_programming_preference_state(state)
            for memory_id in state.source_memory_ids
        }
        if unrelated_preference_memory_ids:
            memories=[record for record in memories
                      if not isinstance(record,dict)
                      or not isinstance(record.get("id") or record.get("memory_id"),str)
                      or ((record.get("id") or record.get("memory_id"))
                          not in unrelated_preference_memory_ids)
                      or ((record.get("id") or record.get("memory_id"))
                          in programming_memory_ids)]
    if most_recent_preference_intent:
        eligible=[]
        for state in states:
            metadata=state.metadata or {}
            if (state.user_id!=user_id or not _is_programming_preference_state(state)
                    or metadata.get("source_role")!="user"):
                continue
            recorded=metadata.get("source_timestamp") or state.valid_from
            try:
                instant=parse_requested_timestamp(recorded)
            except (TypeError,ValueError):
                continue
            eligible.append((instant,state))
        eligible.sort(key=lambda pair:(pair[0],pair[1].state_id),reverse=True)
        states=[eligible[0][1]] if eligible else []
        linked_ids=set(states[0].source_memory_ids) if states else set()
        memories=[record for record in memories
                  if isinstance(record,dict)
                  and (record.get("id") or record.get("memory_id")) in linked_ids]
    state_started=time.perf_counter()
    if as_of is not None:
        # Mem0 creation time can precede SQLite's transition timestamp because
        # ingestion writes Mem0 first. For memories linked to explicit states,
        # state validity is the temporal authority; keep unlinked records only
        # under the existing Mem0 created_at filter above.
        all_user_states=_from_source("SQLite state history",lambda:db.states(user_id))
        linked_ids={memory_id for state in all_user_states
                    for memory_id in state.source_memory_ids}
        valid_linked_ids={memory_id for state in states
                          for memory_id in state.source_memory_ids}
        memories=[record for record in memories
                  if not isinstance(record,dict)
                  or not isinstance(record.get("id") or record.get("memory_id"),str)
                  or (record.get("id") or record.get("memory_id")) not in linked_ids
                  or (record.get("id") or record.get("memory_id")) in valid_linked_ids]
    ambiguity=({"kind":"inconsistent_state_history"} if inconsistent_history else None)
    state_focus=None
    if not causal_intent and not transition_intent and not predecessor_intent:
        state_focus=_favorite_state_focus(question,states,memories)
    if requested_state_types and not causal_intent and not transition_intent:
        # Keep the complete state snapshot/timeline for compatibility, but
        # identify the matching candidates so answer generation need not treat
        # unrelated active types as answers to a type-specific question.
        all_user_states=_from_source("SQLite states",lambda:db.states(user_id))
        focused_states=[state for state in states if state.state_type in requested_state_types]
        matching_ids={memory_id for state in focused_states for memory_id in state.source_memory_ids}
        unrelated_ids={memory_id for state in all_user_states
                       if state.state_type not in requested_state_types
                       for memory_id in state.source_memory_ids}
        if unrelated_ids and focused_states and not _stored_preferences_query(question):
            memories=[record for record in memories
                      if not isinstance(record,dict)
                      or not isinstance(record.get("id") or record.get("memory_id"),str)
                      or (record.get("id") or record.get("memory_id")) not in unrelated_ids
                      or (record.get("id") or record.get("memory_id")) in matching_ids]
        # A singular current-state question cannot safely choose among several
        # active records of the requested type. Keep the evidence and expose
        # the ambiguity for the answer layer to abstain.
        if focused_states:
            state_focus={"state_types":sorted(requested_state_types),
                         "candidate_state_ids":[state.state_id for state in focused_states],
                         "source_memory_ids":sorted(matching_ids)}
        else:
            state_focus={"state_types":sorted(requested_state_types),
                         "candidate_state_ids":[],
                         "source_memory_ids":_memory_fallback_ids(question,memories),
                         "memory_fallback":True}
        if current_snapshot and len(focused_states)==1 and matching_ids:
            # A single current typed state is the authoritative answer. Do not
            # let an unlinked Mem0 record of the same type compete with it.
            memories=[record for record in memories
                      if not isinstance(record,dict)
                      or not isinstance(record.get("id") or record.get("memory_id"),str)
                      or (record.get("id") or record.get("memory_id")) in matching_ids]
        if (len(focused_states)>1 and ambiguity is None
                and not _stored_preferences_query(question)
                and (current_snapshot or (as_of is None and not use_time))):
            # Every route that returns a current typed snapshot must apply the
            # same conflict rule, not only the STATE category.
            ambiguity={"kind":"multiple_state_candidates",
                       "state_type":sorted(requested_state_types),
                       "candidate_state_ids":[state.state_id for state in focused_states]}
    timings["timeline_db_ms"]=round((time.perf_counter()-state_started)*1000,1)
    graph_started=time.perf_counter()
    if use_graph:
        if as_of is None:
            transitions=_from_source("SQLite transitions",lambda:relevant_transitions(
                user_id,question,causal_only=causal_intent))
        else:
            transitions=_from_source("SQLite transitions",lambda:relevant_transitions(
                user_id,question,causal_only=causal_intent,requested_timestamp=as_of))
    else:
        transitions=[]
    if use_graph and (causal_intent or transition_intent):
        # Include both endpoints of selected causal or direct-change evidence
        # even when temporal lexical filtering returned only one endpoint.
        # Load endpoints through the user-scoped API and preserve order.
        user_states=_from_source("SQLite transition endpoints",lambda:db.states(user_id))
        states_by_id={s.state_id:s for s in user_states}
        selected_ids=set()
        for transition in transitions:
            selected_ids.update((transition["from_state_id"],transition["to_state_id"]))
        if as_of is None and (causal_intent or selected_ids):
            states=[state for state in user_states if state.state_id in selected_ids]
        if causal_intent:
            causal_path=[{"from":states_by_id[t["from_state_id"]].state_value,
                          "to":states_by_id[t["to_state_id"]].state_value,**t} for t in transitions
                         if t["from_state_id"] in states_by_id and t["to_state_id"] in states_by_id]
        else:
            causal_path=[]
        if causal_intent and len(transitions)>1:
            ambiguity={"kind":"multiple_causal_candidates",
                       "candidate_transition_ids":[t["transition_id"] for t in transitions]}
        elif (not causal_intent and transition_direction(question)
              and len(transitions)>1):
            ambiguity={"kind":"multiple_transition_candidates",
                       "candidate_transition_ids":[t["transition_id"] for t in transitions]}
    else:
        causal_path=_from_source("causal path",lambda:relevant_path(
            user_id,question,as_of)) if use_graph else []
    timings["graph_ms"]=round((time.perf_counter()-graph_started)*1000,1)
    finish_timing()
    result={"user_id":user_id,"query_category":routed_category,"memories":memories,"states":[s.model_dump(mode="json") for s in states],"transitions":transitions,"causal_path":causal_path,"retrieval_sources":["Mem0"]+( ["Timeline"] if use_time else [])+( ["Graph"] if use_graph else [])}
    if as_of is not None:
        resolved_timestamp=as_of.isoformat()
        result["requested_timestamp"]=resolved_timestamp
        result["temporal_context"]={
            "mode":"as_of",
            "requested_timestamp":resolved_timestamp,
            "state_validity_rule":"valid_from <= requested_timestamp < valid_until; null valid_until is open-ended",
            "selected_states_valid_at_requested_timestamp":True,
        }
    if ambiguity:
        result["ambiguity"]=ambiguity
    if state_focus:
        result["state_focus"]=state_focus
    preference_conflict=_preference_conflict(question,states,memories,user_id)
    if preference_conflict:
        result["preference_conflict"]=preference_conflict
        if _stored_preferences_query(question):
            candidate_ids=set(preference_conflict.get("state_ids",[]))
            conflict_memory_ids=set(preference_conflict.get("memory_ids",[]))
            state_memory_ids={memory_id for state in states
                              if state.state_id in candidate_ids
                              for memory_id in state.source_memory_ids}
            result["state_focus"]={
                "state_types":["PREFERENCE"],
                "candidate_state_ids":sorted(candidate_ids),
                "source_memory_ids":sorted(conflict_memory_ids|state_memory_ids),
                "stored_preference_inventory":True,
            }
    return result


def _deduplicate_equivalent_states(states):
    """Collapse legacy duplicate current states with the same type and value.

    Distinct values remain separate so genuine conflicts still abstain. The
    most recently established equivalent row represents the pair; no database
    rows are modified here.
    """
    result=[]
    positions={}
    for state in states or []:
        value=re.sub(r"\s+"," ",str(state.state_value or "").strip()).casefold()
        key=(str(state.state_type or "").upper(),value)
        if key not in positions:
            positions[key]=len(result)
            result.append(state)
            continue
        current=result[positions[key]]
        if state.valid_from > current.valid_from:
            result[positions[key]]=state
    return result


def _deduplicate_memories(records):
    """Remove exact repeated Mem0 records only when they have a stable ID."""
    import json
    from collections.abc import Mapping

    result=[]
    seen=set()
    for record in records or []:
        memory_id=record.get("id") if isinstance(record,Mapping) else None
        if not isinstance(memory_id,str) or not memory_id.strip():
            result.append(record)
            continue
        try:
            fingerprint=json.dumps(record,sort_keys=True,separators=(",",":"),ensure_ascii=False,default=str)
        except (TypeError,ValueError):
            result.append(record)
            continue
        key=(memory_id,fingerprint)
        if key in seen:
            continue
        seen.add(key)
        result.append(record)
    return result


def _memory_belongs_to_user(record,user_id):
    """Fail closed unless a Mem0 result explicitly identifies the requested user.

    Current Mem0 OSS promotes payload ``user_id`` to the result's top level.
    ``metadata.user_id`` is accepted for compatible nested result formats. If
    both forms are present, every value must agree with the requested user.
    """
    from collections.abc import Mapping

    if not isinstance(record,Mapping) or not isinstance(user_id,str) or not user_id:
        return False
    owners=[]
    if "user_id" in record:
        owners.append(record["user_id"])
    metadata=record.get("metadata")
    if isinstance(metadata,Mapping) and "user_id" in metadata:
        owners.append(metadata["user_id"])
    return bool(owners) and all(isinstance(owner,str) and owner==user_id for owner in owners)
