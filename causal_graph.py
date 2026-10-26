import networkx as nx
import re
from database import db
from temporal_engine import parse_requested_timestamp

# Words expressing a request for causal or temporal explanation do not identify
# which transition the user means.  Only content words may anchor retrieval.
_GENERIC_TERMS = {
    "why", "what", "caused", "cause", "causes", "change", "changed",
    "changes", "changing", "switch", "switched", "switching", "transition",
    "transitioned", "replacement", "replaced", "previous", "state",
    "from", "to", "into", "about", "this", "that", "did", "does", "was",
    "were", "the", "for", "with", "when", "recorded", "explicitly",
    "because", "due", "reason", "reasons", "unknown", "whether", "know",
    "sure", "explain", "led", "leading", "made",
}
_CAUSAL_QUESTION = re.compile(
    r"\b(?:why|what\s+(?:\w+\s+){0,3}caused|"
    r"what\s+(?:was|is)\s+the\s+(?:\w+\s+){0,2}(?:cause|reason)|"
    r"(?:cause|reason)\s+(?:did|do|was|is)|"
    r"(?:was|is|were|did|does|do)\b[^?]*\bbecause\b|"
    r"(?:did|does|do)\b[^?]*\bcause\b|"
    r"what\s+(?:led|made)\b|"
    r"how\s+come\b)\b", re.I,
)
_TRANSITION_QUESTION = re.compile(
    r"\b(?:switch(?:ed|ing)?|chang(?:e|ed|es|ing)|replac(?:e|ed|es|ing)|"
    r"transition(?:ed|ing)?|mov(?:e|ed|es|ing)|relocat(?:e|ed|es|ing))\b", re.I,
)
_STATE_TYPE_TERMS = {
    "goal": "GOAL", "goals": "GOAL",
    "preference": "PREFERENCE", "preferences": "PREFERENCE", "prefer": "PREFERENCE",
    "interest": "INTEREST", "interests": "INTEREST", "interested": "INTEREST",
    "activity": "ACTIVITY", "activities": "ACTIVITY",
    "sport": "ACTIVITY", "sports": "ACTIVITY", "exercise": "ACTIVITY",
    "exercises": "ACTIVITY",
    "skill": "SKILL", "skills": "SKILL",
    "plan": "PLAN", "plans": "PLAN",
    "fact": "FACT", "facts": "FACT",
    "relationship": "RELATIONSHIP", "relationships": "RELATIONSHIP",
    "location": "LOCATION", "locations": "LOCATION",
    "city": "LOCATION", "cities": "LOCATION", "place": "LOCATION",
    "places": "LOCATION", "relocation": "LOCATION", "relocations": "LOCATION",
    "relocate": "LOCATION", "relocated": "LOCATION", "residence": "LOCATION",
    "status": "STATUS", "statuses": "STATUS",
    "life-event": "LIFE_EVENT", "life-events": "LIFE_EVENT",
    "emotion": "EMOTION", "emotions": "EMOTION",
    "project": "PROJECT", "projects": "PROJECT",
}


def _terms(text):
    return {w.casefold() for w in re.findall(r"[\w'-]+", text or "")
            if len(w) > 2 and w.casefold() not in _GENERIC_TERMS}


def requested_state_types(query):
    tokens={token.casefold() for token in re.findall(r"[\w-]+",query or "")}
    return {_STATE_TYPE_TERMS[token] for token in tokens if token in _STATE_TYPE_TERMS}


def is_transition_question(query):
    return bool(_TRANSITION_QUESTION.search(query or ""))


def transition_direction(query):
    """Return a requested source/destination side for a direct change query."""
    text=query or ""
    action=r"\b(?:switch|chang|replac|transition|mov|relocat)\w*\b"
    if re.search(r"\b(?:from|before)\b",text,re.I) and re.search(action,text,re.I):
        return "from"
    if re.search(r"\b(?:to|into)\b",text,re.I) and re.search(action,text,re.I):
        return "to"
    return None


def is_causal_question(query):
    return bool(_CAUSAL_QUESTION.search(query or ""))


def relevant_transitions(user_id, query, causal_only=None, requested_timestamp=None):
    """Return transitions anchored to query terms in an endpoint or cause.

    Causal questions include only explicit-cause edges with a recorded trigger
    or reason. Cause words can anchor an edge when the question does not repeat
    either endpoint label. A generic question with no anchor intentionally
    returns no evidence. All records are fetched through the user-scoped DB API.
    """
    instant = parse_requested_timestamp(requested_timestamp) if requested_timestamp is not None else None
    states = db.states(user_id)
    by_id = {s.state_id: s for s in states}
    terms = _terms(query)
    if not terms:
        return []
    if causal_only is None:
        causal_only = is_causal_question(query)
    types = requested_state_types(query)
    selected = []
    for edge in db.transitions(user_id):
        source, target = by_id.get(edge.from_state_id), by_id.get(edge.to_state_id)
        if source is None or target is None:
            continue
        if instant is not None:
            try:
                # Neither a future endpoint nor an unparseable endpoint can
                # be proven available at this recorded instant.
                if (parse_requested_timestamp(source.valid_from) > instant
                        or parse_requested_timestamp(target.valid_from) > instant
                        or parse_requested_timestamp(edge.timestamp) > instant):
                    continue
            except (TypeError, ValueError):
                continue
        if causal_only:
            if not (
                edge.relation_kind == "EXPLICIT_CAUSE"
                and bool((edge.trigger and edge.trigger.strip())
                         or (edge.reason and edge.reason.strip()))
            ):
                continue
        elif edge.relation_kind not in {"EXPLICIT_CAUSE", "TEMPORAL_SUCCESSION"}:
            # UNKNOWN and inferred edges are not established changes. Keep
            # direct change questions on the same grounded relations used for
            # causal and temporal validation.
            continue
        endpoint_terms = _terms(source.state_value) | _terms(target.state_value)
        cause_terms = _terms(edge.trigger or "") | _terms(edge.reason or "")
        type_match = bool(types and (
            source.state_type in types or target.state_type in types))
        if terms & (endpoint_terms | cause_terms) or type_match:
            selected.append(edge.model_dump(mode="json"))
    return selected

def build_graph(user_id, requested_timestamp=None):
    instant = parse_requested_timestamp(requested_timestamp) if requested_timestamp is not None else None
    graph=nx.DiGraph()
    states = db.states(user_id)
    state_by_id = {state.state_id: state for state in states}
    for state in states: graph.add_node(state.state_id,**state.model_dump(mode="json"))
    for edge in db.transitions(user_id):
        if instant is not None:
            try:
                source=state_by_id.get(edge.from_state_id); target=state_by_id.get(edge.to_state_id)
                if source is None or target is None:
                    continue
                if (parse_requested_timestamp(source.valid_from) > instant
                        or parse_requested_timestamp(target.valid_from) > instant
                        or parse_requested_timestamp(edge.timestamp) > instant):
                    continue
            except (TypeError, ValueError):
                continue
        if graph.has_node(edge.from_state_id) and graph.has_node(edge.to_state_id): graph.add_edge(edge.from_state_id,edge.to_state_id,**edge.model_dump(mode="json"))
    return graph

def relevant_path(user_id,query,requested_timestamp=None):
    graph=build_graph(user_id,requested_timestamp); terms=[w.casefold() for w in query.split() if len(w)>3]
    nodes=[n for n,d in graph.nodes(data=True) if any(w in d.get("state_value","").casefold() for w in terms)]
    if len(nodes)<2:return []
    try:
        path=nx.shortest_path(graph,nodes[0],nodes[-1])
        return [{"from":graph.nodes[a].get("state_value"),"to":graph.nodes[b].get("state_value"),**graph.edges[a,b]} for a,b in zip(path,path[1:])]
    except (nx.NetworkXNoPath,nx.NodeNotFound): return []
