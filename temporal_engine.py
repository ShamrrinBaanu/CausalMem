import re
from datetime import date, datetime, time, timezone

from database import db

_AS_OF = re.compile(r"\b(?:as\s+of|at|on)\s+(\d{4}-\d{2}-\d{2}(?:[Tt][0-9:.+Zz-]+)?)", re.I)
_AS_OF_MENTION = re.compile(r"\bas\s+of\b", re.I)
_MONTH = (r"January|Jan|February|Feb|March|Mar|April|Apr|May|June|Jun|"
          r"July|Jul|August|Aug|September|Sept|Sep|October|Oct|November|Nov|"
          r"December|Dec")
_MONTH_NUMBERS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_NATURAL_DATE = re.compile(
    rf"\b(?:on|in|during|as\s+of|at)\s+({_MONTH})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\s+UTC\b",
    re.I,
)
_NATURAL_DATETIME = re.compile(
    rf"\b(?:on|in|during|as\s+of|at)\s+({_MONTH})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})"
    rf"(?:\s+at)?\s+(\d{{1,2}}):(\d{{2}})(?::(\d{{2}})(?:\.(\d{{1,6}}))?)?\s+UTC\b",
    re.I,
)
_NATURAL_DATE_MENTION = re.compile(rf"\b(?:{_MONTH})\s+\d{{1,2}}(?:st|nd|rd|th)?\b", re.I)
_DATE_PREFIX = re.compile(rf"\b(?:on|in|during|as\s+of|at)\s+(?:\d{{4}}-\d{{2}}-\d{{2}}|{_MONTH}(?:\s+\d{{1,2}})?)", re.I)
_RELATIVE_PAST_DATE = re.compile(r"\b(?:yesterday|last\s+(?:week|month|year)|\d+\s+(?:days?|weeks?|months?|years?)\s+ago)\b", re.I)
_PREDECESSOR = re.compile(
    r"\b(?:before\s+(?:i|we)\s+(?:chang(?:e|ed|ing)|switch(?:ed|ing)?|replac(?:e|ed|ing))|"
    r"before\s+(?:the\s+)?(?:chang(?:e|ing)|switch(?:ing)?|replacement)|"
    r"prior\s+to\s+(?:the\s+)?(?:chang(?:e|ing)|switch(?:ing)?|replacement)|"
    r"previous\s+(?:state|goal|preference|interest|emotion|project|location|residence)|"
    r"(?:what|which|where)\s+did\s+(?:i|we)\b[^?]*\b(?:prefer\w*|preference\w*|"
    r"goal\w*|interest\w*|activit(?:y|ies)|state\w*|status\w*|emotion\w*|project\w*|"
    r"location\w*|live|lives|living|lived|reside\w*|move\w*)\b|"
    r"\b(?:complete|finish|achiev)\w*\b[^?]*\b(?:before|previously)\b|"
    r"\b(?:goal\w*|preference\w*|interest\w*|activit(?:y|ies)|state\w*|"
    r"status\w*|emotion\w*|project\w*|location\w*|residence)\b[^?]*"
    r"\b(?:did|have|had)\b[^?]*\bbefore\b|"
    r"\b(?:prefer\w*|preference\w*|goal\w*|interest\w*|activit(?:y|ies)|"
    r"state\w*|status\w*|emotion\w*|project\w*|location\w*|live|lives|living|"
    r"lived|reside\w*|move\w*|complete\w*)\b[^?]*\bpreviously\b)\b",
    re.I,
)
_PREDECESSOR_TYPED = re.compile(
    r"\b(?:prefer\w*|preference\w*|goal\w*|interest\w*|activit(?:y|ies)|"
    r"sport\w*|exercise\w*|place\w*|city|cities|relocat\w*|"
    r"state\w*|status\w*|emotion\w*|project\w*|location\w*|live|lives|living|"
    r"lived|reside\w*|move\w*|complete\w*|finish\w*|achiev\w*)\b",
    re.I,
)
_PREDECESSOR_BEFORE_VALUE = re.compile(
    rf"\bbefore\s+(?:(?:my|the)\s+)?(?!(?:i|we|a|an|this|that|then|last|next|"
    rf"yesterday|today|tomorrow|{_MONTH}|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d)\b)"
    r"(?P<anchor>[A-Za-z][\w#+.-]{1,})\b",
    re.I,
)
_PREDECESSOR_STOP = {
    "before", "changed", "change", "changes", "changing", "switched",
    "switching", "replaced", "replacing", "replacement", "transition",
    "transitioned", "prior", "previous", "previously", "state", "goal", "goals",
    "preference", "preferences", "interest", "interests", "what", "which",
    "where", "when", "how", "was", "were", "have", "my", "i", "we", "it",
    "the", "a", "an", "did", "do", "then", "existed", "exist", "emotion",
    "emotions", "project", "projects",
}
_STATE_TYPE_TERMS = {
    "goal": "GOAL", "goals": "GOAL", "preference": "PREFERENCE",
    "preferences": "PREFERENCE", "prefer": "PREFERENCE",
    "interest": "INTEREST", "interests": "INTEREST",
    "activity": "ACTIVITY", "activities": "ACTIVITY",
    "sport": "ACTIVITY", "sports": "ACTIVITY", "exercise": "ACTIVITY",
    "exercises": "ACTIVITY",
    "city": "LOCATION", "cities": "LOCATION", "place": "LOCATION",
    "places": "LOCATION", "relocation": "LOCATION", "relocations": "LOCATION",
    "relocate": "LOCATION", "relocated": "LOCATION",
    "location": "LOCATION", "locations": "LOCATION", "live": "LOCATION",
    "lives": "LOCATION", "living": "LOCATION", "lived": "LOCATION",
    "reside": "LOCATION", "resides": "LOCATION", "residence": "LOCATION",
    "move": "LOCATION", "moved": "LOCATION", "moves": "LOCATION",
    "moving": "LOCATION",
    "complete": "LIFE_EVENT", "completed": "LIFE_EVENT",
    "completion": "LIFE_EVENT", "finish": "LIFE_EVENT",
    "finished": "LIFE_EVENT", "achieve": "LIFE_EVENT",
    "achieved": "LIFE_EVENT", "achievement": "LIFE_EVENT",
    "status": "STATUS", "emotion": "EMOTION", "emotions": "EMOTION",
    "project": "PROJECT", "projects": "PROJECT",
}


def parse_requested_timestamp(value):
    """Parse an aware ISO-8601 instant and normalize it to UTC.

    Naive and malformed values are rejected; unknown timezone context is never
    guessed. This is shared by state and transition as-of filtering.
    """
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.strip().replace("Z", "+00:00").replace("z", "+00:00"))
        except ValueError as exc:
            raise ValueError("requested_timestamp must be a valid ISO-8601 timestamp") from exc
    if not isinstance(value, datetime):
        raise TypeError("requested_timestamp must be an aware datetime or ISO-8601 string")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("requested_timestamp must include a timezone")
    return value.astimezone(timezone.utc)


def _month_number(name):
    month = _MONTH_NUMBERS.get(str(name or "")[:3].casefold())
    if month is None:
        raise ValueError("query contains an invalid month")
    return month


def _parse_natural_datetime(match):
    month_name, day, year, hour, minute, second, fraction = match.groups()
    try:
        microsecond = int((fraction or "").ljust(6, "0") or 0)
        return datetime(int(year), _month_number(month_name), int(day),
                        int(hour), int(minute), int(second or 0), microsecond,
                        tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError("query contains an invalid date or time") from exc


def resolve_requested_timestamp(query, requested_timestamp=None):
    """Resolve an explicit instant or a supported natural UTC date/time.

    A natural date such as ``on October 1, 2025 UTC`` means the final
    representable instant of that UTC calendar day. Adding an explicit UTC time,
    such as ``October 1, 2025 at 19:50 UTC``, means that exact instant. Dates
    without a year or explicit UTC marker are rejected rather than guessed. A
    supplied ``requested_timestamp`` is authoritative and is validated without
    parsing or overriding it from the query text.
    """
    if requested_timestamp is not None:
        return parse_requested_timestamp(requested_timestamp)

    found = _AS_OF.search(query or "")
    natural_datetime = _NATURAL_DATETIME.search(query or "")
    natural = _NATURAL_DATE.search(query or "")
    parsed_query = None
    if natural_datetime:
        parsed_query = _parse_natural_datetime(natural_datetime)
    elif natural:
        month_name, day, year = natural.groups()
        try:
            calendar_day = date(int(year), _month_number(month_name), int(day))
        except ValueError as exc:
            raise ValueError("query contains an invalid date") from exc
        parsed_query = datetime.combine(calendar_day, time.max, tzinfo=timezone.utc)
    if found:
        token = found.group(1).rstrip(".,;)")
        iso_query_time = parse_requested_timestamp(token)
        if parsed_query is not None and parsed_query != iso_query_time:
            raise ValueError("query contains conflicting timestamps")
        parsed_query = iso_query_time
    date_requested = bool(_AS_OF_MENTION.search(query or "") or _DATE_PREFIX.search(query or "")
                          or _NATURAL_DATE_MENTION.search(query or "")
                          or _RELATIVE_PAST_DATE.search(query or ""))
    if date_requested and parsed_query is None:
        raise ValueError(
            "Use a timezone-aware ISO-8601 timestamp or a date like 'October 1, 2025 UTC'"
        )
    return parsed_query


def is_predecessor_query(query):
    text = str(query or "")
    if _PREDECESSOR.search(text):
        return True
    # A state-typed question can name its successor directly ("preference
    # before Python"); a date-before question remains an as-of lookup.
    return bool(_PREDECESSOR_TYPED.search(text)
                and _PREDECESSOR_BEFORE_VALUE.search(text))


def predecessor_state(user_id, query):
    """Resolve one predecessor only when the user's recorded transition is unique.

    Returns ``(state, transition, ambiguous)``. Endpoints and transitions are
    loaded through user-scoped database methods; missing or inconsistent links
    are ignored. No candidate means the caller should return empty evidence.
    """
    states = db.states(user_id)
    by_id = {state.state_id: state for state in states if state.user_id == user_id}
    tokens = {token.casefold() for token in re.findall(r"[\w'-]+", query or "")}
    types = {_STATE_TYPE_TERMS[token] for token in tokens if token in _STATE_TYPE_TERMS}
    anchors = {token for token in tokens if len(token) > 2 and token not in _PREDECESSOR_STOP
               and token not in _STATE_TYPE_TERMS}
    candidates = []
    fallback_candidates = []
    for edge in db.transitions(user_id):
        if edge.user_id != user_id:
            continue
        source, destination = by_id.get(edge.from_state_id), by_id.get(edge.to_state_id)
        if source is None or destination is None:
            continue
        if edge.relation_kind not in {"TEMPORAL_SUCCESSION", "EXPLICIT_CAUSE"}:
            continue
        boundary_link = False
        try:
            transition_time = parse_requested_timestamp(edge.timestamp)
            boundary_link = (
                parse_requested_timestamp(source.valid_until) == transition_time
                and parse_requested_timestamp(destination.valid_from) == transition_time
            )
        except (TypeError, ValueError):
            boundary_link = False
        # At least one endpoint must belong to the requested type. Existing
        # explicit-cause changes legitimately cross types (INTEREST -> ACTIVITY).
        if types and (source.state_type not in types and destination.state_type not in types):
            continue
        if anchors:
            endpoint_text = (source.state_value + " " + destination.state_value).casefold()
            if not any(anchor in endpoint_text for anchor in anchors):
                continue
        if boundary_link:
            candidates.append((source, edge))
        else:
            fallback_candidates.append((source, edge))
    if len(candidates) == 1:
        state, edge = candidates[0]
        return state, edge, False
    if not candidates and len(fallback_candidates) == 1:
        # Older records may have an explicit edge without exact adjacent
        # interval endpoints. A unique user-scoped explicit change is still
        # the recorded predecessor; competing loose candidates remain
        # ambiguous and fail closed.
        state, edge = fallback_candidates[0]
        return state, edge, False
    ambiguous = len(candidates) > 1 or (
        not candidates and len(fallback_candidates) > 1)
    return None, None, ambiguous

def timeline(user_id,query="",requested_timestamp=None,include_quarantined=False):
    if requested_timestamp is not None:
        return states_at(user_id, requested_timestamp, include_quarantined)
    states=db.states(user_id,include_quarantined=include_quarantined)
    if query:
        terms={w.casefold() for w in query.split() if len(w)>3}
        matches=[s for s in states if any(w in s.state_value.casefold() for w in terms)]
        if matches:
            # A query that matches one side of a recorded transition may be
            # asking about its predecessor/successor. Include only that edge's
            # endpoints, using this user's already-scoped state/transition rows.
            matched_ids={s.state_id for s in matches}
            related_ids=set(matched_ids)
            for edge in db.transitions(user_id,include_quarantined=include_quarantined):
                if edge.from_state_id in matched_ids or edge.to_state_id in matched_ids:
                    related_ids.update((edge.from_state_id,edge.to_state_id))
            return [s for s in states if s.state_id in related_ids]
    return states
def current_states(user_id,include_quarantined=False):
    """Return states active and temporally valid at the current instant.

    A stale ``ACTIVE`` label is not sufficient: the half-open interval is the
    authority, so an elapsed ``valid_until`` excludes the row.
    """
    instant=datetime.now(timezone.utc)
    result=[]
    for state in db.states(user_id,include_quarantined=include_quarantined):
        if state.status!="ACTIVE":
            continue
        try:
            valid=parse_requested_timestamp(state.valid_from)<=instant and (
                state.valid_until is None or instant<parse_requested_timestamp(state.valid_until))
        except (TypeError,ValueError):
            valid=False
        if valid:
            result.append(state)
    return result

def states_at(user_id,requested_timestamp,include_quarantined=False):
    """Return this user's states valid at an aware recorded timestamp."""
    return db.states_at(user_id,parse_requested_timestamp(requested_timestamp),
                        include_quarantined=include_quarantined)
