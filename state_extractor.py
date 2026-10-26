
"""Extract and persist grounded user states and explicit cause transitions."""
import re
from datetime import datetime, timedelta, timezone

from database import db
from llm_service import llm
from models import Extraction, State, Transition


STATE_TYPES = {
    "FACT", "PREFERENCE", "ACTIVITY", "GOAL", "PLAN", "SKILL",
    "INTEREST", "RELATIONSHIP", "LOCATION", "LIFE_EVENT", "STATUS",
    "EMOTION", "PROJECT", "OTHER",
}

CHANGE_VERBS = r"switched|changed|moved|shifted|transitioned"

CAUSE_RE = re.compile(
    r"\b(?:due to|because of|because)\s+(.+?)"
    r"(?=\s+(?:i|we|they)\s+(?:" + CHANGE_VERBS + r")\b|[,;.!?]|$)",
    re.IGNORECASE,
)

TARGET_RE = re.compile(
    r"\b(?:" + CHANGE_VERBS + r")\s+"
    r"(?:from\s+(?P<source>.+?)\s+)?to\s+(?P<target>[^,;.!?]+)",
    re.IGNORECASE,
)

NON_ASSERTION_RE = re.compile(
    r"^(?:why|how|what|when|where|who|which|whose|whom|"
    r"do|does|did|is|are|was|were|have|has|had|can|could|would|should|"
    r"please|tell me|explain|remind me|show me|help me)\b",
    re.IGNORECASE,
)

_QUESTION_START_RE = re.compile(
    r"^(?:why|how|what|when|where|who|which|whose|whom|"
    r"do|does|did|is|are|was|were|have|has|had|can|could|would|should|"
    r"will|am)\b",
    re.IGNORECASE,
)

_QUERY_REQUEST_RE = re.compile(
    r"^(?:(?:please|kindly)\s+)?"
    r"(?:tell|explain|remind|show|help|list|describe|summari[sz]e|"
    r"compare|find|search|recall|clarify)\b",
    re.IGNORECASE,
)

_META_CLAUSE_RE = re.compile(
    r"^(?:i\s+(?:don't|do\s+not)\s+(?:know|remember)|"
    r"i(?:'m|\s+am)\s+(?:not\s+sure|unsure|curious|wondering)|"
    r"i\s+(?:wonder|wondered|was\s+wondering|am\s+wondering|"
    r"ask|asked|am\s+asking|was\s+asking)|"
    r"i\s+(?:want|need|would\s+like)\s+to\s+know|"
    r"i\s+(?:have|had)\s+a\s+question|i'd\s+like\s+to\s+ask|"
    r"let\s+me\s+ask|question|not\s+sure|unsure|no\s+idea|curious)\b",
    re.IGNORECASE,
)

_ASSISTANT_FRAMING_RE = re.compile(
    r"^(?:you|your|assistant|causalmem|"
    r"based\s+on\s+(?:your|the)\s+memory|"
    r"according\s+to\s+your\s+memory)\b",
    re.IGNORECASE,
)

_ASSERTION_PREFIX_RE = re.compile(
    r"^(?:(?:please|kindly)\s+)?"
    r"(?:(?:remember|note|keep\s+in\s+mind|for\s+the\s+record|"
    r"for\s+context|just\s+so\s+you\s+know|fyi|by\s+the\s+way|btw|"
    r"you\s+know|"
    r"i\s+(?:want|would\s+like)\s+to\s+(?:tell\s+you|let\s+you\s+know)|"
    r"i\s+need\s+to\s+tell\s+you)"
    r"(?:\s+that)?(?:[,:]\s*|\s+))+",
    re.IGNORECASE,
)

_FIRST_PERSON_RE = re.compile(
    r"\b(?:i|i'm|i\u2019m|i've|i\u2019ve|i'd|i\u2019d|"
    r"my|we|we're|we\u2019re|we've|we\u2019ve|our)\b",
    re.IGNORECASE,
)

_DECLARATIVE_SUBJECT_RE = re.compile(
    r"^(?:i|i'm|i\u2019m|i've|i\u2019ve|i'd|i\u2019d|"
    r"my|we|we're|we\u2019re|we've|we\u2019ve|our)\b",
    re.IGNORECASE,
)

_CLAUSE_SPLIT_RE = re.compile(
    r"(?<=[.!?;])\s+|\n+|,\s+(?=(?:and|but|so|because|although|while)\b)",
    re.IGNORECASE,
)
_CLAUSE_TRIM = "\"' \t\r\n\u2018\u2019\u201c\u201d"

_EXTERNAL_REPORT_RE = re.compile(
    r"\b(?:said|says|told|tells|asked|asks|wrote|writes|reported|reports|"
    r"claimed|claims|according\s+to|quoted|quoting|"
    r"(?:read|heard|learned|learnt|saw)\s+that)\b",
    re.IGNORECASE,
)
_HYPOTHETICAL_RE = re.compile(
    r"\b(?:if|unless|suppose|supposing|imagine|what\s+if|in\s+case|"
    r"hypothetical|hypothetically)\b",
    re.IGNORECASE,
)
_HEDGE_RE = re.compile(
    r"\b(?:maybe|perhaps|probably|possibly|might|may|probably|"
    r"not\s+sure|unsure|i\s+think|i\s+guess|i\s+believe|seems?|"
    r"apparently|likely|hopefully|i\s+suppose)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(?:not|never|no\s+longer|nothing|nobody|none|neither|nor|"
    r"without|anymore)\b|n['\u2019]t\b",
    re.IGNORECASE,
)
_PLAN_RE = re.compile(
    r"\b(?:plan|plans|planned|planning|intend|intends|intended|intending|"
    r"will|going\s+to|gonna|want\s+to|would\s+like\s+to|hope\s+to|"
    r"aim\s+to|expect\s+to|tomorrow|someday|one\s+day)\b",
    re.IGNORECASE,
)
_PAST_HABIT_RE = re.compile(
    r"\b(?:used\s+to|formerly|previously|once|in\s+the\s+past|"
    r"when\s+i\s+was|back\s+then)\b",
    re.IGNORECASE,
)
_FUTURE_TIME_RE = re.compile(
    r"\b(?:next\s+(?:week|month|year)|tomorrow|soon|later|someday|"
    r"one\s+day|in\s+\d+\s+(?:days?|weeks?|months?|years?))\b",
    re.IGNORECASE,
)
_PROGRESSIVE_RE = re.compile(r"\b(?:am|is|are)\b[^.!?;]*\b\w+ing\b", re.IGNORECASE)
_RETRACTION_RE = re.compile(
    r"\b(?:no\s+longer|not\s+anymore|anymore|stopped|quit|gave\s+up|"
    r"given\s+up|forget|forgot|never)\b|n['\u2019]t\b",
    re.IGNORECASE,
)
_GOAL_UPDATE_RE = re.compile(
    r"\b(?:my\s+(?:(?:new|current|primary|main)\s+)?goal\s+is|"
    r"i(?:'m|\s+am)\s+(?:now|currently)?\s*(?:trying|planning|working)\s+to|"
    r"i\s+now\s+aim\s+to)\b",
    re.IGNORECASE,
)
_GOAL_STATEMENT_RE = re.compile(
    r"\b(?:my\s+(?:(?:new|current|primary|main)\s+)?goal\s+is\s+"
    r"(?:to\s+)?|"
    r"i(?:'m|\s+am)\s+(?:now|currently)?\s*(?:trying|planning|working)\s+to\s+|"
    r"i\s+now\s+aim\s+to\s+)"
    r"(?P<value>[^,.;!?]+)",
    re.IGNORECASE,
)
_COMPLETION_RE = re.compile(
    r"\bi\s+(?:(?:have|had|just|recently|already)\s+)*"
    r"(?:completed|finished|achieved|done\s+with)\s+(?P<value>[^,.;!?]+)",
    re.IGNORECASE,
)
_LOCATION_STATEMENT_RE = re.compile(
    r"\b(?:i\s+(?:(?:now|currently|recently|just)\s+)?"
    r"(?:live|reside|am\s+based|have\s+moved|moved|relocated)\s+(?:in|to)|"
    r"i(?:'m|\s+am)\s+from)\s+(?P<target>[^,.;!?\n]+)",
    re.IGNORECASE,
)
_LOCATION_MOVE_FROM_RE = re.compile(
    r"\bi\s+(?:(?:have|had)\s+)?(?:moved|relocated|shifted)\s+"
    r"from\s+(?P<source>[^,.;!?]+?)\s+to\s+(?P<target>[^,.;!?]+)",
    re.IGNORECASE,
)
_LOCATION_MOVE_TO_RE = re.compile(
    r"\bi\s+(?:(?:have|had)\s+)?(?:moved|relocated|shifted)\s+"
    r"to\s+(?P<target>[^,.;!?]+?)\s+from\s+(?P<source>[^,.;!?]+)",
    re.IGNORECASE,
)
_FAVORITE_STATEMENT_RE = re.compile(
    r"\bmy\s+favou?rite\s+(?P<domain>[a-z][\w -]{0,40}?)\s+is\s+"
    r"(?P<value>[^,.;!?]+)",
    re.IGNORECASE,
)
_NAMED_PREFERENCE_RE = re.compile(
    r"\bmy\s+(?:(?:desired|new|current|primary|main)\s+)*"
    r"(?P<domain>[a-z][\w -]{0,40}?)\s+preference\s+is\s+"
    r"(?P<value>[^,.;!?]+)",
    re.IGNORECASE,
)
_PREFERENCE_STATEMENT_RE = re.compile(
    r"\bi\s+(?:(?:now|currently|really|still|also)\s+)?prefer\s+"
    r"(?P<value>.+?)"
    r"(?:\s+(?:over|instead\s+of|to)\s+(?P<old>[^,.;!?]+?))?"
    r"(?:\s+for\s+(?P<domain>[^,.;!?]+?))?"
    r"\s*(?:[.!?]|$)",
    re.IGNORECASE,
)
_COMPARATIVE_PREFERENCE_RE = re.compile(
    r"\bi\s+(?:(?:now|currently|really|still|also)\s+)?"
    r"(?:prefer|like|love)\s+"
    r"(?P<left>.+?)\s+"
    r"(?P<comparator>rather\s+than|instead\s+of|more\s+than|less\s+than|over|to)\s+"
    r"(?P<right>[^,.;!?]+?)"
    r"(?:\s+for\s+(?P<domain>[^,.;!?]+?))?"
    r"\s*(?:[.!?]|$)",
    re.IGNORECASE,
)

_PLAN_STATE_TYPES = {"GOAL", "PLAN"}
_EXCLUSIVE_UPDATE_TYPES = {"LOCATION"}


def _clause_text(clause):
    return _ASSERTION_PREFIX_RE.sub("", clause.strip(_CLAUSE_TRIM)).strip()


def _clause_mode(text):
    """Classify a clause without relying on question-specific vocabulary."""
    if not text or text.endswith("?"):
        return "question"
    if (_QUESTION_START_RE.match(text) or _QUERY_REQUEST_RE.match(text)
            or _META_CLAUSE_RE.match(text) or _ASSISTANT_FRAMING_RE.match(text)):
        return "meta"
    if '"' in text or "\u201c" in text or "\u201d" in text:
        return "external"
    if _EXTERNAL_REPORT_RE.search(text):
        return "external"
    if _HYPOTHETICAL_RE.search(text):
        return "hypothetical"
    if _HEDGE_RE.search(text):
        return "uncertain"
    if _NEGATION_RE.search(text):
        return "negative"
    if _PAST_HABIT_RE.search(text):
        return "historical"
    if _PLAN_RE.search(text) or (_FUTURE_TIME_RE.search(text)
                                 and _PROGRESSIVE_RE.search(text)):
        return "plan"
    return "assertion"


def _clause_records(message):
    records = []
    for clause in _CLAUSE_SPLIT_RE.split(str(message or "").strip()):
        text = _clause_text(clause)
        if text and _FIRST_PERSON_RE.search(text):
            records.append((text, _clause_mode(text)))
    return records


def _durable_clause_texts(message, include_plans=True):
    allowed = {"assertion"} | ({"plan"} if include_plans else set())
    return [text for text, mode in _clause_records(message) if mode in allowed]


def is_ingestable_message(message):
    """Return whether the message asserts a first-person fact or event.

    Questions, recall/explanation requests, uncertainty, hypotheticals, negated
    statements, and assistant-style narration do not become positive durable
    facts even when a language model extracts plausible-looking values from
    them. Explicit retractions are handled separately during reconciliation.
    """
    if not isinstance(message, str) or not message.strip():
        return False
    return bool([text for text, mode in _clause_records(message)
                 if mode in {"assertion", "plan", "negative"}])


def _is_assertion_value(value):
    """Reject question/meta/assistant text used as a state value."""
    text = str(value or "").strip(_CLAUSE_TRIM)
    if not text or "?" in text:
        return False
    if _clause_mode(text) in {"question", "meta", "external", "hypothetical",
                              "uncertain", "negative", "historical"}:
        return False
    return True


def _state_key(state_type, value):
    return (str(state_type or "OTHER").upper(),
            re.sub(r"\s+", " ", str(value or "").strip()).casefold())


def _value_mode(value, message):
    """Resolve an extracted value to the strongest clause containing it."""
    folded = str(value or "").casefold().strip()
    tokens = [token for token in re.findall(r"[\w'-]+", folded) if len(token) > 3]
    matches = []
    for text, mode in _clause_records(message):
        clause = text.casefold()
        if folded and folded in clause:
            matches.append((len(folded), mode))
        elif tokens and any(token in clause for token in tokens):
            matches.append((sum(token in clause for token in tokens), mode))
    if not matches:
        return "assertion"
    return max(matches, key=lambda item: item[0])[1]


def _retraction_targets(message, active_states):
    """Find current states explicitly retracted or negated by this message.

    Only the negative clause is searched. A following positive clause such as
    "but I love cycling" must not close the newly preferred activity.
    """
    phrases = []
    for text, mode in _clause_records(message):
        if mode != "negative":
            continue
        for part in re.split(r"\b(?:and|but)\b", text, maxsplit=1, flags=re.IGNORECASE):
            if _RETRACTION_RE.search(part):
                phrases.append(part.casefold())
    if not phrases:
        return []
    lowered = " ".join(phrases)
    generic = {"that", "this", "with", "from", "love", "like", "enjoy",
               "prefer", "want", "learn", "really", "very"}
    targets = []
    for state in active_states:
        value = str(state.state_value or "").casefold().strip()
        tokens = {token for token in re.findall(r"[\w'-]+", value)
                  if len(token) > 3 and token not in generic}
        if value and value in lowered:
            targets.append(state)
        elif tokens and any(token in lowered for token in tokens):
            targets.append(state)
    return targets

INSTEAD_RE = re.compile(r"\binstead\b", re.IGNORECASE)
GOAL_INSTEAD_RE = re.compile(
    r"\b(?:i\s+want\s+to|my\s+(?:new\s+)?goal\s+is\s+(?:to\s+)?)"
    r"(?P<target>[^.!?;]+?)\s+instead\b",
    re.IGNORECASE,
)


def _recorded_transition_time(destination, source):
    """Keep a replacement interval nonempty when recorded timestamps collide."""
    timestamp = destination.valid_from
    if timestamp <= source.valid_from:
        # ISO timestamps are stored at microsecond precision, so advance by
        # exactly one representable unit only when the clock does not advance.
        timestamp = source.valid_from + timedelta(microseconds=1)
        destination.valid_from = timestamp
    return timestamp


def _goal_replacement(message, raw_transitions, active_states, items):
    """Return a grounded ordinary goal replacement, never a cause inference."""
    if not INSTEAD_RE.search(message):
        return None

    active_goals=[state for state in active_states if state.state_type=="GOAL"]
    if len(active_goals)!=1:
        return None

    source=active_goals[0]
    new_goals=[item for item in items if item["state_type"]=="GOAL"]
    explicit_target=GOAL_INSTEAD_RE.search(message)
    if len(new_goals)==1:
        target=new_goals[0]["state_value"].strip()
    elif not new_goals and explicit_target:
        # A clear first-person "want/goal ... instead" statement is itself a
        # grounded succession signal. Small local models sometimes extract
        # only the object (e.g. "piano") as INTEREST and omit the edge.
        target=explicit_target.group("target").strip()
        target_terms={term.casefold() for term in re.findall(r"[\w'-]+",target)}
        items[:]=[
            item for item in items
            if item.get("state_type")=="GOAL"
            or not ({term.casefold() for term in re.findall(
                r"[\w'-]+",str(item.get("state_value", "")))} <= target_terms)
        ]
        items.append({"state_type":"GOAL","state_value":target,"confidence":0.8})
    else:
        return None
    message_folded=message.casefold()
    # Require the destination label, not merely a generic shared word such as
    # "learn", to be stated in the replacement message.
    if target.casefold() not in message_folded:
        return None

    for candidate in raw_transitions:
        from_value=str(candidate.get("from_state_value","")).strip().casefold()
        to_value=str(candidate.get("to_state_value","")).strip().casefold()
        if (from_value and to_value
                and (from_value==source.state_value.casefold()
                     or from_value in source.state_value.casefold()
                     or source.state_value.casefold() in from_value)
                and (to_value==target.casefold()
                     or to_value in target.casefold()
                     or target.casefold() in to_value)):
            return {"source":source,"target":target}
    # Do not make chronology itself causal: this path is only reachable for a
    # clearly worded, grounded replacement construction containing “instead”.
    if explicit_target and explicit_target.group("target").strip().casefold()==target.casefold():
        return {"source":source,"target":target}
    return None


def _is_supported_utterance(message):
    """Permit fallback only for a simple first-person assertion."""
    text = _clause_text(str(message or ""))
    if not text or NON_ASSERTION_RE.match(text):
        return False
    return bool(_DECLARATIVE_SUBJECT_RE.match(text)
                and _clause_mode(text) == "assertion")


def _goal_update(message, active_states, items):
    """Detect an explicit goal replacement, distinct from a causal change."""
    if INSTEAD_RE.search(message or "") or not _GOAL_UPDATE_RE.search(message or ""):
        return None
    active_goals = [state for state in active_states if state.state_type == "GOAL"]
    new_goals = [item for item in items if item.get("state_type") == "GOAL"]
    if len(active_goals) != 1 or len(new_goals) != 1:
        return None
    source, target = active_goals[0], str(new_goals[0].get("state_value", "")).strip()
    if not target or target.casefold() == source.state_value.casefold():
        return None
    if target.casefold() not in str(message or "").casefold():
        return None
    return {"source": source, "target": target}


def _clean_state_phrase(value):
    value = re.split(r"\b(?:because|since|due\s+to)\b", str(value or ""),
                     maxsplit=1, flags=re.IGNORECASE)[0]
    value = re.sub(r"\b(?:now|currently|these\s+days)\b", "", value,
                   flags=re.IGNORECASE)
    return value.strip(_CLAUSE_TRIM + " ,")


def _location_move(message):
    """Parse an explicit move that names both origin and destination."""
    match = (_LOCATION_MOVE_FROM_RE.search(message or "")
             or _LOCATION_MOVE_TO_RE.search(message or ""))
    if match is None:
        return None
    source = _clean_state_phrase(match.group("source"))
    target = _clean_state_phrase(match.group("target"))
    if not source or not target:
        return None
    return {"source_text": source, "target": target}


def _state_text_matches(state_value, text):
    left = _normalized_preference_label(state_value)
    right = _normalized_preference_label(text)
    if not left or not right:
        return False
    return left == right or left in right or right in left


def _location_update(message, active_states, items):
    """Treat a stated residence or move as replacing one current residence."""
    move = _location_move(message)
    match = _LOCATION_STATEMENT_RE.search(message or "")
    if move is not None:
        target = move["target"]
    elif match is not None:
        target = re.split(r"\b(?:and|but|from)\b", match.group("target"),
                          maxsplit=1, flags=re.IGNORECASE)[0]
        target = re.sub(r"\b(?:now|currently|these\s+days)\b", "", target,
                        flags=re.IGNORECASE).strip(" ,.;!")
    else:
        return None
    if not target:
        return None

    # Small extractors often label the object FACT/OTHER. The explicit
    # residence construction supplies the type without inventing a value.
    matching = next((item for item in items
                     if target.casefold() in item["state_value"].casefold()
                     or item["state_value"].casefold() in target.casefold()), None)
    if matching is None:
        items.append({"state_type": "LOCATION", "state_value": target,
                      "confidence": 0.75, "assertion_mode": "assertion"})
    else:
        matching["state_type"] = "LOCATION"
        matching["state_value"] = target

    active_locations = [state for state in active_states
                        if state.state_type in _EXCLUSIVE_UPDATE_TYPES]
    if move is not None:
        source = next((state for state in active_locations
                       if _state_text_matches(state.state_value,
                                              move["source_text"])), None)
        if source is None:
            return None
    else:
        if len(active_locations) != 1:
            return None
        source = active_locations[0]
    if target.casefold() == source.state_value.strip().casefold():
        return None
    return {"source": source, "target": target}


def _clean_goal_value(value):
    value = re.split(r"\b(?:because|since|so\s+that)\b", str(value or ""),
                     maxsplit=1, flags=re.IGNORECASE)[0]
    value = re.sub(r"\s+\binstead\b.*$", "", value, flags=re.IGNORECASE)
    return value.strip(_CLAUSE_TRIM + " ,")


def _goal_statement(message):
    match = _GOAL_STATEMENT_RE.search(message or "")
    if match is None:
        return None
    value = _clean_goal_value(match.group("value"))
    return value or None


_GOAL_GENERIC_TERMS = {
    "a", "an", "the", "my", "our", "to", "goal", "goals", "current",
    "new", "primary", "main", "complete", "completed", "finish", "finished",
    "achieve", "achieved", "done",
}


def _goal_label_matches(left, right):
    left_terms = {term for term in re.findall(r"[\w'-]+", str(left or "").casefold())
                  if len(term) > 1 and term not in _GOAL_GENERIC_TERMS}
    right_terms = {term for term in re.findall(r"[\w'-]+", str(right or "").casefold())
                   if len(term) > 1 and term not in _GOAL_GENERIC_TERMS}
    if not left_terms or not right_terms:
        return False
    return left_terms <= right_terms or right_terms <= left_terms


def _completion_statement(message):
    match = _COMPLETION_RE.search(message or "")
    if match is None:
        return None
    value = re.split(r"\b(?:and|but)\b", match.group("value"), maxsplit=1,
                     flags=re.IGNORECASE)[0]
    value = _clean_state_phrase(value)
    if not value:
        return None
    return f"completed {value}"


def _completion_update(message, active_states):
    completion = _completion_statement(message)
    if completion is None:
        return None
    source = next((state for state in active_states
                   if state.state_type == "GOAL"
                   and _goal_label_matches(completion, state.state_value)), None)
    if source is None:
        return None
    return {"source": source, "completion": completion}


def _normalized_preference_label(value):
    """Normalize a preference label for replacement matching only."""
    text = re.sub(r"\s+", " ", str(value or "").strip(_CLAUSE_TRIM)).casefold()
    return text.strip(" .,;:!?")


def _preference_key(domain):
    normalized = _normalized_preference_label(domain)
    if not normalized:
        return None
    normalized = re.sub(r"^(?:the|my)\s+", "", normalized)
    if re.fullmatch(r"(?:programming(?:\s+language)?|coding|language)", normalized):
        normalized = "programming language"
    return f"favorite:{normalized}"


def _clean_preference_label(value):
    value = re.split(r"\b(?:because|since)\b", str(value or ""),
                     maxsplit=1, flags=re.IGNORECASE)[0]
    value = re.sub(r"\b(?:now|currently|these\s+days)\b", "", value,
                   flags=re.IGNORECASE)
    return value.strip(_CLAUSE_TRIM + " ,")


def _preference_domain(match):
    domain = re.split(r"\b(?:and|but|because|since)\b",
                      match.group("domain") or "", maxsplit=1,
                      flags=re.IGNORECASE)[0]
    return domain.strip(_CLAUSE_TRIM + " ,")


def _preference_statement(message):
    """Parse explicit favorite/preference constructions without inventing one."""
    clauses = _durable_clause_texts(message) or []
    fallback = None
    for clause in clauses:
        match = _FAVORITE_STATEMENT_RE.search(clause)
        if match:
            value = _clean_preference_label(match.group("value"))
            domain = match.group("domain").strip(_CLAUSE_TRIM + " ,")
            if value and domain:
                return {"value": value, "old": None, "domain": domain,
                        "key": _preference_key(domain), "replacement": True}

        match = _NAMED_PREFERENCE_RE.search(clause)
        if match:
            value = _clean_preference_label(match.group("value"))
            domain = match.group("domain").strip(_CLAUSE_TRIM + " ,")
            if value and domain:
                return {"value": value, "old": None, "domain": domain,
                        "key": _preference_key(domain), "replacement": True}

        comparative = _COMPARATIVE_PREFERENCE_RE.search(clause)
        if comparative:
            left = _clean_preference_label(comparative.group("left"))
            right = _clean_preference_label(comparative.group("right"))
            comparator = re.sub(r"\s+", " ",
                                comparative.group("comparator").casefold())
            value, old = ((right, left) if comparator == "less than"
                          else (left, right))
            domain = _preference_domain(comparative)
            if value:
                return {"value": value, "old": old or None,
                        "domain": domain or None,
                        "key": _preference_key(domain) if domain else None,
                        "replacement": True}

        match = _PREFERENCE_STATEMENT_RE.search(clause)
        if not match:
            continue
        value = _clean_preference_label(match.group("value"))
        old = _clean_preference_label(match.group("old"))
        domain = _preference_domain(match)
        if not value:
            continue
        replacement = bool(old or domain or re.search(
            r"\b(?:now|currently|instead)\b", clause, re.IGNORECASE))
        statement = {"value": value, "old": old or None,
                     "domain": domain or None,
                     "key": _preference_key(domain) if domain else None,
                     "replacement": replacement}
        if replacement:
            return statement
        fallback = fallback or statement
    return fallback


def _preference_key_for_state(state):
    metadata_key = (state.metadata or {}).get("preference_key")
    if isinstance(metadata_key, str) and metadata_key.strip():
        return metadata_key.strip().casefold()
    favorite = _FAVORITE_STATEMENT_RE.search(str(state.state_value or ""))
    if favorite:
        return _preference_key(favorite.group("domain"))
    preference = _PREFERENCE_STATEMENT_RE.search(str(state.state_value or ""))
    if preference and preference.group("domain"):
        return _preference_key(preference.group("domain"))
    return None


def _preference_label_matches(label, value):
    left = _normalized_preference_label(label)
    right = _normalized_preference_label(value)
    if not left or not right:
        return False
    if left == right or left in right or right in left:
        if not re.fullmatch(r"[\w'-]+", left):
            return True
    if re.fullmatch(r"[\w'-]+", left):
        return bool(re.search(rf"(?<!\w){re.escape(left)}(?!\w)", right))
    return left in right or right in left


def _preference_update(message, active_states, items):
    """Normalize a preference and close its explicitly replaced predecessor.

    This records temporal succession only. It never supplies a trigger or an
    EXPLICIT_CAUSE relation, so a preference change cannot become causal proof.
    """
    statement = _preference_statement(message)
    if statement is None:
        return None

    target = statement["value"]
    matching = next((item for item in items
                     if _preference_label_matches(target, item.get("state_value", ""))),
                    None)

    replacement_like = statement["replacement"] or bool(statement["old"])
    if (not replacement_like and matching is not None
            and str(matching.get("state_type", "")).upper() != "OTHER"
            and _normalized_preference_label(matching["state_value"])
            != _normalized_preference_label(target)):
        return None
    if matching is None:
        matching = {"state_type": "PREFERENCE", "state_value": target,
                    "confidence": 0.75, "assertion_mode": "assertion"}
        items.append(matching)
    else:
        matching["state_type"] = "PREFERENCE"
        matching["state_value"] = target
    if statement["key"]:
        matching["preference_key"] = statement["key"]

    candidates = []
    for state in active_states:
        if state.state_type not in {"PREFERENCE", "INTEREST", "FACT", "OTHER"}:
            continue
        if (_normalized_preference_label(state.state_value)
                == _normalized_preference_label(target)):
            continue
        old_match = (statement["old"] is not None
                     and _preference_label_matches(statement["old"], state.state_value))
        key_match = (statement["key"] is not None
                     and _preference_key_for_state(state) == statement["key"])
        if old_match or key_match:
            candidates.append(state)

    if not candidates and not statement["old"] and not statement["key"]:
        # A plain "I prefer coffee" is a global preference update. If exactly one
        # undomained preference is active, close it instead of silently leaving
        # two unrelated active preferences. Multiple candidates remain a real
        # conflict and are reported by the audit/ambiguity path.
        active_preferences = [state for state in active_states
                              if state.state_type == "PREFERENCE"
                              and not _preference_key_for_state(state)
                              and not _preference_label_matches(target, state.state_value)]
        if len(active_preferences) == 1:
            candidates = active_preferences

    # Remove an extracted old value from the new ingestion; it was mentioned
    # only to identify the predecessor, not asserted as a new state.
    if statement["old"]:
        items[:] = [item for item in items
                    if not _preference_label_matches(
                        statement["old"], item.get("state_value", ""))
                    or _preference_label_matches(target, item.get("state_value", ""))]

    if not candidates:
        return None
    candidates.sort(key=lambda state: state.valid_from, reverse=True)
    key = statement["key"] or next(
        (candidate_key for candidate_key in
         (_preference_key_for_state(state) for state in candidates)
         if candidate_key), None)
    if key:
        matching["preference_key"] = key
    source_ids = {state.state_id for state in candidates}
    target_duplicates = []
    for state in active_states:
        if state.state_id in source_ids:
            continue
        if state.state_type not in {"PREFERENCE", "INTEREST", "FACT", "OTHER"}:
            continue
        if not _preference_label_matches(target, state.state_value):
            continue
        state_key = _preference_key_for_state(state)
        if key and state_key and state_key != key:
            continue
        target_duplicates.append(state)
    return {"sources": candidates, "source": candidates[0], "target": target,
            "key": key, "target_duplicates": target_duplicates}


def _grounded_states(raw_states, message):
    """Keep only extracted values grounded in a durable assertive clause."""
    durable = " ".join(_durable_clause_texts(message)).casefold()
    result = []

    for item in raw_states:
        value = str(item.get("state_value", "")).strip()
        if not value or not _is_assertion_value(value):
            continue

        # Require meaningful overlap with a non-question, non-hypothetical
        # clause. A value drawn only from a question or a conditional is not
        # evidence about the user's current life.
        if value.casefold() not in durable and not any(
            token.casefold() in durable
            for token in value.split()
            if len(token) > 3
        ):
            continue

        kind = str(item.get("state_type", "OTHER")).upper()
        if kind not in STATE_TYPES:
            kind = "OTHER"

        mode = _value_mode(value, message)
        if mode == "plan" and kind not in _PLAN_STATE_TYPES:
            # Plans may become structured GOAL/PLAN records, but must not be
            # promoted to completed activities, locations, or life events.
            continue
        if mode not in {"assertion", "plan"}:
            continue

        result.append({
            **item,
            "state_value": value,
            "state_type": kind,
            "assertion_mode": mode,
        })

    return result


def _explicit_change(message, active_states):
    """Identify a simple, explicit change with a stated cause."""
    cause = CAUSE_RE.search(message)
    target_match = TARGET_RE.search(message)

    if not cause or not target_match:
        return None

    target = target_match.group("target").strip()
    target = re.split(
        r"\s+(?:because\s+of|because|due\s+to)\s+",
        target,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()

    source_text = (target_match.group("source") or "").strip()
    source = None

    if source_text:
        # A stated source is required evidence for the edge. If it does not
        # resolve to an active state, do not substitute an unrelated state.
        source = next(
            (
                state for state in reversed(active_states)
                if state.state_value.casefold() in source_text.casefold()
                or source_text.casefold() in state.state_value.casefold()
            ),
            None,
        )
    elif len(active_states) == 1:
        # A source-less change can use the sole active state as its origin.
        source = active_states[0]

    if source is None or not target:
        return None
    if str(source.state_value).strip().casefold() == target.casefold():
        # Repeating "switched to X" while X is already active must not create
        # a self-transition or close/reopen the same value.
        return None

    reason = cause.group(1).strip()

    return {
        "from_state": source,
        "to_state_value": target,
        "trigger": reason,
        "reason": message.strip(),
        "transition_type": "ACTIVITY_CHANGE",
        "relation_kind": "EXPLICIT_CAUSE",
        "confidence": 0.85,
    }


def _is_unresolved_explicit_change(message):
    """Detect a cause-change statement that could not be grounded.

    Such a statement already has structured change semantics. Falling back to
    storing the whole sentence as an ``OTHER`` state would add ungrounded
    history and make a retried ingestion non-idempotent.
    """
    text = str(message or "")
    return bool(CAUSE_RE.search(text) and TARGET_RE.search(text))


def can_extract_deterministically(message):
    """Return whether every durable clause has a safe rule-based parser.

    This is intentionally conservative. It only skips the extractor for simple
    statements whose state, value, and replacement semantics are fully covered
    by the local parsers. Mixed clauses with additional facts still use the
    configured extractor.
    """
    clauses = _durable_clause_texts(message)
    if not clauses:
        return False
    for clause in clauses:
        has_conjunction = bool(re.search(r"\b(?:and|but)\b", clause, re.IGNORECASE))
        if _location_move(clause) or _LOCATION_STATEMENT_RE.search(clause):
            if has_conjunction:
                return False
            continue
        if _preference_statement(clause):
            if has_conjunction:
                return False
            continue
        if _goal_statement(clause):
            continue
        if _completion_statement(clause):
            if has_conjunction and not _goal_statement(clause):
                return False
            continue
        return False
    return True


def ingest_message(message, user_id, memory_ids, memories, source_role="user",
                   allow_llm=True, source_timestamp=None):
    """Extract states and persist state changes atomically."""
    # Only user-authored assertions can become durable personal facts. Question
    # text can contain the same verbs as a factual statement, so provenance and
    # message form must be checked before extraction or any write.
    if source_role != "user" or not is_ingestable_message(message):
        return [], []

    previous = db.states(user_id)
    now = source_timestamp or datetime.now(timezone.utc)
    if isinstance(now,str):
        try:
            now=datetime.fromisoformat(now.replace("Z","+00:00"))
        except ValueError as exc:
            raise ValueError("source_timestamp must be ISO-8601") from exc
    if not isinstance(now,datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("source_timestamp must be timezone-aware")
    now=now.astimezone(timezone.utc)
    active = [state for state in previous
              if state.status == "ACTIVE"
              and state.valid_from <= now
              and (state.valid_until is None or now < state.valid_until)]
    retracted = _retraction_targets(message, active)

    # Extraction happens before opening the write transaction.
    # Provider failures therefore cannot leave partial database writes.
    location_move = _location_move(message)
    if allow_llm:
        raw = llm.extract(
            message,
            memories,
            [state.model_dump(mode="json") for state in active],
        )
    else:
        raw = Extraction()

    items = _grounded_states(raw.states, message)
    goal_statement = _goal_statement(message)
    if goal_statement:
        matching = None
        for item in items:
            if item.get("state_type") == "GOAL":
                matching = item
                break
            if (item.get("state_type") == "OTHER"
                    and _goal_label_matches(goal_statement,
                                            item.get("state_value", ""))):
                matching = item
                break
        if matching is None:
            items.append({"state_type": "GOAL", "state_value": goal_statement,
                          "confidence": 0.8, "assertion_mode": "assertion"})
        else:
            matching["state_type"] = "GOAL"
            matching["state_value"] = goal_statement
    completion_statement = _completion_statement(message)
    if completion_statement:
        matching = None
        retained = []
        for item in items:
            if _goal_label_matches(completion_statement,
                                   item.get("state_value", "")):
                # A completion is a LIFE_EVENT. Extractor output that typed the
                # same object as an interest, goal, activity, or preference is
                # the completion object, not an additional current state.
                if item.get("state_type") in {"LIFE_EVENT", "OTHER"}:
                    matching = matching or item
                continue
            retained.append(item)
        items = retained
        if matching is None:
            items.append({"state_type": "LIFE_EVENT",
                          "state_value": completion_statement,
                          "confidence": 0.8, "assertion_mode": "assertion"})
        else:
            matching["state_type"] = "LIFE_EVENT"
            matching["state_value"] = completion_statement
            items.append(matching)

    explicit = None if location_move else _explicit_change(message, active)
    goal_replacement = (
        _goal_replacement(message, raw.transitions, active, items)
        if explicit is None else None
    )

    if explicit:
        target_value = explicit["to_state_value"].casefold()
        source_value = explicit["from_state"].state_value.casefold()
        normalized_items = []

        for item in items:
            value = item["state_value"].strip()

            # Do not re-create the source as a new state.
            if value.casefold() == source_value:
                continue

            # Split a combined destination and its explanation.
            if (
                target_value in value.casefold()
                and re.search(
                    r"\s+(?:because\s+of|because|due\s+to)\s+",
                    value,
                    re.IGNORECASE,
                )
            ):
                item = {
                    **item,
                    "state_value": explicit["to_state_value"],
                }

            normalized_items.append(item)

        items = normalized_items
                # The explicit destination is grounded in the user's message.
        # Add it even if the extractor omitted the destination.
        if not any(
            item["state_value"].casefold()
            == explicit["to_state_value"].casefold()
            for item in items
        ):
            items.append({
                "state_type": "ACTIVITY",
                "state_value": explicit["to_state_value"],
                "confidence": explicit["confidence"],
            })
    # Fallback for simple first-person assertions only. An unresolved explicit
    # change already failed its grounded source/target checks; persisting the
    # whole sentence as a state would make a retry look like new evidence.
    if (not items and _is_supported_utterance(message)
            and not _is_unresolved_explicit_change(message)):
        items = [{
            "state_type": "OTHER",
            "state_value": message.strip(),
            "confidence": 0.5,
            "assertion_mode": "assertion",
        }]

    preference_update = (_preference_update(message, active, items)
                         if explicit is None else None)
    goal_update = (_goal_update(message, active, items)
                   if explicit is None and goal_replacement is None else None)
    completion_update = (
        _completion_update(message, active)
        if explicit is None and goal_replacement is None and goal_update is None
        else None
    )
    location_update = (_location_update(message, active, items)
                       if explicit is None else None)

    # Deduplicate the extracted states without discarding distinct values.
    deduplicated = []
    seen = set()

    for item in items:
        key = _state_key(item.get("state_type"), item.get("state_value"))
        if key not in seen:
            seen.add(key)
            deduplicated.append(item)

    items = deduplicated

    # A retraction can close existing states without creating a replacement.
    if not items and not retracted:
        return [], []

    now_states = []
    edges = []
    source_state = explicit["from_state"] if explicit else None
    retracted_ids = {state.state_id for state in retracted}
    existing_keys = {_state_key(state.state_type, state.state_value)
                     for state in active
                     if state.state_id not in retracted_ids
                     and (explicit is None
                          or state.state_id != explicit["from_state"].state_id)}
    if preference_update and preference_update.get("target_duplicates"):
        # A replacement must create a fresh target interval even when an older
        # equivalent value is still ACTIVE. Otherwise the duplicate skip below
        # would suppress both the new state and its transition.
        existing_keys.discard(_state_key("PREFERENCE",
                                         preference_update["target"]))

    # All writes, including state closure and edge creation, share one
    # SQLite connection. Any exception rolls back the entire operation.
    with db.connect() as connection:
        for state in retracted:
            if source_state is not None and state.state_id == source_state.state_id:
                continue
            db.close_state(state.state_id, now, connection=connection)
            state.status = "INACTIVE"
            state.valid_until = now

        for item in items:
            key = _state_key(item.get("state_type"), item.get("state_value"))
            if key in existing_keys:
                # Repeated assertions are idempotent; they neither duplicate
                # the active state nor fabricate a transition.
                continue
            state_metadata={"assertion_mode":item.get("assertion_mode","assertion")}
            state_fields={}
            if item.get("state_type")=="PREFERENCE":
                state_fields["valid_from"]=now
                state_metadata.update({"source_role":source_role,
                                       "source_timestamp":now.isoformat()})
            state = State(
                user_id=user_id,
                state_type=item["state_type"],
                state_value=item["state_value"],
                status="ACTIVE",
                confidence=max(
                    0.0,
                    min(1.0, float(item.get("confidence", 0.5))),
                ),
                source_memory_ids=list(memory_ids),
                metadata=state_metadata,
                **state_fields,
            )
            if item.get("preference_key"):
                state.metadata["preference_key"] = item["preference_key"]

            # Close the source only for the grounded explicit change.
            if (
                source_state is not None
                and state.state_value.casefold()
                == explicit["to_state_value"].casefold()
            ):
                timestamp = _recorded_transition_time(state, source_state)
                db.close_state(
                    source_state.state_id,
                    timestamp,
                    connection=connection,
                )

                edge = Transition(
                    user_id=user_id,
                    from_state_id=source_state.state_id,
                    to_state_id=state.state_id,
                    transition_type=explicit["transition_type"],
                    relation_kind=explicit["relation_kind"],
                    trigger=explicit["trigger"],
                    reason=explicit["reason"],
                    timestamp=timestamp,
                    confidence=explicit["confidence"],
                    source_memory_ids=list(memory_ids),
                )

                db.save_state(state, connection=connection)
                db.save_transition(edge, connection=connection)
                edges.append(edge)
                source_state = None
            elif (
                preference_update is not None
                and state.state_type == "PREFERENCE"
                and state.state_value.casefold()
                == preference_update["target"].casefold()
            ):
                primary = preference_update["source"]
                timestamp = _recorded_transition_time(state, primary)
                for source in preference_update["sources"]:
                    if source.state_id == primary.state_id:
                        continue
                    db.close_state(source.state_id, timestamp.isoformat(),
                                   connection=connection)
                    source.status = "INACTIVE"
                    source.valid_until = timestamp
                for duplicate in preference_update.get("target_duplicates", []):
                    if (duplicate.state_id == primary.state_id
                            or duplicate.state_id in retracted_ids):
                        continue
                    db.close_state(duplicate.state_id, timestamp.isoformat(),
                                   connection=connection)
                    duplicate.status = "INACTIVE"
                    duplicate.valid_until = timestamp
                db.close_state(primary.state_id, timestamp.isoformat(),
                               connection=connection)
                edge = Transition(
                    user_id=user_id,
                    from_state_id=primary.state_id,
                    to_state_id=state.state_id,
                    transition_type="PREFERENCE_CHANGE",
                    relation_kind="TEMPORAL_SUCCESSION",
                    trigger=None,
                    reason=None,
                    timestamp=timestamp,
                    confidence=0.5,
                    source_memory_ids=list(memory_ids),
                )
                db.save_state(state, connection=connection)
                db.save_transition(edge, connection=connection)
                edges.append(edge)
                primary.status = "INACTIVE"
                primary.valid_until = timestamp
                preference_update = None
            elif (
                (goal_replacement is not None or goal_update is not None)
                and state.state_type == "GOAL"
                and state.state_value.casefold()
                == (goal_replacement or goal_update)["target"].casefold()
            ):
                source = (goal_replacement or goal_update)["source"]
                timestamp = _recorded_transition_time(state, source)
                db.close_state(source.state_id, timestamp.isoformat(),
                               connection=connection)
                edge = Transition(
                    user_id=user_id,
                    from_state_id=source.state_id,
                    to_state_id=state.state_id,
                    transition_type="GOAL_CHANGE",
                    relation_kind="TEMPORAL_SUCCESSION",
                    trigger=None,
                    reason=None,
                    timestamp=timestamp,
                    confidence=0.5,
                    source_memory_ids=list(memory_ids),
                )
                db.save_state(state, connection=connection)
                db.save_transition(edge, connection=connection)
                edges.append(edge)
                source.status = "INACTIVE"
                source.valid_until = timestamp
                goal_replacement = None
                goal_update = None
            elif (
                location_update is not None
                and state.state_type == "LOCATION"
                and state.state_value.casefold() == location_update["target"].casefold()
            ):
                source = location_update["source"]
                timestamp = _recorded_transition_time(state, source)
                db.close_state(source.state_id, timestamp.isoformat(),
                               connection=connection)
                edge = Transition(
                    user_id=user_id,
                    from_state_id=source.state_id,
                    to_state_id=state.state_id,
                    transition_type="LOCATION_CHANGE",
                    relation_kind="TEMPORAL_SUCCESSION",
                    trigger=None,
                    reason=None,
                    timestamp=timestamp,
                    confidence=0.5,
                    source_memory_ids=list(memory_ids),
                )
                db.save_state(state, connection=connection)
                db.save_transition(edge, connection=connection)
                edges.append(edge)
                source.status = "INACTIVE"
                source.valid_until = timestamp
                location_update = None
            elif (
                completion_update is not None
                and state.state_type == "LIFE_EVENT"
                and state.state_value.casefold()
                == completion_update["completion"].casefold()
            ):
                source = completion_update["source"]
                timestamp = _recorded_transition_time(state, source)
                db.close_state(source.state_id, timestamp.isoformat(),
                               connection=connection)
                edge = Transition(
                    user_id=user_id,
                    from_state_id=source.state_id,
                    to_state_id=state.state_id,
                    transition_type="GOAL_COMPLETION",
                    relation_kind="TEMPORAL_SUCCESSION",
                    trigger=None,
                    reason=None,
                    timestamp=timestamp,
                    confidence=0.5,
                    source_memory_ids=list(memory_ids),
                )
                db.save_state(state, connection=connection)
                db.save_transition(edge, connection=connection)
                edges.append(edge)
                source.status = "INACTIVE"
                source.valid_until = timestamp
                completion_update = None
            else:
                db.save_state(state, connection=connection)

            existing_keys.add(key)
            now_states.append(state)

    return now_states, edges
