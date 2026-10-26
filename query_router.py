import re
from llm_service import llm
from models import QueryAnalysis

# Local models occasionally classify short, unpunctuated questions as facts.
# Route clear question forms deterministically so they never enter ingestion.
QUESTION_START = re.compile(
    r"^(?:(?:please|kindly)\s+)?"
    r"(?:why|how|what|when|where|who|which|whose|whom|"
    r"do|does|did|is|are|was|were|have|has|had|can|could|would|should|"
    r"tell me|explain|remind me|compare|show me|help me|list|describe|"
    r"summari[sz]e|recall|clarify)\b",
    re.IGNORECASE,
)
_PERSONAL = re.compile(r"\b(?:i|me|my|mine|we|our|us)\b", re.I)
_CAUSAL = re.compile(
    r"\b(?:why|what\s+(?:was|is)\s+the\s+(?:\w+\s+){0,2}(?:cause|reason)|"
    r"what\s+(?:\w+\s+){0,2}caused|how\s+come)\b", re.I)
_STORED_PREFERENCES = re.compile(
    r"\b(?:stored|recorded|saved|mentioned)\b[^?]*\bpreferences?\b|"
    r"\bpreferences?\b[^?]*\b(?:stored|recorded|saved)\b", re.I)
_MOST_RECENT_PREFERENCE = re.compile(
    r"\bmost\s+recently\s+(?:say|state|mention|tell)\b|"
    r"\bmost\s+recent(?:ly)?\s+(?:stated|mentioned)\b[^?]*"
    r"\b(?:preferences?|prefer(?:red|s)?)\b",
    re.I)
_HISTORICAL_PREFERENCE = re.compile(
    r"\b(?:past|historical|previously|earlier)\b[^?]*"
    r"\b(?:preferences?|prefer(?:red)?|programming\s+languages?)\b|"
    r"\b(?:preferences?|prefer(?:red)?|programming\s+languages?)\b[^?]*"
    r"\b(?:past|historical|previously|earlier)\b",
    re.I)
_COMPLETION = re.compile(
    r"\b(?:what|which)\b[^?]*\b(?:did|have|had)\s+(?:i|we)\s+"
    r"(?:complete|finish|achieve)\w*\b", re.I)
_CURRENT = re.compile(r"\b(?:current(?:ly)?|now|right\s+now|today|at\s+present)\b", re.I)
_GOAL_PURPOSE = re.compile(
    r"\b(?:trying|planning|working|aiming)\s+to\s+"
    r"(?:learn|study|complete|finish|build|improve|achieve|become|develop)\b", re.I)


def _deterministic_category(question):
    """Route only clear personal-history forms; defer ambiguous wording to the model."""
    text=question.strip()
    if not (text.endswith("?") or QUESTION_START.match(text)) or not _PERSONAL.search(text):
        return None
    if _CAUSAL.search(text):
        return "CAUSAL"
    # Imports are local to keep the router cheap during module initialization.
    from temporal_engine import is_predecessor_query
    from causal_graph import is_transition_question
    if is_predecessor_query(text):
        return "HISTORY"
    if _MOST_RECENT_PREFERENCE.search(text):
        return "HISTORY"
    if _HISTORICAL_PREFERENCE.search(text):
        return "HISTORY"
    if _STORED_PREFERENCES.search(text):
        return "HISTORY"
    if _COMPLETION.search(text):
        return "HISTORY"
    if is_transition_question(text):
        return "EVOLUTION"
    if _CURRENT.search(text) or _GOAL_PURPOSE.search(text):
        return "STATE"
    return None

def normalize_category(value):
    """Canonicalize model-provided intent labels at the routing boundary."""
    text = str(value or "OTHER").strip().upper()
    return text or "OTHER"


def _normalized(result):
    if isinstance(result, QueryAnalysis):
        result.category = normalize_category(result.category)
    return result


def classify_query(question):
    deterministic_category=_deterministic_category(question)
    if deterministic_category:
        return QueryAnalysis(is_question=True, category=deterministic_category,
                             rationale="Clear personal-history intent")
    if question.strip().endswith("?") or QUESTION_START.match(question.strip()):
        try:
            result = llm.classify(question)
            if result.is_question:
                return _normalized(result)
        except Exception:
            pass
        return QueryAnalysis(is_question=True, category="HISTORY", rationale="Recognized question form")
    return _normalized(llm.classify(question))
