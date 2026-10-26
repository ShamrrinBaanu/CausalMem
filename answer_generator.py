from llm_service import llm, model_evidence_projection
from causal_graph import (
    _terms, is_causal_question, is_transition_question, requested_state_types,
)
from temporal_engine import is_predecessor_query
from datetime import datetime, timezone
import re
import time

ABSTENTION="I don't have enough information in your memory to determine that."
_UI_DIAGNOSTIC_REASONS={
    "empty_evidence","ambiguous_evidence","explicit_cause_gate_rejected",
    "missing_explicit_cause_transition","missing_predecessor_transition",
    "ambiguous_predecessor_transition",
    "structured_answer_method_unavailable","malformed_structured_response",
    "no_validated_causal_claim","no_validated_claims","validated_claims",
}
_UI_REJECTION_CATEGORIES={
    "malformed_claim","malformed_evidence_reference","unknown_or_unowned_evidence_id",
    "claim_not_supported_by_source","causal_relation_or_support_rejected",
    "temporal_relation_or_support_rejected","claim_type_not_supported_by_transition",
    "no_valid_evidence_reference","evidence_not_in_model_bundle",
}
_UI_REJECTION_REASON_CODES={
    "claim_not_object","claim_text_missing","claim_text_not_string","claim_text_empty",
    "claim_type_missing","claim_type_invalid","claim_evidence_missing",
    "claim_evidence_not_list","evidence_reference_not_object",
    "evidence_reference_kind_missing","evidence_reference_kind_invalid",
    "evidence_reference_id_missing","evidence_reference_id_not_string",
    "evidence_reference_id_empty","unknown_or_unowned_evidence_id",
    "citation_id_unknown","citation_ref_unknown","citation_ref_conflict",
    "citation_kind_mismatch","citation_foreign_user",
    "citation_owner_unverified","citation_not_in_model_bundle",
    "citation_resolved","citation_structure_malformed","evidence_not_in_model_bundle",
    "fact_lexical_support_failed","claim_not_relevant_to_question",
    "temporal_validity_failed",
    "state_memory_claim_type_not_supported","causal_relation_or_support_rejected",
    "causal_relation_not_explicit","causal_recorded_cause_missing",
    "causal_query_support_missing","causal_claim_missing_cause_terms",
    "causal_claim_lexical_support_failed",
    "temporal_relation_or_support_rejected","claim_type_not_supported_by_transition",
    "no_valid_evidence_reference",
}
_UI_MODEL_STATUSES={"not_called","malformed_structured_response","structured_response"}


def safe_answer_diagnostic_summary(diagnostics):
    """Return an allowlisted, content-free diagnostic summary for a UI."""
    if not isinstance(diagnostics,dict):
        diagnostics={}
    def safe_bool(key):
        value=diagnostics.get(key)
        return value if isinstance(value,bool) else None
    def safe_count(key):
        value=diagnostics.get(key)
        return value if isinstance(value,int) and not isinstance(value,bool) and value>=0 else 0

    categories=diagnostics.get("rejection_categories")
    if not isinstance(categories,dict):
        categories={}
    safe_categories={key:value for key,value in categories.items()
                     if key in _UI_REJECTION_CATEGORIES and isinstance(value,int)
                     and not isinstance(value,bool) and value>0}
    # The detailed validator records richer per-claim structures internally.
    # Expose only its fixed vocabulary of stable reason-code/count pairs.
    rejection_reasons={}
    rejected_claims=diagnostics.get("rejected_claims")
    if isinstance(rejected_claims,list):
        for item in rejected_claims:
            if not isinstance(item,dict):
                continue
            reasons=item.get("rejection_reasons")
            if not isinstance(reasons,list):
                continue
            for code in reasons:
                if isinstance(code,str) and code in _UI_REJECTION_REASON_CODES:
                    rejection_reasons[code]=rejection_reasons.get(code,0)+1
    # Older/internal diagnostic producers may provide counts without the
    # per-claim list. Keep that compatible while never displaying raw entries.
    if not rejection_reasons:
        detail_counts=diagnostics.get("rejection_details")
        if isinstance(detail_counts,dict):
            for code,count in detail_counts.items():
                if (code in _UI_REJECTION_REASON_CODES and isinstance(count,int)
                        and not isinstance(count,bool) and count>0):
                    rejection_reasons[code]=count
    reason=diagnostics.get("answer_reason")
    status=diagnostics.get("model_response_status")
    return {
        "causal_gate_passed":safe_bool("causal_gate_passed"),
        "required_transition_citation_accepted":safe_bool("causal_transition_citation_accepted"),
        "answer_reason":reason if isinstance(reason,str) and reason in _UI_DIAGNOSTIC_REASONS else "unavailable",
        "model_response_status":status if isinstance(status,str) and status in _UI_MODEL_STATUSES else "unavailable",
        "accepted_claim_count":safe_count("accepted_claim_count"),
        "rejected_claim_count":safe_count("rejected_claim_count"),
        "rejection_categories":safe_categories,
        "rejection_reasons":dict(sorted(rejection_reasons.items())),
        **({"citation_validation":[{
            "citation_kind":item.get("citation_kind")
                if isinstance(item.get("citation_kind"),str)
                and item.get("citation_kind") in {"memory","state","transition"} else "invalid",
            "evidence_id_resolves_owned":item.get("evidence_id_resolves_owned")
                if isinstance(item.get("evidence_id_resolves_owned"),bool) else False,
            "kind_matches_resolved_type":item.get("kind_matches_resolved_type")
                if isinstance(item.get("kind_matches_resolved_type"),bool) else None,
            "reason_code":item.get("reason_code")
                if isinstance(item.get("reason_code"),str)
                and item.get("reason_code") in _UI_REJECTION_REASON_CODES else "unavailable",
        } for item in diagnostics.get("citation_validation",[])
            if isinstance(item,dict)]}
           if isinstance(diagnostics.get("citation_validation"),list)
           and diagnostics.get("citation_validation") else {}),
    }


_GENERIC_RELEVANCE_TERMS={
    "current","goal","goals","state","states","change","changed",
    "previous","why","caused","cause","causes","what","did","does",
    "the","my","your","this","that","active","history","memory",
    "because","due","reason","reasons","unknown","whether","know","sure",
    "explain","led","leading","made","how","come",
}
_CAUSAL_CLAIM=re.compile(
    r"\b(?:because|due\s+to|caused\s+by|cause[sd]?|made|led\s+to|"
    r"resulted?\s+in|as\s+a\s+result|reason\s+(?:was|is))\b", re.I,
)


def _subject_terms(text):
    """Use graph tokenization, then discard terms too generic to anchor a state."""
    return _terms(text)-_GENERIC_RELEVANCE_TERMS


def _question_matches_endpoints(question,source_label,destination_label,
                                source_type=None,destination_type=None):
    anchors=_subject_terms(question)
    endpoint_terms=_subject_terms(source_label)|_subject_terms(destination_label)
    if anchors & endpoint_terms:
        return True
    # A typed question such as "what change in activity" may not repeat either
    # endpoint label. The requested state type still anchors the edge.
    types=requested_state_types(question)
    return bool(types and ({source_type,destination_type}&types))


def _question_matches_cause(question,cause_text):
    anchors=_subject_terms(question)
    return bool(anchors and anchors & _subject_terms(cause_text or ""))


def _question_matches_causal_edge(question,resolved,source_type=None,
                                  destination_type=None):
    return (_question_matches_endpoints(
                question,resolved.get("from",""),resolved.get("to",""),
                source_type,destination_type)
            or _question_matches_cause(question,resolved.get("cause","")))


def _has_supported_cause(question,evidence):
    requested_user=evidence.get("user_id")
    anchors=_subject_terms(question)
    if not anchors:
        return False

    state_labels={}
    state_types={}
    foreign_state_ids=set()
    for state in evidence.get("states",[]):
        if not isinstance(state,dict) or not state.get("state_id"):
            continue
        if requested_user is not None and state.get("user_id")!=requested_user:
            foreign_state_ids.add(state["state_id"])
            continue
        if not state.get("state_value"):
            continue
        state_labels[state["state_id"]]=state["state_value"]
        state_types[state["state_id"]]=state.get("state_type")

    paths=evidence.get("causal_path",[])
    for transition in evidence.get("transitions",[]):
        if not isinstance(transition,dict):
            continue
        if transition.get("relation_kind")!="EXPLICIT_CAUSE":
            continue
        if not any(isinstance(transition.get(k),str) and transition[k].strip() for k in ("trigger","reason")):
            continue
        if requested_user is not None and transition.get("user_id")!=requested_user:
            continue
        source_id=transition.get("from_state_id"); destination_id=transition.get("to_state_id")
        if not source_id or not destination_id:
            continue
        if source_id in foreign_state_ids or destination_id in foreign_state_ids:
            continue

        source_label=state_labels.get(source_id); destination_label=state_labels.get(destination_id)
        for path in paths:
            if not isinstance(path,dict):
                continue
            same_edge=(path.get("from_state_id")==source_id and path.get("to_state_id")==destination_id)
            if not same_edge or path.get("relation_kind")!="EXPLICIT_CAUSE":
                continue
            if requested_user is not None and path.get("user_id")!=requested_user:
                continue
            source_label=source_label or path.get("from")
            destination_label=destination_label or path.get("to")
            break
        # Both endpoint IDs must resolve to labels from user-owned state records
        # or the matching user-scoped path, and the query must anchor to an end.
        if not isinstance(source_label,str) or not source_label.strip():
            continue
        if not isinstance(destination_label,str) or not destination_label.strip():
            continue
        cause_text=" ".join(value for value in
                           (transition.get("trigger"),transition.get("reason"))
                           if isinstance(value,str) and value.strip())
        if (_question_matches_endpoints(
                question,source_label,destination_label,
                state_types.get(source_id),state_types.get(destination_id))
                or _question_matches_cause(question,cause_text)):
            return True
    return False


def _supported_cause_destination(question,evidence):
    """Resolve the destination label for a transition accepted by the gate."""
    requested_user=evidence.get("user_id")
    state_labels={state.get("state_id"):state.get("state_value")
                  for state in evidence.get("states",[])
                  if isinstance(state,dict) and state.get("state_id") and state.get("state_value")
                  and (requested_user is None or state.get("user_id")==requested_user)}
    state_types={state.get("state_id"):state.get("state_type")
                 for state in evidence.get("states",[])
                 if isinstance(state,dict) and state.get("state_id")
                 and (requested_user is None or state.get("user_id")==requested_user)}
    for transition in evidence.get("transitions",[]):
        if (not isinstance(transition,dict) or transition.get("relation_kind")!="EXPLICIT_CAUSE"
                or requested_user is not None and transition.get("user_id")!=requested_user
                or not any(isinstance(transition.get(key),str) and transition[key].strip()
                           for key in ("trigger","reason"))):
            continue
        source_id=transition.get("from_state_id"); destination_id=transition.get("to_state_id")
        source=state_labels.get(source_id); destination=state_labels.get(destination_id)
        for path in evidence.get("causal_path",[]):
            if not isinstance(path,dict): continue
            if (path.get("from_state_id")==source_id and path.get("to_state_id")==destination_id
                    and path.get("relation_kind")=="EXPLICIT_CAUSE"
                    and (requested_user is None or path.get("user_id")==requested_user)):
                source=source or path.get("from"); destination=destination or path.get("to")
                break
        if not isinstance(source,str) or not isinstance(destination,str): continue
        cause_text=" ".join(value for value in
                           (transition.get("trigger"),transition.get("reason"))
                           if isinstance(value,str) and value.strip())
        if (_question_matches_endpoints(
                question,source,destination,
                state_types.get(source_id),state_types.get(destination_id))
                or _question_matches_cause(question,cause_text)):
            return destination
    return None


def _causal_model_evidence(question,evidence):
    """Project causal evidence so the model can cite only owned cause edges.

    The complete original bundle remains the authority for validation. This
    model-facing projection removes competing memory/state citation IDs while
    retaining each relevant transition ID, endpoint label, cause, and time.
    """
    user_id=evidence.get("user_id")
    owned=_owned_sources(evidence)
    state_types={state.get("state_id"):state.get("state_type")
                 for state in evidence.get("states",[])
                 if isinstance(state,dict) and state.get("state_id")}
    transitions=[]
    for edge in evidence.get("transitions",[]):
        if (not isinstance(edge,dict) or edge.get("user_id")!=user_id
                or edge.get("relation_kind")!="EXPLICIT_CAUSE"):
            continue
        transition_id=edge.get("transition_id")
        resolved=owned.get(("transition",transition_id)) if isinstance(transition_id,str) else None
        if not isinstance(resolved,dict) or not resolved.get("cause"):
            continue
        if not _question_matches_causal_edge(
                question,resolved,
                state_types.get(edge.get("from_state_id")),
                state_types.get(edge.get("to_state_id"))):
            continue
        transitions.append({
            "transition_id":transition_id,
            "transition_type":edge.get("transition_type"),
            "relation_kind":"EXPLICIT_CAUSE",
            "from":resolved["from"],
            "to":resolved["to"],
            "trigger":edge.get("trigger"),
            "reason":edge.get("reason"),
            "timestamp":edge.get("timestamp"),
        })
    projected={"query_category":"CAUSAL","memories":[],"states":[],
               "transitions":transitions}
    requested=evidence.get("requested_timestamp")
    if requested is not None:
        projected["requested_timestamp"]=requested
        if isinstance(evidence.get("temporal_context"),dict):
            projected["temporal_context"]={
                "mode":evidence["temporal_context"].get("mode"),
                "requested_timestamp":evidence["temporal_context"].get("requested_timestamp"),
                "state_validity_rule":evidence["temporal_context"].get("state_validity_rule"),
            }
    return projected


def _transition_model_evidence(question,evidence):
    """Project direct-change evidence so the model cites a transition.

    Endpoint states alone cannot establish that a change occurred. Keeping the
    model-facing bundle transition-focused avoids selecting an unrelated state
    and preserves the recorded source and destination together.
    """
    user_id=evidence.get("user_id")
    owned=_owned_sources(evidence)
    state_types={state.get("state_id"):state.get("state_type")
                 for state in evidence.get("states",[])
                 if isinstance(state,dict) and state.get("state_id")}
    transitions=[]
    for edge in evidence.get("transitions",[]):
        if not isinstance(edge,dict) or edge.get("user_id")!=user_id:
            continue
        transition_id=edge.get("transition_id")
        resolved=owned.get(("transition",transition_id)) if isinstance(transition_id,str) else None
        if not isinstance(resolved,dict):
            continue
        if not _question_matches_endpoints(
                question,resolved.get("from",""),resolved.get("to",""),
                state_types.get(edge.get("from_state_id")),
                state_types.get(edge.get("to_state_id"))):
            continue
        transitions.append({
            "transition_id":transition_id,
            "transition_type":edge.get("transition_type"),
            "relation_kind":resolved.get("relation_kind"),
            "from":resolved["from"],
            "to":resolved["to"],
            "trigger":edge.get("trigger"),
            "reason":edge.get("reason"),
            "timestamp":edge.get("timestamp"),
        })
    projected={"query_category":"EVOLUTION","memories":[],"states":[],
               "transitions":transitions}
    requested=evidence.get("requested_timestamp")
    if requested is not None:
        projected["requested_timestamp"]=requested
        if isinstance(evidence.get("temporal_context"),dict):
            projected["temporal_context"]={
                "mode":evidence["temporal_context"].get("mode"),
                "requested_timestamp":evidence["temporal_context"].get("requested_timestamp"),
                "state_validity_rule":evidence["temporal_context"].get("state_validity_rule"),
            }
    return projected


def _programming_preference_change_evidence(question,evidence):
    """Expose one owned, typed preference-change edge for its explicit query."""
    if (not re.search(r"\b(?:programming|coding)\b",str(question or ""),re.I)
            or not re.search(r"\b(?:prefer(?:ence|ences|red|s)?)\b",
                             str(question or ""),re.I)):
        return None
    user_id=evidence.get("user_id")
    states={state.get("state_id"):state for state in evidence.get("states",[])
            if isinstance(state,dict) and state.get("user_id")==user_id}
    candidates=[]
    for edge in evidence.get("transitions",[]):
        if (not isinstance(edge,dict) or edge.get("user_id")!=user_id
                or edge.get("transition_type")!="PREFERENCE_CHANGE"
                or edge.get("relation_kind")!="TEMPORAL_SUCCESSION"):
            continue
        source=states.get(edge.get("from_state_id"))
        destination=states.get(edge.get("to_state_id"))
        if (not isinstance(source,dict) or not isinstance(destination,dict)
                or source.get("state_type")!="PREFERENCE"
                or destination.get("state_type")!="PREFERENCE"):
            continue
        keys={(source.get("metadata") or {}).get("preference_key"),
              (destination.get("metadata") or {}).get("preference_key")}
        allowed={"favorite:programming","favorite:programming language",
                 "domain:programming","domain:programming language"}
        if keys.issubset(allowed) and len(keys)==1:
            candidates.append((edge,source,destination))
    if len(candidates)!=1:
        return None
    edge,source,destination=candidates[0]
    return {
        "user_id":user_id,
        "query_category":"EVOLUTION",
        "memories":[],
        "states":[source,destination],
        "transitions":[edge],
        "causal_path":[],
    }


def _owned_sources(evidence):
    """Index only identified evidence records owned by the requested user."""
    from collections.abc import Mapping

    user_id=evidence.get("user_id")
    if not isinstance(user_id,str) or not user_id:
        return {}
    sources={}
    states={}
    for state in evidence.get("states",[]):
        if (not isinstance(state,Mapping) or state.get("user_id")!=user_id
                or not isinstance(state.get("state_id"),str) or not state["state_id"]
                or not isinstance(state.get("state_value"),str)):
            continue
        states[state["state_id"]]=state
        state_type=state.get("state_type")
        typed_value=(str(state_type).replace("_"," ")+" " if isinstance(state_type,str) else "")
        sources[("state",state["state_id"])]=typed_value+state["state_value"]
    paths={}
    for path in evidence.get("causal_path",[]):
        if (isinstance(path,Mapping) and path.get("user_id")==user_id
                and path.get("relation_kind")=="EXPLICIT_CAUSE"
                and path.get("from_state_id") and path.get("to_state_id")
                and isinstance(path.get("from"),str) and isinstance(path.get("to"),str)):
            paths[(path["from_state_id"],path["to_state_id"])]=path
    for memory in evidence.get("memories",[]):
        if not isinstance(memory,Mapping):
            continue
        owners=[]
        if "user_id" in memory: owners.append(memory["user_id"])
        metadata=memory.get("metadata")
        if isinstance(metadata,Mapping) and "user_id" in metadata:
            owners.append(metadata["user_id"])
        memory_id=memory.get("id") or memory.get("memory_id")
        text=memory.get("memory") or memory.get("text") or memory.get("content")
        if (not owners or any(owner!=user_id for owner in owners)
                or not isinstance(memory_id,str) or not memory_id.strip()
                or not isinstance(text,str) or not text.strip()):
            continue
        sources[("memory",memory_id)]=text
    # A state value can omit the predicate that gives a fact its meaning (for
    # example, FACT/Toronto). Include only that state's owned, linked source
    # memories when validating a claim that cites the state.
    for state_id,state in states.items():
        memory_ids=state.get("source_memory_ids")
        if not isinstance(memory_ids,list):
            continue
        linked_text=[sources[("memory",memory_id)]
                     for memory_id in memory_ids
                     if ("memory",memory_id) in sources]
        if linked_text:
            sources[("state",state_id)]+=" "+" ".join(linked_text)
    for transition in evidence.get("transitions",[]):
        if (not isinstance(transition,Mapping) or transition.get("user_id")!=user_id
                or not isinstance(transition.get("transition_id"),str)
                or not transition.get("transition_id")):
            continue
        source=states.get(transition.get("from_state_id"))
        destination=states.get(transition.get("to_state_id"))
        relation=transition.get("relation_kind")
        path=paths.get((transition.get("from_state_id"),transition.get("to_state_id")))
        if source is None or destination is None:
            if relation!="EXPLICIT_CAUSE" or path is None:
                continue
            source_value=path["from"]; destination_value=path["to"]
        else:
            source_value=source["state_value"]; destination_value=destination["state_value"]
        cause=" ".join(value for value in (transition.get("trigger"),transition.get("reason"))
                       if isinstance(value,str) and value.strip())
        cause_anchor=(transition.get("trigger") if isinstance(transition.get("trigger"),str)
                      and transition.get("trigger").strip() else transition.get("reason"))
        source_type=(str(source.get("state_type","")).replace("_"," ").strip()
                     if source is not None else "")
        destination_type=(str(destination.get("state_type","")).replace("_"," ").strip()
                          if destination is not None else "")
        endpoint_text=" ".join(part for part in
                               (source_type,source_value,destination_type,destination_value)
                               if part)
        sources[("transition",transition["transition_id"])]={
            "relation_kind":relation,"from":source_value,"to":destination_value,
            "from_type":source_type,"to_type":destination_type,
            "cause":cause,"cause_anchor":cause_anchor,
            "endpoint_text":endpoint_text,
            "cause_text":cause,
            "source_text":(endpoint_text+" "+cause),
        }
    return sources


def _citation_inventory(evidence):
    """Index evidence identifiers without retaining their values in diagnostics."""
    from collections.abc import Mapping

    user_id=evidence.get("user_id") if isinstance(evidence,Mapping) else None
    inventory={}
    def add(kind,identifier,owner):
        if not isinstance(identifier,str) or not identifier.strip():
            return
        record=inventory.setdefault(identifier,{"kinds":set(),"owned_kinds":set(),
                                                 "foreign_owner":False,"unknown_owner":False})
        record["kinds"].add(kind)
        if owner==user_id and isinstance(user_id,str) and user_id:
            record["owned_kinds"].add(kind)
        elif isinstance(owner,str) and owner:
            record["foreign_owner"]=True
        else:
            record["unknown_owner"]=True

    for memory in evidence.get("memories",[]) if isinstance(evidence,Mapping) else []:
        if not isinstance(memory,Mapping):
            continue
        owners=[]
        if "user_id" in memory:
            owners.append(memory.get("user_id"))
        metadata=memory.get("metadata")
        if isinstance(metadata,Mapping) and "user_id" in metadata:
            owners.append(metadata.get("user_id"))
        if owners and isinstance(user_id,str) and all(value==user_id for value in owners):
            owner=user_id
        elif any(isinstance(value,str) and value and value!=user_id for value in owners):
            owner="__foreign_owner__"
        else:
            owner=None
        add("memory",memory.get("id") or memory.get("memory_id"),owner)
    for collection,key,kind in (("states","state_id","state"),
                                ("transitions","transition_id","transition")):
        for item in evidence.get(collection,[]) if isinstance(evidence,Mapping) else []:
            if isinstance(item,Mapping):
                add(kind,item.get(key),item.get("user_id"))
    return inventory


def _model_citation_keys(evidence):
    """Return typed citation keys explicitly present in the model-facing bundle."""
    from collections.abc import Mapping

    keys=set()
    if not isinstance(evidence,Mapping):
        return keys
    for memory in evidence.get("memories",[]):
        if isinstance(memory,Mapping):
            identifier=memory.get("id") or memory.get("memory_id")
            if isinstance(identifier,str) and identifier.strip():
                keys.add(("memory",identifier))
    for state in evidence.get("states",[]):
        if isinstance(state,Mapping):
            identifier=state.get("state_id")
            if isinstance(identifier,str) and identifier.strip():
                keys.add(("state",identifier))
    for transition in evidence.get("transitions",[]):
        if isinstance(transition,Mapping):
            identifier=transition.get("transition_id")
            if isinstance(identifier,str) and identifier.strip():
                keys.add(("transition",identifier))
    return keys


def _resolve_reference_token(reference,citation_refs):
    """Resolve a short citation_ref to the real typed evidence identifier.

    Returns ``(normalized_reference, reason)``. A reference may cite an exact
    kind/id pair for compatibility, but when it supplies a short token that
    token must resolve to that same kind and identifier. Unknown tokens and
    conflicts fail closed; they are never guessed or fuzzy-matched.
    """
    if not isinstance(reference,dict) or "ref" not in reference:
        return reference,None
    token=reference.get("ref")
    if not isinstance(token,str) or not token.strip():
        return None,"evidence_reference_id_missing"
    resolved=(citation_refs or {}).get(token)
    if resolved is None:
        return None,"citation_ref_unknown"
    kind,identifier=resolved
    supplied_kind=reference.get("kind")
    supplied_id=reference.get("id")
    if supplied_kind is not None and supplied_kind!=kind:
        return None,"citation_ref_conflict"
    if supplied_id is not None and supplied_id!=identifier:
        return None,"citation_ref_conflict"
    return {"kind":kind,"id":identifier},None


def _citation_validation_diagnostic(reference,evidence,model_evidence):
    """Return only allowlisted citation metadata; never return or retain the ID."""
    valid_kinds={"memory","state","transition"}
    kind=reference.get("kind") if isinstance(reference,dict) else None
    kind=kind if isinstance(kind,str) and kind in valid_kinds else "invalid"
    result={"citation_kind":kind,"evidence_id_resolves_owned":False,
            "kind_matches_resolved_type":None,"reason_code":"citation_structure_malformed"}
    if (not isinstance(reference,dict) or kind=="invalid"
            or not isinstance(reference.get("id"),str) or not reference["id"].strip()):
        return result

    identifier=reference["id"]
    inventory=_citation_inventory(evidence)
    record=inventory.get(identifier)
    if record is None:
        result["reason_code"]="citation_id_unknown"
        return result
    owned_kinds=record["owned_kinds"]
    if not owned_kinds:
        result["reason_code"]=("citation_foreign_user" if record["foreign_owner"]
                                else "citation_owner_unverified")
        return result

    result["evidence_id_resolves_owned"]=True
    result["kind_matches_resolved_type"]=kind in owned_kinds
    if kind not in owned_kinds:
        result["reason_code"]="citation_kind_mismatch"
        return result
    if (kind,identifier) not in _model_citation_keys(model_evidence):
        result["reason_code"]="citation_not_in_model_bundle"
        return result
    result["reason_code"]="citation_resolved"
    return result


_CLAIM_STOP_WORDS={"i","me","my","mine","you","your","yours","user","the","a","an",
                   "is","are","was","were","be","been","being","do","does","did",
                   "has","have","had","to","of","in","on","at","for","from","with",
                   "and","or","that","this","it","as","their","theirs","our","ours",
                   "its"}
_FAVORITE_FOCUS=re.compile(r"\bfavou?rite\s+(.+)$",re.I)
_QUESTION_FOCUS_STOP=_CLAIM_STOP_WORDS|{
    "what","which","who","whose","where","when","why","how","about","tell","me",
    "remind","explain","compare","favorite","favourite","prefer","preferred",
    "preference","current","currently","now","today","right","also","but","by",
    "can","could","should","would","will","one",
}
_NEGATION=re.compile(
    r"\b(?:not|never|no\s+longer|none|nothing|nobody|neither|nor|without|"
    r"hardly|barely|cannot)\b|n['\u2019]t\b",
    re.I,
)
_CLAUSE_BOUNDARY=re.compile(r"[,;.!?]+|\b(?:but|however|instead|yet)\b",re.I)
_CAUSAL_FRAMING_TERMS={
    # These words express a relation already established by an owned,
    # EXPLICIT_CAUSE transition; they are not additional personal facts.
    "because","due","cause","caus","reason","record","recorded",
    "change","chang","transition","result","lead","led",
    "made","make","makes","switch","switched","switching",
    "replace","replaced","replacing","replacement",
}


def _claim_lemma(word):
    """Normalize common regular English inflections conservatively."""
    if len(word)>5 and word.endswith("ies"):
        return word[:-3]+"y"
    if len(word)>5 and word.endswith("ing"):
        stem=word[:-3]
        if len(stem)>2 and stem[-1]==stem[-2]:
            stem=stem[:-1]
        return stem
    if len(word)>4 and word.endswith("ed"):
        stem=word[:-2]
        if stem.endswith("e"):
            return stem
        if len(stem)>2 and stem[-1]==stem[-2]:
            stem=stem[:-1]
        return stem
    if len(word)>4 and word.endswith(("ches","shes","xes","zes")):
        return word[:-2]
    if len(word)>3 and word.endswith("s") and not word.endswith(("ss","us","is")):
        return word[:-1]
    return word


def _claim_terms(text):
    terms=set()
    for word in re.findall(r"[\w'-]+",text or ""):
        folded=word.casefold()
        if folded.endswith(("'s","\u2019s")):
            folded=folded[:-2]
        elif folded.endswith(("'","\u2019")):
            folded=folded[:-1]
        if len(folded)>2 and folded not in _CLAIM_STOP_WORDS:
            terms.add(_claim_lemma(folded))
    return terms


_COMPARISON_RE=re.compile(
    r"\b(?:prefer|prefers|preferred|like|likes|liked|love|loves|loved)\s+"
    r"(?P<left>.+?)\s+"
    r"(?P<comparator>rather\s+than|instead\s+of|more\s+than|less\s+than|over|to)\s+"
    r"(?P<right>[^,.;!?]+?)(?=\s*(?:[.;!?]|$))",re.I)
_COMPARISON_FRAMING={
    "prefer","like","love","lik","lov","rather","than","more","less","over",
    "instead","to","now","currently","today",
}
_HISTORICAL_MARKER=re.compile(
    r"\b(?:used\s+to|formerly|previously|once|in\s+the\s+past|when\s+i\s+was|"
    r"back\s+then|was|were|had|did)\b|"
    r"\b(?:have|has)\s+(?:been|lived|worked|studied|visited|traveled|travelled|"
    r"owned|known|played|done|seen|eaten|made|written|run)\b",re.I)


def _comparison_relation(text):
    match=_COMPARISON_RE.search(text or "")
    if match is None:
        return None
    left=match.group("left").strip()
    right=match.group("right").strip()
    comparator=re.sub(r"\s+"," ",match.group("comparator").casefold())
    preferred,dispreferred=((right,left) if comparator=="less than"
                            else (left,right))
    if not preferred or not dispreferred:
        return None
    return preferred,dispreferred


def _comparison_content(text):
    return _claim_terms(text)-_COMPARISON_FRAMING


def _is_historical(text):
    return bool(_HISTORICAL_MARKER.search(text or ""))


def _lexically_supported(claim,source):
    claim_relation=_comparison_relation(claim)
    source_relation=_comparison_relation(source)
    if claim_relation is not None:
        if source_relation is None or claim_relation!=source_relation:
            return False
        if bool(_NEGATION.search(claim))!=bool(_NEGATION.search(source)):
            return False
        if not _comparison_content(claim).issubset(_comparison_content(source)):
            return False
        return _is_historical(claim)==_is_historical(source)
    if _is_historical(claim)!=_is_historical(source):
        return False
    claim_terms=_claim_terms(claim)
    source_terms=_claim_terms(source)
    if not claim_terms or not claim_terms.issubset(source_terms):
        return False
    def polarities(text):
        return {bool(_NEGATION.search(clause)) for clause in _CLAUSE_BOUNDARY.split(text)
                if claim_terms.issubset(_claim_terms(clause))}
    claim_polarities=polarities(claim)
    source_polarities=polarities(source)
    # A negation elsewhere in a multi-clause source must not reverse the
    # polarity of this claim. Require a clause containing all claim terms on
    # both sides; if the relation cannot be localized, fail closed.
    return bool(claim_polarities and source_polarities
                and claim_polarities.intersection(source_polarities))


def _favorite_focus_terms(question):
    """Extract the noun focus of a "favorite X" question conservatively."""
    match=_FAVORITE_FOCUS.search(question or "")
    if match is None:
        return set()
    focus=[]
    for word in re.findall(r"[\w'-]+",match.group(1)):
        folded=word.casefold()
        if len(word)<=2 or folded in _QUESTION_FOCUS_STOP:
            break
        focus.append(_claim_lemma(folded))
        if len(focus)>=3:
            break
    return set(focus)


def _source_text(value):
    if isinstance(value,str):
        return value
    if isinstance(value,dict):
        return " ".join(str(value.get(key)) for key in
                        ("source_text","endpoint_text","cause_text","from","to","cause")
                        if isinstance(value.get(key),str))
    return ""


def _question_focus_supported(question,claim,source,evidence):
    """Require a requested focus to be tied to the answer, not merely present.

    A lexical match can prove that a cited record says "Java" while doing
    nothing to prove the user was asked about food or movies. For a
    "favorite X" question, X must appear in the cited source/claim, or an
    owned record must connect X to a claim value. This is deliberately
    narrow: questions without that form keep the existing claim checks.
    """
    focus=_favorite_focus_terms(question)
    if not focus:
        return True
    claim_terms=_claim_terms(claim)
    cited_terms=_claim_terms(source)
    if focus.issubset(cited_terms):
        return True
    answer_terms=claim_terms-focus-_QUESTION_FOCUS_STOP
    if not answer_terms:
        return False
    for owned in _owned_sources(evidence).values():
        owned_terms=_claim_terms(_source_text(owned))
        if focus.issubset(owned_terms) and answer_terms.intersection(owned_terms):
            return True
    return False


def _direction_label(value):
    text=re.sub(r"^\s*(?:due\s+to|because\s+of|because)\b[^,;]*[,;]\s*",
                "",str(value or ""),flags=re.I)
    text=re.split(r"\b(?:because|since|due\s+to)\b",text,maxsplit=1,
                  flags=re.I)[0]
    text=re.sub(r"^(?:the|my|a|an)\s+","",text.strip(),flags=re.I)
    return re.sub(r"\s+"," ",text.strip(" .,;:!?")).casefold()


def _direction_labels_match(claim_label,endpoint_label):
    left=_direction_label(claim_label)
    right=_direction_label(endpoint_label)
    if not left or not right:
        return False
    if left==right:
        return True
    if re.fullmatch(r"[\w'-]+",left) and re.search(
            rf"(?<!\w){re.escape(left)}(?!\w)",right):
        return True
    if re.fullmatch(r"[\w'-]+",right) and re.search(
            rf"(?<!\w){re.escape(right)}(?!\w)",left):
        return True
    return False


def _transition_direction_supported(claim,source):
    """Reject a directional claim that reverses or substitutes an endpoint."""
    text=re.sub(r"^\s*(?:due\s+to|because\s+of|because)\b[^,;]*[,;]\s*",
                "",str(claim or ""),flags=re.I)
    text=re.split(r"\b(?:because|since|due\s+to)\b",text,maxsplit=1,
                  flags=re.I)[0]
    from_to=re.search(r"\bfrom\s+(.+?)\s+to\s+(.+?)(?=\s*(?:[.;!?]|$))",
                      text,re.I)
    if from_to:
        source_label=from_to.group(1)
        target_label=from_to.group(2)
        if (_direction_label(source_label)==_direction_label(target_label)):
            return False
        return (_direction_labels_match(source_label,source.get("from",""))
                and _direction_labels_match(target_label,source.get("to","")))
    to_match=re.search(r"\bto\s+(.+?)(?=\s*(?:[.;!?]|$))",text,re.I)
    if to_match:
        label=to_match.group(1)
        return (_direction_labels_match(label,source.get("to",""))
                and not _direction_labels_match(label,source.get("from","")))
    from_match=re.search(r"\bfrom\s+(.+?)(?=\s*(?:[.;!?]|$))",text,re.I)
    if from_match:
        label=from_match.group(1)
        return (_direction_labels_match(label,source.get("from",""))
                and not _direction_labels_match(label,source.get("to","")))
    return True


def _causal_transition_claim_supported(claim,source):
    """Check causal claim terms against the recorded cause and endpoints separately.

    The transition's reason may put a cause and its resulting state change in
    separate comma-delimited clauses. Validate each claim term against the
    endpoint or cause field that records it, retaining polarity checks per
    term, instead of requiring all terms to co-occur in one clause.
    """
    terms=_claim_terms(claim)-_CAUSAL_FRAMING_TERMS
    endpoint_text=source.get("endpoint_text","")
    cause_text=source.get("cause_text","")
    if not terms:
        return False
    if not _transition_direction_supported(claim,source):
        return False
    if not terms.issubset(_claim_terms(endpoint_text+" "+cause_text)):
        return False
    return all(_lexically_supported(term,endpoint_text)
               or _lexically_supported(term,cause_text)
               for term in terms)


_TEMPORAL_CLAIM_WORDS={"as","at","before","after","earlier","historical",
                       "history","previous","previously","then","time","date",
                       "change","changed","changes","chang","replacement","replaced",
                       "current","currently","now","today","right","switch",
                       "switched","switching","transition","transitioned","record",
                       "recorded","relation","tell","told","mention","mentioned",
                       "about","move","moved","moves","moving","mov",
                       "relocate","relocated","reloc","shift","shifted",
                       "complete","completed","complet","finish","finished",
                       "live","lives","living","lived","liv","use","used","reside","resides",
                       "residence","resid"}
_TEMPORAL_EXPRESSION=re.compile(
    r"\b\d{4}-\d{2}-\d{2}(?:[tT]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
    r"(?:[zZ]|[+-]\d{2}:?\d{2})?)?\b"
    r"|\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\b\.?\s*\d{1,2}(?:st|nd|rd|th)?"
    r"(?:,?\s*(?:19|20)\d{2})?"
    r"|\b(?:19|20)\d{2}\b"
    r"|\b\d{1,2}(?::\d{2}){1,2}(?:\.\d+)?(?:\s*(?:am|pm))?"
    r"(?:\s*(?:utc|gmt|z))?\b"
    r"|\b(?:utc|gmt)\b",re.I)


def _temporal_transition_claim_supported(claim,source):
    """Validate a source/destination statement without requiring a cause."""
    terms=_claim_terms(claim)-_TEMPORAL_CLAIM_WORDS
    endpoint_text=source.get("endpoint_text","")
    if not terms or not terms.issubset(_claim_terms(endpoint_text)):
        return False
    if not _transition_direction_supported(claim,source):
        return False
    return all(_lexically_supported(term,endpoint_text) for term in terms)


def _programming_preference_transition_supported(question,claim,source,evidence):
    """Validate a specifically typed programming-preference supersession edge."""
    question_text=str(question or "")
    if (not re.search(r"\b(?:programming|coding)\b",question_text,re.I)
            or not re.search(r"\b(?:prefer(?:ence|ences|red|s)?)\b",question_text,re.I)
            or re.search(r"\b(?:because|since|due\s+to|cause|reason)\b",claim,re.I)):
        return False
    # Resolve the transition's endpoint IDs from the same user's typed states.
    if source.get("relation_kind")!="TEMPORAL_SUCCESSION":
        return False
    from_to=re.search(r"\bfrom\s+(.+?)\s+to\s+(.+?)(?=\s*(?:[.;!?]|$))",claim,re.I)
    if not from_to:
        return False
    if (not _direction_labels_match(from_to.group(1),source.get("from",""))
            or not _direction_labels_match(from_to.group(2),source.get("to",""))
            or _direction_label(source.get("from"))==_direction_label(source.get("to"))):
        return False
    if not re.search(r"\b(?:changed|change|switched|replaced|shifted)\b",claim,re.I):
        return False
    terms=_claim_terms(claim)-_TEMPORAL_CLAIM_WORDS
    endpoints=_claim_terms(str(source.get("from") or "")+" "+str(source.get("to") or ""))
    if not terms or not endpoints.issubset(terms):
        return False
    allowed_domain_terms={"programming","program","language","preference","prefer"}
    if terms-endpoints-allowed_domain_terms:
        return False
    states=[state for state in evidence.get("states",[])
            if isinstance(state,dict) and state.get("user_id")==evidence.get("user_id")
            and state.get("state_type")=="PREFERENCE"]
    source_states=[state for state in states
                   if state.get("state_value")==source.get("from")]
    destination_states=[state for state in states
                        if state.get("state_value")==source.get("to")]
    if len(source_states)!=1 or len(destination_states)!=1:
        return False
    edge_matches=[edge for edge in evidence.get("transitions",[])
                  if isinstance(edge,dict)
                  and edge.get("user_id")==evidence.get("user_id")
                  and edge.get("from_state_id")==source_states[0].get("state_id")
                  and edge.get("to_state_id")==destination_states[0].get("state_id")
                  and edge.get("transition_type")=="PREFERENCE_CHANGE"
                  and edge.get("relation_kind")=="TEMPORAL_SUCCESSION"]
    if len(edge_matches)!=1:
        return False
    source_key=(source_states[0].get("metadata") or {}).get("preference_key")
    destination_key=(destination_states[0].get("metadata") or {}).get("preference_key")
    valid_keys={"favorite:programming","favorite:programming language",
                "domain:programming","domain:programming language"}
    if source_key not in valid_keys or destination_key not in valid_keys:
        return False
    source_end=source_states[0].get("valid_until")
    destination_start=destination_states[0].get("valid_from")
    try:
        parsed_end=datetime.fromisoformat(str(source_end).replace("Z","+00:00"))
        parsed_start=datetime.fromisoformat(str(destination_start).replace("Z","+00:00"))
        transition_time=datetime.fromisoformat(
            str(edge_matches[0].get("timestamp")).replace("Z","+00:00"))
    except (TypeError,ValueError):
        return False
    return (source_states[0].get("status")=="INACTIVE"
            and source_end is not None and parsed_end==parsed_start
            and transition_time>=parsed_end)


def _focused_current_state_claim_terms(question,claim,state,evidence):
    """Drop query framing from a claim about one focused current state.

    A present-tense typed question supplies the requested state type and the
    predicate it is asking about (for example, "what activity ... enjoy"). The
    authoritative answer is the selected state's value; words the claim merely
    repeats from the question are not additional personal facts. Remaining
    claim terms must still be lexically supported by the state or its linked
    memories. This path applies only to a unique focused current-state
    candidate, never to historical, predecessor, or causal evidence.
    """
    if str(evidence.get("query_category","")).upper()!="STATE":
        return None
    focus=evidence.get("state_focus")
    candidate_ids=(focus.get("candidate_state_ids")
                   if isinstance(focus,dict) else None)
    if not isinstance(candidate_ids,list):
        requested_types=requested_state_types(question)
        if not requested_types:
            return None
        candidate_ids=[item.get("state_id")
                       for item in evidence.get("states",[])
                       if isinstance(item,dict)
                       and item.get("state_type") in requested_types]
    if (len(candidate_ids)!=1
            or candidate_ids[0]!=state.get("state_id")):
        return None
    claim_terms=(_claim_terms(_TEMPORAL_EXPRESSION.sub(" ",claim or ""))
                 -_TEMPORAL_CLAIM_WORDS)
    question_terms=_claim_terms(question)-_TEMPORAL_CLAIM_WORDS
    return claim_terms-question_terms


def _temporal_state_claim_supported(question,claim,reference_kind,reference_id,evidence,
                                    diagnostics=None):
    """Require an owned, linked state valid as-of or at a predecessor boundary."""
    from temporal_engine import parse_requested_timestamp

    details={
        "requested_timestamp_normalized":False,
        "state_reference_resolved":False,
        "state_validity_timestamps_normalized":False,
        "half_open_interval_passed":False,
        "predecessor_boundary_passed":False,
        "lexical_support_passed":False,
        "state_label_value_lexical_support_passed":False,
        "linked_memory_lexical_support_passed":False,
        "combined_lexical_support_passed":False,
        "both_sources_required_for_lexical_support":False,
        "neither_source_supports_lexically":True,
        "final_temporal_validation_passed":False,
        "reason_codes":[],
    }

    def add_reason(code):
        if code not in details["reason_codes"]:
            details["reason_codes"].append(code)

    def finish(result):
        details["final_temporal_validation_passed"]=bool(result)
        if result:
            add_reason("temporal_validation_passed")
        if diagnostics is not None:
            diagnostics.update(details)
        return bool(result)

    user_id=evidence.get("user_id")
    states={state.get("state_id"):state for state in evidence.get("states",[])
            if isinstance(state,dict) and state.get("user_id")==user_id
            and isinstance(state.get("state_id"),str)}
    owned_sources=_owned_sources(evidence)
    if reference_kind=="state":
        state=states.get(reference_id)
    elif reference_kind=="memory":
        owned_memory_ids={key[1] for key in owned_sources if key[0]=="memory"}
        linked=[item for item in states.values()
                if reference_id in (item.get("source_memory_ids") or [])]
        state=linked[0] if reference_id in owned_memory_ids and len(linked)==1 else None
    else:
        state=None
    details["state_reference_resolved"]=state is not None
    if state is None:
        add_reason("state_reference_unresolved_or_unowned")
    state_id=state["state_id"] if state is not None else None

    requested=(evidence.get("requested_timestamp")
               or (evidence.get("temporal_context") or {}).get("requested_timestamp"))
    implicit_current=(requested is None
                      and str(evidence.get("query_category","")).upper()=="STATE")
    instant=None
    if requested is None and implicit_current:
        # A current snapshot is evidence as of now. Treat temporal framing such
        # as "currently" as query context, but still verify the half-open
        # interval and citation ownership against this instant.
        instant=datetime.now(timezone.utc)
        details["requested_timestamp_normalized"]=True
        details["implicit_current_timestamp"]=True
    elif requested is None:
        add_reason("requested_timestamp_missing")
    else:
        try:
            instant=parse_requested_timestamp(requested)
            details["requested_timestamp_normalized"]=True
        except (TypeError,ValueError):
            add_reason("requested_timestamp_invalid")

    valid=False
    if state is not None and instant is not None:
        try:
            valid_from=parse_requested_timestamp(state.get("valid_from"))
            valid_until=(parse_requested_timestamp(state["valid_until"])
                         if state.get("valid_until") is not None else None)
            details["state_validity_timestamps_normalized"]=True
            valid=valid_from<=instant and (valid_until is None or instant<valid_until)
            if implicit_current and state.get("status") not in {None,"ACTIVE"}:
                valid=False
                add_reason("state_not_active_at_current_timestamp")
        except (TypeError,ValueError):
            add_reason("state_validity_timestamp_invalid")
    details["half_open_interval_passed"]=valid
    if state is not None and instant is not None and not details["state_validity_timestamps_normalized"]:
        add_reason("state_validity_timestamp_invalid")
    if state is not None and instant is not None and details["state_validity_timestamps_normalized"] and not valid:
        add_reason("state_interval_not_valid_at_requested_timestamp")

    predecessor_valid=False
    if (state is not None and not valid and is_predecessor_query(question)
            and "Timeline" in evidence.get("retrieval_sources",[])):
        for edge in evidence.get("transitions",[]):
            if (not isinstance(edge,dict) or edge.get("user_id")!=user_id
                    or edge.get("from_state_id")!=state_id
                    or edge.get("relation_kind") not in {"TEMPORAL_SUCCESSION","EXPLICIT_CAUSE"}):
                continue
            destination=states.get(edge.get("to_state_id"))
            try:
                boundary=parse_requested_timestamp(edge.get("timestamp"))
                valid_until=parse_requested_timestamp(state.get("valid_until"))
            except (TypeError,ValueError):
                continue
            if destination is None:
                linked=valid_until==boundary
            else:
                try:
                    destination_start=parse_requested_timestamp(destination.get("valid_from"))
                except (TypeError,ValueError):
                    continue
                linked=valid_until==boundary==destination_start
            if linked:
                predecessor_valid=True
                break
        if not predecessor_valid:
            # A legacy row may lack exact adjacent interval timestamps. Accept
            # only when retrieval supplied one unique, owned transition whose
            # source is this state; competing or unlinked edges still fail.
            linked=[edge for edge in evidence.get("transitions",[])
                    if isinstance(edge,dict) and edge.get("user_id")==user_id
                    and edge.get("from_state_id")==state_id
                    and edge.get("relation_kind") in {"TEMPORAL_SUCCESSION","EXPLICIT_CAUSE"}]
            if len(linked)==1:
                predecessor_valid=True
                add_reason("predecessor_transition_link_accepted")
    details["predecessor_boundary_passed"]=predecessor_valid
    if (state is not None and not valid and is_predecessor_query(question)
            and "Timeline" in evidence.get("retrieval_sources",[]) and not predecessor_valid):
        add_reason("predecessor_boundary_not_verified")

    lexical_supported=False
    if state is not None:
        # Date/time framing is validated against the requested instant and the
        # state interval, not against the state's factual text.
        terms=(_claim_terms(_TEMPORAL_EXPRESSION.sub(" ",claim or ""))
               -_TEMPORAL_CLAIM_WORDS)
        focused_terms=_focused_current_state_claim_terms(question,claim,state,evidence)
        if focused_terms is not None:
            terms=focused_terms
        state_type=state.get("state_type")
        state_source=(str(state_type).replace("_"," ")+" " if isinstance(state_type,str) else "")
        state_source+=state.get("state_value","")
        # State values are normalized summaries. Let only this state's explicitly
        # linked, same-user Mem0 records supply additional lexical support.
        source_memory_ids=state.get("source_memory_ids")
        linked_text=[]
        if isinstance(source_memory_ids,list):
            linked_text=[owned_sources[("memory",memory_id)]
                         for memory_id in source_memory_ids
                         if isinstance(memory_id,str) and ("memory",memory_id) in owned_sources]
        claim_terms_text=" ".join(sorted(terms))
        state_supported=bool(terms) and _lexically_supported(claim_terms_text,state_source)
        linked_source=" ".join(linked_text)
        linked_supported=(bool(terms) and bool(linked_text)
                          and _lexically_supported(claim_terms_text,linked_source))
        combined_source=state_source+" "+linked_source
        lexical_supported=(bool(terms)
                           and _lexically_supported(claim_terms_text,combined_source))
        details["state_label_value_lexical_support_passed"]=state_supported
        details["linked_memory_lexical_support_passed"]=linked_supported
        details["combined_lexical_support_passed"]=lexical_supported
        details["both_sources_required_for_lexical_support"]=(
            lexical_supported and not state_supported and not linked_supported)
        details["neither_source_supports_lexically"]=not lexical_supported
        details["lexical_support_passed"]=lexical_supported
        if not lexical_supported:
            add_reason("temporal_lexical_support_failed")
    return finish(state is not None and (valid or predecessor_valid) and lexical_supported)


def _programming_preference_history_representatives(question,evidence):
    if (not re.search(r"\b(?:programming|coding)\b",question,re.I)
            or not re.search(r"\b(?:prefer(?:ence|ences|red|s)?)\b",question,re.I)
            or not re.search(r"\b(?:past|history|historical|previously|earlier)\b",
                             question,re.I)):
        return {}
    allowed_keys={"favorite:programming","favorite:programming language",
                  "domain:programming","domain:programming language"}
    representatives={}
    for state in evidence.get("states",[]):
        if (not isinstance(state,dict)
                or state.get("user_id")!=evidence.get("user_id")
                or state.get("state_type")!="PREFERENCE"
                or state.get("status")!="INACTIVE"
                or (state.get("metadata") or {}).get("preference_key") not in allowed_keys
                or not isinstance(state.get("state_value"),str)
                or not state["state_value"].strip()):
            continue
        value=re.sub(r"\s+"," ",state["state_value"].strip()).casefold()
        previous=representatives.get(value)
        if previous is None:
            representatives[value]=state
            continue
        try:
            current_time=datetime.fromisoformat(
                str(state.get("valid_from")).replace("Z","+00:00"))
            previous_time=datetime.fromisoformat(
                str(previous.get("valid_from")).replace("Z","+00:00"))
        except (TypeError,ValueError):
            continue
        if current_time>previous_time:
            representatives[value]=state
    return representatives


def _validate_claims(question,evidence,response,rejection_counts=None,detail_counts=None,
                    rejected_claims=None,temporal_diagnostics=None,model_evidence=None,
                    citation_diagnostics=None,citation_refs=None):
    """Return claim text plus only validated, actual same-user source IDs."""
    if not isinstance(response,dict) or not isinstance(response.get("claims"),list):
        return []
    def rejected(reason):
        if rejection_counts is not None:
            rejection_counts[reason]=rejection_counts.get(reason,0)+1
    def detailed(reason):
        if detail_counts is not None:
            detail_counts[reason]=detail_counts.get(reason,0)+1

    sources=_owned_sources(evidence)
    model_citation_keys=_model_citation_keys(
        evidence if model_evidence is None else model_evidence)
    accepted=[]

    def programming_history_representatives():
        return _programming_preference_history_representatives(question,evidence)

    def programming_history_claim(text,refs,kind):
        if (kind!="fact" or not isinstance(text,str)
                or not re.search(r"\b(?:previously|past|historical|earlier|history)\b",
                                 text,re.I)):
            return False
        representatives=programming_history_representatives()
        if not representatives:
            return False
        values={value:state["state_value"]
                for value,state in representatives.items()}
        folded=text.casefold()
        if (not re.search(r"\bprogram(?:ming)?\s+preference\b|"
                          r"\bfor\s+programming\b",text,re.I)
                or any(not re.search(rf"(?<!\w){re.escape(value)}(?!\w)",folded)
                       for value in values)):
            return False
        resolved=[]
        for reference in refs:
            item,reason=_resolve_reference_token(reference,citation_refs)
            if (reason is not None or not isinstance(item,dict)
                    or item.get("kind")!="state"):
                return False
            resolved.append(item.get("id"))
        required={state.get("state_id") for state in representatives.values()}
        return (len(resolved)==len(required) and set(resolved)==required
                and all(("state",identifier) in model_citation_keys
                        for identifier in required))

    def stored_conflict_claim(text,refs,kind):
        """Require a conflict summary to cite all same-domain recorded sources."""
        if (kind!="fact" or not isinstance(text,str)
                or not re.search(r"\b(?:stored|recorded|saved|mentioned)\b[^?]*\bpreferences?\b|"
                                 r"\bpreferences?\b[^?]*\b(?:stored|recorded|saved)\b",
                                 question,re.I)
                or not re.search(r"\b(?:conflict(?:s|ing)?|contradict(?:s|ory|ing)?|disagree)\b",
                                 text,re.I)):
            return False
        conflict=evidence.get("preference_conflict")
        state_ids=conflict.get("state_ids") if isinstance(conflict,dict) else None
        memory_ids=conflict.get("memory_ids") if isinstance(conflict,dict) else None
        if not isinstance(state_ids,list) or not isinstance(memory_ids,list):
            return False
        if (len(set(state_ids))!=len(state_ids)
                or len(set(memory_ids))!=len(memory_ids)
                or len(state_ids)+len(memory_ids)<2):
            return False
        states={state.get("state_id"):state for state in evidence.get("states",[])
                if isinstance(state,dict) and state.get("user_id")==evidence.get("user_id")
                and state.get("state_type")=="PREFERENCE" and state.get("status")=="ACTIVE"}
        selected=[states.get(identifier) for identifier in state_ids]
        if any(state is None for state in selected):
            return False
        memory_values=conflict.get("memory_values")
        if not isinstance(memory_values,dict) or set(memory_values)!=set(memory_ids):
            return False
        source_values=[]
        for identifier in memory_ids:
            source=sources.get(("memory",identifier))
            value=memory_values.get(identifier)
            if not isinstance(source,str) or not isinstance(value,str) or not value.strip():
                return False
            match=re.search(
                r"\b(?:i|user)\s+(?:now\s+)?prefers?\s+"
                r"(?P<value>[\w+#.-]+(?:\s+[\w+#.-]+){0,2})\s+"
                r"(?:for|as)\s+(?P<domain>[^,.;!?]+)",source,re.I)
            if (not match
                    or re.sub(r"\s+"," ",match.group("value")).strip().casefold()
                       !=re.sub(r"\s+"," ",value).strip().casefold()
                    or str(conflict.get("domain") or "").casefold()
                       not in match.group("domain").casefold()):
                return False
            source_values.append(value)
        values=[state.get("state_value") for state in selected]+source_values
        normalized={re.sub(r"\s+"," ",value.strip()).casefold()
                    for value in values if isinstance(value,str) and value.strip()}
        if len(normalized)<2:
            return False
        conflict_values=conflict.get("values")
        if (not isinstance(conflict_values,list)
                or {value.casefold() for value in conflict_values
                    if isinstance(value,str)}!=normalized):
            return False
        text_folded=text.casefold()
        if any(not re.search(rf"(?<!\w){re.escape(value.casefold())}(?!\w)",text_folded)
               for value in values):
            return False
        domain=str(conflict.get("domain") or "").casefold()
        if domain and domain not in text_folded:
            return False
        resolved=[]
        for reference in refs:
            item,reason=_resolve_reference_token(reference,citation_refs)
            if (reason is not None or not isinstance(item,dict)
                    or item.get("kind") not in {"state","memory"}):
                return False
            resolved.append((item["kind"],item.get("id")))
        required={("state",identifier) for identifier in state_ids}
        required.update(("memory",identifier) for identifier in memory_ids)
        ok=(len(resolved)==len(required) and set(resolved)==required
            and required.issubset(model_citation_keys))
        return ok

    def memory_fallback_claim(text,kind,ref_kind,source_id,source):
        """Allow exact legacy source text only in an explicit typed fallback."""
        focus=evidence.get("state_focus")
        if (kind!="fact" or ref_kind!="memory" or not isinstance(focus,dict)
                or not focus.get("memory_fallback") or source_id not in
                set(focus.get("source_memory_ids",[])) or not isinstance(source,str)
                or text.strip()!=source.strip()):
            return False
        requested=set(focus.get("state_types",[]))
        if "GOAL" in requested:
            explicit_goal=bool(re.search(
                r"\b(?:goal|plan|trying\s+to|planning\s+to|working\s+to|aiming\s+to)\b",
                source,re.I))
            current_request=bool(re.search(
                r"\b(?:current(?:ly)?|now|right\s+now|today)\b",question,re.I))
            current_record=bool(re.search(
                r"\b(?:new|current|now|currently|trying|planning|working|aiming)\b",
                source,re.I))
            return explicit_goal and (not current_request or current_record)
        if "LIFE_EVENT" in requested:
            return bool(re.search(r"\b(?:complete\w*|finish\w*|achiev\w*)\b",source,re.I)
                        and re.search(r"\bproject\b",question,re.I)
                        and re.search(r"\bproject\b",source,re.I))
        return False
    def report_rejected_claim(index,claim_type,reference_diagnostics,reasons):
        if rejected_claims is None:
            return
        codes=sorted(set(reasons)) or ["no_valid_evidence_reference"]
        rejected_claims.append({
            "claim_index":index,
            "claim_type":claim_type if isinstance(claim_type,str) and claim_type in {"fact","temporal","causal"} else None,
            "rejection_reasons":codes,
            "evidence_references":reference_diagnostics,
        })

    for claim_index,claim in enumerate(response["claims"]):
        if not isinstance(claim,dict):
            rejected("malformed_claim")
            detailed("claim_not_object")
            report_rejected_claim(claim_index,None,[],["claim_not_object"])
            continue
        text=claim.get("text")
        kind=claim.get("claim_type")
        reported_kind=kind
        refs=claim.get("evidence")
        malformed=[]
        if "text" not in claim:
            malformed.append("claim_text_missing")
        elif not isinstance(text,str):
            malformed.append("claim_text_not_string")
        elif not text.strip():
            malformed.append("claim_text_empty")
        if "claim_type" not in claim:
            malformed.append("claim_type_missing")
        elif not isinstance(kind,str) or kind not in {"fact","temporal","causal"}:
            malformed.append("claim_type_invalid")
        if "evidence" not in claim:
            malformed.append("claim_evidence_missing")
        elif not isinstance(refs,list):
            malformed.append("claim_evidence_not_list")
        if malformed:
            rejected("malformed_claim")
            for reason in malformed:
                detailed(reason)
            reference_diagnostics=[]
            if isinstance(refs,list):
                for ref in refs:
                    if citation_diagnostics is not None:
                        citation_diagnostics.append(
                            _citation_validation_diagnostic(ref,evidence,
                                evidence if model_evidence is None else model_evidence))
                    item={"evidence_id_resolved":False,"rejection_reason":malformed[0]}
                    if not isinstance(ref,dict):
                        item["rejection_reason"]="evidence_reference_not_object"
                    else:
                        ref_kind=ref.get("kind")
                        if isinstance(ref_kind,str) and ref_kind in {"memory","state","transition"}:
                            item["kind"]=ref_kind
                            ref_id=ref.get("id")
                            if isinstance(ref_id,str) and ref_id.strip():
                                item["evidence_id_resolved"]=(ref_kind,ref_id) in sources
                            elif "id" not in ref:
                                item["rejection_reason"]="evidence_reference_id_missing"
                            elif not isinstance(ref_id,str):
                                item["rejection_reason"]="evidence_reference_id_not_string"
                            else:
                                item["rejection_reason"]="evidence_reference_id_empty"
                        elif "kind" not in ref:
                            item["rejection_reason"]="evidence_reference_kind_missing"
                        else:
                            item["rejection_reason"]="evidence_reference_kind_invalid"
                    reference_diagnostics.append(item)
            report_rejected_claim(claim_index,reported_kind,reference_diagnostics,
                malformed+[item["rejection_reason"] for item in reference_diagnostics
                           if item.get("rejection_reason")])
            continue
        causal_text=bool(_CAUSAL_CLAIM.search(text))
        if causal_text:
            kind="causal"
        group_supported=(stored_conflict_claim(text,refs,kind)
                         or programming_history_claim(text,refs,kind))
        validated=[]
        claim_rejections=set()
        claim_details=set()
        reference_diagnostics=[]
        for ref in refs:
            ref,resolution_reason=_resolve_reference_token(ref,citation_refs)
            if resolution_reason is not None:
                provided_kind=("invalid"
                    if not isinstance(ref,dict)
                    or not isinstance(ref.get("kind"),str)
                    or ref.get("kind") not in {"memory","state","transition"}
                    else ref["kind"])
                if resolution_reason=="evidence_reference_id_missing":
                    claim_rejections.add("malformed_evidence_reference")
                    claim_details.add(resolution_reason)
                else:
                    claim_rejections.add("unknown_or_unowned_evidence_id")
                    claim_details.add(resolution_reason)
                reference_diagnostics.append({"evidence_id_resolved":False,
                    "rejection_reason":resolution_reason})
                if citation_diagnostics is not None:
                    citation_diagnostics.append({
                        "citation_kind":provided_kind,
                        "evidence_id_resolves_owned":False,
                        "kind_matches_resolved_type":None,
                        "reason_code":(resolution_reason
                            if resolution_reason in _UI_REJECTION_REASON_CODES
                            else "citation_structure_malformed"),
                    })
                continue
            citation_info=_citation_validation_diagnostic(
                ref,evidence,evidence if model_evidence is None else model_evidence)
            if citation_diagnostics is not None:
                citation_diagnostics.append(citation_info)
            if not isinstance(ref,dict):
                claim_rejections.add("malformed_evidence_reference")
                claim_details.add("evidence_reference_not_object")
                reference_diagnostics.append({"evidence_id_resolved":False,
                    "rejection_reason":"evidence_reference_not_object"})
                continue
            if "kind" not in ref:
                claim_rejections.add("malformed_evidence_reference")
                claim_details.add("evidence_reference_kind_missing")
                reference_diagnostics.append({"evidence_id_resolved":False,
                    "rejection_reason":"evidence_reference_kind_missing"})
                continue
            if (not isinstance(ref.get("kind"),str)
                    or ref.get("kind") not in {"memory","state","transition"}):
                claim_rejections.add("malformed_evidence_reference")
                claim_details.add("evidence_reference_kind_invalid")
                reference_diagnostics.append({"evidence_id_resolved":False,
                    "rejection_reason":"evidence_reference_kind_invalid"})
                continue
            if "id" not in ref:
                claim_rejections.add("malformed_evidence_reference")
                claim_details.add("evidence_reference_id_missing")
                reference_diagnostics.append({"kind":ref["kind"],"evidence_id_resolved":False,
                    "rejection_reason":"evidence_reference_id_missing"})
                continue
            if not isinstance(ref.get("id"),str):
                claim_rejections.add("malformed_evidence_reference")
                claim_details.add("evidence_reference_id_not_string")
                reference_diagnostics.append({"kind":ref["kind"],"evidence_id_resolved":False,
                    "rejection_reason":"evidence_reference_id_not_string"})
                continue
            if not ref["id"].strip():
                claim_rejections.add("malformed_evidence_reference")
                claim_details.add("evidence_reference_id_empty")
                reference_diagnostics.append({"kind":ref["kind"],"evidence_id_resolved":False,
                    "rejection_reason":"evidence_reference_id_empty"})
                continue
            source_id=ref.get("id")
            source=sources.get((ref["kind"],source_id))
            reference_diag={"kind":ref["kind"],"evidence_id_resolved":source is not None}
            if source is None:
                if ref["kind"] in {"memory","state"} and kind in {"fact","temporal"}:
                    temporal_check={}
                    _temporal_state_claim_supported(question,text,ref["kind"],source_id,
                                                    evidence,temporal_check)
                    if temporal_diagnostics is not None:
                        temporal_diagnostics.append(temporal_check)
                claim_rejections.add("unknown_or_unowned_evidence_id")
                reference_diag["rejection_reason"]="unknown_or_unowned_evidence_id"
                reference_diagnostics.append(reference_diag)
                continue
            if (ref["kind"],source_id) not in model_citation_keys:
                claim_rejections.add("evidence_not_in_model_bundle")
                reference_diag["rejection_reason"]="evidence_not_in_model_bundle"
                reference_diagnostics.append(reference_diag)
                detailed("evidence_not_in_model_bundle")
                continue
            if ref["kind"]=="memory" or ref["kind"]=="state":
                lexical_supported=(kind=="fact" and _lexically_supported(text,source))
                memory_fallback_supported=memory_fallback_claim(
                    text,kind,ref["kind"],source_id,source)
                temporal_supported=False
                if kind in {"fact","temporal"}:
                    temporal_check={}
                    temporal_supported=_temporal_state_claim_supported(
                        question,text,ref["kind"],source_id,evidence,temporal_check)
                    if temporal_diagnostics is not None:
                        temporal_diagnostics.append(temporal_check)
                conflict_query=(bool(re.search(
                    r"\b(?:stored|recorded|saved|mentioned)\b[^?]*\bpreferences?\b|"
                    r"\bpreferences?\b[^?]*\b(?:stored|recorded|saved)\b",
                    question,re.I))
                    and isinstance(evidence.get("preference_conflict"),dict))
                supported=(group_supported or memory_fallback_supported
                           or (not conflict_query
                               and (lexical_supported or temporal_supported)))
                if (supported and kind=="fact"
                        and not group_supported and not memory_fallback_supported
                        and not _question_focus_supported(question,text,source,evidence)):
                    claim_rejections.add("claim_not_supported_by_source")
                    claim_details.add("claim_not_relevant_to_question")
                    reference_diag["rejection_reason"]="claim_not_relevant_to_question"
                    reference_diagnostics.append(reference_diag)
                    continue
                if not supported:
                    claim_rejections.add("claim_not_supported_by_source")
                    if kind=="fact":
                        reason="fact_lexical_support_failed"
                    elif kind=="temporal":
                        reason="temporal_validity_failed"
                    else:
                        reason="state_memory_claim_type_not_supported"
                    reference_diag["rejection_reason"]=reason
                    reference_diagnostics.append(reference_diag)
                    continue
            else:
                relation=source["relation_kind"]
                if kind=="causal":
                    claim_terms=_claim_terms(text)
                    cause_terms=_claim_terms(source.get("cause_anchor",""))
                    endpoint_terms=_claim_terms(source["from"]+" "+source["to"])
                    grounded_cause_terms=cause_terms-endpoint_terms
                    if relation!="EXPLICIT_CAUSE":
                        causal_reason="causal_relation_not_explicit"
                    elif not source["cause"]:
                        causal_reason="causal_recorded_cause_missing"
                    elif not _has_supported_cause(question,evidence):
                        causal_reason="causal_query_support_missing"
                    elif not claim_terms.intersection(grounded_cause_terms):
                        causal_reason="causal_claim_missing_cause_terms"
                    elif not _causal_transition_claim_supported(text,source):
                        causal_reason="causal_claim_lexical_support_failed"
                    else:
                        causal_reason=None
                    if causal_reason is not None:
                        claim_rejections.add("causal_relation_or_support_rejected")
                        reference_diag["rejection_reason"]=causal_reason
                        reference_diagnostics.append(reference_diag)
                        continue
                elif kind=="temporal":
                    if (relation not in {"TEMPORAL_SUCCESSION","EXPLICIT_CAUSE"}
                            or causal_text
                            or not (_temporal_transition_claim_supported(text,source)
                                    or _programming_preference_transition_supported(
                                        question,text,source,evidence))):
                        claim_rejections.add("temporal_relation_or_support_rejected")
                        reference_diag["rejection_reason"]="temporal_relation_or_support_rejected"
                        reference_diagnostics.append(reference_diag)
                        continue
                else:
                    claim_rejections.add("claim_type_not_supported_by_transition")
                    reference_diag["rejection_reason"]="claim_type_not_supported_by_transition"
                    reference_diagnostics.append(reference_diag)
                    continue
            reference_diag["rejection_reason"]=None
            reference_diagnostics.append(reference_diag)
            item={"kind":ref["kind"],"id":source_id}
            if item not in validated:
                validated.append(item)
        for reason in claim_details:
            detailed(reason)
        if validated:
            accepted.append({"text":text.strip(),"claim_type":kind,"evidence":validated})
        else:
            if not claim_rejections:
                claim_rejections.add("no_valid_evidence_reference")
        # Record per-reference failures even when another citation in the same
        # claim is valid. This keeps mixed valid/invalid citations observable
        # without rendering or citing the rejected record.
        for reason in claim_rejections:
            rejected(reason)
        if not validated:
            specific_reasons=[item["rejection_reason"] for item in reference_diagnostics
                              if item.get("rejection_reason")]
            report_rejected_claim(claim_index,reported_kind,reference_diagnostics,
                                  specific_reasons or claim_rejections)
    return accepted


def _render_claims(claims):
    def natural_text(claim):
        text=claim["text"].strip()
        if claim.get("claim_type")=="causal":
            text=re.sub(
                r"^the recorded reason for (?:the )?change was\s+",
                "The change happened because of ",text,count=1,flags=re.I)
        return text

    return " ".join(
        f'{natural_text(claim)} [Sources: '+", ".join(
            f'{ref["kind"]}:{ref["id"]}' for ref in claim["evidence"])+"]"
        for claim in claims)


def _render_natural_causal_claims(claims,evidence):
    """Render validated causal evidence with its grounded endpoints and cause."""
    user_id=evidence.get("user_id")
    states={state.get("state_id"):state for state in evidence.get("states",[])
            if isinstance(state,dict) and state.get("user_id")==user_id}
    transitions={edge.get("transition_id"):edge for edge in evidence.get("transitions",[])
                 if isinstance(edge,dict) and edge.get("user_id")==user_id
                 and edge.get("relation_kind")=="EXPLICIT_CAUSE"}
    paths={(path.get("from_state_id"),path.get("to_state_id")):path
           for path in evidence.get("causal_path",[])
           if isinstance(path,dict) and path.get("user_id")==user_id
           and path.get("relation_kind")=="EXPLICIT_CAUSE"}
    for claim in claims:
        if claim.get("claim_type")!="causal":
            continue
        for reference in claim.get("evidence",[]):
            if not isinstance(reference,dict) or reference.get("kind")!="transition":
                continue
            edge=transitions.get(reference.get("id"))
            if edge is None:
                continue
            source=states.get(edge.get("from_state_id"))
            destination=states.get(edge.get("to_state_id"))
            path=paths.get((edge.get("from_state_id"),edge.get("to_state_id")),{})
            cause=edge.get("trigger") or ""
            reason=edge.get("reason") or ""
            reason_cause=re.search(
                r"\b(?:because(?:\s+of)?|due\s+to)\s+(.+?)(?:[,;.!?]|$)",
                reason,re.I)
            if reason_cause and len(reason_cause.group(1).strip())>len(cause.strip()):
                cause=reason_cause.group(1).strip()
            if not cause:
                cause=reason
            source_value=(source.get("state_value") if source else path.get("from"))
            destination_value=(destination.get("state_value") if destination else path.get("to"))
            if ((source is None and not path) or (destination is None and not path)
                    or (source is not None and source.get("user_id")!=user_id)
                    or (destination is not None and destination.get("user_id")!=user_id)
                    or not isinstance(source_value,str)
                    or not isinstance(destination_value,str)
                    or not isinstance(cause,str) or not cause.strip()):
                continue
            if (source_value.casefold() in claim.get("text","").casefold()
                    and destination_value.casefold() in claim.get("text","").casefold()):
                # The fully validated model wording already states both
                # endpoints; retain its wording and its exact claim citation.
                return None
            cause=re.sub(r"^(?:because\s+of|due\s+to|because)\s+","",cause.strip(),flags=re.I)
            source_value=source_value.strip()
            destination_value=destination_value.strip()
            source_type=source.get("state_type") if source else None
            destination_type=destination.get("state_type") if destination else None
            if source_type is None and destination_type is None and path:
                source_type=path.get("from_type")
                destination_type=path.get("to_type")
            verb="moved" if source_type==destination_type=="LOCATION" else "switched"
            causal_clause=bool(re.search(
                r"\b(?:cause[sd]?|made|led|resulted|forced|required|prompted)\b",
                cause,re.I))
            causal_phrase=("because " if causal_clause else "because of ")+cause.rstrip(" .;:!?")
            text=(f"You {verb} from {source_value} to {destination_value} "
                  f"{causal_phrase}.")
            citations=", ".join(f'{ref["kind"]}:{ref["id"]}'
                                 for ref in claim.get("evidence",[])
                                 if isinstance(ref,dict) and ref.get("kind") in
                                 {"memory","state","transition"})
            return f"{text} [Sources: {citations}]" if citations else None
    return None


def generate_answer_with_diagnostics(question, evidence):
    """Generate the usual answer plus text-free validation diagnostics.

    Diagnostics contain counts, safe structural fields, and reason codes only.
    They do not retain claim text, evidence IDs/content, prompts, or raw output.
    """
    started=time.perf_counter()
    timings={"model_ms":0.0,"validation_ms":0.0,"render_ms":0.0}
    diagnostics={"model_response_status":"not_called","proposed_claim_count":None,
                 "accepted_claim_count":0,"rejected_claim_count":0,
                 "rejection_categories":{},"rejection_details":{},"rejected_claims":[],
                 "temporal_validation":[],"citation_validation":[],
                 "causal_gate_passed":None,
                 "causal_transition_citation_accepted":None,
                 "answer_reason":None}
    def finish_timing():
        timings["total_ms"]=round((time.perf_counter()-started)*1000,1)
        diagnostics["timings"]={key:round(value,1) for key,value in timings.items()}
    def abstain(reason):
        finish_timing()
        diagnostics["answer_reason"]=reason
        return ABSTENTION,diagnostics

    causal = (
        str(evidence.get("query_category", "")).upper() == "CAUSAL"
        or is_causal_question(question)
    )
    diagnostics["causal_transition_citation_accepted"]=False if causal else None

    if (
        not evidence.get("memories")
        and not evidence.get("states")
        and (not causal or not evidence.get("transitions"))
    ):
        retrieval_reason=evidence.get("retrieval_reason")
        if retrieval_reason in {"missing_predecessor_transition",
                                "ambiguous_predecessor_transition"}:
            return abstain(retrieval_reason)
        if causal:
            return abstain("missing_explicit_cause_transition")
        return abstain("empty_evidence")

    # Retrieval preserves all plausible candidates. If it explicitly marks
    # them ambiguous, do not ask the LLM to choose one without evidence.
    if evidence.get("ambiguity"):
        return abstain("ambiguous_evidence")

    diagnostics["causal_gate_passed"]=(bool(_has_supported_cause(question,evidence)) if causal else None)
    if causal and not diagnostics["causal_gate_passed"]:
        if evidence.get("transitions") or evidence.get("causal_path"):
            return abstain("explicit_cause_gate_rejected")
        return abstain("missing_explicit_cause_transition")

    answer_evidence=evidence
    focus=evidence.get("state_focus")
    if isinstance(focus,dict) and focus.get("stored_preference_inventory"):
        candidate_ids=set(focus.get("candidate_state_ids",[]))
        source_ids=set(focus.get("source_memory_ids",[]))
        answer_evidence=dict(evidence)
        answer_evidence["states"]=[state for state in evidence.get("states",[])
                                    if isinstance(state,dict) and state.get("state_id") in candidate_ids]
        answer_evidence["memories"]=[record for record in evidence.get("memories",[])
                                      if isinstance(record,dict)
                                      and (record.get("id") or record.get("memory_id")) in source_ids]
    elif isinstance(focus,dict) and focus.get("candidate_state_ids"):
        # Retrieval may retain independent active states for history and
        # compatibility. For a type-specific answer, expose only the matched
        # state candidates and their attributed Mem0 records to the LLM.
        candidate_ids=set(focus["candidate_state_ids"])
        source_ids=set(focus.get("source_memory_ids",[]))
        answer_evidence=dict(evidence)
        answer_evidence["states"]=[state for state in evidence.get("states",[])
                                    if isinstance(state,dict) and state.get("state_id") in candidate_ids]
        answer_evidence["memories"]=[record for record in evidence.get("memories",[])
                                      if not isinstance(record,dict)
                                      or not isinstance(record.get("id") or record.get("memory_id"),str)
                                      or (record.get("id") or record.get("memory_id")) in source_ids]
    elif isinstance(focus,dict) and focus.get("memory_fallback"):
        # If extraction left no state of the requested type, use only the
        # query-matched, already ownership-checked Mem0 records. Do not expose
        # unrelated OTHER/legacy states as tempting substitutes; citations to
        # fallback memories still pass the same exact-ID and lexical checks.
        fallback_ids=set(focus.get("source_memory_ids",[]))
        answer_evidence=dict(evidence)
        answer_evidence["states"]=[]
        answer_evidence["transitions"]=[]
        answer_evidence["memories"]=[record for record in evidence.get("memories",[])
                                      if isinstance(record,dict)
                                      and (record.get("id") or record.get("memory_id")) in fallback_ids]
    if is_predecessor_query(question):
        # A predecessor question asks for the closed state (or the edge's
        # from-state). Exposing the edge itself invites the model to restate the
        # whole change instead of the predecessor value. Keep transitions in the
        # original bundle for boundary validation, but not in the model bundle.
        answer_evidence=dict(answer_evidence)
        answer_evidence["transitions"]=[]
    if causal:
        answer_evidence=_causal_model_evidence(question,evidence)
    elif (is_transition_question(question) and evidence.get("transitions")
          and not is_predecessor_query(question)):
        answer_evidence=_transition_model_evidence(question,evidence)
        if not answer_evidence.get("transitions"):
            preference_change_evidence=_programming_preference_change_evidence(
                question,evidence)
            if preference_change_evidence is not None:
                answer_evidence=preference_change_evidence

    # The model receives short citation_ref tokens instead of long UUIDs.
    # Build the same mapping here so a returned token can be resolved to the
    # actual record for ownership, provenance, and out-of-bundle validation.
    _model_payload,citation_refs=model_evidence_projection(answer_evidence)

    answer_method=getattr(llm,"answer_with_provenance",None)
    if not callable(answer_method):
        return abstain("structured_answer_method_unavailable")
    model_started=time.perf_counter()
    response=answer_method(question,answer_evidence)
    timings["model_ms"]=(time.perf_counter()-model_started)*1000
    if not isinstance(response,dict) or not isinstance(response.get("claims"),list):
        diagnostics["model_response_status"]="malformed_structured_response"
        return abstain("malformed_structured_response")
    diagnostics["model_response_status"]="structured_response"
    diagnostics["proposed_claim_count"]=len(response["claims"])
    rejection_counts={}
    rejection_details={}
    rejected_claims=[]
    validation_started=time.perf_counter()
    claims=_validate_claims(question,evidence,response,rejection_counts,rejection_details,
                            rejected_claims,diagnostics["temporal_validation"],answer_evidence,
                            diagnostics["citation_validation"],citation_refs)
    if not claims:
        fallback_claim=None
        conflict=evidence.get("preference_conflict")
        if (isinstance(conflict,dict)
                and re.search(r"\b(?:stored|recorded|saved|mentioned)\b[^?]*\bpreferences?\b|"
                              r"\bpreferences?\b[^?]*\b(?:stored|recorded|saved)\b",
                              question,re.I)):
            values=list(dict.fromkeys(value for value in conflict.get("values",[])
                                      if isinstance(value,str) and value.strip()))
            refs=([{"kind":"state","id":identifier}
                   for identifier in conflict.get("state_ids",[])]
                  +[{"kind":"memory","id":identifier}
                    for identifier in conflict.get("memory_ids",[])])
            domain=str(conflict.get("domain") or "preferences")
            if len(values)>1 and refs:
                fallback_claim={"text":f"Your recorded {domain} preferences conflict: "
                               +" and ".join(values)+".",
                               "claim_type":"fact","evidence":refs}
        if (fallback_claim is None
                and re.search(r"\b(?:programming|coding)\b",question,re.I)
                and re.search(r"\b(?:prefer(?:ence|ences|red|s)?)\b",question,re.I)):
            preference_edges=[edge for edge in evidence.get("transitions",[])
                              if isinstance(edge,dict)
                              and edge.get("user_id")==evidence.get("user_id")
                              and edge.get("transition_type")=="PREFERENCE_CHANGE"
                              and edge.get("relation_kind")=="TEMPORAL_SUCCESSION"]
            if len(preference_edges)==1:
                edge=preference_edges[0]
                source=_owned_sources(evidence).get(
                    ("transition",edge.get("transition_id")))
                if isinstance(source,dict):
                    fallback_claim={
                        "text":f"Your programming preference changed from {source['from']} to {source['to']}.",
                        "claim_type":"temporal",
                        "evidence":[{"kind":"transition",
                                     "id":edge.get("transition_id")}]}
        if fallback_claim is None:
            historical=_programming_preference_history_representatives(
                question,evidence)
            if historical:
                values=[state["state_value"] for state in historical.values()]
                joined=(values[0] if len(values)==1 else
                        ", ".join(values[:-1])+" and "+values[-1])
                fallback_claim={
                    "text":f"You previously preferred {joined} for programming.",
                    "claim_type":"fact",
                    "evidence":[{"kind":"state","id":state["state_id"]}
                                for state in historical.values()],
                }
        if fallback_claim is None:
            direction_from=bool(re.search(r"\bfrom\b",question,re.I))
            direction_to=bool(re.search(r"\bto\b",question,re.I))
            if (is_transition_question(question) and not is_causal_question(question)
                    and direction_from != direction_to
                    and bool(requested_state_types(question))
                    and len(evidence.get("transitions",[]))==1):
                edge=evidence["transitions"][0]
                transition_source=_owned_sources(evidence).get(
                    ("transition",edge.get("transition_id")))
                if (isinstance(transition_source,dict)
                        and edge.get("relation_kind") in {"TEMPORAL_SUCCESSION","EXPLICIT_CAUSE"}):
                    side="to" if direction_to else "from"
                    value=transition_source.get(side)
                    if isinstance(value,str) and value.strip():
                        fallback_claim={"text":f"You moved to {value}" if side=="to"
                                        else f"You moved from {value}",
                                        "claim_type":"temporal",
                                        "evidence":[{"kind":"transition",
                                                     "id":edge.get("transition_id")}]}
        if fallback_claim is None:
            focus=evidence.get("state_focus")
            if isinstance(focus,dict) and focus.get("memory_fallback"):
                eligible=set(focus.get("source_memory_ids",[]))
                record=next((record for record in evidence.get("memories",[])
                             if isinstance(record,dict)
                             and (record.get("id") or record.get("memory_id")) in eligible
                             and isinstance(record.get("memory") or record.get("text"),str)),None)
                if record is not None:
                    identifier=record.get("id") or record.get("memory_id")
                    source=record.get("memory") or record.get("text")
                    fallback_claim={"text":source,"claim_type":"fact",
                                    "evidence":[{"kind":"memory","id":identifier}]}
        if fallback_claim is not None:
            diagnostics["model_claim_rejections"]={
                "rejection_categories":dict(sorted(rejection_counts.items())),
                "rejection_details":dict(sorted(rejection_details.items())),
            }
            rejection_counts={}; rejection_details={}; rejected_claims=[]
            diagnostics["temporal_validation"]=[]
            diagnostics["citation_validation"]=[]
            claims=_validate_claims(
                question,evidence,{"claims":[fallback_claim]},rejection_counts,
                rejection_details,rejected_claims,diagnostics["temporal_validation"],
                answer_evidence,diagnostics["citation_validation"],citation_refs)
            diagnostics["fallback_claim_used"]=bool(claims)
    timings["validation_ms"]=(time.perf_counter()-validation_started)*1000
    diagnostics["accepted_claim_count"]=len(claims)
    diagnostics["rejected_claim_count"]=(
        len(response["claims"]) if not claims and "fallback_claim_used" in diagnostics
        else max(0,len(response["claims"])-len(claims)))
    if causal:
        diagnostics["causal_transition_citation_accepted"]=any(
            claim["claim_type"]=="causal"
            and any(reference["kind"]=="transition" for reference in claim["evidence"])
            for claim in claims)
    diagnostics["rejection_categories"]=dict(sorted(rejection_counts.items()))
    diagnostics["rejection_details"]=dict(sorted(rejection_details.items()))
    diagnostics["rejected_claims"]=rejected_claims
    if causal and not any(claim["claim_type"]=="causal" for claim in claims):
        return abstain("no_validated_causal_claim")
    if not claims:
        return abstain("no_validated_claims")

    render_started=time.perf_counter()
    answer=_render_claims(claims)
    if causal:
        answer=_render_natural_causal_claims(claims,evidence) or answer
    timings["render_ms"]=(time.perf_counter()-render_started)*1000
    finish_timing()
    diagnostics["answer_reason"]="validated_claims"
    return answer,diagnostics


def generate_answer(question, evidence):
    answer,_diagnostics=generate_answer_with_diagnostics(question,evidence)
    return answer

