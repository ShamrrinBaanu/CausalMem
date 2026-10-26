"""Configurable OpenAI-compatible LLM calls with JSON validation."""
import json
from typing import Any
from openai import OpenAI
from config import settings
from models import Extraction, QueryAnalysis

class LLMUnavailable(RuntimeError): pass


def model_evidence_projection(evidence):
    """Build a model-facing evidence bundle with short citation references.

    Long database identifiers are not useful to the answer model and small
    local models occasionally mistranscribe a UUID. Replace them with stable
    per-kind tokens (M1, S1, T1) and return the token-to-record index. The
    caller keeps the original evidence and identifiers for ownership,
    provenance, and out-of-bundle validation.
    """
    from collections.abc import Mapping

    refs={}
    payload={}
    if not isinstance(evidence,Mapping):
        return payload,refs
    for key in ("query_category","requested_timestamp","temporal_context"):
        if key in evidence:
            payload[key]=evidence[key]

    states_by_id={state.get("state_id"):state
                  for state in evidence.get("states",[])
                  if isinstance(state,Mapping)
                  and isinstance(state.get("state_id"),str)
                  and isinstance(state.get("state_value"),str)}
    memories=[]
    memory_refs_by_id={}
    for index,record in enumerate(evidence.get("memories",[]) or [],1):
        if not isinstance(record,Mapping):
            continue
        identifier=record.get("id") or record.get("memory_id")
        text=record.get("memory") or record.get("text") or record.get("content")
        if (not isinstance(identifier,str) or not identifier.strip()
                or not isinstance(text,str) or not text.strip()):
            continue
        ref=f"M{index}"
        refs[ref]=("memory",identifier)
        record_metadata=record.get("metadata")
        record_owner=(record.get("user_id")
                      or (record_metadata.get("user_id")
                          if isinstance(record_metadata,Mapping) else None))
        if record_owner==evidence.get("user_id"):
            memory_refs_by_id[identifier]=ref
        memories.append({"citation_ref":ref,"memory":text})

    states=[]
    state_refs_by_id={}
    for index,state in enumerate(evidence.get("states",[]) or [],1):
        if not isinstance(state,Mapping):
            continue
        identifier=state.get("state_id")
        value=state.get("state_value")
        if (not isinstance(identifier,str) or not identifier.strip()
                or not isinstance(value,str) or not value.strip()):
            continue
        ref=f"S{index}"
        refs[ref]=("state",identifier)
        state_refs_by_id[identifier]=ref
        states.append({key:state.get(key) for key in
                       ("state_type","state_value","status","valid_from",
                        "valid_until","confidence")
                       if key in state} | {"citation_ref":ref})

    conflict=evidence.get("preference_conflict")
    if isinstance(conflict,Mapping):
        conflict_ids=conflict.get("state_ids")
        selected=[state for state in evidence.get("states",[]) or []
                  if isinstance(state,Mapping) and isinstance(conflict_ids,list)
                  and state.get("state_id") in conflict_ids
                  and state.get("user_id")==evidence.get("user_id")
                  and state.get("state_type")=="PREFERENCE"
                  and state.get("status")=="ACTIVE"]
        conflict_memory_ids=conflict.get("memory_ids")
        memory_values=conflict.get("memory_values")
        if not isinstance(memory_values,Mapping):
            memory_values={}
        if not isinstance(conflict_memory_ids,list):
            conflict_memory_ids=[]
        selected_memory_ids=[identifier for identifier in conflict_memory_ids
                             if isinstance(identifier,str)
                             and identifier in memory_refs_by_id
                             and identifier in memory_values]
        all_values={str(state.get("state_value") or "").strip().casefold()
                    for state in selected}
        all_values.update(str(memory_values[identifier]).strip().casefold()
                          for identifier in selected_memory_ids)
        expected_ids=len(conflict_ids) if isinstance(conflict_ids,list) else 0
        if (len(selected)==expected_ids
                and len(selected_memory_ids)==len(conflict_memory_ids or [])
                and len(all_values)>1 and all(all_values)
                and len(selected)+len(selected_memory_ids)>=2):
            payload["preference_conflict"]={
                "domain":conflict.get("domain"),
                "state_refs":[state_refs_by_id[identifier]
                              for identifier in conflict_ids or []
                              if identifier in state_refs_by_id],
                "memory_refs":[memory_refs_by_id[identifier]
                               for identifier in selected_memory_ids],
                "values":conflict.get("values",[]),
            }

    transitions=[]
    for index,edge in enumerate(evidence.get("transitions",[]) or [],1):
        if not isinstance(edge,Mapping):
            continue
        identifier=edge.get("transition_id")
        if not isinstance(identifier,str) or not identifier.strip():
            continue
        ref=f"T{index}"
        refs[ref]=("transition",identifier)
        source=edge.get("from")
        destination=edge.get("to")
        if not isinstance(source,str):
            source=(states_by_id.get(edge.get("from_state_id")) or {}).get("state_value")
        if not isinstance(destination,str):
            destination=(states_by_id.get(edge.get("to_state_id")) or {}).get("state_value")
        item={"citation_ref":ref,
              "transition_type":edge.get("transition_type"),
              "relation_kind":edge.get("relation_kind"),
              "from":source,"to":destination,
              "trigger":edge.get("trigger"),"reason":edge.get("reason"),
              "timestamp":edge.get("timestamp")}
        transitions.append(item)

    payload["memories"]=memories
    payload["states"]=states
    payload["transitions"]=transitions
    return payload,refs


def user_facing_error(exc: Exception) -> str:
    """Turn provider errors into actionable text without dumping API responses."""
    source=getattr(exc,"source",None)
    if isinstance(source,str) and source:
        return f"Could not retrieve {source} evidence. No answer was generated; check the local store/provider and try again."
    status=getattr(exc,"status_code",None)
    body=getattr(exc,"body",None)
    detail=body.get("error",body) if isinstance(body,dict) else {}
    code=str(detail.get("code", "")) if isinstance(detail,dict) else ""
    message=str(detail.get("message", "")) if isinstance(detail,dict) else ""
    if code in {"insufficient_quota","credit_balance_exhausted"} or "no credits remaining" in message.lower():
        return "Your OpenAI API account has no credits remaining. Add API credits or configure Ollama in .env. This message may not have been stored."
    if status==401 or code=="invalid_api_key":
        return "OpenAI rejected the API key. Check OPENAI_API_KEY in .env, then restart the app."
    if status==429:
        return "The LLM provider is rate limiting requests. Wait briefly and try again."
    if isinstance(exc,ValueError) and "shapes" in str(exc).lower() and "not aligned" in str(exc).lower():
        return "Mem0 found an embedding-size mismatch in its local database. Restart the app; Ollama now uses a separate store. The previous message may not have been saved."
    if isinstance(exc,LLMUnavailable):
        return str(exc)
    return "The memory operation failed. Check the provider configuration and the application terminal for details."

class LLMService:
    def __init__(self):
        if settings.provider == "openai" and not settings.api_key: self.client = None
        else:
            kwargs: dict[str, Any] = {"api_key": settings.effective_api_key or "missing"}
            if settings.effective_base_url: kwargs["base_url"] = settings.effective_base_url
            self.client = OpenAI(**kwargs)
    def _json(self, system, payload):
        if not self.client: raise LLMUnavailable("Configure OPENAI_API_KEY or an available Ollama model.")
        kwargs: dict[str, Any] = {}
        if settings.provider == "ollama":
            # Keep the local model resident between requests when the Ollama
            # OpenAI-compatible endpoint supports extra body fields.
            kwargs["extra_body"] = {"keep_alive": "30m"}
        response = self.client.chat.completions.create(model=settings.model, messages=[{"role":"system","content":system},{"role":"user","content":json.dumps(payload,default=str)}], response_format={"type":"json_object"}, temperature=0, timeout=90, **kwargs)
        return json.loads(response.choices[0].message.content or "{}")
    def extract(self, message, memories, active_states):
        prompt=("Extract meaningful personal facts asserted in NEW_MESSAGE. Return one JSON object with exactly these keys: "
                '{"states":[{"state_type":"INTEREST","state_value":"<short concept from the message>","confidence":0.9}],'
                '"transitions":[{"from_state_value":"<prior concept>","to_state_value":"<new concept>",'
                '"transition_type":"ACTIVITY_CHANGE","relation_kind":"EXPLICIT_CAUSE",'
                '"trigger":"<stated cause>","reason":"<brief evidence summary>","confidence":0.9}]}. '
                "For an informative personal message, do not return empty states. If the user says they love or enjoy something, "
                "record that concept as INTEREST; if they switch activities, record the destination as ACTIVITY. Use state types "
                "such as FACT, PREFERENCE, ACTIVITY, GOAL, SKILL, INTEREST, LIFE_EVENT, STATUS, PROJECT, or OTHER; never put a "
                "transition type in state_type. Each state value and transition endpoint must be grounded in NEW_MESSAGE or an "
                "active state; never copy concepts from instructions or unrelated memories. Use active states to identify prior "
                "endpoints. EXPLICIT_CAUSE requires the user's message to explicitly state a cause, for example with 'because' or "
                "'due to'. Otherwise use TEMPORAL_SUCCESSION or UNKNOWN; do not invent causes. Avoid unchanged duplicates. "
                "If NEW_MESSAGE is a question, recall/explanation request, command, quotation of an assistant reply, or "
                "does not contain a first-person statement of fact, return empty states and transitions. Never output "
                "question text or meta/retrieval language as a state_value.")
        return Extraction.model_validate(self._json(prompt,{"message":message,"memories":memories,"active_states":active_states}))
    def classify(self, question):
        prompt="Classify this chat message. JSON {is_question:boolean, category: FACTUAL|PREFERENCE|TEMPORAL|CAUSAL|HISTORY|EVOLUTION|MULTI_HOP|COMPARISON|STATE|OTHER, rationale:string}. is_question is true for requests to recall, explain, compare, or answer about the user's personal history, even without a question mark. Statements sharing or correcting personal information are false. Use semantic intent; multi-hop means multiple linked steps."
        return QueryAnalysis.model_validate(self._json(prompt,{"question":question}))
    def answer(self, question, evidence):
        result=self.answer_with_provenance(question,evidence)
        claims=result.get("claims",[]) if isinstance(result,dict) else []
        return " ".join(claim["text"] for claim in claims
                        if isinstance(claim,dict) and isinstance(claim.get("text"),str))

    def answer_with_provenance(self, question, evidence):
        prompt=("Answer using only supplied personal memory evidence. Return JSON with exactly this shape: "
                '{"claims":[{"text":"one factual claim","claim_type":"fact",'
                '"evidence":[{"kind":"state","ref":"exact supplied citation_ref"}]}]}. '
                'Each claim_type must be exactly one of "fact", "temporal", or "causal". '
                "Write concise, natural wording that directly answers the user's question. "
                "Avoid meta phrases such as 'the recorded reason' unless the user asks what was recorded. "
                "Use the shortest wording supported by the cited evidence; do not add locations, qualifiers, or details absent from it. "
                'Each evidence reference kind must be exactly one of "memory", "state", or "transition". '
                'Every supplied memory, state, and transition has a short citation_ref such as M1, S1, or T1. '
                'Put that exact citation_ref in the reference "ref" field. Do not copy or invent long IDs or UUIDs; '
                'a mistyped identifier is rejected even when the underlying record exists. '
                'Use these literal field values; do not output pipe-delimited enum descriptions or placeholders. '
                'For example, a supported past-state claim may be {"text":"My goal was to learn guitar",'
                '"claim_type":"temporal","evidence":[{"kind":"state","ref":"S1"}]}; '
                'use this only when the supplied as-of validity context or predecessor transition supports that state. '
                'A state marked INACTIVE can still be valid historical evidence when the supplied interval supports it. '
                'When a selected state supports a historical/as-of question, propose the answer as a temporal claim instead '
                'of returning an empty claims list, and cite that state reference (or its uniquely linked memory reference). '
                'For a causal claim, cite the supplied EXPLICIT_CAUSE transition citation_ref; memory or endpoint-state references '
                'alone do not establish causation. Include the recorded cause and transition endpoints only when supported. '
                'For a direct question about which activity changed to or from, cite the supplied transition and use '
                'claim_type temporal for the source/destination statement; do not add a cause unless asked. '
                'For example, a supported causal claim uses {"text":"The recorded reason for the change was a knee injury",'
                '"claim_type":"causal","evidence":[{"kind":"transition",'
                '"ref":"T1"}]}; never substitute a memory or endpoint-state citation. '
                "Cite only references present in the supplied evidence. Separate claims when they need different sources. "
                "Interpret relation_kind strictly: EXPLICIT_CAUSE supports a causal claim only when its recorded trigger or reason "
                "and both same-user endpoint states are supplied; TEMPORAL_SUCCESSION supports ordering only, not causation; "
                "UNKNOWN and INFERRED_RELATIONSHIP are not established causal facts. Never infer a cause from chronology. For an "
                "as-of question, do not use transitions recorded after the requested timestamp. Do not include a claim if no supplied "
                "source supports it; return an empty claims list when evidence is insufficient or ambiguous.")
        model_evidence,_refs=model_evidence_projection(evidence)
        if "preference_conflict" in model_evidence:
            prompt += (" For a question asking what preferences are stored, if preference_conflict is supplied, "
                       "state that the recorded preferences conflict, name each supplied value, and cite every "
                       "listed state_ref and memory_ref in one fact claim. Do not choose one as definitive or current.")
        raw=self._json(prompt,{"question":question,"evidence":model_evidence})
        return raw if isinstance(raw,dict) and isinstance(raw.get("claims"),list) else {"claims":[]}

llm=LLMService()
