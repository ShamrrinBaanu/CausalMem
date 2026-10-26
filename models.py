"""Typed records shared by the causal extension."""
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4
from pydantic import BaseModel, Field, field_validator

def utc_now(): return datetime.now(timezone.utc)

class State(BaseModel):
    state_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    state_type: str = "OTHER"
    state_value: str
    status: str = "ACTIVE"
    valid_from: datetime = Field(default_factory=utc_now)
    valid_until: datetime | None = None
    confidence: float = Field(default=0.5, ge=0, le=1)
    source_memory_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

class Transition(BaseModel):
    transition_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    from_state_id: str
    to_state_id: str
    transition_type: str = "TEMPORAL_SUCCESSION"
    relation_kind: str = "TEMPORAL_SUCCESSION"
    trigger: str | None = None
    reason: str | None = None
    timestamp: datetime = Field(default_factory=utc_now)
    confidence: float = Field(default=0.5, ge=0, le=1)
    source_memory_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

class QueryAnalysis(BaseModel):
    category: str = "OTHER"
    rationale: str = ""
    is_question: bool = False

    @field_validator("category")
    @classmethod
    def category_is_canonical(cls, value: str) -> str:
        text = str(value or "OTHER").strip().upper()
        return text or "OTHER"

class Extraction(BaseModel):
    states: list[dict[str, Any]] = Field(default_factory=list)
    transitions: list[dict[str, Any]] = Field(default_factory=list)
