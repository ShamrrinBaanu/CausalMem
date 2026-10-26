"""Thin HTTP API for the CausalMem research console."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AliasChoices, BaseModel, Field, field_validator

from chat_service import process_chat
from config import settings
from database import db
from hybrid_retriever import RetrievalFailure
from llm_service import LLMUnavailable, user_facing_error
from mem0_engine import get_memories
from temporal_engine import timeline

logger = logging.getLogger("causalmem.api")

app = FastAPI(
    title="CausalMem API",
    version="1.0.0",
    description="User-scoped conversational memory, state history, and evidence validation.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    user_id: str = Field(default_factory=lambda: settings.default_user, min_length=1, max_length=128)
    requested_timestamp: str | None = None
    source_role: Literal["user", "assistant", "system", "tool"] = Field(
        default="user",
        validation_alias=AliasChoices("source_role", "role"),
    )

    @field_validator("message")
    @classmethod
    def message_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must contain text")
        return value.strip()

    @field_validator("user_id")
    @classmethod
    def user_id_is_a_clean_namespace(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or any(character.isspace() and character != " " for character in cleaned):
            raise ValueError("user_id must be a non-empty namespace")
        return cleaned

    @field_validator("requested_timestamp")
    @classmethod
    def timestamp_is_aware_iso(cls, value: str | None) -> str | None:
        if value is None:
            return None
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("requested_timestamp must be ISO-8601") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("requested_timestamp must include a timezone")
        return parsed.astimezone(timezone.utc).isoformat()


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, RetrievalFailure):
        return user_facing_error(exc)
    if isinstance(exc, LLMUnavailable):
        return str(exc)
    return user_facing_error(exc)


def _require_user(user_id: str) -> str:
    cleaned = (user_id or "").strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="user_id is required")
    if len(cleaned) > 128:
        raise HTTPException(status_code=400, detail="user_id is too long")
    return cleaned


@app.get("/api/health")
def health():
    return {"status": "ok", "provider": settings.provider, "model": settings.model}


@app.get("/api/config")
def config():
    return {
        "provider": settings.provider,
        "model": settings.model,
        "default_user": settings.default_user,
        "has_api_key": bool(settings.api_key),
        "temporal_rule": "valid_from <= t < valid_until",
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    try:
        return process_chat(
            request.message,
            request.user_id,
            request.requested_timestamp,
            source_role=request.source_role,
        )
    except RetrievalFailure as exc:
        logger.error("Chat retrieval failed (source=%s, type=%s)", exc.source, exc.cause_type)
        raise HTTPException(status_code=502, detail=_safe_error(exc)) from None
    except LLMUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except ValueError as exc:
        logger.info("Chat validation rejected request (%s)", type(exc).__name__)
        raise HTTPException(status_code=400, detail=_safe_error(exc)) from None
    except Exception as exc:
        logger.error("Chat operation failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=502, detail=_safe_error(exc)) from None


@app.get("/api/memories")
def memories(user_id: str = Query(default=settings.default_user, min_length=1, max_length=128)):
    user_id = _require_user(user_id)
    try:
        records = get_memories(user_id)
    except Exception as exc:
        logger.error("Memory listing failed (%s)", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Mem0 is unavailable. Check the local provider.") from None
    return {
        "user_id": user_id,
        "memories": [
            {
                "id": record.get("id") or record.get("memory_id"),
                "memory": record.get("memory") or record.get("text") or record.get("content"),
                "created_at": record.get("created_at"),
                "updated_at": record.get("updated_at"),
                "attribution": record.get("attributed_to") or record.get("attribution"),
            }
            for record in records
        ],
    }


@app.get("/api/states")
def states(
    user_id: str = Query(default=settings.default_user, min_length=1, max_length=128),
    include_inactive: bool = Query(default=True),
    include_quarantined: bool = Query(default=True),
):
    user_id = _require_user(user_id)
    return {
        "user_id": user_id,
        "states": [state.model_dump(mode="json") for state in db.states(
            user_id, include_inactive, include_quarantined=include_quarantined)],
    }


@app.get("/api/transitions")
def transitions(
    user_id: str = Query(default=settings.default_user, min_length=1, max_length=128),
    include_quarantined: bool = Query(default=True),
):
    user_id = _require_user(user_id)
    return {
        "user_id": user_id,
        "transitions": [transition.model_dump(mode="json") for transition in db.transitions(
            user_id, include_quarantined=include_quarantined)],
    }


@app.get("/api/timeline")
def state_timeline(
    user_id: str = Query(default=settings.default_user, min_length=1, max_length=128),
    include_quarantined: bool = Query(default=True),
):
    user_id = _require_user(user_id)
    states_history = timeline(user_id, include_quarantined=include_quarantined)
    edges = db.transitions(user_id, include_quarantined=include_quarantined)
    return {
        "user_id": user_id,
        "states": [state.model_dump(mode="json") for state in states_history],
        "transitions": [transition.model_dump(mode="json") for transition in edges],
    }


@app.get("/api/diagnostics")
def diagnostics(
    user_id: str = Query(default=settings.default_user, min_length=1, max_length=128),
    include_quarantined: bool = Query(default=True),
):
    user_id = _require_user(user_id)
    state_rows = db.states(user_id, include_quarantined=include_quarantined)
    edge_rows = db.transitions(user_id, include_quarantined=include_quarantined)
    try:
        memory_rows = get_memories(user_id)
        mem0_error = None
    except Exception as exc:
        logger.error("Diagnostic memory count failed (%s)", type(exc).__name__)
        memory_rows = []
        mem0_error = "Mem0 unavailable"
    return {
        "user_id": user_id,
        "provider": settings.provider,
        "model": settings.model,
        "counts": {
            "memories": len(memory_rows),
            "states": len(state_rows),
            "active_states": sum(1 for state in state_rows if state.status == "ACTIVE"),
            "transitions": len(edge_rows),
            "explicit_causes": sum(1 for edge in edge_rows if edge.relation_kind == "EXPLICIT_CAUSE"),
            "temporal_successions": sum(1 for edge in edge_rows
                                        if edge.relation_kind == "TEMPORAL_SUCCESSION"),
        },
        "store_status": {"mem0": "ok" if mem0_error is None else "unavailable"},
        "errors": [mem0_error] if mem0_error else [],
        "metrics_note": "Counts are measured directly from the configured stores for this user namespace.",
    }


WEB_DIST = Path(__file__).resolve().parent / "web" / "dist"
if WEB_DIST.is_dir():
    assets = WEB_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def frontend(full_path: str):
        candidate = (WEB_DIST / full_path).resolve()
        if candidate.is_file() and WEB_DIST in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(WEB_DIST / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def root():
        return {
            "service": "CausalMem API",
            "web": "not built",
            "build": "cd web && npm install && npm run build",
            "docs": "/docs",
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
