from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Chat schemas
# --------------------------------------------------------------------------- #


class ChatRequest(BaseModel):
    message: str
    chat_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    channel_id: str = "wa"
    case: Literal["off_hours", "low_scoring", "overflow", "max_retries"] = "off_hours"
    user_name: str = "Usuario"


class ChatResponse(BaseModel):
    reply: str
    chat_id: str
    guardrail_allowed: bool
    guardrail_reason: str
    language: str | None = None


# --------------------------------------------------------------------------- #
# A2A schemas
# --------------------------------------------------------------------------- #


class A2ATaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class A2ATaskRequest(BaseModel):
    message: str
    chat_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    channel_id: str = "wa"
    case: Literal["off_hours", "low_scoring", "overflow", "max_retries"] = "off_hours"
    user_name: str = "Usuario"


class A2ATask(BaseModel):
    task_id: str
    status: A2ATaskStatus
    created_at: datetime
    completed_at: datetime | None = None
    result: ChatResponse | None = None
    error: str | None = None


class A2ACapabilities(BaseModel):
    name: str
    description: str
    version: str
    supported_cases: list[str]
    endpoints: list[str]
