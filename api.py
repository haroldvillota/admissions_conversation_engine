from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from admissions_conversation_engine.infrastructure.agent_builder import AgentBuilder
from admissions_conversation_engine.infrastructure.config.config_bootstrap import get_app_config
from admissions_conversation_engine.infrastructure.langfuse_factory import build_langfuse_client
from admissions_conversation_engine.infrastructure.postgres_checkpointer_manager import (
    PostgresCheckpointerManager,
)

# --------------------------------------------------------------------------- #
# Schemas
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
# Shared application state
# --------------------------------------------------------------------------- #


class _AppState:
    graph = None
    observability_handler = None
    checkpointer_manager: PostgresCheckpointerManager | None = None


_state = _AppState()


# --------------------------------------------------------------------------- #
# Lifespan
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_config = get_app_config()
    langfuse, handler = build_langfuse_client(app_config)

    checkpointer_manager = PostgresCheckpointerManager(app_config.checkpointer)
    checkpointer = await checkpointer_manager.aget_checkpointer()

    builder = AgentBuilder(
        app_config=app_config,
        checkpointer=checkpointer,
        langfuse_client=langfuse,
    )
    compiled = builder.build()

    _state.graph = compiled.with_config({"callbacks": [handler]})
    _state.observability_handler = handler
    _state.checkpointer_manager = checkpointer_manager

    yield

    await checkpointer_manager.ashutdown()


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Admissions Conversation Engine API",
    description="REST API that exposes the admissions conversational agent.",
    version="0.1.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@app.get("/health", tags=["Ops"])
async def health() -> dict:
    """Returns service liveness status."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to the agent and receive the full response.

    The `chat_id` acts as the conversation thread identifier — reuse the same
    value across calls to continue an existing conversation.
    """
    context = {
        "chat_id": request.chat_id,
        "channel_id": request.channel_id,
        "case": request.case,
        "user_name": request.user_name,
    }

    config: RunnableConfig = {
        "configurable": {"thread_id": request.chat_id},
        "callbacks": [_state.observability_handler],
    }

    try:
        agent_state = await _state.graph.ainvoke(
            {"messages": [{"type": "human", "content": request.message}]},
            config=config,
            context=context,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    reply = agent_state["messages"][-1].content

    return ChatResponse(
        reply=reply,
        chat_id=request.chat_id,
        guardrail_allowed=agent_state.get("guardrail_allowed", True),
        guardrail_reason=agent_state.get("guardrail_reason", "OK"),
        language=agent_state.get("language"),
    )


@app.post("/chat/stream", tags=["Agent"])
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Send a message to the agent and receive the response as a token stream.

    Streams raw text tokens via `text/plain` as they are produced by the LLM.
    The `chat_id` acts as the conversation thread identifier.
    """
    context = {
        "chat_id": request.chat_id,
        "channel_id": request.channel_id,
        "case": request.case,
        "user_name": request.user_name,
    }

    config: RunnableConfig = {
        "configurable": {"thread_id": request.chat_id},
        "callbacks": [_state.observability_handler],
    }

    async def token_generator() -> AsyncGenerator[str, None]:
        try:
            async for event in _state.graph.astream_events(
                {"messages": [{"type": "human", "content": request.message}]},
                config=config,
                context=context,
                version="v2",
            ):
                if event["event"] == "on_chat_model_stream":
                    chunk: AIMessageChunk = event["data"]["chunk"]
                    if chunk.content:
                        yield chunk.content
        except Exception as exc:
            yield f"\n[ERROR] {exc}"

    return StreamingResponse(token_generator(), media_type="text/plain")


# --------------------------------------------------------------------------- #
# Dev runner
# --------------------------------------------------------------------------- #

def main() -> None:
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
