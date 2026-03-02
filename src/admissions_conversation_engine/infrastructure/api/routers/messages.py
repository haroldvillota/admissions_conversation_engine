from __future__ import annotations

from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import RunnableConfig

from ..dependencies import get_current_token
from ..schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/messages", tags=["Agent"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: Request,
    body: ChatRequest,
    _: dict = Depends(get_current_token),
) -> ChatResponse:
    """Send a message to the agent and receive the full response.

    The `chat_id` acts as the conversation thread identifier — reuse the same
    value across calls to continue an existing conversation.
    """
    context = {
        "chat_id": body.chat_id,
        "channel_id": body.channel_id,
        "case": body.case,
        "user_name": body.user_name,
    }

    config: RunnableConfig = {
        "configurable": {"thread_id": body.chat_id},
        "callbacks": [request.app.state.observability_handler],
    }

    try:
        agent_state = await request.app.state.graph.ainvoke(
            {"messages": [{"type": "human", "content": body.message}]},
            config=config,
            context=context,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    reply = agent_state["messages"][-1].content

    return ChatResponse(
        reply=reply,
        chat_id=body.chat_id,
        guardrail_allowed=agent_state.get("guardrail_allowed", True),
        guardrail_reason=agent_state.get("guardrail_reason", "OK"),
        language=agent_state.get("language"),
    )


@router.post("/stream")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    _: dict = Depends(get_current_token),
) -> StreamingResponse:
    """Send a message to the agent and receive the response as a token stream.

    Streams raw text tokens via `text/plain` as they are produced by the LLM.
    The `chat_id` acts as the conversation thread identifier.
    """
    context = {
        "chat_id": body.chat_id,
        "channel_id": body.channel_id,
        "case": body.case,
        "user_name": body.user_name,
    }

    config: RunnableConfig = {
        "configurable": {"thread_id": body.chat_id},
        "callbacks": [request.app.state.observability_handler],
    }

    async def token_generator() -> AsyncGenerator[str, None]:
        try:
            async for event in request.app.state.graph.astream_events(
                {"messages": [{"type": "human", "content": body.message}]},
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
