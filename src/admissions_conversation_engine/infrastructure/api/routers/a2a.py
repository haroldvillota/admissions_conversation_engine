from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from langchain_core.runnables import RunnableConfig

from ..dependencies import get_current_token
from ..schemas import (
    A2ACapabilities,
    A2ATask,
    A2ATaskRequest,
    A2ATaskStatus,
    ChatResponse,
)

router = APIRouter(prefix="/a2a", tags=["A2A"])


async def _execute_task(task_id: str, body: A2ATaskRequest, app_state) -> None:
    """Run the agent graph for an A2A task and update the task store on completion."""
    app_state.tasks[task_id] = app_state.tasks[task_id].model_copy(
        update={"status": A2ATaskStatus.running}
    )

    context = {
        "chat_id": body.chat_id,
        "channel_id": body.channel_id,
        "case": body.case,
        "user_name": body.user_name,
    }

    config: RunnableConfig = {
        "configurable": {"thread_id": body.chat_id},
        "callbacks": [app_state.observability_handler],
    }

    try:
        agent_state = await app_state.graph.ainvoke(
            {"messages": [{"type": "human", "content": body.message}]},
            config=config,
            context=context,
        )
        result = ChatResponse(
            reply=agent_state["messages"][-1].content,
            chat_id=body.chat_id,
            guardrail_allowed=agent_state.get("guardrail_allowed", True),
            guardrail_reason=agent_state.get("guardrail_reason", "OK"),
            language=agent_state.get("language"),
        )
        app_state.tasks[task_id] = app_state.tasks[task_id].model_copy(
            update={
                "status": A2ATaskStatus.completed,
                "completed_at": datetime.now(tz=timezone.utc),
                "result": result,
            }
        )
    except Exception as exc:
        app_state.tasks[task_id] = app_state.tasks[task_id].model_copy(
            update={
                "status": A2ATaskStatus.failed,
                "completed_at": datetime.now(tz=timezone.utc),
                "error": str(exc),
            }
        )


@router.post("/tasks", status_code=202, response_model=A2ATask)
async def create_task(
    request: Request,
    body: A2ATaskRequest,
    _: dict = Depends(get_current_token),
) -> A2ATask:
    """Submit an A2A task for asynchronous processing.

    Returns immediately (HTTP 202) with the initial task record.
    Poll `GET /a2a/tasks/{task_id}` to retrieve the result once the task completes.
    """
    task_id = str(uuid.uuid4())
    task = A2ATask(
        task_id=task_id,
        status=A2ATaskStatus.pending,
        created_at=datetime.now(tz=timezone.utc),
    )
    request.app.state.tasks[task_id] = task
    asyncio.create_task(_execute_task(task_id, body, request.app.state))
    return task


@router.get("/tasks/{task_id}", response_model=A2ATask)
async def get_task(
    task_id: str,
    request: Request,
    _: dict = Depends(get_current_token),
) -> A2ATask:
    """Retrieve the current status and result of an A2A task."""
    task = request.app.state.tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return task


@router.get("/capabilities", response_model=A2ACapabilities)
async def get_capabilities(
    _: dict = Depends(get_current_token),
) -> A2ACapabilities:
    """Describe the capabilities exposed by this agent for A2A integration."""
    return A2ACapabilities(
        name="Admissions Conversation Engine",
        description=(
            "Conversational agent that handles university admissions inquiries "
            "across multiple case scenarios using RAG-powered responses."
        ),
        version="0.1.0",
        supported_cases=["off_hours", "low_scoring", "overflow", "max_retries"],
        endpoints=[
            "POST /a2a/tasks",
            "GET /a2a/tasks/{task_id}",
            "GET /a2a/capabilities",
        ],
    )
