from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from admissions_conversation_engine.infrastructure.agent_builder import AgentBuilder
from admissions_conversation_engine.infrastructure.config.config_bootstrap import get_app_config
from admissions_conversation_engine.infrastructure.langfuse_factory import build_langfuse_client
from admissions_conversation_engine.infrastructure.postgres_checkpointer_manager import (
    PostgresCheckpointerManager,
)

from admissions_conversation_engine.infrastructure.api.routers.a2a import router as a2a_router
from admissions_conversation_engine.infrastructure.api.routers.messages import router as messages_router

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Lifespan
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app_config = get_app_config()
        langfuse, handler = build_langfuse_client(app_config)

        checkpointer_manager = PostgresCheckpointerManager(app_config.checkpointer)
        await checkpointer_manager.aprobe_connection()
        checkpointer = await checkpointer_manager.aget_checkpointer()

        builder = AgentBuilder(
            app_config=app_config,
            checkpointer=checkpointer,
            langfuse_client=langfuse,
        )
        compiled = builder.build()
    except Exception as exc:
        logger.critical("La aplicación no pudo iniciar: %s", exc)
        raise

    app.state.graph = compiled.with_config({"callbacks": [handler]})
    app.state.observability_handler = handler
    app.state.checkpointer_manager = checkpointer_manager
    app.state.tasks = {}

    yield

    await checkpointer_manager.ashutdown()


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #


app = FastAPI(
    title="Admissions Conversation Engine API",
    description="API REST que expone el agente conversacional de admissions optimizer.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(messages_router)
app.include_router(a2a_router)


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #


@app.get("/health", tags=["Ops"])
async def health() -> dict:
    """Returns service liveness status."""
    return {"status": "ok"}
