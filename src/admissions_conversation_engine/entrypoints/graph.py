from __future__ import annotations

from admissions_conversation_engine.infrastructure.langfuse_factory import (
    build_langfuse_client,
)
from admissions_conversation_engine.infrastructure.agent_builder import (
    AgentBuilder,
)
from admissions_conversation_engine.infrastructure.config.config_bootstrap import get_app_config

app_config = get_app_config()

langfuse, observability_handler = build_langfuse_client(app_config)

builder = AgentBuilder(
    app_config=app_config,
    checkpointer=None,
    langfuse_client=langfuse
)

compiled_graph = builder.build()

graph = compiled_graph.with_config(config={"callbacks": [observability_handler]})
