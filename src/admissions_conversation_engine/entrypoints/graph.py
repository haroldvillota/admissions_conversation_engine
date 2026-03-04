from __future__ import annotations

import logging
import sys

from admissions_conversation_engine.infrastructure.langfuse_factory import (
    build_langfuse_client,
)
from admissions_conversation_engine.infrastructure.agent_builder import (
    AgentBuilder,
)
from admissions_conversation_engine.infrastructure.config.config_bootstrap import get_app_config

logger = logging.getLogger(__name__)

try:
    app_config = get_app_config()

    langfuse, observability_handler = build_langfuse_client(app_config)

    builder = AgentBuilder(
        app_config=app_config,
        checkpointer=None,
        langfuse_client=langfuse
    )

    compiled_graph = builder.build()

    graph = compiled_graph.with_config(config={"callbacks": [observability_handler]})
except Exception as exc:
    logger.critical("Error al exportar el grafo para LangGraph Studio: %s", exc)
    sys.exit(1)
