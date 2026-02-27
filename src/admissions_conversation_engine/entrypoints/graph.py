from __future__ import annotations

import os
from dotenv import load_dotenv

from admissions_conversation_engine.infrastructure.langfuse_factory import (
    build_langfuse_client,
)
from admissions_conversation_engine.infrastructure.agent_builder import (
    AgentBuilder,
)
from admissions_conversation_engine.infrastructure.config.config_bootstrap import get_app_config


load_dotenv()

use_vault = os.getenv("USE_VAULT", "0") == "1"
vault_path = os.getenv("VAULT_PATH", "secret/myapp")

app_config = get_app_config(use_vault=use_vault, vault_path=vault_path)

langfuse, observability_handler = build_langfuse_client(app_config)


builder = AgentBuilder(
    app_config=app_config,
    checkpointer=None,
    langfuse_client=langfuse
)

compiled_graph = builder.build()

graph = compiled_graph.with_config(config={"callbacks": [observability_handler]})
