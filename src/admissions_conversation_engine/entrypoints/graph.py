from __future__ import annotations

import os
from dotenv import load_dotenv

from admissions_conversation_engine.infrastructure.agent_builder import (
    AgentBuilder,
)
from admissions_conversation_engine.infrastructure.config.config_bootstrap import get_app_config

from langfuse import get_client
from langfuse.langchain import CallbackHandler


load_dotenv()

use_vault = os.getenv("USE_VAULT", "0") == "1"
vault_path = os.getenv("VAULT_PATH", "secret/myapp")

app_config = get_app_config(use_vault=use_vault, vault_path=vault_path)
print(app_config.model_dump_json(indent=2))
langfuse = get_client()
observability_handler = CallbackHandler()

# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")


builder = AgentBuilder(
    app_config=app_config,
    checkpointer=None
)

compiled_graph = builder.build()

graph = compiled_graph.with_config(config={"callbacks": [observability_handler]})
