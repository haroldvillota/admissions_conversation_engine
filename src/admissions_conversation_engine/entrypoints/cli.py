from __future__ import annotations

import os
from dotenv import load_dotenv

from admissions_conversation_engine.infrastructure.agent_builder import (
    AgentBuilder
)
from admissions_conversation_engine.infrastructure.config.config_bootstrap import get_app_config
from admissions_conversation_engine.infrastructure.postgres_checkpointer_manager import PostgresCheckpointerManager
from langchain_core.runnables import RunnableConfig
from langfuse import get_client
from langfuse.langchain import CallbackHandler

class ConsoleConversationRunner:

    def __init__(self) -> None:

        load_dotenv()

        use_vault = os.getenv("USE_VAULT", "0") == "1"
        vault_path = os.getenv("VAULT_PATH", "secret/myapp")

        app_config = get_app_config(use_vault=use_vault, vault_path=vault_path)

        langfuse = get_client()
        self.observability_handler = CallbackHandler()

        # Verify connection
        if langfuse.auth_check():
            print("Langfuse client is authenticated and ready!")
        else:
            print("Authentication failed. Please check your credentials and host.")

        self.checkpointerManager = PostgresCheckpointerManager(app_config.checkpointer)
        self.checkpointer = self.checkpointerManager.get_checkpointer()
        
        chat_id = input("Chat ID (default new_id): ").strip() or "new_id"
        channel_id = input("Channel ID (default wa): ").strip() or "wa"
        case = input("Case (off_hours | low_scoring | overflow | max_retries) (default off_hours): ").strip()  or "off_hours"
        user_name = input("Nombre del usuario (default Pedro): ").strip() or "Pedro"

        builder = AgentBuilder(
            app_config=app_config,
            checkpointer=self.checkpointer
        )

        self._graph = builder.build()
        
        self._graph = self._graph.with_config(config={"callbacks": [self.observability_handler]})

        self._context = {
            "chat_id": chat_id,
            "channel_id": channel_id,
            "case": case,
            "user_name": user_name,
        }

    def run(self) -> None:

        print("Conversacion iniciada. Escribe 'salir' para terminar.")
        while True:
            user_input = input("Tu: ").strip()
            if user_input.lower() in {"salir", "exit", "quit"}:
                print("Hasta luego.")
                self.checkpointerManager.shutdown()
                return
            
            config: RunnableConfig = {
                "configurable": {"thread_id": self._context["chat_id"]},
                "callbacks": [self.observability_handler]
            }
            self._state = self._graph.invoke(
                {"messages": [{"type": "human", "content": user_input}]},
                config = config,
                context = self._context,
            )

            assistant_message = self._state["messages"][-1]
            print(f"Bot: {assistant_message.content}")
        

def main() -> None:
    ConsoleConversationRunner().run()
