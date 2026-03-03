from __future__ import annotations

import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from admissions_conversation_engine.infrastructure.langfuse_factory import (
    build_langfuse_client,
)
from admissions_conversation_engine.infrastructure.agent_builder import (
    AgentBuilder
)
from admissions_conversation_engine.infrastructure.config.config_bootstrap import get_app_config
from admissions_conversation_engine.infrastructure.postgres_checkpointer_manager import PostgresCheckpointerManager
from langchain_core.runnables import RunnableConfig

class ConsoleConversationRunner:

    def __init__(self) -> None:

        app_config = get_app_config()
        
        self.langfuse, self.observability_handler = build_langfuse_client(app_config)

        self.checkpointerManager = PostgresCheckpointerManager(app_config.checkpointer)
        self.checkpointer = None
        
        self._app_config = app_config

    async def run(self) -> None:
        self.checkpointer = await self.checkpointerManager.aget_checkpointer()

        chat_id = (await asyncio.to_thread(input, "Chat ID (default new_id): ")).strip() or "new_id"
        channel_id = (await asyncio.to_thread(input, "Channel ID (default wa): ")).strip() or "wa"
        case = (
            (await asyncio.to_thread(
                input,
                "Case (off_hours | low_scoring | overflow | max_retries) (default off_hours): ",
            ))
            .strip()
            or "off_hours"
        )
        user_name = (await asyncio.to_thread(input, "Nombre del usuario (default Pedro): ")).strip() or "Pedro"

        builder = AgentBuilder(
            app_config=self._app_config,
            checkpointer=self.checkpointer,
            langfuse_client=self.langfuse
        )

        self._graph = builder.build()
        self._graph = self._graph.with_config(config={"callbacks": [self.observability_handler]})

        self._context = {
            "chat_id": chat_id,
            "channel_id": channel_id,
            "case": case,
            "user_name": user_name,
        }

        print("Conversacion iniciada. Escribe 'salir' para terminar.")
        while True:
            user_input = (await asyncio.to_thread(input, "Tu: ")).strip()
            if user_input.lower() in {"salir", "exit", "quit"}:
                print("Hasta luego.")
                await self.checkpointerManager.ashutdown()
                return

            config: RunnableConfig = {
                "configurable": {"thread_id": self._context["chat_id"]},
                "callbacks": [self.observability_handler],
            }
            self._state = await self._graph.ainvoke(
                {"messages": [{"type": "human", "content": user_input}]},
                config=config,
                context=self._context,
            )

            assistant_message = self._state["messages"][-1]
            print(f"Bot: {assistant_message.content}")
        

def main() -> None:
    asyncio.run(ConsoleConversationRunner().run())
