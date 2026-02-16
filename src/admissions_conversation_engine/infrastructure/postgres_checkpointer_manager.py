from __future__ import annotations

from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from admissions_conversation_engine.infrastructure.config.app_config import CheckpointerConfig

class PostgresCheckpointerManager:
    def __init__(self, config: CheckpointerConfig):
        self.config = config
        self._pool = None

    def get_checkpointer(self) -> PostgresSaver:
        if self._pool is None:
            self._pool = ConnectionPool(
                conninfo=self.config.dsn,
                max_size=10,
                kwargs={"autocommit": True}
            )
        
        saver = PostgresSaver(self._pool)
        saver.setup()
        return saver

    def shutdown(self):
        if self._pool:
            self._pool.close()