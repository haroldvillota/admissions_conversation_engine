from __future__ import annotations

from psycopg_pool import AsyncConnectionPool, ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from admissions_conversation_engine.infrastructure.config.app_config import CheckpointerConfig

class PostgresCheckpointerManager:
    def __init__(self, config: CheckpointerConfig):
        self.config = config
        self._pool: ConnectionPool | None = None
        self._apool: AsyncConnectionPool | None = None

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

    async def aget_checkpointer(self) -> AsyncPostgresSaver:
        if self._apool is None:
            self._apool = AsyncConnectionPool(
                conninfo=self.config.dsn,
                max_size=10,
                kwargs={"autocommit": True},
            )

        saver = AsyncPostgresSaver(self._apool)
        await saver.setup()
        return saver

    def shutdown(self):
        if self._pool:
            self._pool.close()

    async def ashutdown(self):
        if self._apool:
            await self._apool.close()