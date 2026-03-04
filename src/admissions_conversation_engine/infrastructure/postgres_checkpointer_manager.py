from __future__ import annotations

import logging

import psycopg
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from admissions_conversation_engine.infrastructure.config.app_config import CheckpointerConfig

logger = logging.getLogger(__name__)


class PostgresCheckpointerManager:
    def __init__(self, config: CheckpointerConfig):
        self.config = config
        self._pool: ConnectionPool | None = None
        self._apool: AsyncConnectionPool | None = None

    def probe_connection(self) -> None:
        """Verifica que la base de datos PostgreSQL del checkpointer sea alcanzable.

        Lanza RuntimeError con un mensaje descriptivo si la conexión falla,
        para que los entrypoints puedan detener la aplicación de forma temprana.
        """
        dsn = self.config.dsn
        try:
            with psycopg.connect(dsn) as conn:
                conn.execute("SELECT 1")
        except Exception as exc:
            raise RuntimeError(
                f"Checkpointer PostgreSQL no disponible (DSN: '{dsn}'): {exc}"
            ) from exc

    async def aprobe_connection(self) -> None:
        """Versión asíncrona de probe_connection."""
        dsn = self.config.dsn
        try:
            async with await psycopg.AsyncConnection.connect(dsn) as conn:
                await conn.execute("SELECT 1")
        except Exception as exc:
            raise RuntimeError(
                f"Checkpointer PostgreSQL no disponible (DSN: '{dsn}'): {exc}"
            ) from exc

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
                open=False,
            )
            await self._apool.open()

        saver = AsyncPostgresSaver(self._apool)
        await saver.setup()
        return saver

    def shutdown(self):
        if self._pool:
            self._pool.close()

    async def ashutdown(self):
        if self._apool:
            await self._apool.close()