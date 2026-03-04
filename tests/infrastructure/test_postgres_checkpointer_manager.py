import pytest

from admissions_conversation_engine.infrastructure.config.app_config import CheckpointerConfig
from admissions_conversation_engine.infrastructure.postgres_checkpointer_manager import (
    PostgresCheckpointerManager,
)


def _checkpointer_config() -> CheckpointerConfig:
    return CheckpointerConfig(kind="postgresql", dsn="postgresql://u:p@h:5432/db")


def test_get_checkpointer_creates_pool_and_returns_saver(monkeypatch) -> None:
    # Verifica que get_checkpointer crea un pool de conexiones sincrónico y devuelve un PostgresSaver configurado.
    created_pools: list = []
    setup_called: list[bool] = []

    class FakePool:
        def __init__(self, conninfo, max_size, kwargs):
            created_pools.append(self)
            self.conninfo = conninfo

        def close(self):
            pass

    class FakeSaver:
        def __init__(self, pool):
            self.pool = pool

        def setup(self):
            setup_called.append(True)

    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.postgres_checkpointer_manager.ConnectionPool",
        FakePool,
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.postgres_checkpointer_manager.PostgresSaver",
        FakeSaver,
    )

    manager = PostgresCheckpointerManager(_checkpointer_config())
    saver = manager.get_checkpointer()

    assert len(created_pools) == 1
    assert created_pools[0].conninfo == "postgresql://u:p@h:5432/db"
    assert isinstance(saver, FakeSaver)
    assert setup_called == [True]


def test_get_checkpointer_reuses_existing_pool_on_second_call(monkeypatch) -> None:
    # Verifica que get_checkpointer no crea un nuevo pool si ya existe uno (reutilización del pool).
    created_pools: list = []

    class FakePool:
        def __init__(self, conninfo, max_size, kwargs):
            created_pools.append(self)

        def close(self):
            pass

    class FakeSaver:
        def __init__(self, pool):
            pass

        def setup(self):
            pass

    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.postgres_checkpointer_manager.ConnectionPool",
        FakePool,
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.postgres_checkpointer_manager.PostgresSaver",
        FakeSaver,
    )

    manager = PostgresCheckpointerManager(_checkpointer_config())
    manager.get_checkpointer()
    manager.get_checkpointer()

    assert len(created_pools) == 1


@pytest.mark.asyncio
async def test_aget_checkpointer_creates_async_pool_and_returns_saver(monkeypatch) -> None:
    # Verifica que aget_checkpointer crea un pool asíncrono, lo abre y devuelve un AsyncPostgresSaver configurado.
    created_apools: list = []
    opened: list[bool] = []
    async_setup_called: list[bool] = []

    class FakeAsyncPool:
        def __init__(self, conninfo, max_size, kwargs, open):
            created_apools.append(self)
            self.conninfo = conninfo

        async def open(self):
            opened.append(True)

        async def close(self):
            pass

    class FakeAsyncSaver:
        def __init__(self, pool):
            self.pool = pool

        async def setup(self):
            async_setup_called.append(True)

    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.postgres_checkpointer_manager.AsyncConnectionPool",
        FakeAsyncPool,
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.postgres_checkpointer_manager.AsyncPostgresSaver",
        FakeAsyncSaver,
    )

    manager = PostgresCheckpointerManager(_checkpointer_config())
    saver = await manager.aget_checkpointer()

    assert len(created_apools) == 1
    assert created_apools[0].conninfo == "postgresql://u:p@h:5432/db"
    assert opened == [True]
    assert isinstance(saver, FakeAsyncSaver)
    assert async_setup_called == [True]


@pytest.mark.asyncio
async def test_aget_checkpointer_reuses_existing_async_pool_on_second_call(monkeypatch) -> None:
    # Verifica que aget_checkpointer no abre un nuevo pool si ya existe uno (reutilización del pool asíncrono).
    created_apools: list = []

    class FakeAsyncPool:
        def __init__(self, conninfo, max_size, kwargs, open):
            created_apools.append(self)

        async def open(self):
            pass

        async def close(self):
            pass

    class FakeAsyncSaver:
        def __init__(self, pool):
            pass

        async def setup(self):
            pass

    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.postgres_checkpointer_manager.AsyncConnectionPool",
        FakeAsyncPool,
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.postgres_checkpointer_manager.AsyncPostgresSaver",
        FakeAsyncSaver,
    )

    manager = PostgresCheckpointerManager(_checkpointer_config())
    await manager.aget_checkpointer()
    await manager.aget_checkpointer()

    assert len(created_apools) == 1


def test_shutdown_closes_sync_pool_when_pool_exists(monkeypatch) -> None:
    # Verifica que shutdown() cierra el pool sincrónico si estaba abierto.
    closed: list[bool] = []

    class FakePool:
        def __init__(self, conninfo, max_size, kwargs):
            pass

        def close(self):
            closed.append(True)

    class FakeSaver:
        def __init__(self, pool):
            pass

        def setup(self):
            pass

    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.postgres_checkpointer_manager.ConnectionPool",
        FakePool,
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.postgres_checkpointer_manager.PostgresSaver",
        FakeSaver,
    )

    manager = PostgresCheckpointerManager(_checkpointer_config())
    manager.get_checkpointer()
    manager.shutdown()

    assert closed == [True]


def test_shutdown_does_nothing_when_pool_is_none() -> None:
    # Verifica que shutdown() no lanza error si no se creó ningún pool previamente.
    manager = PostgresCheckpointerManager(_checkpointer_config())
    manager.shutdown()  # debe ejecutarse sin error


@pytest.mark.asyncio
async def test_ashutdown_closes_async_pool_when_pool_exists(monkeypatch) -> None:
    # Verifica que ashutdown() cierra el pool asíncrono si estaba abierto.
    closed: list[bool] = []

    class FakeAsyncPool:
        def __init__(self, conninfo, max_size, kwargs, open):
            pass

        async def open(self):
            pass

        async def close(self):
            closed.append(True)

    class FakeAsyncSaver:
        def __init__(self, pool):
            pass

        async def setup(self):
            pass

    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.postgres_checkpointer_manager.AsyncConnectionPool",
        FakeAsyncPool,
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.postgres_checkpointer_manager.AsyncPostgresSaver",
        FakeAsyncSaver,
    )

    manager = PostgresCheckpointerManager(_checkpointer_config())
    await manager.aget_checkpointer()
    await manager.ashutdown()

    assert closed == [True]


@pytest.mark.asyncio
async def test_ashutdown_does_nothing_when_async_pool_is_none() -> None:
    # Verifica que ashutdown() no lanza error si no se creó ningún pool asíncrono previamente.
    manager = PostgresCheckpointerManager(_checkpointer_config())
    await manager.ashutdown()  # debe ejecutarse sin error


def test_probe_connection_succeeds_when_database_is_reachable(monkeypatch) -> None:
    # Verifica que probe_connection no lanza excepción cuando la conexión sincrónica es exitosa.
    import admissions_conversation_engine.infrastructure.postgres_checkpointer_manager as mod

    class FakeConn:
        def execute(self, query: str) -> None:
            pass

        def __enter__(self) -> "FakeConn":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(mod.psycopg, "connect", lambda dsn: FakeConn())

    manager = PostgresCheckpointerManager(_checkpointer_config())
    manager.probe_connection()  # No debe lanzar


def test_probe_connection_raises_runtime_error_when_database_is_unreachable(monkeypatch) -> None:
    # Verifica que probe_connection lanza RuntimeError con mensaje claro cuando la conexión sincrónica falla.
    import admissions_conversation_engine.infrastructure.postgres_checkpointer_manager as mod

    monkeypatch.setattr(mod.psycopg, "connect", lambda dsn: (_ for _ in ()).throw(Exception("Connection refused")))

    manager = PostgresCheckpointerManager(_checkpointer_config())

    try:
        manager.probe_connection()
        assert False, "Expected RuntimeError"
    except RuntimeError as error:
        assert "Checkpointer PostgreSQL no disponible" in str(error)
        assert "Connection refused" in str(error)


@pytest.mark.asyncio
async def test_aprobe_connection_succeeds_when_database_is_reachable(monkeypatch) -> None:
    # Verifica que aprobe_connection no lanza excepción cuando la conexión asíncrona es exitosa.
    import admissions_conversation_engine.infrastructure.postgres_checkpointer_manager as mod

    class FakeAsyncConn:
        async def execute(self, query: str) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncConn":
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

    class FakeAsyncConnection:
        @staticmethod
        async def connect(dsn: str) -> FakeAsyncConn:
            return FakeAsyncConn()

    monkeypatch.setattr(mod.psycopg, "AsyncConnection", FakeAsyncConnection)

    manager = PostgresCheckpointerManager(_checkpointer_config())
    await manager.aprobe_connection()  # No debe lanzar


@pytest.mark.asyncio
async def test_aprobe_connection_raises_runtime_error_when_database_is_unreachable(monkeypatch) -> None:
    # Verifica que aprobe_connection lanza RuntimeError con mensaje claro cuando la conexión asíncrona falla.
    import admissions_conversation_engine.infrastructure.postgres_checkpointer_manager as mod

    class FakeAsyncConnection:
        @staticmethod
        async def connect(dsn: str) -> None:
            raise Exception("Connection refused")

    monkeypatch.setattr(mod.psycopg, "AsyncConnection", FakeAsyncConnection)

    manager = PostgresCheckpointerManager(_checkpointer_config())

    try:
        await manager.aprobe_connection()
        assert False, "Expected RuntimeError"
    except RuntimeError as error:
        assert "Checkpointer PostgreSQL no disponible" in str(error)
        assert "Connection refused" in str(error)
