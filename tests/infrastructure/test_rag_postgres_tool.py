from admissions_conversation_engine.infrastructure.config.app_config import RagConfig
from admissions_conversation_engine.infrastructure.rag_postgres_tool import (
    PostgresVectorStoreTool,
)


def _rag_config() -> RagConfig:
    return RagConfig.model_validate(
        {
            "vector_store": {
                "kind": "postgresql",
                "dsn": "postgresql://u:p@h:5432/db",
                "collection": "rag_docs",
                "top_k": 2,
            },
            "embeddings": {
                "provider": "openai",
                "model": "text-embedding-3-large",
                "batch_size": 128,
                "api_key": "secret",
            },
        }
    )


def test_rag_postgres_tool_returns_joined_page_content(monkeypatch) -> None:
    class FakeDoc:
        def __init__(self, page_content: str):
            self.page_content = page_content

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def embed_query(self, query: str):
            assert query == "vector_size_probe"
            return [0.1, 0.2, 0.3]

    class FakeEngine:
        def __init__(self):
            self.init_calls = []

        def init_vectorstore_table(self, table_name: str, vector_size: int):
            self.init_calls.append((table_name, vector_size))

    class FakeVectorStore:
        def similarity_search(self, query: str, k: int):
            assert query == "admisiones"
            assert k == 2
            return [FakeDoc("doc 1"), FakeDoc("doc 2")]

    fake_engine = FakeEngine()
    fake_store = FakeVectorStore()

    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.rag_postgres_tool.OpenAIEmbeddings",
        FakeEmbeddings,
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.rag_postgres_tool.PGEngine.from_connection_string",
        lambda url: fake_engine,
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.rag_postgres_tool.PGVectorStore.create_sync",
        lambda **kwargs: fake_store,
    )

    tool = PostgresVectorStoreTool(rag_config=_rag_config())

    result = tool._run("admisiones")

    assert result == "doc 1\n\ndoc 2"
    assert fake_engine.init_calls == [("rag_docs", 3)]


def test_rag_postgres_tool_returns_empty_string_for_blank_query() -> None:
    tool = PostgresVectorStoreTool(rag_config=_rag_config())

    result = tool._run("   ")

    assert result == ""


def test_rag_postgres_tool_raises_for_unsupported_vector_store_kind() -> None:
    config = _rag_config().model_copy(
        update={
            "vector_store": _rag_config().vector_store.model_copy(update={"kind": "memory"}),
        }
    )
    tool = PostgresVectorStoreTool(rag_config=config)

    try:
        tool._run("hola")
        assert False, "Expected ValueError for unsupported vector store kind"
    except ValueError as error:
        assert "Unsupported vector store kind" in str(error)
