from admissions_conversation_engine.infrastructure.agent_builder import AgentBuilder
from admissions_conversation_engine.infrastructure.config.app_config import AppConfig
from admissions_conversation_engine.infrastructure.prompt_provider import FormattedPrompts


class FakeStateGraph:
    last_instance = None

    def __init__(self, state_type, context_schema=None):
        self.state_type = state_type
        self.context_schema = context_schema
        self.nodes = {}
        self.edges = []
        self.conditional_edges = []
        self.compile_checkpointer = None
        FakeStateGraph.last_instance = self

    def add_node(self, name, node):
        self.nodes[name] = node

    def add_edge(self, source, target):
        self.edges.append((source, target))

    def add_conditional_edges(self, source, router, mapping):
        self.conditional_edges.append((source, mapping))

    def compile(self, checkpointer=None):
        self.compile_checkpointer = checkpointer
        return {"graph": "compiled", "checkpointer": checkpointer}


def _app_config() -> AppConfig:
    return AppConfig.model_validate(
        {
            "rag": {
                "vector_store": {
                    "kind": "postgresql",
                    "dsn": "postgresql://u:p@h:5432/db",
                    "collection": "rag_docs",
                    "top_k": 4,
                },
                "embeddings": {
                    "provider": "openai",
                    "api_key": "embeddings-key",
                    "model": "text-embedding-3-large",
                    "batch_size": 128,
                },
            },
            "llm": {
                "default": {
                    "provider": "openai",
                    "api_key": "default-key",
                    "model": "gpt-4.1-mini",
                    "temperature": 0.0,
                },
                "guardrail": {
                    "provider": "openai",
                    "api_key": "guardrail-key",
                    "model": "gpt-4.1-mini",
                    "temperature": 0.0,
                },
                "react": {
                    "provider": "openai",
                    "api_key": "react-key",
                    "model": "gpt-4.1",
                    "temperature": 0.0,
                },
                "translator": {
                    "provider": "openai",
                    "api_key": "translator-key",
                    "model": "gpt-4.1-nano",
                    "temperature": 0.0,
                },
            },
            "checkpointer": {
                "kind": "postgresql",
                "dsn": "postgresql://u:p@h:5432/db",
            },
            "observability": {
                "provider": "langfuse",
                "public_key": "pk",
                "secret_key": "sk",
                "base_url": "https://cloud.langfuse.com",
            },
            "tenant": {
                "institution": "Universidad Europea",
                "terms_of_service": "https://example.com/tos",
                "allowed_topics": "Admisiones",
                "tone": "Empatico",
                "language_fallback": "en-US",
                "allowed_languages": "es-ES,en-US",
            },
            "auth": {
                "jwt_secret_key": "test-secret-key",
                "jwt_algorithm": "HS256",
                "jwt_expire_minutes": 60,
            },
        }
    )


def test_agent_builder_builds_graph_with_expected_nodes_and_compiles(monkeypatch) -> None:
    # Verifica que AgentBuilder construye el grafo con los 9 nodos esperados y lo compila con el checkpointer.
    class FakeLLM:
        def __init__(self, model: str):
            self.model = model

        def bind_tools(self, _tools):
            return self

    class FakeLLMFactory:
        def __init__(self, profile):
            self._profile = profile

        def probe_connection(self) -> None:
            pass

        def build_llm(self):
            return FakeLLM(self._profile.model)

    class FakePromptProvider:
        def __init__(self, langfuse_client, tenant):
            pass

        def get_formatted_prompts(self):
            return FormattedPrompts(
                guardrail="guardrail-prompt",
                language_detector="language-detector-prompt",
                off_hours="off-hours-prompt",
                low_scoring="low-scoring-prompt",
                overflow="overflow-prompt",
                max_retries="max-retries-prompt",
            )

    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.agent_builder.StateGraph",
        FakeStateGraph,
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.agent_builder.LLMFactory",
        FakeLLMFactory,
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.agent_builder.PromptProvider",
        FakePromptProvider,
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.agent_builder.PostgresVectorStoreTool.probe_connection",
        lambda self: None,
    )

    checkpointer = object()
    builder = AgentBuilder(
        app_config=_app_config(),
        checkpointer=checkpointer,
        langfuse_client=None,
    )

    compiled = builder.build()
    graph = FakeStateGraph.last_instance

    assert compiled == {"graph": "compiled", "checkpointer": checkpointer}
    assert graph is not None
    assert graph.compile_checkpointer is checkpointer
    assert set(graph.nodes) == {
        "setup",
        "language_detector",
        "guardrail",
        "case_router",
        "off_hours_node",
        "tools",
        "low_scoring_node",
        "overflow_node",
        "max_retries_node",
    }
    assert ("setup", "language_detector") in graph.edges
    assert ("language_detector", "guardrail") in graph.edges


def test_agent_builder_fetches_all_prompts_from_langfuse(monkeypatch) -> None:
    # Verifica que AgentBuilder solicita los 6 prompts esperados al cliente de Langfuse cuando está disponible.
    class FakeLLM:
        def bind_tools(self, _tools):
            return self

    class FakeLLMFactory:
        def __init__(self, profile):
            pass

        def probe_connection(self) -> None:
            pass

        def build_llm(self):
            return FakeLLM()

    class FakeLangfusePrompt:
        def __init__(self, name):
            self.prompt = f"{name}-from-langfuse"

    class FakeLangfuse:
        def __init__(self):
            self.fetched = []

        def get_prompt(self, name):
            self.fetched.append(name)
            return FakeLangfusePrompt(name)

    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.agent_builder.StateGraph",
        FakeStateGraph,
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.agent_builder.LLMFactory",
        FakeLLMFactory,
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.prompt_provider.render_guardrail_prompt",
        lambda *_: "guardrail-prompt",
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.prompt_provider.render_language_detector_prompt",
        lambda *_: "language-detector-prompt",
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.prompt_provider.render_case_off_hours_prompt",
        lambda *_: "off-hours-prompt",
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.prompt_provider.render_case_low_scoring_prompt",
        lambda *_: "low-scoring-prompt",
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.prompt_provider.render_case_overflow_prompt",
        lambda *_: "overflow-prompt",
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.prompt_provider.render_case_max_retries_prompt",
        lambda *_: "max-retries-prompt",
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.agent_builder.PostgresVectorStoreTool.probe_connection",
        lambda self: None,
    )

    langfuse_client = FakeLangfuse()
    builder = AgentBuilder(
        app_config=_app_config(),
        checkpointer=object(),
        langfuse_client=langfuse_client,
    )

    builder.build()

    assert set(langfuse_client.fetched) == {
        "guardrail",
        "language_detector",
        "case_off_hours",
        "case_low_scoring",
        "case_overflow",
        "case_max_retries",
    }
