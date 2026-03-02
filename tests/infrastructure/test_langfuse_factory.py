from admissions_conversation_engine.infrastructure.config.app_config import AppConfig
from admissions_conversation_engine.infrastructure.langfuse_factory import build_langfuse_client


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
                "public_key": "pk-test",
                "secret_key": "sk-test",
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


def test_build_langfuse_client_sets_env_vars_from_observability_config(monkeypatch) -> None:
    # Verifica que build_langfuse_client configura las variables de entorno de Langfuse con los valores de la config.
    env_set: dict = {}

    class FakeLangfuseClient:
        def auth_check(self) -> bool:
            return True

    class FakeCallbackHandler:
        pass

    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.langfuse_factory.get_client",
        lambda: FakeLangfuseClient(),
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.langfuse_factory.CallbackHandler",
        lambda: FakeCallbackHandler(),
    )

    import os
    original_environ = os.environ.copy()

    build_langfuse_client(_app_config())

    assert os.environ.get("LANGFUSE_SECRET_KEY") == "sk-test"
    assert os.environ.get("LANGFUSE_PUBLIC_KEY") == "pk-test"
    assert os.environ.get("LANGFUSE_BASE_URL") == "https://cloud.langfuse.com"

    # Restore original environment
    for key in ["LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_BASE_URL"]:
        if key in original_environ:
            os.environ[key] = original_environ[key]
        elif key in os.environ:
            del os.environ[key]


def test_build_langfuse_client_returns_client_and_handler(monkeypatch) -> None:
    # Verifica que build_langfuse_client devuelve el cliente Langfuse y el CallbackHandler como tupla.
    class FakeLangfuseClient:
        def auth_check(self) -> bool:
            return True

    class FakeCallbackHandler:
        pass

    fake_client = FakeLangfuseClient()
    fake_handler = FakeCallbackHandler()

    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.langfuse_factory.get_client",
        lambda: fake_client,
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.langfuse_factory.CallbackHandler",
        lambda: fake_handler,
    )

    client, handler = build_langfuse_client(_app_config())

    assert client is fake_client
    assert handler is fake_handler


def test_build_langfuse_client_calls_auth_check(monkeypatch) -> None:
    # Verifica que build_langfuse_client verifica la autenticación del cliente Langfuse al inicializarse.
    auth_check_called: list[bool] = []

    class FakeLangfuseClient:
        def auth_check(self) -> bool:
            auth_check_called.append(True)
            return True

    class FakeCallbackHandler:
        pass

    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.langfuse_factory.get_client",
        lambda: FakeLangfuseClient(),
    )
    monkeypatch.setattr(
        "admissions_conversation_engine.infrastructure.langfuse_factory.CallbackHandler",
        lambda: FakeCallbackHandler(),
    )

    build_langfuse_client(_app_config())

    assert auth_check_called == [True]
