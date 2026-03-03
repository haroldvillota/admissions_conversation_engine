from __future__ import annotations

import os
import logging

from langfuse import Langfuse, get_client
from langfuse.langchain import CallbackHandler

from admissions_conversation_engine.infrastructure.config.app_config import AppConfig


def build_langfuse_client(app_config: AppConfig) -> tuple[Langfuse, CallbackHandler]:
    if app_config.observability.provider == "langfuse":
        if app_config.observability.secret_key is not None:
            os.environ["LANGFUSE_SECRET_KEY"] = app_config.observability.secret_key
        if app_config.observability.public_key is not None:
            os.environ["LANGFUSE_PUBLIC_KEY"] = app_config.observability.public_key
        if app_config.observability.base_url is not None:
            os.environ["LANGFUSE_BASE_URL"] = app_config.observability.base_url

    langfuse = get_client()
    observability_handler = CallbackHandler()

    logger = logging.getLogger(__name__)

    if langfuse.auth_check():
        logger.info("Langfuse client is authenticated and ready!")
    else:
        logger.error("Authentication failed. Please check your credentials and host.")

    return langfuse, observability_handler
