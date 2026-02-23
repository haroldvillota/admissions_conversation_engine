from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TenantConfig(BaseModel):
    """
    Configuración de negocio del tenant (institución, términos, temas permitidos, etc.).
    Pertenece al dominio por representar entidades de configuración del negocio.
    """

    model_config = ConfigDict(frozen=True)

    institution: str
    terms_of_service: str
    allowed_topics: str
    tone: str
    language_fallback: str
    allowed_languages: str
