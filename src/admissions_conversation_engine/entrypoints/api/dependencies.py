from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from admissions_conversation_engine.infrastructure.config.config_bootstrap import get_app_config

_bearer = HTTPBearer()


def get_current_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    """Validate the Bearer JWT and return the decoded payload.

    Raises HTTP 401 if the token is missing, malformed, or expired.
    """
    app_config = get_app_config()
    try:
        payload: dict = jwt.decode(
            credentials.credentials,
            app_config.auth.jwt_secret_key,
            algorithms=[app_config.auth.jwt_algorithm],
        )
        return payload
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
