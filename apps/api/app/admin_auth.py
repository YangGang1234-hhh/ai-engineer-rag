"""最小管理员 API Key 鉴权，供内部管理接口复用。"""

from hmac import compare_digest
from typing import Annotated

from fastapi import Header, HTTPException, status

from app.config import get_settings


def require_admin_api_key(
    x_admin_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """校验 X-Admin-API-Key；未配置密钥时 fail closed。"""

    expected_key = get_settings().admin_api_key.strip()
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="管理员鉴权尚未配置。",
        )
    if x_admin_api_key is None or not compare_digest(
        x_admin_api_key,
        expected_key,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="管理员凭据无效。",
            headers={"WWW-Authenticate": "ApiKey"},
        )
