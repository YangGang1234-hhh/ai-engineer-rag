from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app import admin_auth
from app.config import PROJECT_ROOT, Settings


def test_require_admin_api_key_allows_matching_key(monkeypatch) -> None:
    monkeypatch.setattr(
        admin_auth,
        "get_settings",
        lambda: SimpleNamespace(admin_api_key="test-admin-secret"),
    )

    admin_auth.require_admin_api_key(x_admin_api_key="test-admin-secret")


@pytest.mark.parametrize("provided_key", [None, "wrong-secret"])
def test_require_admin_api_key_rejects_missing_or_invalid_key(
    monkeypatch,
    provided_key,
) -> None:
    monkeypatch.setattr(
        admin_auth,
        "get_settings",
        lambda: SimpleNamespace(admin_api_key="test-admin-secret"),
    )

    with pytest.raises(HTTPException) as error:
        admin_auth.require_admin_api_key(x_admin_api_key=provided_key)

    assert error.value.status_code == 401


def test_require_admin_api_key_fails_closed_when_not_configured(monkeypatch) -> None:
    monkeypatch.setattr(
        admin_auth,
        "get_settings",
        lambda: SimpleNamespace(admin_api_key=""),
    )

    with pytest.raises(HTTPException) as error:
        admin_auth.require_admin_api_key(x_admin_api_key="anything")

    assert error.value.status_code == 503


def test_settings_uses_absolute_root_env_file() -> None:
    """配置路径应由模块位置决定，不受 Uvicorn 当前工作目录影响。"""

    assert Settings.model_config["env_file"] == PROJECT_ROOT / ".env"


def test_settings_resolves_local_data_paths_from_project_root() -> None:
    """SQLite 与 Qdrant 路径不得因 Uvicorn 启动目录变化而漂移。"""

    settings = Settings(
        database_url="sqlite:///./data/test.db",
        qdrant_path="./data/test-qdrant",
    )

    assert settings.database_url == f"sqlite:///{(PROJECT_ROOT / 'data/test.db').as_posix()}"
    assert settings.qdrant_path == str(PROJECT_ROOT / "data/test-qdrant")
