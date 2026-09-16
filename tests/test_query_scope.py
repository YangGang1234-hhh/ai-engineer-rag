from types import SimpleNamespace

import pytest

from app import query_scope
from app.query_guard import QueryGuardDecision
from app.query_scope import classify_query_scope, parse_scope_judgment


class FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def configure_router(monkeypatch, content: str) -> FakeCompletions:
    completions = FakeCompletions(content)
    monkeypatch.setattr(
        query_scope,
        "get_llm_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    monkeypatch.setattr(
        query_scope,
        "get_settings",
        lambda: SimpleNamespace(
            llm_model="qwen3.5-flash", query_guard_llm_model="qwen-plus"
        ),
    )
    return completions


def test_scope_router_allows_in_scope_query(monkeypatch) -> None:
    completions = configure_router(
        monkeypatch, '{"decision":"in_scope","reason":"属于 RAG 检索问题。"}'
    )

    result = classify_query_scope("RRF 如何融合检索结果？")

    assert result.decision == QueryGuardDecision.ALLOW
    assert completions.request["model"] == "qwen-plus"
    assert completions.request["temperature"] == 0


def test_scope_router_rejects_chitchat(monkeypatch) -> None:
    configure_router(
        monkeypatch, '{"decision":"out_of_scope","reason":"天气问题不属于知识库范围。"}'
    )

    result = classify_query_scope("深圳今天天气怎么样？")

    assert result.decision == QueryGuardDecision.OUT_OF_SCOPE
    assert "仅回答知识库" in result.user_message


def test_parse_scope_judgment_rejects_invalid_decision() -> None:
    with pytest.raises(ValueError, match="invalid scope"):
        parse_scope_judgment('{"decision":"maybe","reason":"不合法"}')
