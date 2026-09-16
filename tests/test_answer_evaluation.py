import pytest
from types import SimpleNamespace

from app import answer_evaluation
from app.answer_evaluation import (
    judge_answer_faithfulness,
    judge_answer_relevance,
    parse_faithfulness_judgment,
    parse_relevance_judgment,
)


class FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.received_arguments = None

    def create(self, **kwargs):
        self.received_arguments = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


def configure_fake_judge(monkeypatch, content: str) -> FakeCompletions:
    completions = FakeCompletions(content)
    monkeypatch.setattr(
        answer_evaluation,
        "get_llm_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    monkeypatch.setattr(
        answer_evaluation,
        "get_settings",
        lambda: SimpleNamespace(
            llm_model="qwen3.5-flash",
            judge_llm_model="qwen-plus",
        ),
    )
    return completions


def test_parse_relevance_judgment_accepts_a_complete_point_partition() -> None:
    judgment = parse_relevance_judgment(
        """{
          "relevance_score": 4,
          "covered_points": ["外部知识库检索"],
          "missing_points": ["作为上下文交给大模型生成"],
          "reason": "回答覆盖了检索，但遗漏生成阶段。"
        }""",
        ["外部知识库检索", "作为上下文交给大模型生成"],
    )

    assert judgment.relevance_score == 4
    assert judgment.missing_points == ["作为上下文交给大模型生成"]


def test_parse_relevance_judgment_rejects_unknown_or_unclassified_points() -> None:
    with pytest.raises(ValueError, match="partition expected answer points"):
        parse_relevance_judgment(
            """{
              "relevance_score": 3,
              "covered_points": ["模型幻觉"],
              "missing_points": [],
              "reason": "返回了不属于本题黄金集的要点。"
            }""",
            ["外部知识库检索"],
        )


def test_parse_relevance_judgment_rejects_perfect_score_with_missing_points() -> None:
    with pytest.raises(ValueError, match="cannot be 5"):
        parse_relevance_judgment(
            """{
              "relevance_score": 5,
              "covered_points": ["外部知识库检索"],
              "missing_points": ["作为上下文交给大模型生成"],
              "reason": "遗漏了一个要点却错误给满分。"
            }""",
            ["外部知识库检索", "作为上下文交给大模型生成"],
        )


def test_parse_faithfulness_judgment_validates_score_range() -> None:
    judgment = parse_faithfulness_judgment(
        """{
          "faithfulness_score": 5,
          "unsupported_claims": [],
          "invalid_citations": [],
          "reason": "所有结论均能由证据支持。"
        }"""
    )

    assert judgment.faithfulness_score == 5

    with pytest.raises(ValueError, match="invalid faithfulness"):
        parse_faithfulness_judgment(
            """{
              "faithfulness_score": 6,
              "unsupported_claims": [],
              "invalid_citations": [],
              "reason": "分数超出范围。"
            }"""
        )


def test_judge_answer_relevance_sends_exact_expected_points(monkeypatch) -> None:
    completions = configure_fake_judge(
        monkeypatch,
        """{
          "relevance_score": 5,
          "covered_points": ["外部知识库检索"],
          "missing_points": [],
          "reason": "回答完整。"
        }""",
    )

    judgment = judge_answer_relevance(
        query="什么是 RAG？",
        answer="RAG 会检索外部知识库。",
        expected_answer_points=["外部知识库检索"],
    )

    assert judgment.relevance_score == 5
    request = completions.received_arguments
    assert request["temperature"] == 0
    assert request["model"] == "qwen-plus"
    assert request["response_format"] == {"type": "json_object"}
    assert "标准答案要点" in request["messages"][1]["content"]


def test_judge_answer_faithfulness_hides_expected_points(monkeypatch) -> None:
    completions = configure_fake_judge(
        monkeypatch,
        """{
          "faithfulness_score": 5,
          "unsupported_claims": [],
          "invalid_citations": [],
          "reason": "证据充分。"
        }""",
    )

    judgment = judge_answer_faithfulness(
        query="什么是 RAG？",
        answer="RAG 会检索外部资料[1]。",
        citations=[1],
        evidence_text="[证据 1] RAG 会从外部知识库检索资料。",
    )

    assert judgment.faithfulness_score == 5
    request = completions.received_arguments
    assert "标准答案要点" not in request["messages"][1]["content"]
    assert "实际检索证据" in request["messages"][1]["content"]
