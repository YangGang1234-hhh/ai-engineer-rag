from types import SimpleNamespace  # 用简单对象模拟 OpenAI SDK 响应和上下文对象。
import pytest  # 用于断言非法引用会抛出异常。
from app import generation  # 导入待测试的回答生成模块。
from app.generation import (  # 导入回答模型和引用校验函数。
    GeneratedAnswer,
    validate_citations,
)

class FakeCompletions:
    """模拟 OpenAI chat.completions 接口。"""

    def __init__(self) -> None:
        self.received_arguments = None  # 保存发送给模型的请求参数。

    def create(self, **kwargs):
        """记录请求，并返回一份模拟的结构化回答。"""

        self.received_arguments = kwargs

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"answer":"混合检索结合了向量检索和关键词检索[1]，'
                            '通常比单一检索更稳定[1]。",'
                            '"citations":[1]}'
                        )
                    )
                )
            ]
        )


class FakeLLMClient:
    """模拟百炼等 OpenAI 兼容客户端。"""

    def __init__(self) -> None:
        self.chat = SimpleNamespace(
            completions=FakeCompletions()
        )


def test_generate_answer_uses_context_and_returns_citations(monkeypatch) -> None:
    """回答生成应携带证据，并返回结构化引用。"""

    fake_client = FakeLLMClient()  # 创建假的 LLM 客户端。

    # 替换真实客户端，避免测试访问网络。
    monkeypatch.setattr(
        generation,
        "get_llm_client",
        lambda: fake_client,
    )

    # 提供测试所需的模型配置。
    monkeypatch.setattr(
        generation,
        "get_settings",
        lambda: SimpleNamespace(
            llm_model="qwen3.5-flash",
            generation_llm_model="qwen-plus",
            llm_max_retries=2,
        ),
    )

    # 模拟上下文组装结果。
    fake_context = SimpleNamespace(
    context_text=(
                "[证据 1]\n"
                "章节：混合检索\n"
                "内容：混合检索结合向量检索和关键词检索。"
            ),
            # 与 context_text 中的 [证据 1] 对应。
            # 引用校验需要通过 chunks 数量判断引用是否越界。
            chunks=[
                SimpleNamespace(chunk_id="chunk-1"),
            ],
        )

    result = generation.generate_answer(
        query="什么是混合检索？",
        context=fake_context,
    )

    request = fake_client.chat.completions.received_arguments

    # 验证调用了正确的模型和稳定的生成参数。
    assert request["model"] == "qwen-plus"
    assert request["temperature"] == 0
    assert request["response_format"] == {"type": "json_object"}

    # 验证用户问题和证据都被发送给模型。
    user_content = request["messages"][1]["content"]
    assert "什么是混合检索？" in user_content
    assert "[证据 1]" in user_content
    assert "混合检索结合向量检索和关键词检索。" in user_content

    # 验证回答和引用被正确解析。
    assert result.answer.startswith("混合检索结合了")
    assert result.citations == [1]

def test_validate_citations_deduplicates_valid_citations() -> None:
    """合法引用应保留，重复引用应被去重。"""

    answer = GeneratedAnswer(
        answer="混合检索结合了两种检索方式。[1][2]",
        citations=[1, 2, 1],
    )

    context = SimpleNamespace(
        chunks=[
            SimpleNamespace(chunk_id="chunk-1"),
            SimpleNamespace(chunk_id="chunk-2"),
        ]
    )

    validated_answer = validate_citations(
        answer=answer,
        context=context,
    )

    # 引用顺序保留，但重复的 1 只保留一次。
    assert validated_answer.citations == [1, 2]


def test_validate_citations_rejects_out_of_range_citation() -> None:
    """超出实际证据范围的引用应被拒绝。"""

    answer = GeneratedAnswer(
        answer="这是一个没有依据的引用。[3]",
        citations=[3],
    )

    context = SimpleNamespace(
        chunks=[
            SimpleNamespace(chunk_id="chunk-1"),
            SimpleNamespace(chunk_id="chunk-2"),
        ]
    )

    # 当前只有两条证据，引用 [3] 必须抛出 ValueError。
    with pytest.raises(ValueError, match="Invalid citation numbers"):
        validate_citations(
            answer=answer,
            context=context,
        )
