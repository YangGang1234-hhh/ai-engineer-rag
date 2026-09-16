from types import SimpleNamespace  # 用简单对象模拟 OpenAI SDK 的响应结构。

from app import query_rewrite  # 导入待测试的查询改写模块。


class FakeCompletions:
    """模拟 chat.completions 接口。"""

    def __init__(self) -> None:
        self.received_arguments = None  # 保存实际发送给模型的参数。

    def create(self, **kwargs):
        """记录请求参数，并返回模拟的 JSON 响应。"""

        self.received_arguments = kwargs  # 保存请求参数，供测试断言。

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"search_query":"混合检索与向量检索的区别",'
                            '"keywords":["混合检索","向量检索"]}'
                        )
                    )
                )
            ]
        )


class FakeLLMClient:
    """模拟 OpenAI 兼容客户端。"""

    def __init__(self) -> None:
        self.chat = SimpleNamespace(
            completions=FakeCompletions()
        )


def test_rewrite_query_with_llm_sends_structured_request(monkeypatch) -> None:
    """查询改写应按约定参数调用 LLM，并解析返回结果。"""

    fake_client = FakeLLMClient()  # 创建假的 LLM 客户端。

    # 替换真实客户端，避免测试访问百炼网络。
    monkeypatch.setattr(
        query_rewrite,
        "get_llm_client",
        lambda: fake_client,
    )

    # 提供测试所需的模型配置。
    monkeypatch.setattr(
        query_rewrite,
        "get_settings",
        lambda: SimpleNamespace(llm_model="qwen-plus"),
    )

    result = query_rewrite.rewrite_query_with_llm(
        "请解释一下混合检索和向量检索有什么区别？"
    )

    request = fake_client.chat.completions.received_arguments

    # 验证调用了正确的模型。
    assert request["model"] == "qwen-plus"

    # 查询改写应使用稳定的低随机性参数。
    assert request["temperature"] == 0

    # 验证要求模型返回 JSON 对象。
    assert request["response_format"] == {"type": "json_object"}

    # 验证系统提示词和用户问题都已发送。
    assert request["messages"][0]["role"] == "system"
    assert request["messages"][1] == {
        "role": "user",
        "content": "请解释一下混合检索和向量检索有什么区别？",
    }

    # 验证最终结果已被解析成结构化对象。
    assert result.search_query == "混合检索与向量检索的区别"
    assert result.keywords == ["混合检索", "向量检索"]