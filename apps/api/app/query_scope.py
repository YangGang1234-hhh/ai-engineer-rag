"""使用低成本 LLM 将问题路由到知识库范围内或范围外。"""

from typing import Literal

from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm_client import get_llm_client
from app.query_guard import QueryGuardDecision, QueryGuardResult


KNOWLEDGE_BASE_SCOPE = (
    "本知识库只覆盖 AI 应用工程实践：RAG、文档切分、向量/关键词/混合检索、"
    "重排序、Embedding、上下文构建、引用回答、评测、Agent、工具调用与提示注入防护。"
)

SCOPE_ROUTER_SYSTEM_PROMPT = f"""
你是知识库问答的范围路由器。

{KNOWLEDGE_BASE_SCOPE}

只返回 JSON：
{{"decision":"in_scope","reason":"简短中文理由"}}

规则：
1. 只判断用户问题是否适合由上述知识库回答，不回答问题本身。
2. 天气、情感闲聊、写作、娱乐、与 AI 工程无关的编程或生活问题为 out_of_scope。
3. 即使问题提到 RAG/Agent，但需要某个未收录产品的最新资料，仍为 in_scope；是否有证据由后续检索判断。
4. decision 只能是 in_scope 或 out_of_scope。
""".strip()


class ScopeJudgment(BaseModel):
    """范围路由模型的严格结构化输出。"""

    decision: Literal["in_scope", "out_of_scope"]
    reason: str = Field(min_length=1)


def parse_scope_judgment(response_text: str) -> ScopeJudgment:
    """解析模型的范围路由 JSON。"""

    try:
        return ScopeJudgment.model_validate_json(response_text)
    except ValueError as exc:
        raise ValueError("LLM returned invalid scope routing JSON") from exc


def classify_query_scope(query: str) -> QueryGuardResult:
    """只分类问题范围；模型不可用时由调用方决定是否降级到证据准入。"""

    if not query.strip():
        raise ValueError("query cannot be empty")

    client = get_llm_client()
    if client is None:
        raise RuntimeError("LLM client is not configured")

    settings = get_settings()
    try:
        response = client.chat.completions.create(
            model=(
                settings.query_guard_llm_model.strip()
                or settings.llm_model
            ),
            messages=[
                {"role": "system", "content": SCOPE_ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        raise RuntimeError("LLM scope routing request failed") from exc

    response_text = response.choices[0].message.content
    if response_text is None:
        raise RuntimeError("LLM scope routing returned no content")

    judgment = parse_scope_judgment(response_text)
    if judgment.decision == "in_scope":
        return QueryGuardResult(
            decision=QueryGuardDecision.ALLOW,
            reason=f"scope_in_scope:{judgment.reason}",
        )
    return QueryGuardResult(
        decision=QueryGuardDecision.OUT_OF_SCOPE,
        reason=f"scope_out_of_scope:{judgment.reason}",
        user_message=(
            "当前助手仅回答知识库中与 RAG、检索增强、Agent 工程实践相关的问题。"
            "请换一个与知识库相关的问题。"
        ),
    )
