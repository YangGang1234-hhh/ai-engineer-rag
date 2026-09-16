"""RAG 问答的查询准入层：先做确定性输入安全校验。"""

from enum import StrEnum
import re

from pydantic import BaseModel


class QueryGuardDecision(StrEnum):
    """准入层的标准决策；范围与证据判定将在后续步骤补充。"""

    ALLOW = "allow"
    INVALID_INPUT = "invalid_input"
    UNSAFE = "unsafe"
    OUT_OF_SCOPE = "out_of_scope"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class QueryGuardResult(BaseModel):
    """查询准入结果，供 API、日志与后续评测共同使用。"""

    decision: QueryGuardDecision
    reason: str
    user_message: str | None = None


MAX_QUERY_CHARACTERS = 1_000

# 仅命中明确要求绕过系统边界、泄露系统指令的指令式短语。
# 不拦截“什么是提示注入”等知识性问题，避免误伤知识库正常问答。
UNSAFE_QUERY_PATTERNS = (
    re.compile(r"忽略\s*(之前|以上|所有).{0,12}(指令|规则|要求)"),
    re.compile(r"无视\s*(之前|以上|所有).{0,12}(指令|规则|要求)"),
    re.compile(r"ignore\s+(all\s+)?(previous|above)\s+instructions?", re.I),
    re.compile(r"(输出|泄露|展示|告诉我).{0,8}(系统提示词|system\s+prompt)", re.I),
    re.compile(r"(reveal|show|print).{0,16}(system\s+prompt|hidden\s+instructions)", re.I),
)


def check_basic_query_safety(
    query: str,
    *,
    max_characters: int = MAX_QUERY_CHARACTERS,
) -> QueryGuardResult:
    """阻断无效输入与明显越权指令；不调用模型，也不依赖知识库。"""

    normalized_query = query.strip()
    if not normalized_query:
        return QueryGuardResult(
            decision=QueryGuardDecision.INVALID_INPUT,
            reason="empty_query",
            user_message="请输入一个与知识库相关的具体问题。",
        )
    if len(normalized_query) > max_characters:
        return QueryGuardResult(
            decision=QueryGuardDecision.INVALID_INPUT,
            reason="query_too_long",
            user_message="问题过长，请将问题控制在 1000 个字符以内。",
        )
    if any(pattern.search(normalized_query) for pattern in UNSAFE_QUERY_PATTERNS):
        return QueryGuardResult(
            decision=QueryGuardDecision.UNSAFE,
            reason="prompt_injection_or_system_prompt_request",
            user_message="该请求包含不支持的指令，因此无法处理。",
        )
    return QueryGuardResult(
        decision=QueryGuardDecision.ALLOW,
        reason="basic_safety_passed",
    )
