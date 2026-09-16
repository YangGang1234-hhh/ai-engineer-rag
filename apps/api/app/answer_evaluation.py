"""回答质量评测的数据契约：相关性与忠实度分别判定。"""

from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm_client import get_llm_client


RELEVANCE_JUDGE_SYSTEM_PROMPT = """
你是严格的 RAG 回答相关性裁判。你只评估回答是否回答了用户问题，以及是否覆盖给定的标准答案要点。

只返回 JSON，不要返回 Markdown 或解释：
{
  "relevance_score": 1,
  "covered_points": ["从给定标准要点中原样选择"],
  "missing_points": ["从给定标准要点中原样选择"],
  "reason": "简短中文理由"
}

规则：
1. relevance_score 只能是 1 到 5 的整数；5 表示完整、直接回答，1 表示基本没有回答。
2. 每个标准要点必须且只能出现在 covered_points 或 missing_points 之一。
3. 不得改写、补充或编造标准要点。
4. 只评估问题与答案，不评估答案引用是否真实。
5. 只有 missing_points 为空时 relevance_score 才能为 5；只要遗漏任何标准要点，最高只能为 4。
""".strip()


FAITHFULNESS_JUDGE_SYSTEM_PROMPT = """
你是严格的 RAG 忠实度裁判。你只能根据提供的检索证据判断答案是否有依据。

只返回 JSON，不要返回 Markdown 或解释：
{
  "faithfulness_score": 1,
  "unsupported_claims": ["答案中无法被证据支持的具体结论"],
  "invalid_citations": [1],
  "reason": "简短中文理由"
}

规则：
1. faithfulness_score 只能是 1 到 5 的整数；5 表示所有事实性结论均能由证据支持。
2. unsupported_claims 只列出证据无法推出的事实性结论；没有则返回空列表。
3. invalid_citations 只列出引用了不存在证据编号，或该编号不能支撑关联结论的引用编号；没有则返回空列表。
4. 不得使用自身知识补全证据，也不要根据答案是否流畅来提高分数。
""".strip()


class RelevanceJudgment(BaseModel):
    """问题回答相关性与标准要点覆盖度的裁判结果。"""

    relevance_score: int = Field(ge=1, le=5)
    covered_points: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)


class FaithfulnessJudgment(BaseModel):
    """回答是否可被实际检索证据支撑的裁判结果。"""

    faithfulness_score: int = Field(ge=1, le=5)
    unsupported_claims: list[str] = Field(default_factory=list)
    invalid_citations: list[int] = Field(default_factory=list)
    reason: str = Field(min_length=1)


def parse_relevance_judgment(
    response_text: str,
    expected_answer_points: list[str],
) -> RelevanceJudgment:
    """解析相关性裁判结果，并验证要点恰好覆盖黄金集中的全部项。"""

    try:
        judgment = RelevanceJudgment.model_validate_json(response_text)
    except ValueError as exc:
        raise ValueError("LLM returned invalid relevance evaluation JSON") from exc

    expected = set(expected_answer_points)
    covered = set(judgment.covered_points)
    missing = set(judgment.missing_points)

    if len(covered) != len(judgment.covered_points):
        raise ValueError("relevance evaluation contains duplicate covered points")
    if len(missing) != len(judgment.missing_points):
        raise ValueError("relevance evaluation contains duplicate missing points")
    if covered | missing != expected or covered & missing:
        raise ValueError("relevance evaluation must partition expected answer points")
    if judgment.missing_points and judgment.relevance_score == 5:
        raise ValueError("relevance score cannot be 5 when answer points are missing")

    return judgment


def parse_faithfulness_judgment(response_text: str) -> FaithfulnessJudgment:
    """解析忠实度裁判结果；事实判断留给后续 LLM 裁判提示词。"""

    try:
        return FaithfulnessJudgment.model_validate_json(response_text)
    except ValueError as exc:
        raise ValueError("LLM returned invalid faithfulness evaluation JSON") from exc


def judge_answer_relevance(
    *,
    query: str,
    answer: str,
    expected_answer_points: list[str],
) -> RelevanceJudgment:
    """调用 LLM 裁判，评估答案相关性和黄金要点覆盖度。"""

    if not query.strip() or not answer.strip() or not expected_answer_points:
        raise ValueError("query, answer, and expected answer points are required")

    response_text = _request_judgment(
        system_prompt=RELEVANCE_JUDGE_SYSTEM_PROMPT,
        user_message=(
            f"用户问题：\n{query}\n\n"
            f"系统回答：\n{answer}\n\n"
            "标准答案要点（必须原样分类）：\n"
            + "\n".join(f"- {point}" for point in expected_answer_points)
        ),
    )
    return parse_relevance_judgment(response_text, expected_answer_points)


def judge_answer_faithfulness(
    *,
    query: str,
    answer: str,
    citations: list[int],
    evidence_text: str,
) -> FaithfulnessJudgment:
    """调用 LLM 裁判，只以真实检索证据评估回答忠实度。"""

    if not query.strip() or not answer.strip() or not evidence_text.strip():
        raise ValueError("query, answer, and evidence text are required")

    response_text = _request_judgment(
        system_prompt=FAITHFULNESS_JUDGE_SYSTEM_PROMPT,
        user_message=(
            f"用户问题：\n{query}\n\n"
            f"系统回答：\n{answer}\n\n"
            f"回答声明使用的引用编号：{citations}\n\n"
            f"实际检索证据：\n{evidence_text}"
        ),
    )
    return parse_faithfulness_judgment(response_text)


def _request_judgment(*, system_prompt: str, user_message: str) -> str:
    """通过现有百炼兼容客户端请求一份严格 JSON 裁判结果。"""

    client = get_llm_client()
    if client is None:
        raise RuntimeError("LLM client is not configured")

    try:
        response = client.chat.completions.create(
            model=(
                get_settings().judge_llm_model.strip()
                or get_settings().llm_model
            ),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        raise RuntimeError("LLM answer evaluation request failed") from exc

    response_text = response.choices[0].message.content
    if response_text is None:
        raise RuntimeError("LLM answer evaluation returned no content")

    return response_text
