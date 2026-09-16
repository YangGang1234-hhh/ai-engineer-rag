import time

from pydantic import BaseModel,Field

from app.config import get_settings
from app.context import ContextAssembly
from app.llm_client import get_llm_client

class GeneratedAnswer(BaseModel):
    """LLM 根据检索证据生成最终的回答。"""

    # 面向用户展示的自然语言答案。
    answer:str = Field(min_length=1)

    # 答案使用的证据编号。
    citations:list[int] = Field(default_factory=list)

GENERATION_SYSTEM_PROMPT = """
你是一个基于证据回答问题的 RAG 助手。

你只能依据用户提供的证据回答，不得使用证据之外的知识补充结论。

请严格只返回 JSON，不要返回 Markdown，不要添加解释：
{
  "answer": "基于证据的简洁回答，并在相关陈述后使用 [1]、[2] 等引用标记",
  "citations": [1, 2]
}

请严格遵守以下规则：

1. 如果证据足够，直接回答用户问题。
2. 关键结论后必须添加对应的证据引用，例如 [1] 或 [1][2]。
3. citations 只能填写实际使用过的证据编号。
4. 如果证据不足，必须明确说明“知识库中没有足够证据回答该问题”。
5. 不得把模型自身知识伪装成知识库证据。
6. 不要虚构证据编号。
7. 不要回答证据之外的问题。
""".strip()

def parse_generated_answer(response_text:str) -> GeneratedAnswer:
    """解析并校验LLM 返回的答案 JSON。"""

    response_text = response_text.strip()

    if not response_text:
        raise ValueError("LLM return an empty answer")

    try:
        # 使用Pydantic 校验JSON 字段和数据类型。
        return GeneratedAnswer.model_validate_json(response_text)
    except ValueError as exc:
        raise ValueError("LLM returned invalid answer JSON") from exc

def validate_citations(
            answer: GeneratedAnswer,
            context: ContextAssembly,
        ) -> GeneratedAnswer:
    """验证 LLM 返回的引用编号是否对应真实证据。"""

    evidence_count = len(context.chunks)  # 当前上下文实际包含的证据数量。

    # 只允许引用 1 到 evidence_count 范围内的证据。
    invalid_citations = [
        citation
        for citation in answer.citations
        if citation < 1 or citation > evidence_count
    ]

    if invalid_citations:
        raise ValueError(
            f"Invalid citation numbers: {invalid_citations}"
        )

    # 去重但保留引用原有顺序。
    unique_citations = list(dict.fromkeys(answer.citations))

    return GeneratedAnswer(
        answer=answer.answer,
        citations=unique_citations,
    )

def generate_answer(
        query:str,
        context:ContextAssembly,
) -> GeneratedAnswer:
    """根据检索证据调用LLM生成带引用的回答。"""

    query = query.strip()

    if not query:
        raise ValueError("query cannot be empty")

    if not context.context_text.strip():
        raise ValueError("context cannot be empty")

    client = get_llm_client()

    if client is None:
        raise RuntimeError("llm client is not configured")

    settings = get_settings()

    user_message = (
        f"用户问题：\n{query}\n\n"
        f"检索证据：\n{context.context_text}"
    )

    if settings.llm_max_retries < 0:
        raise ValueError("llm_max_retries must not be negative")

    response = None
    last_error: Exception | None = None
    for attempt in range(settings.llm_max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=(
                    settings.generation_llm_model.strip()
                    or settings.llm_model
                ),
                messages=[
                    {"role":"system", "content":GENERATION_SYSTEM_PROMPT},
                    {"role":"user", "content":user_message},
                ],
                temperature=0,
                response_format={"type":"json_object"},
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt < settings.llm_max_retries:
                time.sleep(0.5 * (attempt + 1))

    if response is None:
        raise RuntimeError("LLM answer generation request failed") from last_error

    response_text =response.choices[0].message.content

    if response_text is None:
        raise RuntimeError("LLM return no answer content")

    # 先解析 JSON，再验证 citation 是否对应真实证据。
    parsed_answer = parse_generated_answer(response_text)

    return validate_citations(
        answer=parsed_answer,
        context=context,
    )
