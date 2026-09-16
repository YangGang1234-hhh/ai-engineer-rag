from pydantic import BaseModel,Field #用于校验和解析LLM返回的结构化数据。
from app.config import get_settings
from app.llm_client import get_llm_client

class QueryRewriteResult(BaseModel):
    """LLM 查询改写后的标准结果。"""

    # 用于向量检索的完整、简洁查询
    search_query:str = Field(min_length=1)

    # 用于关键词检索的技术关键词列表。
    # 最多保留10个，避免模型输出过多噪声词。
    keywords:list[str] = Field(default_factory=list,max_length=10)

    # 最多两个补充检索问题，用于受限两跳检索。
    follow_up_queries: list[str] = Field(default_factory=list, max_length=2)

QUERY_REWRITE_SYSTEM_PROMPT = """
你是一个 RAG 检索查询分析器。

你的任务是分析用户问题并生成检索表达，不要回答问题。

请严格只返回 JSON，不要返回 Markdown，不要添加任何解释：
{
  "search_query": "适合语义检索的简洁查询",
  "keywords": ["适合关键词检索的术语"],
  "follow_up_queries": ["补充检索子问题"]
}

请严格遵守以下规则：

1. search_query 只能对用户原问题进行压缩、改写和去除问句表达，
   不得引入用户问题中没有出现的新技术概念。

2. keywords 只能提取用户问题中明确出现的技术术语、实体、产品名、
   函数名、指标名、版本号或其他具有检索价值的词语。

3. 不要根据常识补充相关但未被用户提及的概念。

4. 关键词提取删除没有检索价值的问句表达，例如：
   “请问”“请解释一下”“什么是”“如何”“有什么区别”。

5. 不要把“什么是”“为什么”“如何”“区别”“作用”等问法词
   作为关键词，除非它们本身是用户要检索的专业术语。

6. keywords 最多返回 5 个；如果用户问题中没有明确的技术关键词，
   返回空列表。

7. 不要回答用户问题，不要解释关键词选择过程。

8. follow_up_queries 最多两个。只有复杂比较、流程、多条件问题才生成，
   每个子问题必须由原问题直接推出，不能引入新概念；简单问题返回空列表。
""".strip()

def parse_query_rewrite_response(response_text:str) -> QueryRewriteResult:
    """解析并检验LLM返回的查询改写JSON."""

    response_text = response_text.strip()

    if not response_text:
        raise ValueError("LLM return an empty response")

    try:
        # 使用Pydantic 校验JSON 结构和字段类型。
        return QueryRewriteResult.model_validate_json(response_text)
    except ValueError as exc:
        # 将底层解析错误转换成更容易理解的业务错误。
        raise ValueError("LLM returned invalid query rewrite JSON") from exc

def rewrite_query_with_llm(query:str) -> QueryRewriteResult:
    """调用LLM改写用户问题，并提取关键词。"""

    query = query.strip()

    if not query: 
        raise ValueError("Query cannot be empty")

    client = get_llm_client()

    if client is None:
        raise RuntimeError("LLM client is not configured")

    settings = get_settings()

    try:
        response=client.chat.completions.create(
            model=settings.llm_model,
            messages=[
            {
                "role":"system",
                "content":QUERY_REWRITE_SYSTEM_PROMPT,

            },
            {
                "role":"user",
                "content":query,
            },
            ],
            temperature=0,
            response_format={"type":"json_object"},
        )
    except Exception as exc:
        # 保留原始异常作为cause,方便日志排查具体的API问题。
        raise RuntimeError("LLM query rewrite request failed" ) from exc

    # 读取模型返回的文本内容
    response_text = response.choices[0].message.content

    if response_text is None:
        raise RuntimeError("llm returned no content")

    return parse_query_rewrite_response(response_text)
