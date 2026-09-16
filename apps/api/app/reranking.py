# from pydantic import BaseModel,Field

# class RerankItem(BaseModel):
#     """单个chunk的重排序结果。"""

#     # 用于关联原始RetrievedChunk.
#     chunk_id:str = Field(min_length=1)

#     # 相关性分数范围固定为0到1.
#     relevance_score:float = Field(ge=0.0,le=1.0)

# class RerankResponse(BaseModel):
#     """llm 返回重排序结果。"""

#     # 每个候选chunk 应该对应一个评分结果。
#     result:list[RerankItem]

# RERANK_SYSTEM_PROMPT="""
# 你是一个 RAG 检索结果重排序器。

# 你的任务是判断每个候选 chunk 与用户问题的相关程度，不要回答用户问题。

# 请严格只返回 JSON，不要返回 Markdown，不要添加解释：
# {
#   "results": [
#     {
#       "chunk_id": "候选 chunk 的原始 ID",
#       "relevance_score": 0.95
#     }
#   ]
# }

# 评分规则：
# 1. 只根据用户问题和候选 chunk 的内容评分。
# 2. chunk 直接回答问题核心内容时，分数应较高。
# 3. 只有背景相关但不能直接支持回答时，分数应为中等。
# 4. 与问题无关时，分数应接近 0。
# 5. 不要因为 chunk 中出现相同词语就盲目给高分，必须判断语义相关性。
# 6. 必须返回每个候选 chunk 的结果。
# 7. 必须原样保留每个 chunk_id，不得修改或生成新的 ID。
# 8. 不要回答用户问题，不要添加评分理由。
# """.strip()

from functools import lru_cache
from sentence_transformers import CrossEncoder #重排序模型

from app.config import get_settings 
from typing import TYPE_CHECKING  # 仅在类型检查阶段导入类型，避免循环导入。
if TYPE_CHECKING:
    from app.retrieval import RetrievedChunk  # 只供编辑器和类型检查使用。
@lru_cache
def get_reranker_model() -> CrossEncoder:
    """加载并缓存本地 CrossEncoder 重排序模型"""

    settings = get_settings() #读取RERANK_MODEL配置。

    # Cross-Encoder 会同时读取query 和chunk,
    # 直接输出二者之间的相关性分数
    return CrossEncoder(settings.reranker_model)

def rerank_chunks(
    query: str,
    chunks: list["RetrievedChunk"],
    top_k: int = 5,
) -> list["RetrievedChunk"]:
    """使用cross-encoder 对候选chunks 重新排序。"""
    from app.retrieval import RetrievedChunk
    query = query.strip()

    if not query:
        raise ValueError("query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not chunks:
        return []

    model = get_reranker_model()

    # Cross-Encoder 的每个输入都是第一对：用户问题以及候选chunk
    pairs=[
        (query,chunk.content)
        for chunk in chunks
    ]

    #对所有候选一次性打分，避免逐条调用模型。
    settings = get_settings()
    if settings.reranker_batch_size <= 0:
        raise ValueError("reranker_batch_size must be greater than zero")

    scores = model.predict(
        pairs,
        show_progress_bar=False,
        batch_size=settings.reranker_batch_size,
    )

    # 将原始分数与chunk绑定，并按照分数从高到低排序
    scored_chunks = sorted(
        zip(chunks,scores,strict=True),
        key = lambda item:float(item[1]),
        reverse=True,
    )

    # 先过滤低于阈值的弱相关 chunk，再按 top_k 截取结果。
    filtered_chunks = [
        (chunk, float(score))
        for chunk, score in scored_chunks
        if float(score) >= settings.reranker_min_score
    ]

    # 返回新的统一结果对象，并把 score 更新为 Cross-Encoder 分数。
    return [
        RetrievedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            content=chunk.content,
            section_path=chunk.section_path,
            position=chunk.position,
            token_count=chunk.token_count,
            score=score,
            source="reranked",
            parent_chunk_id=chunk.parent_chunk_id,
            chunk_level=chunk.chunk_level,
            block_type=chunk.block_type,
            page_number=chunk.page_number,
        )
        for chunk, score in filtered_chunks[:top_k]
    ]
