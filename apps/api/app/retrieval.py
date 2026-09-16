from dataclasses import dataclass
from typing import Any
from sqlalchemy import select
from sqlalchemy.orm import Session  # 表示一次 SQLite 数据库会话。
from app.models import DocumentChunk
from app.query_rewrite import rewrite_query_with_llm  # 调用 LLM 改写用户问题。
from app.full_text import search_chunks_by_keyword  # 导入 SQLite FTS5 关键词检索。
from app.vector_store import search_similar_chunks  # 导入 Qdrant 向量检索。
from app.reranking import rerank_chunks # 导入统一的检索结果类型。
from app.chunking import count_tokens, split_paragraph_into_sentences

@dataclass
class RetrievedChunk:
    """关键词检索和向量检索都能使用统一的结果结构。"""

    chunk_id:str 
    document_id:str
    content:str
    section_path:str
    position:int
    token_count:int
    score:float
    source:str
    parent_chunk_id: str | None = None
    chunk_level: str = "child"
    block_type: str = "prose"
    page_number: int | None = None

def from_keyword_result(result:Any) -> RetrievedChunk:
    """把SQLite 关键词检索结果转换为统一结构。"""

    return RetrievedChunk(
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        content=result.content,
        section_path = result.section_path,
        position=result.position,
        token_count=result.token_count,
        score=float(result.score),
        source="keyword",
        parent_chunk_id=getattr(result, "parent_chunk_id", None),
        chunk_level=getattr(result, "chunk_level", "child"),
        block_type=getattr(result, "block_type", "prose"),
        page_number=getattr(result, "page_number", None),
    )

def from_vector_result(result:Any) -> RetrievedChunk:
    """把 Qdrant 向量检索结果转换为统一结构。"""

    payload = result.payload or {} 

    return RetrievedChunk(
        chunk_id=str(payload["chunk_id"]),
        document_id=str(payload["document_id"]),
        content=str(payload["content"]),
        section_path=str(payload["section_path"]),
        position=int(payload["position"]),
        token_count=int(payload["token_count"]),
        score=float(result.score),
        source="vector",
        parent_chunk_id=payload.get("parent_chunk_id"),
        chunk_level=str(payload.get("chunk_level", "child")),
        block_type=str(payload.get("block_type", "prose")),
        page_number=payload.get("page_number"),
    )

def fuse_with_rrf(
        result_lists:list[list[RetrievedChunk]],
        limit:int=5,
        k:int=60,
) -> list[RetrievedChunk]:
    """使用RRF融合多路检索结果，并按融合分数降序返回。"""

    if limit <=0:
        raise ValueError("limit must be greater than zero")

    if k<=0:
        raise ValueError("k must be greater than zero")

    fused_scores:dict[str,float] = {} #保存每个chunk 的最终融合分数
    chunks_by_id:dict[str,RetrievedChunk] = {} # 保存chunk的完整信息

    # 依次处理关键词检索、向量检索的每一路结果。
    for results in result_lists:
        for rank,chunk in enumerate(results,start=1):
            # RRF 公式：排名越靠前，贡献分数越高。
            rrf_score=1/(k+rank)

            #同一个chunk被多路检索命中时，累加RRF分数。
            fused_scores[chunk.chunk_id]=(
                fused_scores.get(chunk.chunk_id,0.0) +rrf_score
            )

            # 保存第一次见到的chunk内容，避免重复返回。
            chunks_by_id.setdefault(chunk.chunk_id,chunk)

    # 按融合分数从高到低排序，再截取Top-k.
    ranked_chunks=sorted(
        chunks_by_id.values(),
        key=lambda chunk:fused_scores[chunk.chunk_id],
        reverse=True,
    )

    # 创建新对象：score 改为RRF 融合分数，source 标记为hybrid.
    return [
        RetrievedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            content=chunk.content,
            section_path=chunk.section_path,
            position=chunk.position,
            token_count=chunk.token_count,
            score=fused_scores[chunk.chunk_id],
            source="hybrid",
            parent_chunk_id=chunk.parent_chunk_id,
            chunk_level=chunk.chunk_level,
            block_type=chunk.block_type,
            page_number=chunk.page_number,
        )
        for chunk in ranked_chunks[:limit]
    ]

def hybrid_search(
    query: str,
    db: Session,
    limit: int = 5,
    candidate_limit: int = 20,
    enable_query_rewrite: bool = True,
) -> list[RetrievedChunk]:
    """同时执行关键词检索和向量检索，并使用 RRF 融合结果。"""

    query = query.strip()  # 去掉问题两侧多余的空白。

    if not query:  # 空问题不应发起检索。
        raise ValueError("Query cannot be empty")

    if limit <= 0:  # 最终返回数量必须为正数。
        raise ValueError("limit must be greater than zero")

    if candidate_limit <= 0:  # 每路召回的候选数量必须为正数。
        raise ValueError("candidate_limit must be greater than zero")

        # 默认使用原始问题，保证 LLM 不可用时仍能检索。
    search_query = query
    extracted_keywords: list[str] | None = None
    follow_up_queries: list[str] = []

    try:
        if not enable_query_rewrite:
            raise RuntimeError("LLM query rewrite disabled")

        # 优先调用 LLM，获得语义查询和明确关键词。
        rewrite_result = rewrite_query_with_llm(query)

        # 向量检索使用 LLM 改写后的简洁查询。
        search_query = rewrite_result.search_query

        # 关键词检索使用 LLM 提取出的技术关键词。
        extracted_keywords = rewrite_result.keywords
        follow_up_queries = getattr(rewrite_result, "follow_up_queries", [])

    except (RuntimeError, ValueError):
        # LLM 未配置、网络失败或返回格式错误时自动降级。
        # 保留原问题，并让 FTS5 使用旧的 trigram 查询。
        search_query = query
        extracted_keywords = None
        follow_up_queries = []

    # SQLite FTS5：LLM 成功时使用 keywords，
    # LLM 失败时使用原有 trigram 兜底。
    # 第一跳使用主查询；第二跳最多使用两个 LLM 子问题补充候选。
    hop_queries = [search_query, *follow_up_queries[:2]]
    keyword_results = []
    vector_results = []
    for hop_index, hop_query in enumerate(hop_queries):
        keyword_results.extend(search_chunks_by_keyword(
            query=hop_query,
            keywords=extracted_keywords if hop_index == 0 else None,
            db=db,
            limit=candidate_limit,
        ))
        vector_results.extend(search_similar_chunks(
            query=hop_query,
            limit=candidate_limit,
        ))

    # 先把两种不同来源的返回对象转换为统一结构。
    keyword_chunks = [
        from_keyword_result(result)
        for result in keyword_results
    ]
    vector_chunks = [
        from_vector_result(result)
        for result in vector_results
    ]

     # 先保留较多候选，让 Reranker 有机会重新比较更多结果。
    fused_candidates = fuse_with_rrf(
        [keyword_chunks, vector_chunks],
        limit=candidate_limit,
    )

    try:
        # 使用改写后的查询和 RRF 候选进行 Cross-Encoder 精排。
        reranked_chunks = rerank_chunks(
            query=search_query,
            chunks=fused_candidates,
            top_k=limit,
        )
    except Exception:
        # Reranker 模型加载或推理失败时，退回 RRF 排序结果。
        # 这样本地模型异常不会让整个检索接口不可用。
        reranked_chunks = fused_candidates[:limit]

    # 检索接口只返回精排后的 child。这样页面展示的是精确命中的小证据，
    # 而不是被父块放大的整段内容。
    return reranked_chunks


def expand_chunks_for_context(
    children: list[RetrievedChunk],
    db: Session,
    *,
    max_evidence: int = 3,
    local_max_tokens: int = 320,
) -> list[RetrievedChunk]:
    """将精排 child 扩展为有限的局部上下文，供回答阶段使用。"""

    if max_evidence <= 0 or local_max_tokens <= 0:
        raise ValueError("context limits must be greater than zero")

    if not children or not hasattr(db, "scalars"):
        return children[:max_evidence]

    parent_ids = list(dict.fromkeys(child.parent_chunk_id or child.chunk_id for child in children))
    parents = {
        parent.id: parent
        for parent in db.scalars(select(DocumentChunk).where(DocumentChunk.id.in_(parent_ids))).all()
    }
    expanded: list[RetrievedChunk] = []
    resolved_parent_ids: set[str] = set()
    for child in children:
        if len(expanded) >= max_evidence:
            break
        parent_id = child.parent_chunk_id or child.chunk_id
        if parent_id in resolved_parent_ids:
            continue
        parent = parents.get(parent_id)
        if parent is None:
            # 兼容尚未完成父子块迁移的旧记录：继续使用精确 child，
            # 而不是因为父块缺失把本次回答变成“无证据”。
            resolved_parent_ids.add(parent_id)
            expanded.append(child)
            continue
        resolved_parent_ids.add(parent_id)
        content = (
            parent.content
            if parent.block_type == "table"
            else expand_child_with_parent(
                parent_content=parent.content,
                child_content=child.content,
                max_tokens=local_max_tokens,
            )
        )
        expanded.append(RetrievedChunk(
            # 引用保留真正命中的 child ID，便于追踪精确召回位置。
            chunk_id=child.chunk_id, document_id=child.document_id,
            content=content, section_path=parent.section_path,
            position=child.position, token_count=count_tokens(content),
            score=child.score, source="context", parent_chunk_id=parent.id,
            chunk_level="child", block_type=parent.block_type,
            page_number=parent.page_number,
        ))
    return expanded


def expand_child_with_parent(
    parent_content: str,
    child_content: str,
    max_tokens: int = 320,
) -> str:
    """从父块中找 child，并在不截断句子的前提下扩展邻近句子。"""

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")

    parent_sentences = split_paragraph_into_sentences(parent_content)
    child_sentences = split_paragraph_into_sentences(child_content)
    if not parent_sentences or not child_sentences:
        return child_content.strip()

    start = _find_sentence_sequence(parent_sentences, child_sentences)
    if start is None:
        # 数据已重建或清洗规则变更时，宁可保留命中的 child，也不能塞入无关父块。
        return child_content.strip()

    end = start + len(child_sentences)
    selected = parent_sentences[start:end]
    left = start - 1
    right = end

    # 交替向两侧扩展，优先保留命中位置附近的完整语句。
    while left >= 0 or right < len(parent_sentences):
        added = False
        for index, use_left in ((left, True), (right, False)):
            if index < 0 or index >= len(parent_sentences):
                continue
            candidate = (
                [parent_sentences[index], *selected]
                if use_left
                else [*selected, parent_sentences[index]]
            )
            if count_tokens("\n".join(candidate)) <= max_tokens:
                selected = candidate
                if use_left:
                    left -= 1
                else:
                    right += 1
                added = True
                break
        if not added:
            break

    return "\n".join(selected)


def _find_sentence_sequence(sentences: list[str], target: list[str]) -> int | None:
    """返回 target 在 sentences 中的起点；child 由同一父块生成时应能命中。"""

    for index in range(len(sentences) - len(target) + 1):
        if sentences[index:index + len(target)] == target:
            return index
    return None


def resolve_parent_chunks(children: list[RetrievedChunk], db: Session) -> list[RetrievedChunk]:
    """兼容旧调用名：现在返回受预算限制的局部上下文，而不是完整父块。"""

    return expand_chunks_for_context(children, db)
