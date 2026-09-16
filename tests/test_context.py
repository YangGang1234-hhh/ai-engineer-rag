from app.context import assemble_context  # 导入待测试的上下文组装函数。
from app.retrieval import RetrievedChunk  # 导入统一检索结果类型。
from app.chunking import count_tokens  # 用于计算测试上下文的 token 数。


def create_chunk(
    chunk_id: str,
    content: str,
    position: int,
) -> RetrievedChunk:
    """创建测试用的检索 chunk。"""

    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="document-1",
        content=content,
        section_path="RAG 基础",
        position=position,
        token_count=count_tokens(content),
        score=0.03 - position * 0.001,
        source="hybrid",
    )


def test_assemble_context_formats_evidence_and_counts_tokens() -> None:
    """上下文应保留证据编号、章节路径和完整正文。"""

    chunks = [
        create_chunk(
            chunk_id="chunk-1",
            content="混合检索会融合向量检索和关键词检索。",
            position=0,
        ),
        create_chunk(
            chunk_id="chunk-2",
            content="RRF 会根据不同检索列表中的排名融合结果。",
            position=1,
        ),
    ]

    result = assemble_context(
        chunks=chunks,
        max_tokens=500,
    )

    # 两条短 chunk 都应被完整加入上下文。
    assert len(result.chunks) == 2
    assert result.skipped_count == 0

    # 每条证据都应有编号、章节和正文。
    assert "[证据 1]" in result.context_text
    assert "[证据 2]" in result.context_text
    assert "章节：RAG 基础" in result.context_text
    assert "混合检索会融合向量检索和关键词检索。" in result.context_text
    assert "RRF 会根据不同检索列表中的排名融合结果。" in result.context_text

    # 返回的 token 数必须与最终上下文完全一致。
    assert result.token_count == count_tokens(result.context_text)


def test_assemble_context_skips_complete_chunk_when_budget_is_insufficient() -> None:
    """预算不足时应跳过完整 chunk，而不是截断正文。"""

    first_chunk = create_chunk(
        chunk_id="chunk-1",
        content="混合检索会融合向量检索和关键词检索。",
        position=0,
    )
    second_chunk = create_chunk(
        chunk_id="chunk-2",
        content="这是一段较长的补充证据，用于验证上下文预算限制。",
        position=1,
    )

    # 计算只容纳第一条完整证据时所需的预算。
    first_only = assemble_context(
        chunks=[first_chunk],
        max_tokens=500,
    )

    result = assemble_context(
        chunks=[first_chunk, second_chunk],
        max_tokens=first_only.token_count,
    )

    # 第一条应完整保留，第二条应整体跳过。
    assert [chunk.chunk_id for chunk in result.chunks] == ["chunk-1"]
    assert result.skipped_count == 1

    # 第一条正文不能被截断。
    assert first_chunk.content in result.context_text

    # 最终上下文不能超过预算。
    assert result.token_count <= first_only.token_count