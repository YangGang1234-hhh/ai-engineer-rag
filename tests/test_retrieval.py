from types import SimpleNamespace  # 用简单对象模拟不同检索系统的返回结果。

from app.retrieval import (  # 导入统一结构、转换函数和 RRF 融合函数。
    RetrievedChunk,
    from_keyword_result,
    from_vector_result,
    fuse_with_rrf,
    hybrid_search,
    expand_child_with_parent,
)

def test_from_keyword_result_converts_to_unified_chunk() -> None:
    """关键词检索结果应转换为统一的 RetrievedChunk。"""

    # 模拟 SQLite FTS5 的关键词检索结果。
    keyword_result = SimpleNamespace(
        chunk_id="chunk-1",
        document_id="document-1",
        content="混合检索会融合多个检索器的结果。",
        section_path="RAG 基础 > 混合检索",
        position=3,
        token_count=18,
        score=-2.5,
    )

    chunk = from_keyword_result(keyword_result)  # 执行转换。

    assert chunk.chunk_id == "chunk-1"
    assert chunk.document_id == "document-1"
    assert chunk.score == -2.5
    assert chunk.source == "keyword"


def test_from_vector_result_converts_payload_to_unified_chunk() -> None:
    """Qdrant 向量结果应从 payload 中提取统一字段。"""

    # 模拟 Qdrant 的 ScoredPoint。
    vector_result = SimpleNamespace(
        score=0.82,
        payload={
            "chunk_id": "chunk-2",
            "document_id": "document-1",
            "content": "向量检索根据语义相似度召回内容。",
            "section_path": "RAG 基础 > 向量检索",
            "position": 4,
            "token_count": 16,
        },
    )

    chunk = from_vector_result(vector_result)  # 执行转换。

    assert chunk.chunk_id == "chunk-2"
    assert chunk.document_id == "document-1"
    assert chunk.score == 0.82
    assert chunk.source == "vector"

def test_fuse_with_rrf_deduplicates_and_rewards_multi_source_hits() -> None:
    """同一 chunk 被多路命中时应去重，并累加 RRF 分数。"""

    # 关键词检索：chunk-a 排第一，chunk-b 排第二。
    keyword_results = [
        RetrievedChunk(
            chunk_id="chunk-a",
            document_id="document-1",
            content="混合检索内容",
            section_path="混合检索",
            position=0,
            token_count=10,
            score=-3.0,
            source="keyword",
        ),
        RetrievedChunk(
            chunk_id="chunk-b",
            document_id="document-1",
            content="关键词检索内容",
            section_path="关键词检索",
            position=1,
            token_count=10,
            score=-2.0,
            source="keyword",
        ),
    ]

    # 向量检索：chunk-b 排第一，chunk-a 排第二。
    vector_results = [
        RetrievedChunk(
            chunk_id="chunk-b",
            document_id="document-1",
            content="关键词检索内容",
            section_path="关键词检索",
            position=1,
            token_count=10,
            score=0.85,
            source="vector",
        ),
        RetrievedChunk(
            chunk_id="chunk-a",
            document_id="document-1",
            content="混合检索内容",
            section_path="混合检索",
            position=0,
            token_count=10,
            score=0.80,
            source="vector",
        ),
    ]

    fused_results = fuse_with_rrf(
        [keyword_results, vector_results],
        limit=5,
        k=60,
    )

    # 两路都命中相同两个 chunk，因此最终只能返回两个去重后的结果。
    assert len(fused_results) == 2

    # 两个 chunk 都在两路中命中，但排名总和相同；
    # 此处预期保持关键词列表先出现的 chunk-a 排在前面。
    assert fused_results[0].chunk_id == "chunk-a"
    assert fused_results[1].chunk_id == "chunk-b"

    # 最终结果不再标记为单一路，而是 hybrid。
    assert fused_results[0].source == "hybrid"

    # chunk-a 的融合分数应为两路贡献之和。
    expected_score = 1 / (60 + 1) + 1 / (60 + 2)
    assert fused_results[0].score == expected_score

def test_hybrid_search_fuses_keyword_and_vector_results(monkeypatch) -> None:
    """混合检索应调用两路召回，并返回 RRF 融合后的结果。"""

    # 模拟 SQLite FTS5 返回的关键词检索结果。
    keyword_results = [
        SimpleNamespace(
            chunk_id="chunk-a",
            document_id="document-1",
            content="混合检索融合多个检索器的结果。",
            section_path="混合检索",
            position=0,
            token_count=12,
            score=-3.0,
        ),
        SimpleNamespace(
            chunk_id="chunk-b",
            document_id="document-1",
            content="关键词检索适合精确术语。",
            section_path="关键词检索",
            position=1,
            token_count=10,
            score=-2.0,
        ),
    ]

    # 模拟 Qdrant 返回的向量检索结果。
    vector_results = [
        SimpleNamespace(
            score=0.85,
            payload={
                "chunk_id": "chunk-b",
                "document_id": "document-1",
                "content": "关键词检索适合精确术语。",
                "section_path": "关键词检索",
                "position": 1,
                "token_count": 10,
            },
        ),
        SimpleNamespace(
            score=0.78,
            payload={
                "chunk_id": "chunk-c",
                "document_id": "document-1",
                "content": "向量检索可以理解语义相近的表达。",
                "section_path": "向量检索",
                "position": 2,
                "token_count": 14,
            },
        ),
    ]

    # 记录两路检索实际收到的参数，验证 candidate_limit 被正确传递。
    received_limits = []

    def fake_keyword_search(
            *,
            query: str,
            db,
            limit: int,
            keywords=None,
        ):
        """替代真实 FTS5 检索。"""

        received_limits.append(
        ("keyword", query, db, limit, keywords)
        )
        return keyword_results

    def fake_vector_search(*, query: str, limit: int):
        """替代真实 Qdrant 检索。"""

        received_limits.append(("vector", query, limit))
        return vector_results

    # 模拟 LLM 成功返回的查询改写结果。
    monkeypatch.setattr(
    "app.retrieval.rewrite_query_with_llm",
    lambda query: SimpleNamespace(
        search_query="混合检索与向量检索的区别",
        keywords=["混合检索", "向量检索"],
    ),
)
    # 将 retrieval 模块内部引用的真实检索函数替换为测试替身。
    monkeypatch.setattr(
        "app.retrieval.search_chunks_by_keyword",
        fake_keyword_search,
    )
    monkeypatch.setattr(
        "app.retrieval.search_similar_chunks",
        fake_vector_search,
    )
    rerank_arguments = []  # 记录 Reranker 收到的问题和候选结果。
    
    def fake_rerank_chunks(
        *,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """替代真实 Cross-Encoder，并模拟精排结果。"""

        rerank_arguments.append((query, chunks, top_k))

        # 模拟 Cross-Encoder 判断 chunk-b 最相关。
        chunks_by_id = {
            chunk.chunk_id: chunk
            for chunk in chunks
        }

        return [
            chunks_by_id["chunk-b"],
            chunks_by_id["chunk-a"],
        ][:top_k]

    # 替换真实 Cross-Encoder，避免测试下载或调用模型。
    # 注意：这段代码必须和 def fake_rerank_chunks 对齐。
    monkeypatch.setattr(
        "app.retrieval.rerank_chunks",
        fake_rerank_chunks,
    )
    
    fake_db = object()  # 本测试不使用真实数据库，只传入占位对象。

    results = hybrid_search(
        query="混合检索有什么作用？",
        db=fake_db,
        limit=2,
        candidate_limit=10,
    )

    # 验证两路检索都收到相同问题，并各自取前 10 个候选。
    assert received_limits == [
            (
                "keyword",
                "混合检索与向量检索的区别",
                fake_db,
                10,
                ["混合检索", "向量检索"],
            ),
            (
                "vector",
                "混合检索与向量检索的区别",
                10,
            ),
        ]

    # chunk-b 同时被两路命中，因此 RRF 分数最高，应排在第一。
    assert [result.chunk_id for result in results] == ["chunk-b", "chunk-a"]

    # 验证 Cross-Encoder 收到的是 LLM 改写后的问题。
    assert rerank_arguments[0][0] == "混合检索与向量检索的区别"

    # 验证最终只返回 limit=2 条结果。
    assert rerank_arguments[0][2] == 2

    # 验证 Reranker 收到的是 RRF 候选，而不是只收到最终两条。
    rerank_candidate_ids = {
        chunk.chunk_id
        for chunk in rerank_arguments[0][1]
    }
    assert rerank_candidate_ids == {"chunk-a", "chunk-b", "chunk-c"}


def test_expand_child_with_parent_keeps_local_complete_sentences() -> None:
    """回答上下文应只扩展命中子块附近的完整句子。"""

    parent = "第一句介绍背景。第二句说明 RAG 的检索挑战。第三句说明查询可能模糊。第四句介绍无关的部署内容。"
    child = "第二句说明 RAG 的检索挑战。第三句说明查询可能模糊。"

    expanded = expand_child_with_parent(parent, child, max_tokens=30)

    assert "第二句说明 RAG 的检索挑战。" in expanded
    assert "第三句说明查询可能模糊。" in expanded
    assert "第四句介绍无关的部署内容。" not in expanded

def test_hybrid_search_falls_back_when_llm_fails(monkeypatch) -> None:
    """LLM 调用失败时，应使用原问题继续检索。"""

    received_arguments = []

    def fake_rewrite(query: str):
        """模拟 LLM 调用失败。"""

        raise RuntimeError("temporary LLM failure")

    def fake_keyword_search(
        *,
        query: str,
        db,
        limit: int,
        keywords=None,
    ):
        """记录关键词检索收到的参数。"""

        received_arguments.append(
            ("keyword", query, db, limit, keywords)
        )
        return []

    def fake_vector_search(*, query: str, limit: int):
        """记录向量检索收到的参数。"""

        received_arguments.append(("vector", query, limit))
        return []

    monkeypatch.setattr(
        "app.retrieval.rewrite_query_with_llm",
        fake_rewrite,
    )
    monkeypatch.setattr(
        "app.retrieval.search_chunks_by_keyword",
        fake_keyword_search,
    )
    monkeypatch.setattr(
        "app.retrieval.search_similar_chunks",
        fake_vector_search,
    )


    fake_db = object()
    query = "请解释一下混合检索？"

    results = hybrid_search(
        query=query,
        db=fake_db,
        limit=3,
        candidate_limit=10,
    )

    # 两路都没有结果时，最终结果应为空。
    assert results == []

    # 失败后关键词检索收到原问题和 None，
    # None 表示使用 trigram 兜底，而不是 LLM keywords。
    assert received_arguments == [
        ("keyword", query, fake_db, 10, None),
        ("vector", query, 10),
    ]


def test_hybrid_search_can_disable_llm_query_rewrite(monkeypatch) -> None:
    """离线评测可跳过 LLM 改写，直接测本地检索和重排序。"""

    received = []
    monkeypatch.setattr(
        "app.retrieval.rewrite_query_with_llm",
        lambda query: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    monkeypatch.setattr(
        "app.retrieval.search_chunks_by_keyword",
        lambda **kwargs: received.append(("keyword", kwargs["query"])) or [],
    )
    monkeypatch.setattr(
        "app.retrieval.search_similar_chunks",
        lambda **kwargs: received.append(("vector", kwargs["query"])) or [],
    )

    results = hybrid_search(
        query="什么是混合检索？",
        db=object(),
        enable_query_rewrite=False,
    )

    assert results == []
    assert received == [("keyword", "什么是混合检索？"), ("vector", "什么是混合检索？")]
