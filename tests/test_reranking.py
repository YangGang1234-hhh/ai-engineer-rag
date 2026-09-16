from app.reranking import rerank_chunks  # 导入待测试的重排序函数。
from app.retrieval import RetrievedChunk  # 导入统一检索结果类型。
from types import SimpleNamespace  # 用于模拟项目配置。

class FakeRerankerModel:
    """用于测试的假的 Cross-Encoder 模型。"""

    def __init__(self) -> None:
        self.received_pairs = None  # 保存模型收到的问题和 chunk 配对。

    def predict(self, pairs, show_progress_bar: bool, batch_size: int):
        """记录输入，并返回固定的测试分数。"""

        self.received_pairs = pairs  # 保存实际输入，供测试断言。

        # 分数故意设置成不同顺序，验证函数会重新排序。
        return [0.20, 0.95, 0.60]


def create_chunk(chunk_id: str, content: str) -> RetrievedChunk:
    """创建测试用的检索 chunk。"""

    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="document-1",
        content=content,
        section_path="RAG 基础",
        position=0,
        token_count=10,
        score=0.01,
        source="hybrid",
    )


def test_rerank_chunks_sorts_by_cross_encoder_score(monkeypatch) -> None:
    """Cross-Encoder 应按相关性分数降序返回候选 chunk。"""

    fake_model = FakeRerankerModel()  # 创建假的重排序模型。

    # 替换真实模型，避免下载或加载本地 Reranker。
    monkeypatch.setattr(
        "app.reranking.get_reranker_model",
        lambda: fake_model,
    )
    monkeypatch.setattr(
        "app.reranking.get_settings",
        lambda: SimpleNamespace(reranker_min_score=0.20, reranker_batch_size=4),
    )

    chunks = [
        create_chunk("chunk-a", "RAG 的基础概念。"),
        create_chunk("chunk-b", "混合检索融合向量检索和关键词检索。"),
        create_chunk("chunk-c", "文档切分可以保留章节结构。"),
    ]

    results = rerank_chunks(
        query="什么是混合检索？",
        chunks=chunks,
        top_k=2,
    )

    # 验证 query 和 chunk 正文被组成成对输入。
    assert fake_model.received_pairs == [
        ("什么是混合检索？", "RAG 的基础概念。"),
        ("什么是混合检索？", "混合检索融合向量检索和关键词检索。"),
        ("什么是混合检索？", "文档切分可以保留章节结构。"),
    ]

    # 分数顺序为 chunk-b、chunk-c、chunk-a，只取前两条。
    assert [chunk.chunk_id for chunk in results] == [
        "chunk-b",
        "chunk-c",
    ]

    # score 应替换为 Cross-Encoder 的分数。
    assert results[0].score == 0.95
    assert results[1].score == 0.60

    # 重排序后的结果使用新的来源标识。
    assert all(chunk.source == "reranked" for chunk in results)

class FakeThresholdModel:
    """返回包含低分结果的假的 Cross-Encoder 模型。"""

    def predict(self, pairs, show_progress_bar: bool, batch_size: int):
        """返回两个高分和一个低分结果。"""

        return [0.91, 0.19, 0.75]


def test_rerank_chunks_filters_scores_below_threshold(monkeypatch) -> None:
    """低于最低相关性分数的 chunk 应被过滤。"""

    # 使用假的模型，避免加载真实 Reranker。
    monkeypatch.setattr(
        "app.reranking.get_reranker_model",
        lambda: FakeThresholdModel(),
    )

    # 显式设置测试阈值，避免测试依赖真实 .env。
    monkeypatch.setattr(
        "app.reranking.get_settings",
        lambda: SimpleNamespace(reranker_min_score=0.20, reranker_batch_size=4),
    )

    chunks = [
        create_chunk("chunk-a", "高相关内容。"),
        create_chunk("chunk-b", "低相关内容。"),
        create_chunk("chunk-c", "中高相关内容。"),
    ]

    results = rerank_chunks(
        query="什么是混合检索？",
        chunks=chunks,
        top_k=5,
    )

    # chunk-b 分数为 0.19，小于 0.20，应被过滤。
    assert [chunk.chunk_id for chunk in results] == [
        "chunk-a",
        "chunk-c",
    ]

    # 保留下来的结果仍应使用 Cross-Encoder 分数。
    assert [chunk.score for chunk in results] == [0.91, 0.75]
