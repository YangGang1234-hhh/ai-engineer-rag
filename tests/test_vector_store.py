from types import SimpleNamespace  # 用简单对象模拟数据库 chunk 和应用配置。

from app import vector_store  # 导入待测试的向量库模块。


class FakeQdrantClient:
    """用于测试的假的 Qdrant 客户端，不会操作真实向量库。"""

    def __init__(self) -> None:
        self.created_collection = None  # 记录是否创建过 collection。
        self.upserted_points = []  # 记录写入的向量点。
        self.deleted_selector = None  # 记录按条件删除向量时使用的过滤器。

    def collection_exists(self, collection_name: str) -> bool:
        """测试开始时假设 collection 还不存在。"""

        return False

    def create_collection(self, collection_name: str, vectors_config) -> None:
        """记录 collection 创建参数。"""

        self.created_collection = {
            "name": collection_name,
            "vectors_config": vectors_config,
        }

    def upsert(self, collection_name: str, points: list) -> None:
        """记录写入的 collection 和向量点。"""

        self.upserted_points = points

    def delete(self, collection_name: str, points_selector) -> None:
        """记录删除操作使用的 collection 和过滤条件。"""

        self.deleted_selector = {
            "collection_name": collection_name,
            "points_selector": points_selector,
        }


def test_upsert_chunk_vectors_writes_vectors_and_payload(monkeypatch) -> None:
    """验证 chunk 能转换为向量，并写入完整的来源信息。"""

    fake_client = FakeQdrantClient()  # 创建假的 Qdrant 客户端。

    # 让被测试模块使用假的客户端，而不是打开真实 Qdrant。
    monkeypatch.setattr(
        vector_store,
        "get_qdrant_client",
        lambda: fake_client,
    )

    # 提供测试所需的最小配置对象。
    monkeypatch.setattr(
        vector_store,
        "get_settings",
        lambda: SimpleNamespace(qdrant_collection="document_chunks"),
    )

    # 用固定向量替代真实 Embedding 模型，测试会更快、更稳定。
    monkeypatch.setattr(
        vector_store,
        "embed_texts",
        lambda texts: [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
    )

    # 使用简单对象模拟两个数据库 chunk。
    chunks = [
        SimpleNamespace(
            id="chunk-1",
            document_id="document-1",
            content="RAG 通过检索外部知识辅助回答。",
            section_path="RAG 基础",
            position=0,
            token_count=12,
        ),
        SimpleNamespace(
            id="chunk-2",
            document_id="document-1",
            content="向量检索可以找到语义相近的内容。",
            section_path="RAG 基础 > 向量检索",
            position=1,
            token_count=13,
        ),
    ]

    vector_store.upsert_chunk_vectors(chunks)  # 执行待测试的向量写入函数。

    # 验证 collection 已按预期创建。
    assert fake_client.created_collection["name"] == "document_chunks"

    # 验证两个 chunk 产生了两个向量点。
    assert len(fake_client.upserted_points) == 2

    first_point = fake_client.upserted_points[0]  # 读取第一条向量点。

    # 验证向量内容和 Qdrant 点 ID。
    assert first_point.id == "chunk-1"
    assert first_point.vector == [0.1, 0.2, 0.3]

    # 验证检索结果需要的来源信息都写入了 payload。
    assert first_point.payload == {
        "document_id": "document-1",
        "chunk_id": "chunk-1",
        "content": "RAG 通过检索外部知识辅助回答。",
        "section_path": "RAG 基础",
        "position": 0,
        "token_count": 12,
    }

def test_delete_document_vectors_uses_document_id_filter(monkeypatch) -> None:
    """删除旧向量时，应只按指定 document_id 过滤。"""

    fake_client = FakeQdrantClient()  # 创建不会操作真实 Qdrant 的替身。

    # 将真实客户端替换为测试替身。
    monkeypatch.setattr(
        vector_store,
        "get_qdrant_client",
        lambda: fake_client,
    )

    # 提供删除函数所需的最小配置。
    monkeypatch.setattr(
        vector_store,
        "get_settings",
        lambda: SimpleNamespace(qdrant_collection="document_chunks"),
    )

    vector_store.delete_document_vectors("document-123")  # 执行待测试的删除操作。

    # 验证删除目标是正确的 collection。
    assert fake_client.deleted_selector["collection_name"] == "document_chunks"

    # 读取 Qdrant 过滤器中的第一个必须满足条件。
    condition = fake_client.deleted_selector["points_selector"].must[0]

    # 验证过滤字段和值都来自 document_id，而不是 chunk_id。
    assert condition.key == "document_id"
    assert condition.match.value == "document-123"