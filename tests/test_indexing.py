from hashlib import sha256  # 用于生成测试文档的内容哈希。

from sqlalchemy import create_engine, select  # 创建内存数据库并查询数据。
from sqlalchemy.orm import Session, sessionmaker  # 创建测试数据库会话。

from app import indexing  # 导入待测试的索引编排模块。
from app.database import Base  # 导入数据库表基类。
from app.models import Document, DocumentChunk  # 导入文档和 chunk 模型。


def create_test_session() -> Session:
    """创建一个不会影响真实数据库的内存 SQLite 会话。"""

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    # 在内存数据库中创建项目需要的表。
    Base.metadata.create_all(bind=engine)

    return sessionmaker(bind=engine)()


def test_index_document_reads_chunks_and_updates_status(monkeypatch) -> None:
    """验证文档 chunks 能被读取并完成索引状态更新。"""

    db = create_test_session()  # 创建独立的测试数据库。
    indexed_chunks = []  # 用于记录传给向量层的 chunks。
    deleted_document_ids = []  # 用于记录索引前请求清理的文档 ID。

    def fake_delete_document_vectors(document_id: str) -> None:
        """模拟删除旧向量，不操作真实 Qdrant。"""

        deleted_document_ids.append(document_id)  # 记录被请求清理的文档。

    def fake_upsert_chunk_vectors(chunks) -> None:
        """模拟向量写入，不真正加载模型或操作 Qdrant。"""

        indexed_chunks.extend(chunks)  # 保存传入的 chunks，供后续断言。

    # 把真实的旧向量删除函数替换成测试替身。
    monkeypatch.setattr(
    indexing,
    "delete_document_vectors",
    fake_delete_document_vectors,
)
    # 把真实的向量写入函数替换成测试替身。
    monkeypatch.setattr(
        indexing,
        "upsert_chunk_vectors",
        fake_upsert_chunk_vectors,
    )

    try:
        content = "# RAG 基础\n\nRAG 是检索增强生成。"

        document = Document(
            title="测试文档",
            source_type="markdown",
            raw_content=content,
            content_hash=sha256(content.encode("utf-8")).hexdigest(),
            status="chunked",
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        # 故意按照乱序加入 chunks，验证业务函数会按 position 排序。
        db.add_all(
            [
                DocumentChunk(
                    document_id=document.id,
                    section_path="RAG 基础 > 第二节",
                    content="第二个 chunk",
                    position=1,
                    token_count=6,
                    chunk_level="child",
                ),
                DocumentChunk(
                    document_id=document.id,
                    section_path="RAG 基础 > 第一节",
                    content="第一个 chunk",
                    position=0,
                    token_count=6,
                    chunk_level="child",
                ),
            ]
        )
        db.commit()

        result = indexing.index_document(
            document_id=document.id,
            db=db,
        )

        # 验证返回结果中的索引数量。
        assert result.chunks_indexed == 2

        # 验证向量层收到了全部 chunks。
        assert len(indexed_chunks) == 2

        # 验证传给向量层的顺序是原文顺序，而不是数据库加入顺序。
        assert indexed_chunks[0].position == 0
        assert indexed_chunks[1].position == 1

        assert deleted_document_ids == [document.id]
        # 验证向量索引成功后，文档状态变为 indexed。
        saved_document = db.get(Document, document.id)
        assert saved_document is not None
        assert saved_document.status == "indexed"

    finally:
        # 无论测试成功或失败，都关闭测试数据库会话。
        db.close()
