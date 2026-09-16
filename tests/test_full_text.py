# sha256 用于生成测试 Document 所需的内容哈希
from hashlib import sha256

# create_engine 创建测试专用内存数据库。
# text 用于创建和查询 FTS5 虚拟表。
from sqlalchemy import create_engine, text

# Session、sessionmaker 用于创建测试数据库会话。
from sqlalchemy.orm import Session, sessionmaker

# 导入 FTS 同步函数。
from app.full_text import (
    build_fts_match_query_from_keywords,
    build_fts_match_query,
    search_chunks_by_keyword,
    sync_document_chunks_to_fts,
)

# Base 用于创建普通业务表。
from app.database import Base

# 导入测试中需要写入的业务模型。
from app.models import Document, DocumentChunk


def create_test_session() -> Session:
    """创建包含业务表和 FTS5 表的临时内存数据库。"""

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    # 创建 documents 和 document_chunks 两张普通表。
    Base.metadata.create_all(bind=engine)

    # FTS5 是 SQLite 虚拟表，不属于 SQLAlchemy 模型，
    # 所以使用原生 SQL 单独创建。
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE VIRTUAL TABLE document_chunks_fts
                USING fts5(
                    document_id UNINDEXED,
                    chunk_id UNINDEXED,
                    content,
                    section_path,
                    tokenize = 'trigram'
                )
                """
            )
        )

    return sessionmaker(bind=engine)()


def test_sync_document_chunks_to_fts_is_idempotent() -> None:
    """同步应写入索引；重复同步后索引记录数不应增加。"""

    content = "# RAG 基础\n\nRAG 会先检索资料。"
    db = create_test_session()

    try:
        # 先创建一份原始文档。
        document = Document(
            title="FTS 测试文档",
            source_type="markdown",
            raw_content=content,
            content_hash=sha256(content.encode("utf-8")).hexdigest(),
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        # 再创建两个已经完成切分的 chunk。
        db.add_all(
            [
                DocumentChunk(
                    document_id=document.id,
                    section_path="RAG 基础",
                    content="RAG 会先检索资料。",
                    position=0,
                    token_count=10,
                    chunk_level="child",
                ),
                DocumentChunk(
                    document_id=document.id,
                    section_path="RAG 基础 > 混合检索",
                    content="RRF 可以融合多个排序结果。",
                    position=1,
                    token_count=10,
                    chunk_level="child",
                ),
            ]
        )
        db.commit()

        # 第一次同步应写入两个 FTS 索引记录。
        first_count = sync_document_chunks_to_fts(
            document_id=document.id,
            db=db,
        )
        db.commit()

        assert first_count == 2

        indexed_count_after_first_sync = db.execute(
            text("SELECT COUNT(*) FROM document_chunks_fts")
        ).scalar_one()

        assert indexed_count_after_first_sync == 2

        # 第二次同步前会先删除该文档旧索引，
        # 因此最终记录数仍是两个，而不会累积为四个。
        second_count = sync_document_chunks_to_fts(
            document_id=document.id,
            db=db,
        )
        db.commit()

        assert second_count == 2

        indexed_count_after_second_sync = db.execute(
            text("SELECT COUNT(*) FROM document_chunks_fts")
        ).scalar_one()

        assert indexed_count_after_second_sync == 2

    finally:
        # 测试结束后关闭临时数据库连接。
        db.close()

def test_search_chunks_by_keyword_returns_relevant_chunk() -> None:
    """关键词检索应返回匹配的 chunk，并带回业务表中的完整信息。"""

    content = "# RAG 笔记\n\n混合检索会融合多个召回结果。"
    db = create_test_session()

    try:
        # 创建一份测试文档。
        document = Document(
            title="关键词检索测试",
            source_type="markdown",
            raw_content=content,
            content_hash=sha256(content.encode("utf-8")).hexdigest(),
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        # 创建一条匹配“混合检索”的 chunk，
        # 再创建一条不相关 chunk，验证搜索能正确筛选。
        db.add_all(
            [
                DocumentChunk(
                    document_id=document.id,
                    section_path="RAG > 混合检索",
                    content="混合检索会融合向量检索与关键词检索。",
                    position=0,
                    token_count=20,
                    chunk_level="child",
                ),
                DocumentChunk(
                    document_id=document.id,
                    section_path="RAG > 评测",
                    content="评测可以计算 Recall 和 MRR。",
                    position=1,
                    token_count=20,
                    chunk_level="child",
                ),
            ]
        )
        db.commit()

        # 先将普通业务表中的 chunk 同步进 FTS5 索引。
        sync_document_chunks_to_fts(
            document_id=document.id,
            db=db,
        )
        db.commit()

        # 使用关键词执行检索。
        results = search_chunks_by_keyword(
            query="什么是混合检索？",
            db=db,
            limit=5,
        )

        # 只应命中包含该关键词的 chunk。
        assert len(results) == 1
        assert results[0].content == "混合检索会融合向量检索与关键词检索。"
        assert results[0].section_path == "RAG > 混合检索"
        assert results[0].document_id == document.id

    finally:
        db.close()

def test_build_fts_match_query_uses_or_for_chinese_question() -> None:
    """中文自然语言问题应拆为多个 trigram，并使用 OR 连接。"""

    query = build_fts_match_query("什么是混合检索？")

    assert query == (
        '"什么是" OR "么是混" OR "是混合" OR "混合检" OR "合检索"'
    )

def test_build_fts_match_query_from_keywords_uses_explicit_terms() -> None:
    """明确关键词应直接组成 OR 查询，而不是生成 trigram。"""

    match_query = build_fts_match_query_from_keywords(
        ["混合检索", "向量检索"]
    )

    assert match_query == '"混合检索" OR "向量检索"'


def test_search_chunks_by_keyword_accepts_llm_keywords() -> None:
    """关键词检索应支持使用 LLM 提取出的明确关键词。"""

    content = "# RAG 笔记\n\n混合检索会融合向量检索与关键词检索。"
    db = create_test_session()

    try:
        # 创建测试文档。
        document = Document(
            title="LLM 关键词测试",
            source_type="markdown",
            raw_content=content,
            content_hash=sha256(content.encode("utf-8")).hexdigest(),
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        # 创建一个包含“混合检索”的 chunk。
        db.add(
            DocumentChunk(
                document_id=document.id,
                section_path="RAG > 混合检索",
                content="混合检索会融合向量检索与关键词检索。",
                position=0,
                token_count=20,
                chunk_level="child",
            )
        )
        db.commit()

        # 将 chunk 同步到 FTS5。
        sync_document_chunks_to_fts(
            document_id=document.id,
            db=db,
        )
        db.commit()

        # query 使用自然语言，keywords 使用 LLM 提取结果。
        results = search_chunks_by_keyword(
            query="请解释一下它们有什么区别？",
            keywords=["混合检索", "向量检索"],
            db=db,
            limit=5,
        )

        # FTS5 应根据明确关键词命中该 chunk。
        assert len(results) == 1
        assert results[0].content == (
            "混合检索会融合向量检索与关键词检索。"
        )

    finally:
        # 关闭临时数据库会话。
        db.close()
