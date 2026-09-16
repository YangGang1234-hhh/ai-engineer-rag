# sha256 用于生成测试文档必须具备的内容哈希
from hashlib import sha256

# create_engine 创建仅供测试使用的内存 SQLite 数据库
# select 用于读取测试结果
from sqlalchemy import create_engine, select

# Session、sessionmaker 用于创建测试数据库会话
from sqlalchemy.orm import Session, sessionmaker

# Base 包含 Document 和 DocumentChunk 两张表的定义
from app.database import Base

# 导入本次要测试的业务函数
from app.document_processing import chunk_document

# 导入数据库模型，用于创建文档和检查 chunk 记录
from app.models import Document, DocumentChunk, DocumentTable


def create_test_session() -> Session:
    """创建一个临时内存数据库，不影响项目真实 SQLite 文件。"""

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    # 在内存数据库中创建 documents 和 document_chunks 表。
    Base.metadata.create_all(bind=engine)

    return sessionmaker(bind=engine)()


def test_chunk_document_creates_chunks_and_can_be_run_again() -> None:
    """切分应写入 chunk；重复执行不应产生重复记录。"""

    content = (
        "# RAG 基础\n\n"
        "RAG 会先检索相关资料，再基于资料生成回答。\n\n"
        "## 混合检索\n\n"
        "混合检索会融合向量检索与关键词检索的结果。"
    )

    # 创建独立测试数据库，并预先写入一份原始文档。
    db = create_test_session()

    try:
        document = Document(
            title="RAG 测试资料",
            source_type="markdown",
            raw_content=content,
            content_hash=sha256(content.encode("utf-8")).hexdigest(),
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        # 第一次切分：两个标题路径，因此预期创建两个 chunk。
        first_result = chunk_document(
            document_id=document.id,
            db=db,
            max_tokens=400,
        )

        assert first_result.chunks_created == 4
        assert first_result.document.status == "chunked"

        chunks_after_first_run = db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document.id)
            .order_by(DocumentChunk.position)
        ).all()

        assert len(chunks_after_first_run) == 4
        parents = [chunk for chunk in chunks_after_first_run if chunk.chunk_level == "parent"]
        children = [chunk for chunk in chunks_after_first_run if chunk.chunk_level == "child"]
        assert [chunk.section_path for chunk in parents] == ["RAG 基础", "RAG 基础 > 混合检索"]
        assert len(children) == 2
        assert all(chunk.parent_chunk_id for chunk in children)

        # 第二次切分：旧 chunk 应先被删除，再重新创建。
        # 最终数量仍应为 2，而不是累积成 4。
        second_result = chunk_document(
            document_id=document.id,
            db=db,
            max_tokens=400,
        )

        assert second_result.chunks_created == 4

        chunks_after_second_run = db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document.id)
        ).all()

        assert len(chunks_after_second_run) == 4

    finally:
        # 无论测试成功或失败，都关闭数据库会话。
        db.close()

def test_chunk_document_saves_cleaned_content_and_excludes_html_noise() -> None:
    """切分前应保存清洗结果，脚本噪声不应进入检索 chunk。"""

    content = (
        "# RAG 清洗测试\n\n"
        "<script>alert('tracking')</script>\n\n"
        "混合检索会融合向量检索和关键词检索，"
        "让系统同时保留语义匹配和精确术语匹配的优势。\n\n"
        "<style>.hidden { display: none; }</style>"
    )

    db = create_test_session()  # 创建独立内存数据库。

    try:
        document = Document(
            title="RAG 清洗测试",
            source_type="markdown",
            raw_content=content,
            content_hash=sha256(content.encode("utf-8")).hexdigest(),
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        # 执行完整的“清洗 -> 保存 -> 切分”流程。
        chunk_document(
            document_id=document.id,
            db=db,
            max_tokens=400,
        )

        # 从数据库重新读取文档，确认清洗结果已被持久化。
        saved_document = db.get(Document, document.id)

        assert saved_document is not None
        assert saved_document.cleaned_content is not None
        assert "<script>" not in saved_document.cleaned_content
        assert "<style>" not in saved_document.cleaned_content
        assert "混合检索会融合向量检索和关键词检索" in (
            saved_document.cleaned_content
        )

        # 最终可检索 chunks 也不能包含已清理的 HTML 噪声。
        chunks = db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document.id)
        ).all()

        assert len(chunks) == 2
        assert all("<script>" not in chunk.content for chunk in chunks)
        assert all("<style>" not in chunk.content for chunk in chunks)
        assert "混合检索会融合向量检索和关键词检索" in (
            chunks[0].content
        )

    finally:
        # 无论测试结果如何，都关闭测试数据库会话。
        db.close()

import pytest  # 用于验证质量检查失败时会抛出异常。

def test_chunk_document_marks_document_failed_when_cleaned_content_is_too_short() -> None:
    """清洗后内容过短时，文档不应产生 chunk，并标记为 failed。"""

    # script 被清洗后只剩“短内容”，不足 30 个有效字符。
    content = (
        "# 短文档\n\n"
        "<script>alert('tracking')</script>\n\n"
        "短内容。"
    )

    db = create_test_session()  # 创建独立内存数据库。

    try:
        document = Document(
            title="短文档",
            source_type="markdown",
            raw_content=content,
            content_hash=sha256(content.encode("utf-8")).hexdigest(),
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        # 质量检查失败时，切分流程应明确抛出业务错误。
        with pytest.raises(ValueError, match="too short"):
            chunk_document(
                document_id=document.id,
                db=db,
                max_tokens=400,
            )

        # 重新读取文档，确认失败状态和清洗结果均已保存。
        saved_document = db.get(Document, document.id)

        assert saved_document is not None
        assert saved_document.status == "failed"
        assert saved_document.cleaned_content is not None
        assert "<script>" not in saved_document.cleaned_content

        # 质量检查失败时，不能产生可检索 chunk。
        chunks = db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document.id)
        ).all()

        assert chunks == []

    finally:
        # 无论测试成功或失败，都关闭数据库会话。
        db.close()

def test_chunk_document_adds_pdf_table_as_independent_chunk() -> None:
    """PDF 表格应作为独立完整 chunk，而不是混入正文。"""

    content = (
        "RAG 通过从外部知识库检索资料，"
        "再把相关证据交给大语言模型生成可追溯回答。"
    )
    db = create_test_session()

    try:
        document = Document(
            title="PDF 测试资料",
            source_type="pdf",
            raw_content=content,
            content_hash=sha256(
                content.encode("utf-8")
            ).hexdigest(),
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        table = DocumentTable(
            document_id=document.id,
            page_number=3,
            table_index=1,
            rows=[
                ["检索方式", "优势"],
                ["向量检索", "语义匹配"],
            ],
            markdown_content=(
                "| 检索方式 | 优势 |\n"
                "| --- | --- |\n"
                "| 向量检索 | 语义匹配 |"
            ),
        )
        db.add(table)
        db.commit()

        result = chunk_document(
            document_id=document.id,
            db=db,
            max_tokens=400,
        )

        chunks = db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document.id)
            .order_by(DocumentChunk.position)
        ).all()

        assert result.chunks_created == 4
        assert len(chunks) == 4

        table_chunk = next(
            chunk for chunk in chunks
            if chunk.chunk_level == "parent" and chunk.block_type == "table"
        )
        assert table_chunk.section_path == "PDF 第 3 页 > 表格 1"
        assert table_chunk.content == table.markdown_content

    finally:
        db.close()
