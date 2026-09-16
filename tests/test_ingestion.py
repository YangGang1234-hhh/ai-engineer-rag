from pathlib import Path 

from sqlalchemy import create_engine,select 
from sqlalchemy.orm import Session,sessionmaker

from app.database import Base 

from app.ingestion import extract_markdown_title,import_markdown_file

from app.models import Document 

def create_test_session() -> Session:
    """创建只存在于内存中的测试数据库会话"""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread":False},
    )

    #根据Dodument等模型，在这个临时数据库中创建表
    Base.metadata.create_all(bind=engine)

    #创建并返回一次测试数据库会话
    return sessionmaker(bind=engine)()

def test_extract_markdown_title() -> None:
    """应从第一个一级标题中提取文档标题。"""
    content = "# RAG 检索基础\n\n这里是正文。"

    title = extract_markdown_title(
        content = content,
        fallback_title="fallback",
    )

    assert title == "RAG 检索基础"

def test_extract_markdown_title_uses_fallback() -> None:
    """没有一级标题时，应使用文件名作为兜底标题。"""

    content = "这是一份没有一级标题的 Markdown 文档。"

    title = extract_markdown_title(
        content = content,
        fallback_title="fallback-title",
    )

    assert title == "fallback-title"

def test_import_markdown_file_creates_document_and_deduplicates(
        tmp_path:Path,
) ->None:
    """首次导入创建文档，再次导入相同内容时返回已有的文档。"""

    markdown_file = tmp_path/"rag-note.md"
    markdown_file.write_text(
        "# RAG 测试资料\n\n这是一份用于测试导入器的资料。",
        encoding="utf-8",
    )
    db = create_test_session()

    try:
        first_result = import_markdown_file(
            file_path=markdown_file,
            db=db,
        )

        assert first_result.created is True
        assert first_result.document.title == "RAG 测试资料"
        assert first_result.document.source_type== "markdown"

        #第二次导入同一个文件，应命中内容相同的哈希去重逻辑。
        second_result = import_markdown_file(
            file_path=markdown_file,
            db=db
        )

        assert second_result.created is False
        assert second_result.document.id == first_result.document.id

        #数据库中最终只有有一条Document记录
        documents = db.scalars(select(Document)).all()
        assert len(documents) == 1

    finally:
        db.close()