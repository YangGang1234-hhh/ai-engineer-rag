from app.database import Base,engine
from sqlalchemy import text

#导入Document,这样SQLAlchemy才知道documents表属于Base.metadata.
from app.models import Document, DocumentChunk, DocumentTable, QueryAuditLog

def ensure_cleaned_content_column() -> None:
    """为已有 documents 表补充 cleaned_content 字段。"""
    with engine.begin() as connection:
        result = connection.execute(
            text("PRAGMA table_info(documents)")
        )
        column_names = {row[1] for row in result}

        if "cleaned_content" not in column_names:
            connection.execute(
                text(
                    "ALTER TABLE documents "
                    "ADD COLUMN cleaned_content TEXT"
                )
            )
            print("Added documents.cleaned_content column.")


def ensure_document_chunk_columns() -> None:
    """为已有 document_chunks 表补充父子块检索需要的字段。"""

    required_columns = {
        "parent_chunk_id": "VARCHAR(36)",
        "chunk_level": "VARCHAR(20) NOT NULL DEFAULT 'parent'",
        "block_type": "VARCHAR(30) NOT NULL DEFAULT 'prose'",
        "page_number": "INTEGER",
    }

    with engine.begin() as connection:
        result = connection.execute(text("PRAGMA table_info(document_chunks)"))
        column_names = {row[1] for row in result}

        for column_name, definition in required_columns.items():
            if column_name not in column_names:
                connection.execute(
                    text(
                        "ALTER TABLE document_chunks "
                        f"ADD COLUMN {column_name} {definition}"
                    )
                )
                print(f"Added document_chunks.{column_name} column.")

def create_full_text_index() -> None:
    """创建用于关键词检索的 SQLite FTS5 虚拟表"""
    # engine.begin() 会开启一个事物；
    # 成功时自动提交，发生异常时自动回滚
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts
                USING fts5(
                    -- 原始文档 ID，只用于重建或删除该文档的索引。
                    document_id UNINDEXED,

                    -- 对应 document_chunks 表的 chunk ID，只用于关联，不参与全文匹配。
                    chunk_id UNINDEXED,

                    -- chunk 正文，关键词检索的主要字段。
                    content,

                    -- 标题路径也纳入索引，
                    -- 让“混合检索”“RAG 基础”等章节名可以被检索到。
                    section_path,

                    -- 使用 trigram，改善中文、英文术语与代码标识符的检索体验。
                    tokenize = 'trigram'
                )
                """
            )
        )
def main() -> None:
    #根据已导入的所有模型，创建上不存在的表
    #如果表存在，不会被删除或覆盖。
    Base.metadata.create_all(bind=engine)

    # 为已经存在的 documents 表补充本次新增的清洗结果字段。
    ensure_cleaned_content_column()
    ensure_document_chunk_columns()

    # 创建 FTS5 全文索引虚拟表。
    # 当前只创建表结构，下一步才同步已有 chunk 数据。
    create_full_text_index()

    print(
    "Database initialized. Verified tables: "
    "documents, document_chunks, document_tables, query_audit_logs, "
    "document_chunks_fts"
        )

if __name__ == "__main__":
    main()
