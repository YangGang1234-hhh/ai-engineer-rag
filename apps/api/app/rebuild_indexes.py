"""按当前处理策略重建全部文档的 chunk、FTS 与向量索引。"""

from sqlalchemy import select

from app.database import SessionLocal
from app.document_processing import chunk_document
from app.full_text import sync_document_chunks_to_fts
from app.indexing import index_document
from app.models import Document
from app.vector_store import get_qdrant_client


def main() -> None:
    """对所有知识库文档执行可重复的完整重建。"""

    db = SessionLocal()
    try:
        documents = db.scalars(
            select(Document).order_by(Document.created_at)
        ).all()

        for document in documents:
            chunk_result = chunk_document(document.id, db)
            fts_count = sync_document_chunks_to_fts(document.id, db)
            db.commit()
            index_result = index_document(document.id, db)
            print(
                f"{document.title}: parents and children={chunk_result.chunks_created}, "
                f"fts children={fts_count}, vectors={index_result.chunks_indexed}"
            )
    finally:
        db.close()
        client = get_qdrant_client()
        client.close()
        get_qdrant_client.cache_clear()


if __name__ == "__main__":
    main()
