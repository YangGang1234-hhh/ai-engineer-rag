"""用最新 PDF 提取与切分策略替换一份已入库 PDF。"""

from pathlib import Path

from sqlalchemy import select

from app.database import SessionLocal
from app.document_processing import chunk_document
from app.full_text import delete_document_fts_records, sync_document_chunks_to_fts
from app.indexing import index_document
from app.ingestion import import_pdf_file
from app.models import Document
from app.vector_store import delete_document_vectors, get_qdrant_client


PDF_PATH = Path("datasets/raw/pdf/RAG应用开发手册.pdf")


def main() -> None:
    db = SessionLocal()
    try:
        old_documents = db.scalars(
            select(Document).where(Document.source_type == "pdf")
        ).all()
        for document in old_documents:
            delete_document_vectors(document.id)
            delete_document_fts_records(document.id, db)
            db.delete(document)
        db.commit()

        imported = import_pdf_file(PDF_PATH, db)
        chunked = chunk_document(imported.document.id, db)
        fts_count = sync_document_chunks_to_fts(imported.document.id, db)
        db.commit()
        indexed = index_document(imported.document.id, db)
        print(
            f"refreshed={imported.document.title}, "
            f"records={chunked.chunks_created}, "
            f"fts_children={fts_count}, vectors={indexed.chunks_indexed}"
        )
    finally:
        db.close()
        client = get_qdrant_client()
        client.close()
        get_qdrant_client.cache_clear()


if __name__ == "__main__":
    main()
