from dataclasses import dataclass

from sqlalchemy import select 
from sqlalchemy.orm import Session

from app.models import Document,DocumentChunk
from app.vector_store import (  # 导入删除旧向量和写入新向量的操作。
    delete_document_vectors,
    upsert_chunk_vectors,
)

@dataclass
class DocumentIndexingResult:
    """一份文档完成向量索引后的结果。"""

    document:Document
    chunks_indexed:int


def index_document(
        document_id:str,
        db:Session,
) -> DocumentIndexingResult:
    """读取文档 chunks,生成向量并写入Qdrant。"""

    document = db.get(Document,document_id)

    if document is None:
        raise ValueError(f"Document not found:{document_id}")

    # 按原文顺序读取这份文档的全部chunks。
    chunks = list(
        db.scalars(
            select(DocumentChunk)
            .where(
                DocumentChunk.document_id == document.id,
                DocumentChunk.chunk_level == "child",
            )
            .order_by(DocumentChunk.position)
        ).all()
    )

    if not chunks:
        raise ValueError(f"Document has no chunks:{document_id}")

    try:
        #只有向量成功写入后，才把文档标记为 indexed。
        # 重切分会产生新的 chunk ID，因此先清除该文档的旧向量。
        # 删除条件只匹配当前 document_id，不会影响其他文档。
        delete_document_vectors(document.id)

        # 将当前 SQLite chunks 生成向量并写入 Qdrant。
        upsert_chunk_vectors(chunks)

        document.status = "indexed"
        db.commit()

    except Exception:
        # 如果Embedding 或 Qdrant 写入失败，撤销本次数据库事务。
        db.rollback()
        raise

    return DocumentIndexingResult(
        document=document,
        chunks_indexed=len(chunks),
    )
