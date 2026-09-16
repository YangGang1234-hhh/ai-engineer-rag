from dataclasses import dataclass 
from app.cleaning import (  # 导入文档清洗和清洗后质量检查。
    clean_markdown,
    validate_cleaned_content,
)
# delete 用于删除某文档之前生成的旧 chunk
from sqlalchemy import delete , select

# Session 代表一次数据库会话
from sqlalchemy.orm import Session 

# 导入纯切分函数
from app.chunking import (
    StructureBlock,
    build_chunk_drafts,
    count_tokens,
    parse_markdown_blocks,
)
from app.parent_child_chunking import build_child_contents
# 导入两张数据库表对应的python 模型
from app.models import Document, DocumentChunk, DocumentTable

@dataclass
class DocumentChunkingResult:
    """一份文档完成切分并写入数据库后的结果。"""

    # 被处理的原始文档
    document:Document

    # 本次实际创建的chunk 数量
    chunks_created:int

def chunk_document(
        document_id:str,
        db:Session,
        max_tokens:int = 350,
) -> DocumentChunkingResult:
    """读取指定文档，切分内容，并将最终chunk 写入document_chunks 表。"""

    # 根据主键查询原始文档。
    document = db.get(Document,document_id)

    # 找不到文档立即报错，避免后续出现难以理解的空值错误。
    if document is None:
        raise ValueError(f"Document not found:{document_id}")

    # 从原始内容生成清洗结果，但不覆盖 raw_content，保留可追溯的原文。
    cleaned_content = clean_markdown(document.raw_content)

    # 将清洗结果保存到数据库，方便检查与后续重新处理。
    document.cleaned_content = cleaned_content

    # 只有满足基础质量要求的内容，才允许继续切分和建立索引。
    quality = validate_cleaned_content(cleaned_content)

    if not quality.is_valid:
        # 保存清洗结果和失败状态，便于后续排查或调整规则后重新处理。
        document.status = "failed"
        db.commit()

        raise ValueError(
            quality.reason or "Document content failed quality validation."
        )

    # 只从清洗后的内容解析段落、列表、代码块和表格。
    blocks = parse_markdown_blocks(cleaned_content)

    # PDF 表格已单独保存在 document_tables。
    # 每张表作为完整结构块加入检索，而不是混进普通正文。
    if document.source_type == "pdf":
        tables = db.scalars(
            select(DocumentTable)
            .where(DocumentTable.document_id == document.id)
            .order_by(
                DocumentTable.page_number,
                DocumentTable.table_index,
            )
        ).all()

        blocks.extend(
            StructureBlock(
                block_type="table",
                content=table.markdown_content,
                section_path=(
                    f"PDF 第 {table.page_number} 页"
                    f" > 表格 {table.table_index}"
                ),
            )
            for table in tables
            if table.markdown_content.strip()
        )

    #再按标题路径和token 预算，把结构块聚合成最终检索的chunk。
    drafts = build_chunk_drafts(
        blocks=blocks,
        max_tokens=max_tokens,
    )

    # 可对同一份文档进行重复切分
    # 重新切分前，先删除旧chunk，避免产生重复数据
    db.execute(
        delete(DocumentChunk).where(
            DocumentChunk.document_id == document.id
        )
    )

    # 先保存完整父块。child 命中后会回溯到父块组装回答证据。
    parent_chunks = [
        DocumentChunk(
            document_id=document.id,
            section_path=draft.section_path,
            content=draft.content,
            position = draft.position,
            token_count = draft.token_count,
            chunk_level="parent",
            block_type="table" if "表格" in draft.section_path else "prose",
            page_number=_extract_pdf_page_number(draft.section_path),
        )
        for draft in drafts
    ]
    db.add_all(parent_chunks)
    db.flush()

    # child 是实际建立 FTS、向量索引和 Cross-Encoder 精排的单元。
    child_chunks: list[DocumentChunk] = []
    child_position = 0
    for parent_chunk in parent_chunks:
        child_contents = (
            [parent_chunk.content]
            if parent_chunk.block_type == "table"
            else build_child_contents(parent_chunk.content)
        )

        for child_content in child_contents:
            child_chunks.append(
                DocumentChunk(
                    document_id=document.id,
                    parent_chunk_id=parent_chunk.id,
                    section_path=parent_chunk.section_path,
                    content=child_content,
                    position=child_position,
                    token_count=count_tokens(child_content),
                    chunk_level="child",
                    block_type=parent_chunk.block_type,
                    page_number=parent_chunk.page_number,
                )
            )
            child_position += 1

    db.add_all(child_chunks)

    # 已完成切分，但还没生成embedding，没写入Qdrant.
    document.status = "chunked"

    # 一次提交文档状态和所有新 chunk。
    db.commit()

    # 返回本次处理结果。
    return DocumentChunkingResult(
        document=document,
        chunks_created=len(parent_chunks) + len(child_chunks)
    )


def _extract_pdf_page_number(section_path: str) -> int | None:
    """从 PDF 章节路径中读取页码；Markdown 文档保持为空。"""

    if not section_path.startswith("PDF 第 "):
        return None

    try:
        return int(
            section_path.split(" 页", maxsplit=1)[0].removeprefix("PDF 第 ")
        )
    except ValueError:
        return None
