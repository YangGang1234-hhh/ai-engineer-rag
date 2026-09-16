from dataclasses import dataclass
from pathlib import Path

#sha256用于计算文档内容哈希，识别重复导入
from hashlib import sha256

# select 用于查询数据库中是否已有相同内容的文档
from sqlalchemy import select 

from sqlalchemy.orm import Session
from app.models import Document

from app.cleaning import clean_markdown
from app.models import Document, DocumentTable
from app.pdf_extraction import extract_pdf_content

@dataclass 
class MarkdownImportResult:
    """Markdown导入结果数据类"""
    document: Document
    created: bool

@dataclass
class PdfImportResult:
    """PDF 导入结果，包含主文档和本次保存的表格数量。"""

    document: Document
    created: bool
    tables_created: int

def extract_markdown_title(content:str,fallback_title:str) -> str:
    """从Markdown内容中提取标题，如果没有标题则使用备用标题"""
    #spitlines() 会把全文按行拆开，方便逐行查找标题
    for line in content.splitlines():
        #Markdown标题通常以#开头，后面跟一个空格和标题文本
        if line.startswith("# "):
            title = line.removeprefix("# ").strip()

            #避免只有# 而没有实际标题的情况。
            if title:
                return title

    #如果没有一级标题，就使用文件名作为标题
    return fallback_title

def import_markdown_file(
        file_path:Path,
        db:Session,
) -> MarkdownImportResult:
    """读取一份本地的Markdown文件，并将其保存为Document记录。"""
    # 检查路径是否存在，避免后面读取文件时出现不清楚的报错
    if not file_path.is_file():
        raise FileNotFoundError(f"Markdown file not found:{file_path}")

    # 名曲指定UTF-8,保证中文Markdown能正确读取
    content = file_path.read_text(encoding="utf-8")

    #如果文件为空，拒绝导入
    if not content.strip():
        raise ValueError("Markdown file is empty.")

    #计算文本哈希值，用于识别重复导入的相同资料
    content_hash = sha256(content.encode("utf-8")).hexdigest()

    # 从documents表中查询是否已经导入过相同内容。
    existing_document=db.scalar(
        select(Document).where(Document.content_hash==content_hash)
    )

    #已存在，直接返回已有记录
    if existing_document is not None:
        return MarkdownImportResult(
            document=existing_document,
            created=False,
        )

    #提取标题，如果没有标题就使用文件名
    fallback_title=file_path.stem

    title = extract_markdown_title(content,fallback_title)

    document = Document(
        title=title,
        source_type="markdown",
        raw_content=content,
        content_hash=content_hash,
        status="pending",
    )
    # 将记录加入当前数据库会话
    db.add(document)

    #提交
    db.commit()

    #刷新
    db.refresh(document)

    return MarkdownImportResult(
        document=document,
        created=True
    )

def import_pdf_file(
    file_path: Path,
    db: Session,
) -> PdfImportResult:
    """提取 PDF 正文和表格，并保存为知识库文档。"""

    if not file_path.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    extraction = extract_pdf_content(file_path)

    # PDF 正文和表格 Markdown 共同参与去重。
    # 避免同一份资料只是重新上传为另一个文件名时重复入库。
    # 表格 Markdown 也要经过同一套清洗和 Unicode 规范化规则。
    # 后续存储、哈希和建立检索 chunk 都使用清洗后的表格文本。
    extracted_tables = [
        (
            table,
            clean_markdown(table.markdown_content),
        )
        for table in extraction.tables
        if table.markdown_content
    ]
    table_contents = [
        markdown_content
        for _, markdown_content in extracted_tables
        if markdown_content
    ]
    content_for_hash = "\n\n".join(
        [extraction.text_content, *table_contents]
    ).strip()

    if not content_for_hash:
        raise ValueError(
            "PDF has no extractable text or tables. "
            "Scanned PDFs require OCR support."
        )

    content_hash = sha256(
        content_for_hash.encode("utf-8")
    ).hexdigest()

    existing_document = db.scalar(
        select(Document).where(
            Document.content_hash == content_hash
        )
    )

    if existing_document is not None:
        return PdfImportResult(
            document=existing_document,
            created=False,
            tables_created=0,
        )

    # 当前先使用文件名作为标题；后续可扩展为读取 PDF metadata。
    document = Document(
        title=file_path.stem,
        source_type="pdf",
        # raw_content 保存提取到的正文，表格保存在 document_tables。
        raw_content=extraction.text_content,
        content_hash=content_hash,
        status="pending",
    )
    db.add(document)
    db.flush()

    tables = [
        DocumentTable(
            document_id=document.id,
            page_number=table.page_number,
            table_index=table.table_index,
            rows=table.rows,
            markdown_content=markdown_content,
        )
        for table, markdown_content in extracted_tables
        if markdown_content
    ]
    db.add_all(tables)

    db.commit()
    db.refresh(document)

    return PdfImportResult(
        document=document,
        created=True,
        tables_created=len(tables),
    )