from datetime import datetime,timezone

from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column,relationship

from app.database import Base

class DocumentTable(Base):
    """从 PDF 中独立提取并保存的一张表格。"""

    __tablename__ = "document_tables"

    # 表格记录唯一 ID。
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    # 所属文档；删除文档时，表格记录也会级联删除。
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # PDF 页码，从 1 开始。
    page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    # 该页中的第几张表，从 1 开始。
    table_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    # 原始二维单元格数据，例如 [["字段", "说明"], ["RRF", "融合算法"]]。
    rows: Mapped[list[list[str]]] = mapped_column(
        JSON,
        nullable=False,
    )

    # 转为 Markdown 后的文本，供后续建立独立检索 chunk。
    markdown_content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # 可通过 table.document 返回所属文档。
    document: Mapped["Document"] = relationship(
        back_populates="tables",
    )
    
class Document(Base):
    #指定数据库中的表名
    __tablename__ = "documents"

    #文档字段创建
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    #文档标题
    title: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    #原始来源Url
    #本地markdown 文件没有Url，允许为空
    source_url: Mapped[str | None] = mapped_column(
        String(2048),
        nullable=True, 
    )

    #资料来源类型
    source_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default = "markdown",
    )

    #文档内容
    raw_content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # 清洗后的文档内容，用于后续切分和建立检索索引。
    # 设为可为空，是为了兼容已经存在但尚未重新处理的旧文档。
    cleaned_content: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    #内容哈希值，用于识别重复导入的相同资料
    content_hash:Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
    )

    #导入状态，例如pending,indexed,failed
    #现在默认pending,后续做索引时再更新。
    status:Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="pending",
    )

    #创建时间
    created_at:Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    #最近一次更新记录的时间
    #onupdate 表示修改该条记录时，SQLAchemy 会自动更新该字段
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default= lambda:datetime.now(timezone.utc),
        onupdate=lambda:datetime.now(timezone.utc),
    )

    #一份原始文档可以被切分成多个chunk
    #cascade 表示删除文档时，同时删除它所属的所有chunk.
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates = "document",
        cascade="all,delete-orphan",
    )

    # 一份 PDF 文档可包含多张独立提取的表格。
    tables: Mapped[list["DocumentTable"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )

#DocumentChunk 对应SQLite 中未来的document_chunks表
#一条记录代表从原始文档中切分出来的一个可检索片段。

class DocumentChunk(Base):
    # 指定数据库表名
    __tablename__ = "document_chunks"

    # 主键ID
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key = True,
        default=lambda:str(uuid4())
    )

    #外键，用于关联原始文档
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id",ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # 父块 ID。parent 块为空，child 块指向实际提供给 LLM 的父块。
    parent_chunk_id: Mapped[str | None] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # parent 用于上下文组装，child 用于 FTS、向量检索和重排序。
    chunk_level: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="parent",
        index=True,
    )

    # prose、table、image_ocr 等，供检索和引用展示使用。
    block_type: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="prose",
    )

    # PDF 块对应的页码；Markdown 文档没有页码时保持为空。
    page_number: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    #chunk 在原始文本中的路径
    section_path:Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
        default="",
    )

    #chunk 内容
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    #chunk 在原文中的顺序，从0开始

    position:Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    #内容大致包含的token数量
    #未来组装LLM上下文时，可以根据token数量来控制上下文长度，避免超过模型的最大上下文长度。
    token_count:Mapped[int] = mapped_column(
        Integer,
        nullable=False,

    )

    #创建时间
    created_at:Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda:datetime.now(timezone.utc)
    )

    #在python 中反向访问所属文档：
    #chunk.document 可以拿到对应的Document 对象。
    document:Mapped["Document"] = relationship(
        back_populates="chunks",
    )


class QueryAuditLog(Base):
    """一条问答请求的隐私友好审计记录。

    不保存原始用户问题或完整回答；问题只保存不可逆哈希，
    以便排查重复请求、Guard 决策和检索链路。
    """

    __tablename__ = "query_audit_logs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    request_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        unique=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # allow、unsafe、out_of_scope、insufficient_evidence 等。
    decision: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    retrieved_scores: Mapped[list[float]] = mapped_column(JSON, nullable=False, default=list)
    citation_numbers: Mapped[list[int]] = mapped_column(JSON, nullable=False, default=list)
    # answered、rejected、failed。
    answer_status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(200), nullable=True)

    
