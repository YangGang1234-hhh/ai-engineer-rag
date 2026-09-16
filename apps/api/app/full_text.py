# select 用于从 document_chunks 表查询某份文档的所有 chunk。
# text 用于执行 FTS5 虚拟表所需的原生 SQL。
from sqlalchemy import select, text
from dataclasses import dataclass
# Session 是一次数据库操作的会话类型。
from sqlalchemy.orm import Session

# 导入已经写入 SQLite 的 chunk 模型。
from app.models import DocumentChunk

import re  # 用正则表达式清理用户问题末尾的问号等标点。

def build_fts_match_query(query: str) -> str:
    """将自然语言问题转换为适合 trigram FTS5 的 OR 查询。"""

    # 只保留中文连续文本、英文、数字和部分技术符号。
    # 例如：“什么是混合检索？”会提取出“什么是混合检索”。
    segments = re.findall(
        r"[\u4e00-\u9fff]+|[A-Za-z0-9_.\-/]+",
        query.strip(),
    )

    fragments: list[str] = []  # 保存交给 FTS5 匹配的候选片段。

    for segment in segments:
        # 中文连续文本长度大于等于 3 时，生成重叠的三个汉字片段。
        # “什么是混合检索”会变成：
        # “什么是”“么是混”“是混合”“混合检”“合检索”。
        if re.fullmatch(r"[\u4e00-\u9fff]+", segment) and len(segment) >= 3:
            fragments.extend(
                segment[index:index + 3]
                for index in range(len(segment) - 2)
            )
        else:
            # 英文技术词（如 Qdrant、BM25、RRF）保留为完整片段。
            fragments.append(segment)

    # dict.fromkeys 会在去重的同时保留原有顺序。
    unique_fragments = list(dict.fromkeys(fragments))

    # 用 OR 连接：任意一个片段命中都可以召回候选。
    # 双引号避免技术术语中的符号被 FTS5 当作查询语法。
    return " OR ".join(
        f'"{fragment}"'
        for fragment in unique_fragments
    )

def build_fts_match_query_from_keywords(
    keywords: list[str],
) -> str:
    """将 LLM 提取的明确关键词转换为 FTS5 OR 查询。"""

    cleaned_keywords: list[str] = []  # 保存清理后的关键词。

    for keyword in keywords:
        keyword = keyword.strip()  # 去掉关键词两侧的空白。

        if not keyword:  # 忽略空字符串。
            continue

        # trigram 至少需要三个字符才能稳定匹配中文短语。
        # 英文技术词通常也至少包含三个字符，例如 RAG、BM25。
        if len(keyword) < 3:
            continue

        # FTS5 短语使用双引号包裹。
        # 将内部双引号替换为两个双引号，避免破坏 MATCH 语法。
        safe_keyword = keyword.replace('"', '""')
        cleaned_keywords.append(f'"{safe_keyword}"')

    # 去重，同时保留 LLM 返回的原始顺序。
    unique_keywords = list(dict.fromkeys(cleaned_keywords))

    # 任意关键词命中即可召回。
    return " OR ".join(unique_keywords)

def delete_document_fts_records(
    document_id: str,
    db: Session,
) -> None:
    """删除指定文档在 FTS5 中的全部索引记录。"""

    # FTS5 虚拟表不受 SQLAlchemy 级联删除管理，
    # 因此按 document_id 显式清理该文档的关键词索引。
    db.execute(
        text(
            """
            DELETE FROM document_chunks_fts
            WHERE document_id = :document_id
            """
        ),
        {"document_id": document_id},
    )

def sync_document_chunks_to_fts(
    document_id: str,
    db: Session,
) -> int:
    """将指定文档的 chunk 同步到 SQLite FTS5 全文索引表。"""

    # 先从普通数据表读取该文档的所有 chunk。
    # position 排序不是 FTS 的硬性要求，但有助于保证同步过程稳定、便于排查。
    chunks = db.scalars(
        select(DocumentChunk)
        .where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.chunk_level == "child",
        )
        .order_by(DocumentChunk.position)
    ).all()

    # 通过 document_id 删除该文档的旧 FTS 索引。
    # 即使 document_chunks 已先删除旧 chunk，这里仍能准确清理旧索引。
    delete_document_fts_records(
        document_id=document_id,
        db=db,
    )

    # 准备批量写入 FTS5 的数据。
    # FTS 表只保存搜索所需字段，不保存完整业务数据。
    rows = [
        {
            # 保存所属文档 ID，支持后续按文档清理旧索引。
            "document_id": document_id,

            # 保存对应 chunk ID，检索命中后可回到业务表取完整信息。
            "chunk_id": chunk.id,

            # 正文与标题路径参与 FTS5 全文检索。
            "content": chunk.content,
            "section_path": chunk.section_path,
        }
        for chunk in chunks
    ]

    # 当前文档确实有 chunk 时，才执行批量插入。
    if rows:
        db.execute(
            text(
                """
                INSERT INTO document_chunks_fts (
                    document_id,
                    chunk_id,
                    content,
                    section_path
                )
                VALUES (
                    :document_id,
                    :chunk_id,
                    :content,
                    :section_path
                )
                """
            ),
            rows,
        )

    # 不在这里 commit。
    # 调用方决定何时把“切分结果 + FTS 索引”作为同一个事务提交。
    return len(rows)

@dataclass
class KeywordSearchResult:
    """一次关键词检索命中的 chunk 结果。"""

    # 命中 chunk 的唯一 ID，用于后续回到业务表或作为引用依据。
    chunk_id: str

    # chunk 所属文档 ID。
    document_id: str

    # chunk 正文内容。
    content: str

    # Markdown 标题路径。
    section_path: str

    # chunk 在原始文档中的顺序。
    position: int

    # chunk 的 token 数。
    token_count: int

    # SQLite FTS5 BM25 分数。
    # 在 FTS5 中，分数通常越小表示越相关。
    score: float
    parent_chunk_id: str | None = None
    chunk_level: str = "child"
    block_type: str = "prose"
    page_number: int | None = None


def search_chunks_by_keyword(
    query: str,
    db: Session,
    limit: int = 5,
    keywords: list[str] | None = None,
) -> list[KeywordSearchResult]:
    """使用 SQLite FTS5 和 BM25 对 chunk 执行关键词检索。"""

    # 有 LLM 关键词时，优先使用明确关键词构造 FTS 查询。
    if keywords is not None:
        match_query = build_fts_match_query_from_keywords(keywords)
    else:
        # LLM 不可用时，使用原有 trigram 查询作为兜底。
        match_query = build_fts_match_query(query)

    # 空查询没有搜索意义，提前返回空结果。
    if not query:
        return []

    # 限制候选数量，避免一次检索返回过多上下文。
    if limit <= 0:
        raise ValueError("limit must be greater than 0.")

    # FTS 表只承担全文匹配；
    # document_chunks 表仍是内容、位置和 token 信息的事实来源。
    rows = db.execute(
        text(
            """
            SELECT
                document_chunks.id AS chunk_id,
                document_chunks.document_id AS document_id,
                document_chunks.content AS content,
                document_chunks.section_path AS section_path,
                document_chunks.position AS position,
                document_chunks.token_count AS token_count,
                document_chunks.parent_chunk_id AS parent_chunk_id,
                document_chunks.chunk_level AS chunk_level,
                document_chunks.block_type AS block_type,
                document_chunks.page_number AS page_number,
                bm25(document_chunks_fts) AS score
            FROM document_chunks_fts
            JOIN document_chunks
                ON document_chunks.id = document_chunks_fts.chunk_id
            WHERE document_chunks_fts MATCH :query
            ORDER BY score
            LIMIT :limit
            """
        ),
        {
            "query": match_query,
            "limit": limit,
        },
    ).mappings().all()

    # 将 SQL 查询结果转换为项目内部统一的数据对象。
    return [
        KeywordSearchResult(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            content=row["content"],
            section_path=row["section_path"],
            position=row["position"],
            token_count=row["token_count"],
            score=row["score"],
            parent_chunk_id=row["parent_chunk_id"],
            chunk_level=row["chunk_level"],
            block_type=row["block_type"],
            page_number=row["page_number"],
        )
        for row in rows
    ]
