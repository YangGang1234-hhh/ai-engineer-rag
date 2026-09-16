"""问答链路的隐私友好审计写入服务。"""

from hashlib import sha256
from typing import Sequence

from sqlalchemy.orm import Session

from app.models import QueryAuditLog


def hash_query(query: str) -> str:
    """返回问题的 SHA-256 哈希，避免把原始输入落入审计库。"""

    return sha256(query.encode("utf-8")).hexdigest()


def record_query_audit(
    *,
    db: Session,
    request_id: str,
    query: str,
    decision: str,
    answer_status: str,
    reason: str | None = None,
    retrieved_chunk_ids: Sequence[str] = (),
    retrieved_scores: Sequence[float] = (),
    citation_numbers: Sequence[int] = (),
    latency_ms: int | None = None,
    error_type: str | None = None,
    model_name: str | None = None,
) -> QueryAuditLog:
    """写入一条审计记录并提交事务。

    调用方在主链路接入时应捕获异常，使日志故障不影响问答结果。
    """

    log = QueryAuditLog(
        request_id=request_id,
        query_hash=hash_query(query),
        decision=decision,
        reason=reason,
        retrieved_chunk_ids=list(retrieved_chunk_ids),
        retrieved_scores=list(retrieved_scores),
        citation_numbers=list(citation_numbers),
        answer_status=answer_status,
        latency_ms=latency_ms,
        error_type=error_type,
        model_name=model_name,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log
