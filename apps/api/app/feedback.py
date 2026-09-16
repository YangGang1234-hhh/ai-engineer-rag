"""用户问答反馈的校验与持久化服务。"""

from datetime import datetime, timezone
from enum import StrEnum

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AnswerFeedback, QueryAuditLog


class FeedbackType(StrEnum):
    UP = "up"
    DOWN = "down"


MAX_FEEDBACK_REASON_CHARACTERS = 500


def record_answer_feedback(
    *,
    db: Session,
    request_id: str,
    feedback_type: FeedbackType,
    reason: str | None = None,
) -> AnswerFeedback:
    """新增或更新一次问答反馈，确保能回溯到真实审计记录。"""

    normalized_reason = reason.strip() if reason else None
    if normalized_reason and len(normalized_reason) > MAX_FEEDBACK_REASON_CHARACTERS:
        raise ValueError(
            f"feedback reason must be at most {MAX_FEEDBACK_REASON_CHARACTERS} characters"
        )

    audit_log = db.scalar(
        select(QueryAuditLog).where(QueryAuditLog.request_id == request_id)
    )
    if audit_log is None:
        raise ValueError("request_id does not reference an audited request")

    feedback = db.scalar(
        select(AnswerFeedback).where(AnswerFeedback.request_id == request_id)
    )
    if feedback is None:
        feedback = AnswerFeedback(
            request_id=request_id,
            feedback_type=str(feedback_type),
            reason=normalized_reason,
        )
        db.add(feedback)
    else:
        feedback.feedback_type = str(feedback_type)
        feedback.reason = normalized_reason
        feedback.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(feedback)
    return feedback
