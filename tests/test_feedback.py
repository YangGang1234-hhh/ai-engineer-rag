from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.audit import record_query_audit
from app.database import Base
from app.feedback import FeedbackType, record_answer_feedback
from app.models import AnswerFeedback


def create_test_session():
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_record_answer_feedback_creates_and_updates_single_request_feedback() -> None:
    db = create_test_session()
    request_id = "bc72db2b-4721-4187-8ea9-5479a84eb4db"

    try:
        record_query_audit(
            db=db,
            request_id=request_id,
            query="什么是混合检索？",
            decision="allow",
            answer_status="answered",
        )

        created = record_answer_feedback(
            db=db,
            request_id=request_id,
            feedback_type=FeedbackType.DOWN,
            reason="没有解释关键词检索的作用。",
        )
        updated = record_answer_feedback(
            db=db,
            request_id=request_id,
            feedback_type=FeedbackType.UP,
        )

        assert created.id == updated.id
        assert updated.feedback_type == "up"
        assert updated.reason is None
        assert db.query(AnswerFeedback).count() == 1
    finally:
        db.close()


def test_record_answer_feedback_requires_existing_audit_request() -> None:
    db = create_test_session()

    try:
        try:
            record_answer_feedback(
                db=db,
                request_id="missing-request-id",
                feedback_type=FeedbackType.DOWN,
            )
        except ValueError as error:
            assert str(error) == "request_id does not reference an audited request"
        else:
            raise AssertionError("unknown request_id must be rejected")
    finally:
        db.close()
