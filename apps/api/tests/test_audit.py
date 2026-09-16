from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.audit import hash_query, record_query_audit
from app.database import Base
from app.models import QueryAuditLog


def create_test_session():
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def test_record_query_audit_persists_privacy_safe_request_metadata() -> None:
    db = create_test_session()
    query = "什么是混合检索？"

    try:
        record_query_audit(
            db=db,
            request_id="aab0f863-c9c4-42b9-8e10-99277dbba2ff",
            query=query,
            decision="allow",
            answer_status="answered",
            reason="evidence_sufficient",
            retrieved_chunk_ids=["chunk-1", "chunk-2"],
            retrieved_scores=[0.91, 0.62],
            citation_numbers=[1],
            latency_ms=123,
            model_name="qwen-plus",
        )

        saved = db.query(QueryAuditLog).one()
        assert saved.query_hash == hash_query(query)
        assert saved.query_hash != query
        assert saved.decision == "allow"
        assert saved.retrieved_chunk_ids == ["chunk-1", "chunk-2"]
        assert saved.retrieved_scores == [0.91, 0.62]
        assert saved.citation_numbers == [1]
        assert saved.answer_status == "answered"
        assert saved.latency_ms == 123
    finally:
        db.close()
