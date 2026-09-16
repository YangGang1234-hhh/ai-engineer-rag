from types import SimpleNamespace

import pytest

from app.evidence_gate import check_retrieval_evidence
from app.query_guard import QueryGuardDecision


def test_evidence_gate_allows_high_scoring_evidence() -> None:
    result = check_retrieval_evidence(
        [SimpleNamespace(score=0.18), SimpleNamespace(score=0.91)],
        minimum_score=0.20,
    )

    assert result.decision == QueryGuardDecision.ALLOW
    assert result.reason == "retrieval_evidence_sufficient:0.9100"


def test_evidence_gate_rejects_empty_or_weak_retrieval() -> None:
    empty = check_retrieval_evidence([], minimum_score=0.20)
    weak = check_retrieval_evidence(
        [SimpleNamespace(score=0.19)], minimum_score=0.20
    )

    assert empty.decision == QueryGuardDecision.INSUFFICIENT_EVIDENCE
    assert empty.reason == "no_retrieved_evidence"
    assert weak.decision == QueryGuardDecision.INSUFFICIENT_EVIDENCE
    assert "below_threshold" in weak.reason


def test_evidence_gate_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        check_retrieval_evidence([], minimum_score=-0.1)
