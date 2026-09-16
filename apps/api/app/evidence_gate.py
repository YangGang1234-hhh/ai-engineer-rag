"""根据实际检索结果判断问题是否有足够证据进入回答生成。"""

from collections.abc import Sequence
from typing import Protocol

from app.query_guard import QueryGuardDecision, QueryGuardResult


class ScoredEvidence(Protocol):
    """证据准入只依赖分数，避免与检索实现强耦合。"""

    score: float


def check_retrieval_evidence(
    chunks: Sequence[ScoredEvidence],
    *,
    minimum_score: float,
) -> QueryGuardResult:
    """要求至少一条证据达到阈值；失败时不得调用生成模型。"""

    if minimum_score < 0:
        raise ValueError("minimum_score must not be negative")
    if not chunks:
        return QueryGuardResult(
            decision=QueryGuardDecision.INSUFFICIENT_EVIDENCE,
            reason="no_retrieved_evidence",
            user_message="知识库中没有足够证据回答该问题。",
        )

    best_score = max(chunk.score for chunk in chunks)
    if best_score < minimum_score:
        return QueryGuardResult(
            decision=QueryGuardDecision.INSUFFICIENT_EVIDENCE,
            reason=f"best_evidence_score_below_threshold:{best_score:.4f}",
            user_message="知识库中没有足够证据回答该问题。",
        )
    return QueryGuardResult(
        decision=QueryGuardDecision.ALLOW,
        reason=f"retrieval_evidence_sufficient:{best_score:.4f}",
    )
