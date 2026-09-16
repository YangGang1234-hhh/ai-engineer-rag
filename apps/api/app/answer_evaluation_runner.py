"""端到端生成评测：执行 RAG、评估相关性与忠实度，并输出可复现实验报告。"""

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.answer_evaluation import (
    FaithfulnessJudgment,
    RelevanceJudgment,
    judge_answer_faithfulness,
    judge_answer_relevance,
)
from app.context import assemble_context
from app.database import SessionLocal
from app.generation import GeneratedAnswer, generate_answer
from app.retrieval import RetrievedChunk, expand_chunks_for_context, hybrid_search
from app.vector_store import get_qdrant_client


DEFAULT_ANSWER_EVAL_SET_PATH = Path("datasets/eval/answer_eval_cases.json")
DEFAULT_REPORTS_DIR = Path("datasets/eval/reports")


@dataclass(frozen=True)
class AnswerEvaluationCase:
    """生成评测所需的、不可变的黄金样本字段。"""

    case_id: str
    query: str
    expected_answer_points: list[str]
    source_document: str
    evidence_anchor: str
    tags: list[str]


def load_answer_evaluation_cases(
    path: Path = DEFAULT_ANSWER_EVAL_SET_PATH,
) -> list[AnswerEvaluationCase]:
    """加载回答黄金集；运行时不写回，保证评测输入可复现。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("answer evaluation set must contain a non-empty cases list")

    cases: list[AnswerEvaluationCase] = []
    seen_ids: set[str] = set()
    for item in raw_cases:
        case_id = item.get("id")
        query = item.get("query")
        points = item.get("expected_answer_points")
        if (
            not isinstance(case_id, str)
            or not case_id.strip()
            or case_id in seen_ids
            or not isinstance(query, str)
            or not query.strip()
            or not isinstance(points, list)
            or not all(isinstance(point, str) and point.strip() for point in points)
        ):
            raise ValueError("answer evaluation cases must have unique ids, query and points")
        seen_ids.add(case_id)
        cases.append(
            AnswerEvaluationCase(
                case_id=case_id,
                query=query,
                expected_answer_points=points,
                source_document=str(item.get("source_document", "")),
                evidence_anchor=str(item.get("evidence_anchor", "")),
                tags=list(item.get("tags", [])),
            )
        )
    return cases


def serialize_evidence(chunks: list[RetrievedChunk]) -> list[dict[str, object]]:
    """保存裁判看到的完整证据，便于人工复核异常样本。"""

    return [
        {
            "citation_number": index,
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "section_path": chunk.section_path,
            "content": chunk.content,
            "score": chunk.score,
            "source": chunk.source,
        }
        for index, chunk in enumerate(chunks, start=1)
    ]


def build_case_report(
    *,
    case: AnswerEvaluationCase,
    generated_answer: GeneratedAnswer,
    evidence: list[RetrievedChunk],
    relevance: RelevanceJudgment,
    faithfulness: FaithfulnessJudgment,
    elapsed_seconds: float,
) -> dict[str, object]:
    """将每题的输入、输出、证据和裁判结果统一成报告记录。"""

    return {
        "case_id": case.case_id,
        "query": case.query,
        "source_document": case.source_document,
        "evidence_anchor": case.evidence_anchor,
        "tags": case.tags,
        "expected_answer_points": case.expected_answer_points,
        "generated_answer": generated_answer.answer,
        "generated_citations": generated_answer.citations,
        "retrieved_evidence": serialize_evidence(evidence),
        "relevance_judgment": relevance.model_dump(),
        "faithfulness_judgment": faithfulness.model_dump(),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def summarize_answer_evaluation(case_reports: list[dict[str, object]]) -> dict[str, object]:
    """汇总回答相关性、忠实度及引用风险，报告均保留逐题明细。"""

    if not case_reports:
        raise ValueError("answer evaluation reports must not be empty")

    completed_reports = [item for item in case_reports if "error" not in item]
    if not completed_reports:
        return {
            "cases": len(case_reports), "completed_cases": 0,
            "failed_cases": len(case_reports), "average_relevance_score": None,
            "average_faithfulness_score": None, "relevance_score_5_rate": None,
            "faithfulness_score_5_rate": None, "unsupported_claims": 0,
            "invalid_citations": 0,
        }

    relevance_scores = [
        item["relevance_judgment"]["relevance_score"]  # type: ignore[index]
        for item in completed_reports
    ]
    faithfulness_scores = [
        item["faithfulness_judgment"]["faithfulness_score"]  # type: ignore[index]
        for item in completed_reports
    ]
    unsupported_claims = sum(
        len(item["faithfulness_judgment"]["unsupported_claims"])  # type: ignore[index]
        for item in completed_reports
    )
    invalid_citations = sum(
        len(item["faithfulness_judgment"]["invalid_citations"])  # type: ignore[index]
        for item in completed_reports
    )
    total = len(completed_reports)

    return {
        "cases": len(case_reports),
        "completed_cases": total,
        "failed_cases": len(case_reports) - total,
        "average_relevance_score": round(sum(relevance_scores) / total, 4),
        "average_faithfulness_score": round(sum(faithfulness_scores) / total, 4),
        "relevance_score_5_rate": round(relevance_scores.count(5) / total, 4),
        "faithfulness_score_5_rate": round(faithfulness_scores.count(5) / total, 4),
        "unsupported_claims": unsupported_claims,
        "invalid_citations": invalid_citations,
    }


def run_answer_evaluation(
    db: Session,
    *,
    cases: list[AnswerEvaluationCase],
    retrieval_limit: int = 3,
    candidate_limit: int = 20,
    context_max_tokens: int = 1200,
    enable_query_rewrite: bool = True,
    continue_on_error: bool = False,
    on_case_complete: Callable[[int, int, dict[str, object]], None] | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """运行真实检索、生成与双裁判；调用失败会交由 CLI 标记并返回非零状态。"""

    if retrieval_limit <= 0 or candidate_limit <= 0 or context_max_tokens <= 0:
        raise ValueError("retrieval, candidate and context limits must be greater than zero")

    reports: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        started_at = time.perf_counter()
        try:
            retrieved = hybrid_search(
                query=case.query, db=db, limit=retrieval_limit,
                candidate_limit=candidate_limit,
                enable_query_rewrite=enable_query_rewrite,
            )
            expanded = expand_chunks_for_context(
                retrieved, db, max_evidence=retrieval_limit, local_max_tokens=320,
            )
            context = assemble_context(expanded, max_tokens=context_max_tokens)
            if not context.chunks:
                raise RuntimeError(f"no evidence retrieved for case: {case.case_id}")
            answer = generate_answer(query=case.query, context=context)
            relevance = judge_answer_relevance(
                query=case.query, answer=answer.answer,
                expected_answer_points=case.expected_answer_points,
            )
            faithfulness = judge_answer_faithfulness(
                query=case.query, answer=answer.answer, citations=answer.citations,
                evidence_text=context.context_text,
            )
            report = build_case_report(
                case=case, generated_answer=answer, evidence=context.chunks,
                relevance=relevance, faithfulness=faithfulness,
                elapsed_seconds=time.perf_counter() - started_at,
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            report = {
                "case_id": case.case_id, "query": case.query,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - started_at, 3),
            }
        reports.append(report)
        if on_case_complete:
            on_case_complete(index, len(cases), report)

    return summarize_answer_evaluation(reports), reports


def main() -> None:
    """CLI：可用 case-id / start-index / count 分批跑，防止长任务中断后重头开始。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_ANSWER_EVAL_SET_PATH)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--case-id")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--count", type=int)
    parser.add_argument("--retrieval-limit", type=int, default=3)
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--context-max-tokens", type=int, default=1200)
    parser.add_argument(
        "--disable-query-rewrite",
        action="store_true",
        help="跳过查询改写，便于隔离评估检索与生成主链路。",
    )
    parser.add_argument(
        "--continue-on-error", action="store_true",
        help="单题失败时记录错误并继续，确保批次报告可用于定位问题。",
    )
    args = parser.parse_args()

    cases = load_answer_evaluation_cases(args.golden_set)
    if args.case_id:
        cases = [case for case in cases if case.case_id == args.case_id]
    else:
        if args.start_index <= 0:
            raise ValueError("start-index must be greater than zero")
        cases = cases[args.start_index - 1:]
        if args.count is not None:
            if args.count <= 0:
                raise ValueError("count must be greater than zero")
            cases = cases[:args.count]
    if not cases:
        raise ValueError("the selected evaluation batch is empty")

    db = SessionLocal()
    try:
        summary, reports = run_answer_evaluation(
            db,
            cases=cases,
            retrieval_limit=args.retrieval_limit,
            candidate_limit=args.candidate_limit,
            context_max_tokens=args.context_max_tokens,
            enable_query_rewrite=not args.disable_query_rewrite,
            continue_on_error=args.continue_on_error,
            on_case_complete=lambda index, total, report: print(
                f"[{index}/{total}] {report['case_id']} "
                + (
                    f"relevance={report['relevance_judgment']['relevance_score']} "  # type: ignore[index]
                    f"faithfulness={report['faithfulness_judgment']['faithfulness_score']}"  # type: ignore[index]
                    if "error" not in report
                    else f"error={report['error']}"
                ),
                flush=True,
            ),
        )
    finally:
        db.close()
        if get_qdrant_client.cache_info().currsize:
            get_qdrant_client().close()
            get_qdrant_client.cache_clear()

    report = {
        "report_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_golden_set": str(args.golden_set),
        "configuration": {
            "retrieval_limit": args.retrieval_limit,
            "candidate_limit": args.candidate_limit,
            "context_max_tokens": args.context_max_tokens,
            "enable_query_rewrite": not args.disable_query_rewrite,
            "continue_on_error": args.continue_on_error,
        },
        "summary": summary,
        "cases": reports,
    }
    report_path = args.report or (
        DEFAULT_REPORTS_DIR
        / f"answer-evaluation-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report written to: {report_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
