"""RAG 检索黄金集评测：先衡量正确证据是否被稳定召回。"""

import argparse
import json
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Document
from app.retrieval import RetrievedChunk, hybrid_search
from app.vector_store import get_qdrant_client


DEFAULT_GOLDEN_SET_PATH = Path("datasets/eval/rag_golden_set.json")


@dataclass(frozen=True)
class EvaluationCase:
    """一条人工标注的检索评测样本。"""

    case_id: str
    source_document: str
    query: str
    evidence_anchor: str
    expected_answer_points: list[str]
    tags: list[str]


@dataclass(frozen=True)
class CaseRetrievalResult:
    """单题的命中情况和正确证据首次出现的位置。"""

    case_id: str
    hit: bool
    first_relevant_rank: int | None


def load_golden_set(path: Path = DEFAULT_GOLDEN_SET_PATH) -> list[EvaluationCase]:
    """读取人工维护的黄金集，并做最低限度的结构校验。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("golden set must contain a non-empty cases list")

    return [
        EvaluationCase(
            case_id=item["id"],
            source_document=item["source_document"],
            query=item["query"],
            evidence_anchor=item["evidence_anchor"],
            expected_answer_points=item["expected_answer_points"],
            tags=item["tags"],
        )
        for item in cases
    ]


def evaluate_case_retrieval(
    case: EvaluationCase,
    results: list[RetrievedChunk],
    document_titles_by_id: dict[str, str],
) -> CaseRetrievalResult:
    """判断结果列表中是否出现人工标注的文档和证据锚点。"""

    for rank, chunk in enumerate(results, start=1):
        if (
            document_titles_by_id.get(chunk.document_id) == case.source_document
            and normalize_evidence_anchor(case.evidence_anchor)
            in normalize_evidence_anchor(chunk.section_path)
        ):
            return CaseRetrievalResult(
                case_id=case.case_id,
                hit=True,
                first_relevant_rank=rank,
            )

    return CaseRetrievalResult(
        case_id=case.case_id,
        hit=False,
        first_relevant_rank=None,
    )


def normalize_evidence_anchor(value: str) -> str:
    """规范化全半角标点和空白，避免清洗后章节路径造成误判。"""

    return " ".join(unicodedata.normalize("NFKC", value).split())


def summarize_retrieval(results: list[CaseRetrievalResult]) -> dict[str, float | int]:
    """汇总 Hit/Recall@K 与 MRR；每题一个锚点时 Hit@K 即 Recall@K。"""

    if not results:
        raise ValueError("evaluation results must not be empty")

    hit_count = sum(result.hit for result in results)
    reciprocal_rank_sum = sum(
        1 / result.first_relevant_rank
        for result in results
        if result.first_relevant_rank is not None
    )
    total = len(results)

    return {
        "cases": total,
        "hits": hit_count,
        "recall_at_k": round(hit_count / total, 4),
        "mrr": round(reciprocal_rank_sum / total, 4),
    }


def run_retrieval_evaluation(
    db: Session,
    *,
    cases: list[EvaluationCase],
    limit: int = 5,
    candidate_limit: int = 20,
    enable_query_rewrite: bool = True,
    on_case_complete: Callable[[int, int, EvaluationCase, CaseRetrievalResult, float], None] | None = None,
) -> tuple[dict[str, float | int], list[CaseRetrievalResult]]:
    """逐题执行真实混合检索，并返回总指标和逐题结果。"""

    if limit <= 0 or candidate_limit <= 0:
        raise ValueError("retrieval limits must be greater than zero")

    document_titles_by_id = {
        document.id: document.title
        for document in db.query(Document).all()
    }
    results = []
    for index, case in enumerate(cases, start=1):
        started_at = time.perf_counter()
        retrieved = hybrid_search(
            query=case.query,
            db=db,
            limit=limit,
            candidate_limit=candidate_limit,
            enable_query_rewrite=enable_query_rewrite,
        )
        case_result = evaluate_case_retrieval(
            case,
            retrieved,
            document_titles_by_id,
        )
        results.append(case_result)
        if on_case_complete:
            on_case_complete(
                index,
                len(cases),
                case,
                case_result,
                time.perf_counter() - started_at,
            )

    return summarize_retrieval(results), results


def main() -> None:
    """命令行入口，输出可保存到实验记录中的 JSON 报告。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET_PATH)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument(
        "--disable-query-rewrite",
        action="store_true",
        help="跳过百炼改写和多跳查询，只评估本地检索与重排序基线。",
    )
    parser.add_argument(
        "--case-id",
        help="只运行指定样本，用于端到端冒烟验证。",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="从第几道题开始运行，序号从 1 开始。",
    )
    parser.add_argument(
        "--count",
        type=int,
        help="本次最多运行多少道题，用于分批评测。",
    )
    args = parser.parse_args()

    cases = load_golden_set(args.golden_set)
    if args.case_id:
        cases = [case for case in cases if case.case_id == args.case_id]
        if not cases:
            raise ValueError(f"case not found: {args.case_id}")
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
        summary, case_results = run_retrieval_evaluation(
            db,
            cases=cases,
            limit=args.limit,
            candidate_limit=args.candidate_limit,
            enable_query_rewrite=not args.disable_query_rewrite,
            on_case_complete=lambda index, total, case, result, seconds: print(
                f"[{index}/{total}] {case.case_id} "
                f"hit={result.hit} rank={result.first_relevant_rank} "
                f"seconds={seconds:.1f}",
                flush=True,
            ),
        )
    finally:
        db.close()
        # Qdrant Local 独占目录；CLI 结束前显式释放锁，便于继续调试。
        if get_qdrant_client.cache_info().currsize:
            get_qdrant_client().close()
            get_qdrant_client.cache_clear()

    print(json.dumps({
        "summary": summary,
        "cases": [result.__dict__ for result in case_results],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
