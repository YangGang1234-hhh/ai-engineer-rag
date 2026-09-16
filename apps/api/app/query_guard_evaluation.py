"""查询准入 Guard 的离线评测运行器。"""

import argparse
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.query_guard import (
    QueryGuardDecision,
    QueryGuardResult,
    check_basic_query_safety,
)
from app.query_scope import classify_query_scope


DEFAULT_GUARD_EVAL_SET_PATH = Path("datasets/eval/query_guard_eval_cases.json")
DEFAULT_REPORTS_DIR = Path("datasets/eval/reports")


@dataclass(frozen=True)
class QueryGuardEvaluationCase:
    case_id: str
    query: str
    expected_decision: QueryGuardDecision
    category: str
    notes: str


def load_query_guard_evaluation_cases(
    path: Path = DEFAULT_GUARD_EVAL_SET_PATH,
) -> list[QueryGuardEvaluationCase]:
    """读取不可变评测集，并在运行前校验数据结构。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("query guard evaluation set must contain a non-empty cases list")

    cases: list[QueryGuardEvaluationCase] = []
    seen_ids: set[str] = set()
    for item in raw_cases:
        case_id = item.get("id")
        query = item.get("query")
        expected = item.get("expected_decision")
        category = item.get("category")
        if (
            not isinstance(case_id, str)
            or not case_id.strip()
            or case_id in seen_ids
            or not isinstance(query, str)
            or not query.strip()
            or not isinstance(category, str)
            or not category.strip()
        ):
            raise ValueError("each query guard case needs unique id, query and category")
        try:
            expected_decision = QueryGuardDecision(expected)
        except ValueError as exc:
            raise ValueError(f"invalid expected decision for case {case_id}") from exc
        seen_ids.add(case_id)
        cases.append(QueryGuardEvaluationCase(
            case_id=case_id,
            query=query,
            expected_decision=expected_decision,
            category=category,
            notes=str(item.get("notes", "")),
        ))
    return cases


def run_query_guard_evaluation(
    cases: list[QueryGuardEvaluationCase],
    *,
    scope_classifier: Callable[[str], QueryGuardResult] = classify_query_scope,
) -> list[dict[str, object]]:
    """运行规则 Guard 与语义范围路由，并保留每题决策链路。"""

    reports: list[dict[str, object]] = []
    for case in cases:
        basic_result = check_basic_query_safety(case.query)
        scope_result: QueryGuardResult | None = None
        error: str | None = None
        actual_decision = basic_result.decision

        if basic_result.decision == QueryGuardDecision.ALLOW:
            try:
                scope_result = scope_classifier(case.query)
                actual_decision = scope_result.decision
            except (RuntimeError, ValueError) as exc:
                error = type(exc).__name__

        reports.append({
            "case_id": case.case_id,
            "query": case.query,
            "category": case.category,
            "notes": case.notes,
            "expected_decision": str(case.expected_decision),
            "actual_decision": str(actual_decision) if error is None else "error",
            "matched": error is None and actual_decision == case.expected_decision,
            "basic_guard": basic_result.model_dump(),
            "scope_guard": scope_result.model_dump() if scope_result else None,
            "error": error,
        })
    return reports


def summarize_query_guard_evaluation(
    reports: list[dict[str, object]],
) -> dict[str, object]:
    """计算 Guard 的整体准确率、误拒绝率与攻击漏拦截率。"""

    if not reports:
        raise ValueError("query guard reports must not be empty")

    completed = [report for report in reports if report["error"] is None]
    matched = [report for report in completed if report["matched"]]
    expected_counts = Counter(str(report["expected_decision"]) for report in reports)
    category_reports: dict[str, list[dict[str, object]]] = {}
    for report in reports:
        category_reports.setdefault(str(report["category"]), []).append(report)

    def ratio(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    unsafe_reports = [
        report for report in reports
        if report["expected_decision"] == QueryGuardDecision.UNSAFE
    ]
    normal_reports = [
        report for report in reports
        if report["expected_decision"] == QueryGuardDecision.ALLOW
    ]
    return {
        "total_cases": len(reports),
        "completed_cases": len(completed),
        "failed_cases": len(reports) - len(completed),
        "overall_accuracy": ratio(len(matched), len(reports)),
        "safety_interception_rate": ratio(
            sum(report["actual_decision"] == "unsafe" for report in unsafe_reports),
            len(unsafe_reports),
        ),
        "unsafe_bypass_rate": ratio(
            sum(report["actual_decision"] == "allow" for report in unsafe_reports),
            len(unsafe_reports),
        ),
        "normal_false_rejection_rate": ratio(
            sum(report["actual_decision"] != "allow" for report in normal_reports),
            len(normal_reports),
        ),
        "expected_decision_counts": dict(sorted(expected_counts.items())),
        "category_accuracy": {
            category: ratio(
                sum(bool(item["matched"]) for item in items),
                len(items),
            )
            for category, items in sorted(category_reports.items())
        },
    }


def write_query_guard_report(
    *,
    cases: list[QueryGuardEvaluationCase],
    reports: list[dict[str, object]],
    output_path: Path,
) -> Path:
    """写出带版本时间和逐题结果的 JSON 报告。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluation_type": "query_guard",
        "generated_at": datetime.now(UTC).isoformat(),
        "cases": [asdict(case) | {"expected_decision": str(case.expected_decision)} for case in cases],
        "summary": summarize_query_guard_evaluation(reports),
        "results": reports,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run query guard evaluation")
    parser.add_argument("--cases", type=Path, default=DEFAULT_GUARD_EVAL_SET_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cases = load_query_guard_evaluation_cases(args.cases)
    reports = run_query_guard_evaluation(cases)
    output_path = args.output or (
        DEFAULT_REPORTS_DIR
        / f"query-guard-evaluation-{datetime.now():%Y%m%d-%H%M%S}.json"
    )
    write_query_guard_report(cases=cases, reports=reports, output_path=output_path)
    print(f"Query guard evaluation report written to: {output_path}")


if __name__ == "__main__":
    main()
