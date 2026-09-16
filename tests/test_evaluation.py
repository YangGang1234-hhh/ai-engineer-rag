from app.evaluation import (
    CaseRetrievalResult,
    EvaluationCase,
    evaluate_case_retrieval,
    load_golden_set,
    summarize_retrieval,
)
from app.retrieval import RetrievedChunk


def create_chunk(*, document_id: str, section_path: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        document_id=document_id,
        content="测试证据",
        section_path=section_path,
        position=0,
        token_count=4,
        score=0.9,
        source="reranked",
    )


def test_evaluate_case_retrieval_requires_document_and_anchor() -> None:
    case = EvaluationCase(
        case_id="case-1",
        source_document="RAG应用开发手册",
        query="RAG 的流程是什么？",
        evidence_anchor="PDF 第 7 页",
        expected_answer_points=["信息检索"],
        tags=["pdf"],
    )
    results = [
        create_chunk(document_id="doc-other", section_path="PDF 第 7 页"),
        create_chunk(document_id="doc-handbook", section_path="PDF 第 7 页 > RAG实现流程"),
    ]

    scored = evaluate_case_retrieval(
        case,
        results,
        {"doc-other": "其他资料", "doc-handbook": "RAG应用开发手册"},
    )

    assert scored.hit is True
    assert scored.first_relevant_rank == 2


def test_evaluate_case_retrieval_normalizes_full_width_anchor_punctuation() -> None:
    """清洗器转换全角冒号后，黄金集章节锚点仍应被判定命中。"""

    case = EvaluationCase(
        case_id="case-punctuation",
        source_document="RAG 检索基础：从召回到引用回答",
        query="什么是 RAG？",
        evidence_anchor="RAG 检索基础：从召回到引用回答 > 1. RAG 是什么",
        expected_answer_points=["检索"],
        tags=["markdown"],
    )
    result = evaluate_case_retrieval(
        case,
        [create_chunk(
            document_id="doc-basics",
            section_path="RAG 检索基础:从召回到引用回答 > 1. RAG 是什么",
        )],
        {"doc-basics": "RAG 检索基础：从召回到引用回答"},
    )

    assert result.hit is True
    assert result.first_relevant_rank == 1


def test_summarize_retrieval_calculates_recall_and_mrr() -> None:
    summary = summarize_retrieval([
        CaseRetrievalResult("case-1", True, 1),
        CaseRetrievalResult("case-2", True, 2),
        CaseRetrievalResult("case-3", False, None),
    ])

    assert summary == {"cases": 3, "hits": 2, "recall_at_k": 0.6667, "mrr": 0.5}


def test_golden_set_contains_thirty_cases() -> None:
    cases = load_golden_set()

    assert len(cases) == 30
    assert sum(case.source_document == "RAG应用开发手册" for case in cases) == 20
    assert sum(case.source_document == "RAG 检索基础：从召回到引用回答" for case in cases) == 10
    assert {case.case_id for case in cases} >= {"handbook-01", "basics-10"}
