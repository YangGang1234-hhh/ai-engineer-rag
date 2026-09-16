from app.answer_evaluation import FaithfulnessJudgment, RelevanceJudgment
from app.answer_evaluation_runner import (
    AnswerEvaluationCase,
    build_case_report,
    load_answer_evaluation_cases,
    summarize_answer_evaluation,
)
from app.generation import GeneratedAnswer
from app.retrieval import RetrievedChunk


def test_load_answer_evaluation_cases_reads_the_existing_golden_set() -> None:
    cases = load_answer_evaluation_cases()

    assert len(cases) == 30
    assert cases[0].case_id == "handbook-01"
    assert cases[-1].case_id == "basics-10"


def test_build_case_report_preserves_reviewable_evidence() -> None:
    case = AnswerEvaluationCase(
        case_id="case-1",
        query="什么是 RAG？",
        expected_answer_points=["检索资料"],
        source_document="测试资料",
        evidence_anchor="第 1 节",
        tags=["test"],
    )
    chunk = RetrievedChunk(
        chunk_id="chunk-1", document_id="doc-1", content="RAG 检索资料。",
        section_path="第 1 节", position=0, token_count=5, score=0.9, source="reranked",
    )
    report = build_case_report(
        case=case,
        generated_answer=GeneratedAnswer(answer="RAG 检索资料[1]。", citations=[1]),
        evidence=[chunk],
        relevance=RelevanceJudgment(
            relevance_score=5, covered_points=["检索资料"], missing_points=[], reason="完整。",
        ),
        faithfulness=FaithfulnessJudgment(
            faithfulness_score=5, unsupported_claims=[], invalid_citations=[], reason="有依据。",
        ),
        elapsed_seconds=1.2349,
    )

    assert report["retrieved_evidence"] == [{
        "citation_number": 1, "chunk_id": "chunk-1", "document_id": "doc-1",
        "section_path": "第 1 节", "content": "RAG 检索资料。", "score": 0.9,
        "source": "reranked",
    }]
    assert report["elapsed_seconds"] == 1.235


def test_summarize_answer_evaluation_calculates_scores_and_risks() -> None:
    reports = [
        {
            "relevance_judgment": {"relevance_score": 5},
            "faithfulness_judgment": {
                "faithfulness_score": 5, "unsupported_claims": [], "invalid_citations": [],
            },
        },
        {
            "relevance_judgment": {"relevance_score": 3},
            "faithfulness_judgment": {
                "faithfulness_score": 4, "unsupported_claims": ["无证据结论"], "invalid_citations": [2],
            },
        },
    ]

    assert summarize_answer_evaluation(reports) == {
        "cases": 2,
        "completed_cases": 2,
        "failed_cases": 0,
        "average_relevance_score": 4.0,
        "average_faithfulness_score": 4.5,
        "relevance_score_5_rate": 0.5,
        "faithfulness_score_5_rate": 0.5,
        "unsupported_claims": 1,
        "invalid_citations": 1,
    }
