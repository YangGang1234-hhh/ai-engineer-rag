from datetime import datetime
from collections import Counter
from math import ceil

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from pydantic import BaseModel, Field  # 定义并校验 API 请求和响应。
from sqlalchemy import select  # 构造数据库查询。
from sqlalchemy.orm import Session  # 表示一次数据库会话。
from app.models import AnswerFeedback, Document, QueryAuditLog  # 读取知识库和审计记录。
from pathlib import Path  # 用于定位项目中的前端 HTML 文件。
from fastapi.responses import FileResponse  # 用于把 HTML 文件返回给浏览器。
from time import perf_counter
from uuid import uuid4

from app.config import get_settings  # 读取应用配置。
from app.database import get_db  # 为每个请求提供 SQLite 会话。
from app.retrieval import (  # 调用混合检索，并为回答构造局部上下文。
    RetrievedChunk,
    expand_chunks_for_context,
    hybrid_search,
)

from app.context import assemble_context  # 将检索结果组装为 LLM 证据上下文。
from app.generation import GeneratedAnswer, generate_answer  # 生成带引用的回答。
from app.evidence_gate import check_retrieval_evidence
from app.query_guard import QueryGuardDecision, check_basic_query_safety
from app.query_scope import classify_query_scope
from app.audit import record_query_audit
from app.admin_auth import require_admin_api_key
from app.feedback import FeedbackType, record_answer_feedback

from app.document_processing import chunk_document  # 负责把文档切分为 chunks。
from app.full_text import (
    delete_document_fts_records,
    sync_document_chunks_to_fts,
)
from app.vector_store import delete_document_vectors
from app.indexing import index_document  # 生成向量并写入 Qdrant。
from app.ingestion import import_markdown_file, import_pdf_file


class HealthResponse(BaseModel):
    status: str
    environment: str
    vector_store: str

class SearchRequest(BaseModel):
    """关键词和向量混合检索请求。"""

    query: str = Field(min_length=1)  # 用户问题，不能为空。
    limit: int = Field(default=5, gt=0, le=20)  # 最终返回数量。
    candidate_limit: int = Field(default=20, gt=0, le=100)  # 每路候选数量。


class SearchResult(BaseModel):
    """对外返回的一条检索结果。"""

    chunk_id: str  # chunk 唯一 ID。
    document_id: str  # 所属文档 ID。
    content: str  # chunk 正文。
    section_path: str  # Markdown 标题路径。
    position: int  # chunk 在原文中的位置。
    token_count: int  # chunk token 数量。
    score: float  # Cross-Encoder 精排分数。
    source: str  # 结果来源，当前应为 reranked。


class SearchResponse(BaseModel):
    """混合检索 API 响应。"""

    results: list[SearchResult]  # 按相关性排序后的结果列表。

class CitationResponse(BaseModel):
    """问答响应中的一条证据引用。"""

    citation_number: int  # 证据编号，对应答案中的 [1]、[2]。
    chunk_id: str  # chunk 唯一 ID。
    section_path: str  # 证据所属章节。
    content: str  # 证据正文。


class AskRequest(BaseModel):
    """RAG 问答请求。"""

    query: str = Field(min_length=1)  # 用户问题。
    retrieval_limit: int = Field(default=3, gt=0, le=20)  # Rerank 后进入回答的证据数量。
    candidate_limit: int = Field(default=20, gt=0, le=100)  # 每路检索候选数量。
    context_max_tokens: int = Field(default=1200, gt=0, le=6000)  # 上下文 token 上限。


class AskResponse(BaseModel):
    """RAG 问答响应。"""

    request_id: str  # 用于后续提交用户反馈并关联审计记录。
    answer: str  # LLM 生成的最终答案。
    citations: list[int]  # 答案使用的证据编号。
    evidence: list[CitationResponse]  # 实际发送给 LLM 的证据。


class FeedbackRequest(BaseModel):
    """用户对一次问答结果的赞/踩反馈。"""

    request_id: str = Field(min_length=36, max_length=36)
    feedback_type: FeedbackType
    reason: str | None = Field(default=None, max_length=500)


class FeedbackResponse(BaseModel):
    request_id: str
    feedback_type: FeedbackType
    reason: str | None
    updated_at: datetime


class FeedbackOptimizationItem(BaseModel):
    """供管理员排查的低质量反馈，始终不返回原问题和完整回答。"""

    request_id: str
    feedback_type: FeedbackType
    reason: str | None
    created_at: datetime
    updated_at: datetime
    decision: str
    answer_status: str
    retrieved_chunk_count: int
    citation_numbers: list[int]
    latency_ms: int | None


class FeedbackOptimizationListResponse(BaseModel):
    items: list[FeedbackOptimizationItem]

class UploadResponse(BaseModel):
    """Markdown 文档上传后的处理结果。"""

    document_id: str  # 文档唯一 ID。
    title: str  # 文档标题。
    created: bool  # 是否是首次导入；重复内容时为 False。
    chunks_created: int  # 本次生成的 chunk 数量。
    chunks_indexed: int  # 成功写入 Qdrant 的向量数量。
    status: str  # 文档最终状态。

class DocumentSummary(BaseModel):
    """文档列表中的单份文档摘要。"""

    document_id: str  # 文档唯一 ID。
    title: str  # 文档标题。
    source_type: str  # 文档来源类型。
    status: str  # 当前处理状态，例如 indexed。


class DocumentListResponse(BaseModel):
    """知识库文档列表响应。"""

    documents: list[DocumentSummary]


class AuditLogResponse(BaseModel):
    """对内排障使用的脱敏审计记录。"""

    request_id: str
    created_at: datetime
    decision: str
    reason: str | None
    retrieved_chunk_ids: list[str]
    retrieved_scores: list[float]
    citation_numbers: list[int]
    answer_status: str
    latency_ms: int | None
    error_type: str | None
    model_name: str | None


class AuditLogListResponse(BaseModel):
    logs: list[AuditLogResponse]


class DecisionCount(BaseModel):
    decision: str
    count: int


class ObservabilitySummaryResponse(BaseModel):
    """从审计日志聚合出的内部观测指标。"""

    total_requests: int
    answered_requests: int
    rejected_requests: int
    failed_requests: int
    answer_rate: float
    rejection_rate: float
    failure_rate: float
    average_latency_ms: float | None
    p95_latency_ms: int | None
    decision_counts: list[DecisionCount]

def to_search_result(chunk: RetrievedChunk) -> SearchResult:
    """将内部检索对象转换为 API 响应对象。"""

    return SearchResult(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        content=chunk.content,
        section_path=chunk.section_path,
        position=chunk.position,
        token_count=chunk.token_count,
        score=chunk.score,
        source=chunk.source,
    )

def to_citation_response(
    citation_number: int,
    chunk: RetrievedChunk,
) -> CitationResponse:
    """将内部 chunk 转换为带证据编号的 API 响应。"""

    return CitationResponse(
        citation_number=citation_number,
        chunk_id=chunk.chunk_id,
        section_path=chunk.section_path,
        content=chunk.content,
    )

def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Evidence-based and evaluable AI engineering learning RAG.",
    )

    # 当前文件位于 apps/api/app/，向上三级可以回到项目根目录。
    web_index = (
        Path(__file__).resolve().parents[3]
        / "apps"
        / "web"
        / "index.html"
    )
    observability_page = web_index.with_name("observability.html")
    document_management_page = web_index.with_name("documents.html")

    @app.get("/", include_in_schema=False)
    def web_page() -> FileResponse:
        """返回 RAG Web 页面。"""

        return FileResponse(web_index)

    @app.get("/observability", include_in_schema=False)
    def observability_page_view() -> FileResponse:
        """返回仅供内部查看的 RAG 运行观测页。"""

        return FileResponse(observability_page)

    @app.get("/documents/manage", include_in_schema=False)
    def document_management_page_view() -> FileResponse:
        """返回知识库文档管理页面。"""

        return FileResponse(document_management_page)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health_check() -> HealthResponse:
        try:
            return HealthResponse(
                status="ok",
                environment=settings.app_env,
                vector_store="qdrant",
            )
        except Exception:
            return HealthResponse(
                status="error",
                environment=settings.app_env,
                vector_store="qdrant",
            )

    @app.get(
        "/audit-logs",
        response_model=AuditLogListResponse,
        tags=["observability"],
        dependencies=[Depends(require_admin_api_key)],
    )
    def list_audit_logs(
        decision: str | None = None,
        limit: int = Query(default=50, gt=0, le=100),
        db: Session = Depends(get_db),
    ) -> AuditLogListResponse:
        """按时间倒序返回脱敏审计记录，供当前 MVP 内部排障使用。"""

        statement = select(QueryAuditLog).order_by(
            QueryAuditLog.created_at.desc()
        ).limit(limit)
        if decision is not None:
            statement = statement.where(QueryAuditLog.decision == decision)

        logs = db.scalars(statement).all()
        return AuditLogListResponse(
            logs=[
                AuditLogResponse(
                    request_id=log.request_id,
                    created_at=log.created_at,
                    decision=log.decision,
                    reason=log.reason,
                    retrieved_chunk_ids=log.retrieved_chunk_ids,
                    retrieved_scores=log.retrieved_scores,
                    citation_numbers=log.citation_numbers,
                    answer_status=log.answer_status,
                    latency_ms=log.latency_ms,
                    error_type=log.error_type,
                    model_name=log.model_name,
                )
                for log in logs
            ]
        )

    @app.get(
        "/feedback/optimization-items",
        response_model=FeedbackOptimizationListResponse,
        tags=["feedback"],
        dependencies=[Depends(require_admin_api_key)],
    )
    def list_feedback_optimization_items(
        feedback_type: FeedbackType = FeedbackType.DOWN,
        limit: int = Query(default=50, gt=0, le=100),
        db: Session = Depends(get_db),
    ) -> FeedbackOptimizationListResponse:
        """返回待优化反馈及可排查的脱敏链路信息。"""

        rows = db.execute(
            select(AnswerFeedback, QueryAuditLog)
            .join(
                QueryAuditLog,
                AnswerFeedback.request_id == QueryAuditLog.request_id,
            )
            .where(AnswerFeedback.feedback_type == str(feedback_type))
            .order_by(AnswerFeedback.updated_at.desc())
            .limit(limit)
        ).all()
        return FeedbackOptimizationListResponse(
            items=[
                FeedbackOptimizationItem(
                    request_id=feedback.request_id,
                    feedback_type=FeedbackType(feedback.feedback_type),
                    reason=feedback.reason,
                    created_at=feedback.created_at,
                    updated_at=feedback.updated_at,
                    decision=audit_log.decision,
                    answer_status=audit_log.answer_status,
                    retrieved_chunk_count=len(audit_log.retrieved_chunk_ids),
                    citation_numbers=audit_log.citation_numbers,
                    latency_ms=audit_log.latency_ms,
                )
                for feedback, audit_log in rows
            ]
        )

    @app.get(
        "/observability/summary",
        response_model=ObservabilitySummaryResponse,
        tags=["observability"],
        dependencies=[Depends(require_admin_api_key)],
    )
    def observability_summary(
        db: Session = Depends(get_db),
    ) -> ObservabilitySummaryResponse:
        """汇总审计日志中的成功、拒答、失败和时延指标。"""

        logs = db.scalars(select(QueryAuditLog)).all()
        total_requests = len(logs)
        status_counts = Counter(log.answer_status for log in logs)
        decision_counts = Counter(log.decision for log in logs)
        latencies = sorted(
            log.latency_ms for log in logs if log.latency_ms is not None
        )
        average_latency_ms = (
            round(sum(latencies) / len(latencies), 2)
            if latencies else None
        )
        p95_latency_ms = (
            latencies[ceil(len(latencies) * 0.95) - 1]
            if latencies else None
        )

        def rate(count: int) -> float:
            return round(count / total_requests, 4) if total_requests else 0.0

        return ObservabilitySummaryResponse(
            total_requests=total_requests,
            answered_requests=status_counts["answered"],
            rejected_requests=status_counts["rejected"],
            failed_requests=status_counts["failed"],
            answer_rate=rate(status_counts["answered"]),
            rejection_rate=rate(status_counts["rejected"]),
            failure_rate=rate(status_counts["failed"]),
            average_latency_ms=average_latency_ms,
            p95_latency_ms=p95_latency_ms,
            decision_counts=[
                DecisionCount(decision=decision, count=count)
                for decision, count in sorted(decision_counts.items())
            ],
        )

    @app.get(
        "/documents",
        response_model=DocumentListResponse,
        tags=["documents"],
        dependencies=[Depends(require_admin_api_key)],
    )
    def list_documents(
        db: Session = Depends(get_db),
    ) -> DocumentListResponse:
        """返回知识库中已导入的文档摘要。"""

        documents = db.scalars(
            select(Document).order_by(
                Document.created_at.desc()
            )
        ).all()

        return DocumentListResponse(
            documents=[
                DocumentSummary(
                    document_id=document.id,
                    title=document.title,
                    source_type=document.source_type,
                    status=document.status,
                )
                for document in documents
            ]
        )

    @app.delete(
        "/documents/{document_id}",
        status_code=204,
        tags=["documents"],
        dependencies=[Depends(require_admin_api_key)],
    )
    def delete_document(
        document_id: str,
        db: Session = Depends(get_db),
    ) -> Response:
        """从 SQLite、FTS5 和 Qdrant 删除指定知识库文档。"""

        document = db.get(Document, document_id)

        if document is None:
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        # 先删除向量；若 Qdrant 操作失败，SQLite 主数据仍保留，便于重试。
        delete_document_vectors(document_id)

        # 删除该文档的关键词索引记录。
        delete_document_fts_records(
            document_id=document_id,
            db=db,
        )

        # 删除主文档；SQLAlchemy 会级联删除其 chunks。
        db.delete(document)
        db.commit()

        return Response(status_code=204)
    
    @app.post(
        "/search",
        response_model=SearchResponse,
        tags=["retrieval"],
    )
    def search(
        request: SearchRequest,
        db: Session = Depends(get_db),
    ) -> SearchResponse:
        """执行混合检索并返回排序后的证据 chunks。"""

        chunks = hybrid_search(
            query=request.query,
            db=db,
            limit=request.limit,
            candidate_limit=request.candidate_limit,
        )

        return SearchResponse(
            results=[
                to_search_result(chunk)
                for chunk in chunks
            ]
        )

    @app.post(
        "/ask",
        response_model=AskResponse,
        tags=["generation"],
    )
    def ask(
        request: AskRequest,
        db: Session = Depends(get_db),
    ) -> AskResponse:
        """执行完整 RAG 问答流程并返回带引用的答案。"""

        request_id = str(uuid4())
        started_at = perf_counter()

        def write_audit(
            *,
            decision: str,
            answer_status: str,
            reason: str | None = None,
            chunks: list[RetrievedChunk] | None = None,
            citations: list[int] | None = None,
            error_type: str | None = None,
        ) -> None:
            """尽力记录链路结果，绝不因审计故障影响主问答。"""

            audit_chunks = chunks or []
            try:
                record_query_audit(
                    db=db,
                    request_id=request_id,
                    query=request.query,
                    decision=decision,
                    answer_status=answer_status,
                    reason=reason,
                    retrieved_chunk_ids=[chunk.chunk_id for chunk in audit_chunks],
                    retrieved_scores=[chunk.score for chunk in audit_chunks],
                    citation_numbers=citations or [],
                    latency_ms=round((perf_counter() - started_at) * 1000),
                    error_type=error_type,
                    model_name=(
                        settings.generation_llm_model.strip()
                        or settings.llm_model
                        or None
                    ),
                )
            except Exception:
                # 写库失败后回滚自身未完成事务，继续返回原有问答结果。
                db.rollback()

        # 第零步：明显越权指令在进入检索和模型调用前直接拦截。
        basic_guard = check_basic_query_safety(request.query)
        if basic_guard.decision != QueryGuardDecision.ALLOW:
            write_audit(
                decision=str(basic_guard.decision),
                answer_status="rejected",
                reason=basic_guard.reason,
            )
            return AskResponse(
                request_id=request_id,
                answer=basic_guard.user_message or "该请求无法处理。",
                citations=[], evidence=[],
            )

        # 第零点五步：范围外闲聊不进入检索/生成。
        # 路由服务临时不可用时选择 fail-open，仍由后续证据准入阻止无依据回答。
        try:
            scope_guard = classify_query_scope(request.query)
        except (RuntimeError, ValueError):
            scope_guard = None
        if (
            scope_guard is not None
            and scope_guard.decision != QueryGuardDecision.ALLOW
        ):
            write_audit(
                decision=str(scope_guard.decision),
                answer_status="rejected",
                reason=scope_guard.reason,
            )
            return AskResponse(
                request_id=request_id,
                answer=scope_guard.user_message or "该请求不属于知识库问答范围。",
                citations=[], evidence=[],
            )

        # 第一步：混合检索、RRF 融合和 Cross-Encoder 精排。
        chunks = hybrid_search(
            query=request.query,
            db=db,
            limit=request.retrieval_limit,
            candidate_limit=request.candidate_limit,
        )

        # 第一点五步：范围内问题也必须有足够实际证据，才能调用生成模型。
        evidence_guard = check_retrieval_evidence(
            chunks,
            minimum_score=settings.query_guard_min_evidence_score,
        )
        if evidence_guard.decision != QueryGuardDecision.ALLOW:
            write_audit(
                decision=str(evidence_guard.decision),
                answer_status="rejected",
                reason=evidence_guard.reason,
                chunks=chunks,
            )
            return AskResponse(
                request_id=request_id,
                answer=evidence_guard.user_message or "知识库中没有足够证据回答该问题。",
                citations=[],
                evidence=[],
            )

        # 对回答阶段按父块去重，但只扩展命中 child 附近的完整句子。
        # 这样既保留必要上下文，也不会把多个完整父块塞进模型。
        chunks = expand_chunks_for_context(
            chunks,
            db,
            max_evidence=3,
            local_max_tokens=320,
        )
        
        # 第二步：按 token 预算组装证据上下文。
        context = assemble_context(
            chunks=chunks,
            max_tokens=request.context_max_tokens,
        )

        # 第三步：让 LLM 只能依据组装后的证据生成答案。
        try:
            answer = generate_answer(
                query=request.query,
                context=context,
            )
        except RuntimeError as exc:
            # LLM 未配置、网络失败或服务异常时，返回明确的 503。
            # 不把底层 API 错误和敏感信息直接暴露给用户。
            write_audit(
                decision="allow",
                answer_status="failed",
                reason="generation_runtime_error",
                chunks=chunks,
                error_type=type(exc).__name__,
            )
            raise HTTPException(
                status_code=503,
                detail="回答生成服务暂时不可用，请稍后重试。",
            ) from exc
        except ValueError as exc:
            # LLM 返回了无法解析或引用不合法的内容。
            # 这是上游响应异常，使用 502 表示网关收到无效响应。
            write_audit(
                decision="allow",
                answer_status="failed",
                reason="generation_invalid_response",
                chunks=chunks,
                error_type=type(exc).__name__,
            )
            raise HTTPException(
                status_code=502,
                detail="回答生成服务返回了无效结果。",
            ) from exc

        # 第四步：返回答案、引用编号和实际使用的证据。
        response = AskResponse(
            request_id=request_id,
            answer=answer.answer,
            citations=answer.citations,
            evidence=[
                to_citation_response(index, chunk)
                for index, chunk in enumerate(context.chunks, start=1)
            ],
        )
        write_audit(
            decision="allow",
            answer_status="answered",
            reason="answer_generated",
            chunks=chunks,
            citations=answer.citations,
        )
        return response

    @app.post(
        "/feedback",
        response_model=FeedbackResponse,
        tags=["feedback"],
    )
    def submit_feedback(
        request: FeedbackRequest,
        db: Session = Depends(get_db),
    ) -> FeedbackResponse:
        """记录用户对一次已审计问答的最新反馈。"""

        try:
            feedback = record_answer_feedback(
                db=db,
                request_id=request.request_id,
                feedback_type=request.feedback_type,
                reason=request.reason,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=404,
                detail="未找到可反馈的问答请求。",
            ) from exc

        return FeedbackResponse(
            request_id=feedback.request_id,
            feedback_type=FeedbackType(feedback.feedback_type),
            reason=feedback.reason,
            updated_at=feedback.updated_at,
        )

    @app.post(
        "/documents/upload",
        response_model=UploadResponse,
        tags=["documents"],
        dependencies=[Depends(require_admin_api_key)],
    )
    async def upload_document(
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
    ) -> UploadResponse:
        """上传 Markdown 文档，并完成切分、全文索引和向量索引。"""


        # 根据扩展名选择 Markdown 或 PDF 导入路径。
        filename = file.filename or ""
        is_markdown = filename.lower().endswith(
            (".md", ".markdown")
        )
        is_pdf = filename.lower().endswith(".pdf")

        if not is_markdown and not is_pdf:
            raise HTTPException(
                status_code=400,
                detail="Only Markdown and PDF files are supported.",
            )

        # 只保留文件名，避免上传文件名包含路径造成路径安全问题。
        safe_filename = Path(filename).name
        source_folder = "markdown" if is_markdown else "pdf"

        # 原始文件按类型保存，便于后续重新处理。
        upload_dir = (
            Path(__file__).resolve().parents[3]
            / "datasets"
            / "raw"
            / source_folder
        )
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / safe_filename

        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        if is_markdown:
            # Markdown 必须是 UTF-8 文本。
            try:
                file_path.write_text(
                    content.decode("utf-8"),
                    encoding="utf-8",
                )
            except UnicodeDecodeError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="The Markdown file must use UTF-8 encoding.",
                ) from exc
        else:
            # PDF 是二进制格式，不能 decode 成 UTF-8 文本。
            file_path.write_bytes(content)
            
        try:
            # 第一步：保存原始文档并按内容哈希去重。
            import_result = (
                import_markdown_file(
                    file_path=file_path,
                    db=db,
                )
                if is_markdown
                else import_pdf_file(
                    file_path=file_path,
                    db=db,
                )
            )
            document = import_result.document

            # 重复上传相同内容时，已有文档已经可能完成索引。
            # 仍然重新处理，保证文件内容变化后索引保持最新。
            chunk_result = chunk_document(
                document_id=document.id,
                db=db,
            )

            # 第二步：同步当前文档的 FTS5 索引。
            fts_count = sync_document_chunks_to_fts(
                document_id=document.id,
                db=db,
            )
            db.commit()

            # 第三步：删除旧向量、生成新向量并写入 Qdrant。
            index_result = index_document(
                document_id=document.id,
                db=db,
            )

            return UploadResponse(
                document_id=document.id,
                title=document.title,
                created=import_result.created,
                chunks_created=chunk_result.chunks_created,
                chunks_indexed=index_result.chunks_indexed,
                status=document.status,
            )

        except ValueError as exc:
            # 文档为空、没有 chunks 等业务错误返回 400。
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc
        
    def search(request: SearchRequest, db: Session = Depends(get_db)):
        """执行混合检索并返回排序后的证据 chunks。"""

        chunks = hybrid_search(
            query=request.query,
            db=db,
            limit=request.limit,
            candidate_limit=request.candidate_limit,
        )

        return SearchResponse(
            results=[
                to_search_result(chunk)
                for chunk in chunks
            ]
        )
    return app


app = create_app()
