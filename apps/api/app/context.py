from dataclasses import dataclass 

from app.chunking import count_tokens
from app.retrieval import RetrievedChunk #导入统一检索结果类型。

@dataclass
class ContextAssembly:
    """检索结果组装成LLM上下文后的结果。"""

    context_text:str #最终发送给LLM的证据文本。
    chunks:list[RetrievedChunk] #实际上进入上下文的chunks.
    token_count:int 
    skipped_count:int #因token 预算未进入上下文的chunk数。


def assemble_context(
        chunks:list[RetrievedChunk],
        max_tokens:int = 1500,
) -> ContextAssembly:
    """将检索结果按排名组装为受token限制的证据上下文。"""

    if max_tokens<=0:
        raise ValueError("max_tokens must be greater than zero")

    selected_chunks:list[RetrievedChunk] = []
    context_parts:list[str] = []
    skipped_count = 0

    for chunk in chunks:
        evidence_number = len(selected_chunks) + 1

        # 给模型的上下文包含证据编号、章节路径和正文。
        evidence_text = (
            f"[证据 {evidence_number}]\n"
            f"章节：{chunk.section_path}\n"
            f"内容：\n{chunk.content}"
        )

        # 计算加入当前证据后的完整上下问题token数。
        candidate_context = "\n\n".join(
            [*context_parts,evidence_text]
        )

        if count_tokens(candidate_context)>max_tokens:
            # 后面的结果通常排名更低，预算不足时跳过。
            skipped_count += 1
            continue
        context_parts.append(evidence_text) #保存当前证据文本。
        selected_chunks.append(chunk) #保存对应的chunk。

    context_text = "\n\n".join(context_parts) #拼接最终上下文。

    return ContextAssembly(
        context_text=context_text,
        chunks=selected_chunks,
        token_count=count_tokens(context_text),
        skipped_count=skipped_count,
    )