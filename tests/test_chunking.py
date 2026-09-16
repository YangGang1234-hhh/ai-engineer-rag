# 导入我们要测试的 Markdown 解析函数
from app.chunking import (
    StructureBlock,
    build_chunk_drafts,
    count_tokens,
    parse_markdown_blocks,
)


def test_parse_heading_path_and_paragraph() -> None:
    """一级标题下的普通正文，应成为带标题路径的段落块。"""

    # 这是一段最简单的 Markdown：
    # 一级标题后面跟一段正文。
    content = "# RAG 基础\n\nRAG 会先检索资料，再生成回答。"

    # 调用我们的解析函数。
    blocks = parse_markdown_blocks(content)

    # 预期只有一个结构块：标题本身不作为 chunk，正文才是结构块。
    assert len(blocks) == 1

    # 验证这个块被识别为普通段落。
    assert blocks[0].block_type == "paragraph"

    # 验证正文没有丢失。
    assert blocks[0].content == "RAG 会先检索资料，再生成回答。"

    # 验证标题被保存为正文的上下文路径。
    assert blocks[0].section_path == "RAG 基础"


def test_parse_list_under_nested_heading() -> None:
    """连续列表应作为一个整体，并保留完整标题路径。"""

    content = (
        "# RAG 基础\n\n"
        "## 检索方式\n\n"
        "- 向量检索\n"
        "- 关键词检索\n"
    )

    blocks = parse_markdown_blocks(content)

    # 标题不单独产生块，因此这里只有一个列表块。
    assert len(blocks) == 1

    # 验证连续的两个列表项没有被拆开。
    assert blocks[0].block_type == "list"
    assert blocks[0].content == "- 向量检索\n- 关键词检索"

    # 验证一级、二级标题共同组成路径。
    assert blocks[0].section_path == "RAG 基础 > 检索方式"

def test_parse_code_block_as_one_complete_block() -> None:
    """代码块内部即使包含 # 等字符，也必须作为一个整体保留。"""

    content = (
        "# 示例代码\n\n"
        "```python\n"
        "def retrieve(query: str) -> list[str]:\n"
        "    # 这里的 # 是 Python 注释，不是 Markdown 标题。\n"
        "    return []\n"
        "```\n"
    )

    blocks = parse_markdown_blocks(content)

    # 代码块应独立成为一个结构块。
    assert len(blocks) == 1
    assert blocks[0].block_type == "code"

    # 起止围栏和内部所有代码必须完整存在。
    assert blocks[0].content == (
        "```python\n"
        "def retrieve(query: str) -> list[str]:\n"
        "    # 这里的 # 是 Python 注释，不是 Markdown 标题。\n"
        "    return []\n"
        "```"
    )

    # 代码块也必须保留它所属的标题上下文。
    assert blocks[0].section_path == "示例代码"

def test_build_chunk_drafts_groups_same_section_and_counts_tokens() -> None:
    """同一标题路径的短结构块应合并，并记录真实 token 数。"""

    # 手动构造三个已经解析好的结构块。
    # 前两个属于同一章节，第三个属于另一个章节。
    blocks = [
        StructureBlock(
            block_type="paragraph",
            content="RAG 会先检索相关资料。",
            section_path="RAG 基础",
        ),
        StructureBlock(
            block_type="list",
            content="- 向量检索\n- 关键词检索",
            section_path="RAG 基础",
        ),
        StructureBlock(
            block_type="paragraph",
            content="RRF 可以融合不同检索结果。",
            section_path="混合检索",
        ),
    ]

    # 上限足够容纳前两个块，但标题路径变化必须产生新 chunk。
    drafts = build_chunk_drafts(
        blocks=blocks,
        max_tokens=100,
    )

    # 应得到两个最终检索 chunk。
    assert len(drafts) == 2

    # 第一个 chunk 合并同一章节内的段落和列表。
    assert drafts[0].content == (
        "RAG 会先检索相关资料。\n\n"
        "- 向量检索\n- 关键词检索"
    )
    assert drafts[0].section_path == "RAG 基础"
    assert drafts[0].position == 0

    # token_count 必须与最终实际内容的 token 数完全一致。
    assert drafts[0].token_count == count_tokens(drafts[0].content)

    # 第二个标题路径应开启新的 chunk。
    assert drafts[1].content == "RRF 可以融合不同检索结果。"
    assert drafts[1].section_path == "混合检索"
    assert drafts[1].position == 1
    assert drafts[1].token_count == count_tokens(drafts[1].content)

def test_build_chunk_drafts_splits_an_oversized_paragraph_by_sentence() -> None:
    """超长普通段落应按句子分开，不能从句子中间截断。"""

    first_sentence = "RAG 会先从知识库中检索与问题相关的资料。"
    second_sentence = "大模型随后依据检索到的证据生成回答。"

    # 一个完整段落由两个句子组成。
    # 设定的上限能容纳任意一个句子，但容纳不下两个句子。
    paragraph = f"{first_sentence}{second_sentence}"
    max_tokens = max(
        count_tokens(first_sentence),
        count_tokens(second_sentence),
    )

    blocks = [
        StructureBlock(
            block_type="paragraph",
            content=paragraph,
            section_path="RAG 基础",
        )
    ]

    drafts = build_chunk_drafts(
        blocks=blocks,
        max_tokens=max_tokens,
    )

    # 原始段落过长，因此应被拆为两个 chunk。
    assert len(drafts) == 2

    # 每个 chunk 保留一个完整句子。
    assert drafts[0].content == first_sentence
    assert drafts[1].content == second_sentence

    # 两个 chunk 仍属于同一个标题路径。
    assert drafts[0].section_path == "RAG 基础"
    assert drafts[1].section_path == "RAG 基础"

    # 每个 chunk 的 token 数都不应超过预算。
    assert drafts[0].token_count <= max_tokens
    assert drafts[1].token_count <= max_tokens

def test_parse_ordered_list_as_one_structure_block() -> None:
    """连续有序列表应被识别为一个完整的 list 结构块。"""

    content = (
        "# RAG 的价值\n\n"
        "1. 可以补充模型训练后出现的新知识。\n"
        "2. 可以为回答提供可核验的证据。\n"
    )

    blocks = parse_markdown_blocks(content)

    assert len(blocks) == 1
    assert blocks[0].block_type == "list"
    assert blocks[0].content == (
        "1. 可以补充模型训练后出现的新知识。\n"
        "2. 可以为回答提供可核验的证据。"
    )
    assert blocks[0].section_path == "RAG 的价值"

def test_parse_markdown_blocks_skips_leading_metadata_block() -> None:
    """文档标题后的字段型引用元数据不应成为可检索结构块。"""

    content = (
        "# RAG 学习资料\n\n"
        "> 类型：学习笔记\n"
        "> 更新时间：2026-09-11\n\n"
        "RAG 通过检索外部知识辅助生成回答。"
    )

    blocks = parse_markdown_blocks(content)

    # 顶部两个元数据引用行应被排除，只保留真实正文。
    assert len(blocks) == 1
    assert blocks[0].content == "RAG 通过检索外部知识辅助生成回答。"
    assert blocks[0].section_path == "RAG 学习资料"