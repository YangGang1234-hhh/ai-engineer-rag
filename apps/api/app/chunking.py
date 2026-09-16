from dataclasses import dataclass
import tiktoken 
import re 

ENCODING = tiktoken.get_encoding("cl100k_base")
# 匹配Markdown 标题 
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
# 匹配 Markdown 引用行，例如：> 类型：学习笔记
BLOCKQUOTE_PATTERN = re.compile(r"^\s*>\s?")

# 匹配元数据形式的引用行，例如：> 类型：学习笔记、> 更新时间：2026-09-08
METADATA_FIELD_PATTERN = re.compile(r"^\s*>\s*[^:：]+[:：]\s*.+$")

@dataclass 
class StructureBlock:
    """Markdown 解析后的最小语义结构块，尚未组成最终的检索 chunk."""

    # 块类型：paragraph、list、code、table
    block_type:str

    # 结构块的原始内容
    content:str

    # 此块所属的Markdown 标题路径
    section_path:str

# 匹配markdown 无序列表和有序列表项
LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")


# 匹配markdown 围栏代码块的开始或结束标记
FENCE_PATTERN = re.compile(r"^\s*(`{3,}|~{3,})")

def strip_leading_metadata_block(content: str) -> str:
    """移除文档标题之后的顶部元数据块，避免它参与检索。"""

    lines = content.splitlines()  # 按行处理，便于识别 Markdown 结构。

    # 找到文档第一个非空行。
    first_content_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip()
        ),
        None,
    )

    # 空文档，或第一个内容不是一级标题时，不做任何处理。
    if first_content_index is None:
        return content

    first_heading = HEADING_PATTERN.match(lines[first_content_index])
    if first_heading is None or len(first_heading.group(1)) != 1:
        return content

    # 从一级标题之后开始，跳过标题和元数据之间的空行。
    index = first_content_index + 1
    while index < len(lines) and not lines[index].strip():
        index += 1

    metadata_start = index  # 记录元数据开始位置，后续用于删除。

    # 连续收集引用行；非引用行意味着元数据块结束。
    metadata_lines: list[str] = []
    while index < len(lines) and BLOCKQUOTE_PATTERN.match(lines[index]):
        metadata_lines.append(lines[index])
        index += 1

    # 至少两行“字段：值”形式的引用，才认定为元数据。
    # 这样不会误删正文中的单条普通引用。
    metadata_field_count = sum(
        bool(METADATA_FIELD_PATTERN.match(line))
        for line in metadata_lines
    )
    if metadata_field_count < 2:
        return content

    # 保留标题、后续正文；仅移除顶部元数据引用行。
    return "\n".join([
        *lines[:metadata_start],
        *lines[index:],
    ])


def is_table_line(line:str) -> bool:
    """判断一行是否看起来像markdown 表格的一部分。 """

    stripped_line = line.strip()

    # Markdown 表格行通常以竖线开始或结束。
    return (
        stripped_line.startswith("|")
        or stripped_line.endswith("|")
    )

def parse_markdown_blocks(content:str) -> list[StructureBlock]:
    """按标题层级解析Markdown,产出段落、列表、代码和表格结构块。"""
    # 原始文档仍完整保存在 SQLite；这里只排除不应参与检索的顶部元数据。
    content = strip_leading_metadata_block(content)
    
    # 保留标题层级
    heading_stack:list[str] = []

    # 最终解析的结构块
    blocks:list[StructureBlock] = []

    # 当前正在收集的结构块类型和文本行
    current_block_type:str|None = None
    current_lines: list[str] = []

    # 是否正在读取一个围栏代码块
    in_code_block = False

    def flush_current_block() -> None:
        """将当前累计的完整结构块写入blocks。"""
        nonlocal current_block_type,current_lines

        block_content = "\n".join(current_lines).strip()

        #空块不保存，例如连续空行产成的空内容
        if current_block_type is not None and block_content:
            blocks.append(
                StructureBlock(
                    block_type=current_block_type,
                    content=block_content,
                    #没有标题时，使用“未分类”作为兜底路径
                    section_path=" > ".join(heading_stack)or "未分类"
                )
            )
        # 清空状态，准备收集下一个结构块。
        current_block_type = None
        current_lines = []

    def start_block(block_type:str,line:str) -> None:
        """开始收集一个新的结构块。"""

        nonlocal current_block_type,current_lines

        current_block_type = block_type
        current_lines = [line]

    for line in content.splitlines():
        # 优先处理代码围栏
        if FENCE_PATTERN.match(line):
            if in_code_block:
                #遇到结束围栏，保留该行，再提交完整代码块
                current_lines.append(line)
                flush_current_block()
                in_code_block = False
            else:
                # 遇到开始围栏：先提交前一个块，再开始代码块
                flush_current_block()
                start_block("code",line)
                in_code_block=True

            continue 

        #代码块内的内容原样保留，不进行任何结构判断
        if in_code_block:
            current_lines.append(line)
            continue 


        #遇到Markdown 标题时，先结束当前块，再更新标题路径。
        heading_math = HEADING_PATTERN.match(line)
        if heading_math is not None:
            flush_current_block()

            heading_level = len(heading_math.group(1))
            heading_text = heading_math.group(2).strip()

            #只保留标题的父级路径
            heading_stack = heading_stack[:heading_level -1]
            heading_stack.append(heading_text)

            continue 

        # 空行是普通段落、列表、表格之间的自然边界。
        if not line.strip():
            flush_current_block()
            continue 

        # 判断当前所属的结构类型。
        if is_table_line(line):
            next_block_type = "table"
        elif LIST_ITEM_PATTERN.match(line):
            next_block_type = "list"
        else:
            next_block_type = "paragraph"

        # 没有正在收集的块时，直接开始
        if current_block_type is None:
            start_block(next_block_type,line)
            continue 

        # 相同类型的相邻行属于同一个结构块。
        if current_block_type == next_block_type:
            current_lines.append(line)
            continue

        # 列表项的缩进行通常是上一个列表项的补充说明，
        # 应继续放入当前列表块，而不是拆成独立段落。
        if current_block_type == "list" and line.startswith((" ", "\t")):
            current_lines.append(line)
            continue

        # 结构类型发生变化时，结束旧块并开始新块。
        flush_current_block()
        start_block(next_block_type, line)

    # 不要遗漏文档末尾尚未提交的块。
    flush_current_block()

    return blocks
# 中文和英文中常见的句末符号。
# 这个规则只用于普通文本段落，不用于代码、列表或表格。
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[。！？!?；;])\s*")


def split_paragraph_into_sentences(text: str) -> list[str]:
    """将普通段落按句子边界拆分，避免从句子中间截断。"""

    # 去掉首尾空白；空段落不应产生任何句子。
    text = text.strip()
    if not text:
        return []

    # 在句末标点之后切分。
    # 例如：
    # “RAG 需要检索。检索结果需要引用。”
    # 会得到两个完整句子。
    sentences = [
        sentence.strip()
        for sentence in SENTENCE_BOUNDARY_PATTERN.split(text)
        if sentence.strip()
    ]

    # 没有识别到句末标点时，例如一句很短的标题式文本，
    # 整段作为一个语义单元返回。
    return sentences or [text]
@dataclass
class ChunkDraft:
    """尚未保存到数据库的检索chunk."""

    #实际参与检索、生成embedding的文本内容
    content:str

    #标题路径
    section_path:str

    # 在原始文档中的顺序，从0开始
    position:int

    # 该chunk 的token数。
    token_count:int

def build_chunk_drafts(
    blocks: list[StructureBlock],
    max_tokens: int = 500,
) -> list[ChunkDraft]:
    """将结构块聚合为受 token 上限控制的最终检索 chunk。"""

    # token 上限必须为正数。
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0.")

    # 最终返回的检索 chunk。
    drafts: list[ChunkDraft] = []

    # 当前正在组合的 chunk 的内容单元。
    current_parts: list[str] = []

    # 当前 chunk 所属标题路径。
    current_section_path: str | None = None

    # 当前 chunk 已使用的 token 数。
    current_token_count = 0

    def flush_current_chunk() -> None:
        """将当前累积内容保存为一个 ChunkDraft。"""

        nonlocal current_parts, current_section_path, current_token_count

        # 没有内容时无需创建 chunk。
        if not current_parts or current_section_path is None:
            return

        # 不同结构块之间用空行分隔，保留原有阅读结构。
        content = "\n\n".join(current_parts)

        drafts.append(
            ChunkDraft(
                content=content,
                section_path=current_section_path,
                position=len(drafts),
                token_count=current_token_count,
            )
        )

        # 清空状态，开始组装下一个 chunk。
        current_parts = []
        current_section_path = None
        current_token_count = 0

    def get_semantic_units(block: StructureBlock) -> list[str]:
        """获取一个结构块中允许被独立组合的语义单元。"""

        block_token_count = count_tokens(block.content)

        # 完整结构块没有超出上限，直接作为一个整体。
        if block_token_count <= max_tokens:
            return [block.content]

        # 只有超长普通段落才按句子拆开。
        # 列表、代码块、表格宁可单独超出上限，也不在中间破坏结构。
        if block.block_type == "paragraph":
            return split_paragraph_into_sentences(block.content)

        return [block.content]

    for block in blocks:
        # 一个结构块可能因“超长普通段落”被拆成多个完整句子。
        for unit in get_semantic_units(block):
            unit_token_count = count_tokens(unit)

            # 标题路径不同，说明进入了新章节。
            # 先提交旧章节，避免一个 chunk 混入两个主题。
            if (
                current_section_path is not None
                and current_section_path != block.section_path
            ):
                flush_current_chunk()

            # 当前 chunk 为空时，直接将当前单元作为起点。
            if not current_parts:
                current_parts = [unit]
                current_section_path = block.section_path
                current_token_count = unit_token_count
                continue

            # BPE tokenizer 的 token 数不是简单可相加的。
            # 因此必须先按最终格式拼接文本，再重新计算真实 token 数。
            next_content = "\n\n".join([*current_parts, unit])
            next_token_count = count_tokens(next_content)

            # 不超过上限时，将完整语义单元加入当前 chunk。
            if next_token_count <= max_tokens:
                current_parts.append(unit)
                current_token_count = next_token_count
                continue

            # 放不下时，先提交已完成的 chunk，
            # 再让当前完整语义单元开启一个新的 chunk。
            flush_current_chunk()
            current_parts = [unit]
            current_section_path = block.section_path
            current_token_count = unit_token_count

    # 不要遗漏循环结束后仍在累积的最后一个 chunk。
    flush_current_chunk()

    return drafts


def count_tokens(text:str) -> int:
    """使用同一的tokenizer 计算一段文本的token数。"""

    return len(ENCODING.encode(text))