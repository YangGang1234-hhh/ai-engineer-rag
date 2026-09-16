import re  # 使用正则表达式处理 HTML 噪声和连续空行。
from dataclasses import dataclass  # 用于定义质量检查结果。
import unicodedata  # 统一 PDF 常见的 Unicode 兼容字符。
# 匹配 HTML 中的 script 块及其全部内容。
SCRIPT_BLOCK_PATTERN = re.compile(
    r"<script\b[^>]*>.*?</script\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)

# 匹配 HTML 中的 style 块及其全部内容。
STYLE_BLOCK_PATTERN = re.compile(
    r"<style\b[^>]*>.*?</style\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)

# 三个及以上连续换行压缩为两个，保留 Markdown 段落边界。
EXCESSIVE_BLANK_LINES_PATTERN = re.compile(r"\n{3,}")

# 匹配围栏代码块的开始行，例如 ```python 或 ~~~html。
FENCE_OPEN_PATTERN = re.compile(r"^\s*(`{3,}|~{3,})")

# 匹配围栏代码块的结束行，例如 ``` 或 ~~~。
FENCE_CLOSE_PATTERN = re.compile(r"^\s*(`{3,}|~{3,})\s*$")

def clean_plain_text(text: str) -> str:
    """清洗不在围栏代码块中的普通 Markdown 文本。"""

     # NFKC 会将 PDF 字体映射产生的兼容字符规范为普通 Unicode。
    # 例如“⽤⼾”会变为“用户”，提升关键词和向量检索的一致性。
    cleaned = unicodedata.normalize("NFKC", text)

    # 删除网页导出时残留的脚本和样式块。
    cleaned = SCRIPT_BLOCK_PATTERN.sub("", cleaned)
    cleaned = STYLE_BLOCK_PATTERN.sub("", cleaned)

    # 移除普通文本每行结尾的空白。
    cleaned = "\n".join(
        line.rstrip()
        for line in cleaned.split("\n")
    )

    # 多个空行只保留一个 Markdown 段落边界。
    cleaned = EXCESSIVE_BLANK_LINES_PATTERN.sub("\n\n", cleaned)

    return cleaned.strip()


def clean_markdown(content: str) -> str:
    """保守清洗 Markdown，并完整保留围栏代码块。"""

    # 去掉 UTF-8 BOM，并将不同系统的换行统一为 \n。
    normalized = content.removeprefix("\ufeff")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # 最终内容由“已清洗的普通文本”和“原样保留的代码块”组成。
    parts: list[str] = []
    normal_lines: list[str] = []
    code_lines: list[str] = []

    in_code_block = False  # 标记当前是否位于围栏代码块中。
    fence_character = ""  # 记录代码围栏使用的是 ` 还是 ~。
    fence_length = 0  # 记录开始围栏至少有几个符号。

    def flush_normal_text() -> None:
        """清洗并保存当前累计的普通文本。"""

        nonlocal normal_lines

        cleaned_text = clean_plain_text("\n".join(normal_lines))

        if cleaned_text:
            parts.append(cleaned_text)

        normal_lines = []

    for line in normalized.split("\n"):
        if not in_code_block:
            opening_fence = FENCE_OPEN_PATTERN.match(line)

            if opening_fence is None:
                normal_lines.append(line)
                continue

            # 遇到代码块开始：先处理它之前的普通文本。
            flush_normal_text()

            marker = opening_fence.group(1)
            in_code_block = True
            fence_character = marker[0]
            fence_length = len(marker)
            code_lines = [line]
            continue

        # 代码块内所有内容都原样保存，包括 script/style 示例和行尾空格。
        code_lines.append(line)

        closing_fence = FENCE_CLOSE_PATTERN.match(line)

        # 结束围栏必须使用同一种符号，且长度不能短于开始围栏。
        if (
            closing_fence is not None
            and closing_fence.group(1)[0] == fence_character
            and len(closing_fence.group(1)) >= fence_length
        ):
            parts.append("\n".join(code_lines))
            code_lines = []
            in_code_block = False
            fence_character = ""
            fence_length = 0

    # 文件末尾如果仍有普通文本，继续清洗。
    if normal_lines:
        flush_normal_text()

    # 未闭合的代码块也保留原样，避免清洗器意外丢失技术内容。
    if code_lines:
        parts.append("\n".join(code_lines))

    # 不同部分之间用一个 Markdown 段落边界连接。
    return "\n\n".join(parts).strip()

@dataclass
class ContentQualityResult:
    """清洗后文档的基础质量检查结果。"""

    is_valid: bool  # 内容是否可以继续进入切分和索引。
    reason: str | None  # 不可用时的原因；可用时为 None.
    character_count: int  # 清洗后可见文本的大致字符数。


def validate_cleaned_content(
    content: str,
    min_characters: int = 30,
) -> ContentQualityResult:
    """检查清洗后的文档是否具备基本检索价值。"""

    if min_characters <= 0:
        raise ValueError("min_characters must be greater than zero")

    # 去掉所有空白后再统计，避免空行被误认为有效内容。
    visible_content = re.sub(r"\s+", "", content)
    character_count = len(visible_content)

    if not visible_content:
        return ContentQualityResult(
            is_valid=False,
            reason="Document has no usable content after cleaning.",
            character_count=0,
        )

    if character_count < min_characters:
        return ContentQualityResult(
            is_valid=False,
            reason=(
                "Document content is too short after cleaning "
                f"({character_count} characters)."
            ),
            character_count=character_count,
        )

    return ContentQualityResult(
        is_valid=True,
        reason=None,
        character_count=character_count,
    )