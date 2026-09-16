from app.cleaning import (  # 导入清洗和质量检查函数。
    clean_markdown,
    validate_cleaned_content,
)

def test_clean_markdown_normalizes_text_and_removes_html_noise() -> None:
    """清洗应规范文本，并移除正文中的 script 和 style 噪声。"""

    content = (
        "\ufeff# RAG 笔记\r\n\r\n"
        "<script>alert('tracking')</script>\r\n\r\n\r\n"
        "⽤⼾使用混合检索会融合多个检索器的结果。  \r\n\r\n"
        "<style>.hidden { display: none; }</style>\r\n"
    )

    cleaned = clean_markdown(content)

    # BOM、HTML 脚本和样式不应出现在清洗结果中。
    assert "\ufeff" not in cleaned
    assert "<script>" not in cleaned
    assert "<style>" not in cleaned

    # 换行应统一为 \n，行尾空格和多余空行应被清理。
    assert cleaned == (
        "# RAG 笔记\n\n"
        "用戶使用混合检索会融合多个检索器的结果。"
    )


def test_clean_markdown_preserves_script_example_inside_code_fence() -> None:
    """代码块中的 script 示例属于技术内容，不能被清洗器删除。"""

    content = (
        "# HTML 示例\n\n"
        "```html\n"
        "<script>\n"
        "console.log('example');\n"
        "</script>\n"
        "```\n\n"
        "上面是一个 JavaScript 示例。"
    )

    cleaned = clean_markdown(content)

    # 代码块和其中的 script 内容必须完整保留。
    assert "```html" in cleaned
    assert "<script>" in cleaned
    assert "console.log('example');" in cleaned
    assert "</script>" in cleaned

def test_validate_cleaned_content_rejects_empty_content() -> None:
    """清洗后只有空白的内容不应进入索引流程。"""

    result = validate_cleaned_content(" \n\t  ")

    assert result.is_valid is False
    assert result.reason == "Document has no usable content after cleaning."
    assert result.character_count == 0


def test_validate_cleaned_content_rejects_short_content() -> None:
    """清洗后过短的内容不应进入索引流程。"""

    result = validate_cleaned_content(
        "混合检索。",
        min_characters=30,
    )

    assert result.is_valid is False
    assert result.character_count < 30
    assert result.reason is not None
    assert "too short" in result.reason


def test_validate_cleaned_content_accepts_sufficient_content() -> None:
    """具备足够正文长度的内容应通过质量检查。"""

    content = "混合检索会融合向量检索和关键词检索，提升不同查询场景下的召回稳定性。"

    result = validate_cleaned_content(content)

    assert result.is_valid is True
    assert result.reason is None
    assert result.character_count >= 30