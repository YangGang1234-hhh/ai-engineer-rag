"""父子块切分：父块保证回答上下文，子块保证检索精度。"""

from dataclasses import dataclass

from app.chunking import count_tokens, split_paragraph_into_sentences


@dataclass
class ParentChildDraft:
    """一个父块及其用于检索的子块文本。"""

    parent_content: str
    child_contents: list[str]


def build_parent_child_drafts(
    content: str,
    parent_max_tokens: int = 500,
    child_max_tokens: int = 220,
    child_overlap_tokens: int = 40,
) -> list[ParentChildDraft]:
    """按完整句子构建父块，并为每个父块生成带 overlap 的子块。"""

    if parent_max_tokens <= 0 or child_max_tokens <= 0:
        raise ValueError("token limits must be greater than zero")

    if child_overlap_tokens < 0:
        raise ValueError("child_overlap_tokens must not be negative")

    sentences = split_paragraph_into_sentences(content)
    if not sentences:
        return []

    parent_units = _pack_sentences(sentences, parent_max_tokens)

    return [
        ParentChildDraft(
            parent_content="\n".join(units),
            child_contents=[
                "\n".join(child_units)
                for child_units in _window_sentences(
                    units,
                    child_max_tokens,
                    child_overlap_tokens,
                )
            ],
        )
        for units in parent_units
    ]


def build_child_contents(
    content: str,
    child_max_tokens: int = 220,
    child_overlap_tokens: int = 40,
) -> list[str]:
    """从一个完整父块生成供检索使用的子块。"""

    sentences = split_paragraph_into_sentences(content)
    if not sentences:
        return []

    return [
        "\n".join(window)
        for window in _window_sentences(
            sentences,
            child_max_tokens,
            child_overlap_tokens,
        )
    ]


def _pack_sentences(sentences: list[str], max_tokens: int) -> list[list[str]]:
    """将完整句子组合到 token 上限内，超长单句保持完整。"""

    groups: list[list[str]] = []
    current: list[str] = []

    for sentence in sentences:
        candidate = [*current, sentence]
        if current and count_tokens("\n".join(candidate)) > max_tokens:
            groups.append(current)
            current = [sentence]
        else:
            current = candidate

    if current:
        groups.append(current)

    return groups


def _window_sentences(
    sentences: list[str],
    max_tokens: int,
    overlap_tokens: int,
) -> list[list[str]]:
    """按句子滑动窗口切分，并让相邻窗口保留完整句子 overlap。"""

    windows: list[list[str]] = []
    start = 0

    while start < len(sentences):
        end = start
        current: list[str] = []

        while end < len(sentences):
            candidate = [*current, sentences[end]]
            if current and count_tokens("\n".join(candidate)) > max_tokens:
                break
            current = candidate
            end += 1

        windows.append(current)

        if end >= len(sentences):
            break

        overlap_start = end
        overlap: list[str] = []
        while overlap_start > start:
            candidate = [sentences[overlap_start - 1], *overlap]
            if overlap and count_tokens("\n".join(candidate)) > overlap_tokens:
                break
            overlap = candidate
            overlap_start -= 1

        # 至少前进一个句子，避免单个超长句导致死循环。
        start = max(end - len(overlap), start + 1)

    return windows
