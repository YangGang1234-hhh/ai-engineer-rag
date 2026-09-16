from dataclasses import dataclass
from pathlib import Path

import pdfplumber 

@dataclass 
class ExtractedPdfPage:
    """PDF 中一页提取出的普通文本。"""

    page_number:int
    text:str

@dataclass
class ExtractedPdfTable:
    """PDF 中一张独立提取出来的表格。"""

    page_number:int
    table_index:int
    rows:list[list[str]]

    @property
    def markdown_content(self) -> str:
        """返回供后续存储和检索使用的 Markdown 表格。"""

        return table_rows_to_markdown(self.rows)

@dataclass
class PdfExtractionResult:
    """一次 PDF 提取的完整结果。"""

    pages:list[ExtractedPdfPage]
    tables:list[ExtractedPdfTable]

    @property
    def text_content(self) -> str:
        """保留页码边界地合并正文，供 PDF 专用切分与引用使用。"""

        return "\n\n".join(
            f"# PDF 第 {page.page_number} 页\n\n{page.text}"
            for page in self.pages
            if page.text
        )

def table_rows_to_markdown(rows: list[list[str]]) -> str:
    """将 PDF 提取出的二维表格转换为 Markdown 表格。"""

    # 过滤完全为空的行，避免写入没有检索价值的空表格。
    non_empty_rows = [
        row
        for row in rows
        if any(cell.strip() for cell in row)
    ]

    if not non_empty_rows:
        return ""

    # 同一张 PDF 表格偶尔会出现缺失单元格；
    # 统一补齐到最大列数，保证 Markdown 每行列数一致。
    column_count = max(
        len(row)
        for row in non_empty_rows
    )

    normalized_rows = [
        row + [""] * (column_count - len(row))
        for row in non_empty_rows
    ]

    def format_cell(cell: str) -> str:
        """将单元格内容转成安全的 Markdown 文本。"""

        # 表格单元格内换行改为 HTML 换行，竖线需要转义，
        # 否则会破坏 Markdown 表格的列边界。
        return (
            cell.strip()
            .replace("\n", "<br>")
            .replace("|", r"\|")
        )

    # 第一行作为表头；PDF 本身没有统一的表头标记，
    # 这里采用 RAG 中常见的“首行即表头”约定。
    header = normalized_rows[0]
    body_rows = normalized_rows[1:]

    lines = [
        "| " + " | ".join(format_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in range(column_count)) + " |",
    ]

    lines.extend(
        "| " + " | ".join(
            format_cell(cell)
            for cell in row
        ) + " |"
        for row in body_rows
    )

    return "\n".join(lines)


def extract_pdf_content(file_path:Path) -> PdfExtractionResult:
    """逐页提取PDF正文和表格结构。"""

    if not file_path.is_file():
        raise FileNotFoundError(
            f"PDF file not found:{file_path}"
        )

    pages:list[ExtractedPdfPage] = []
    tables:list[ExtractedPdfTable] = []

    # pdfplumber 会在with 结束时关闭 PDF 文件。
    with pdfplumber.open(file_path) as pdf:
        for page_number,page in enumerate(pdf.pages,start=1):
            # 提取页面可复制的普通文本；没有文字时返回空字符串。
            # 先识别该页所有表格。后面会从普通正文中排除这些表格区域，
            # 防止同一份表格内容同时进入正文和独立表格记录。
            page_tables = page.find_tables()

            def is_outside_table_area(
                item: dict,
            ) -> bool:
                """判断一个 PDF 对象是否位于任意表格边界之外。"""

                # 这里只过滤文字字符；其他页面对象保留给 pdfplumber。
                if item.get("object_type") != "char":
                    return True

                return not any(
                    (
                        table.bbox[0] <= item["x0"]
                        and item["x1"] <= table.bbox[2]
                        and table.bbox[1] <= item["top"]
                        and item["bottom"] <= table.bbox[3]
                    )
                    for table in page_tables
                )

            # 普通正文只保留表格区域以外的可复制文字。
            page_text = (
                page.filter(is_outside_table_area)
                .extract_text()
                or ""
            ).strip()

            pages.append(
                ExtractedPdfPage(
                    page_number=page_number,
                    text=page_text,
                )
            )

            # page_tables 与表格区域排除使用同一份识别结果。
            for table_index, table in enumerate(
                page_tables,
                start=1,
            ):
                # extract() 返回二维单元表格数据；None 表示空表格。
                rows = [
                    [
                        (cell or "").strip()
                        for cell in row
                    ]
                    for row in table.extract()
                ]

                # 忽略完全没有内容的空表各。
                if not any(
                    any(cell for cell in row)
                    for row in rows
                ):
                    continue

                tables.append(
                    ExtractedPdfTable(
                        page_number= page_number,
                        table_index=table_index,
                        rows=rows,
                    )
                )
    return PdfExtractionResult(
        pages=pages,
        tables=tables,
    )
