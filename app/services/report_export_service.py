"""Phase 29D export renderers: PDF (reportlab), Excel (XlsxWriter), CSV.

Security properties:
* CSV/Excel string cells are guarded against spreadsheet formula injection
  (values starting with = + - @, or tab/CR) without converting genuine numeric
  cells to text.
* Files are written to a private temporary directory (never ``static/``), with
  a safe generated filename; callers stream then delete.
* No caller-supplied path or filename is ever accepted.
* No secret, credential, absolute path or stack detail is written into a file.
"""
from __future__ import annotations

import csv
import io
import re
import tempfile
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.services.report_data_service import REPORTS

# Visual language kept consistent with the Phase 29A palette.
TEAL = colors.HexColor("#0d9488")
NAVY = colors.HexColor("#0f172a")
INK2 = colors.HexColor("#475569")
BORDER = colors.HexColor("#e2e8f0")
SOFT = colors.HexColor("#f8fafc")
RISK_FILL = {
    "Very High": colors.HexColor("#fee2e2"),
    "High": colors.HexColor("#ffedd5"),
    "Moderate": colors.HexColor("#fef3c7"),
    "Low": colors.HexColor("#dcfce7"),
    "No data": colors.HexColor("#f1f5f9"),
}

_FORMULA_PREFIX = ("=", "+", "-", "@")
_UNSAFE_LEAD = ("\t", "\r", "\n")
_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_cell(value: Any) -> Any:
    """Neutralise spreadsheet formula injection for STRING cells only.

    Genuine numeric/bool/None cells are returned untouched so they stay numeric
    in Excel and CSV.
    """
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    if text[:1] in _FORMULA_PREFIX or text[:1] in _UNSAFE_LEAD:
        return "'" + text
    return text


def safe_filename(*parts: Any, extension: str) -> str:
    """Deterministic, sanitised filename. Rejects separators and traversal."""
    cleaned = []
    for part in parts:
        if part in (None, ""):
            continue
        token = _FILENAME_SAFE.sub("_", str(part))
        # Collapse dot runs so no traversal-looking sequence survives.
        token = re.sub(r"\.{2,}", "_", token).strip("._")
        if token:
            cleaned.append(token)
    stem = "_".join(cleaned) or "report"
    stem = stem[:110]
    ext = _FILENAME_SAFE.sub("", extension).lower()[:5]
    return f"{stem}.{ext}"


def temp_path(extension: str) -> Path:
    """Private temp file outside the project's static directory."""
    handle = tempfile.NamedTemporaryFile(prefix="mews_report_", suffix=f".{extension}", delete=False)
    handle.close()
    return Path(handle.name)


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------
def render_csv(headers: list[str], rows: list[list[Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow([sanitize_cell(h) for h in headers])
    for row in rows:
        writer.writerow([sanitize_cell(cell) for cell in row])
    return buffer.getvalue().encode("utf-8-sig")  # BOM so Excel opens UTF-8 correctly


# --------------------------------------------------------------------------
# Excel
# --------------------------------------------------------------------------
def render_xlsx(sheets: list[dict[str, Any]], path: Path) -> None:
    """Write a workbook. Each sheet: {name, headers, rows, widths?, percent_cols?}.

    No macros, no external links, no formulas. Numeric cells stay numeric.
    """
    import xlsxwriter

    book = xlsxwriter.Workbook(str(path), {"in_memory": False, "strings_to_formulas": False,
                                           "strings_to_urls": False})
    header_fmt = book.add_format({"bold": True, "bg_color": "#0e4d4d", "font_color": "#FFFFFF",
                                  "border": 1, "text_wrap": True, "valign": "vcenter"})
    text_fmt = book.add_format({"border": 1})
    num_fmt = book.add_format({"border": 1, "num_format": "#,##0.####"})
    pct_fmt = book.add_format({"border": 1, "num_format": "0.0\\%"})
    title_fmt = book.add_format({"bold": True, "font_size": 12, "font_color": "#0f172a"})

    for sheet in sheets:
        name = _FILENAME_SAFE.sub(" ", str(sheet["name"]))[:31] or "Sheet"
        ws = book.add_worksheet(name)
        headers = sheet.get("headers") or []
        rows = sheet.get("rows") or []
        percent_cols = set(sheet.get("percent_cols") or [])
        note = sheet.get("note")

        start = 0
        if note:
            ws.write(0, 0, sanitize_cell(note), title_fmt)
            start = 2

        if headers:
            for col, head in enumerate(headers):
                ws.write(start, col, sanitize_cell(head), header_fmt)
            for r, row in enumerate(rows, start=start + 1):
                for c, cell in enumerate(row):
                    value = sanitize_cell(cell)
                    if isinstance(value, bool):
                        ws.write_boolean(r, c, value, text_fmt)
                    elif isinstance(value, (int, float)):
                        ws.write_number(r, c, value, pct_fmt if c in percent_cols else num_fmt)
                    elif value is None:
                        ws.write_blank(r, c, None, text_fmt)
                    else:
                        ws.write_string(r, c, value, text_fmt)
            ws.freeze_panes(start + 1, 0)
            if rows:
                ws.autofilter(start, 0, start + len(rows), max(0, len(headers) - 1))
            for col, head in enumerate(headers):
                longest = max([len(str(head))] + [len(str(r[col])) for r in rows if col < len(r)][:400] or [10])
                ws.set_column(col, col, min(max(longest + 3, 11), 48))
        else:
            # Key/value sheet.
            for r, row in enumerate(rows, start=start):
                for c, cell in enumerate(row):
                    value = sanitize_cell(cell)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        ws.write_number(r, c, value, num_fmt)
                    else:
                        ws.write(r, c, value if value is not None else "", text_fmt)
            ws.set_column(0, 0, 42)
            ws.set_column(1, 1, 60)
    book.close()


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------
def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=base["Title"], fontSize=16, textColor=NAVY,
                                spaceAfter=4, alignment=TA_LEFT),
        "sub": ParagraphStyle("s", parent=base["Normal"], fontSize=9, textColor=INK2, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=11.5, textColor=NAVY,
                             spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=8.8, leading=12.6, textColor=NAVY),
        "small": ParagraphStyle("sm", parent=base["Normal"], fontSize=7.6, leading=10.4, textColor=INK2),
        "cell": ParagraphStyle("c", parent=base["Normal"], fontSize=7.6, leading=9.6),
    }


def _page_furniture(meta: dict[str, Any]):
    title = meta.get("report_title", "Report")
    issue = meta.get("issue_label") or ""
    generated = meta.get("generated_at_utc") or ""

    def draw(canvas, doc):
        canvas.saveState()
        width, height = doc.pagesize
        # Header rule + identity
        canvas.setFillColor(NAVY)
        canvas.setFont("Helvetica-Bold", 8.5)
        canvas.drawString(15 * mm, height - 12 * mm, "Republic of Uganda — Ministry of Health")
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(INK2)
        canvas.drawRightString(width - 15 * mm, height - 12 * mm,
                               f"{title} · Issue {issue}")
        canvas.setStrokeColor(TEAL)
        canvas.setLineWidth(1.1)
        canvas.line(15 * mm, height - 14 * mm, width - 15 * mm, height - 14 * mm)
        # Footer
        canvas.setStrokeColor(BORDER)
        canvas.setLineWidth(0.5)
        canvas.line(15 * mm, 13 * mm, width - 15 * mm, 13 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(INK2)
        canvas.drawString(15 * mm, 9 * mm, f"Generated {generated} · Research proof-of-concept")
        canvas.drawRightString(width - 15 * mm, 9 * mm, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    return draw


def _kv_table(pairs, styles, widths=(58 * mm, 105 * mm)):
    data = [[Paragraph(f"<b>{k}</b>", styles["cell"]), Paragraph(str(v), styles["cell"])]
            for k, v in pairs if v not in (None, "")]
    if not data:
        return Spacer(1, 1)
    table = Table(data, colWidths=list(widths), hAlign="LEFT")
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("BACKGROUND", (0, 0), (0, -1), SOFT),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _data_table(headers, rows, styles, col_widths=None, risk_col=None):
    """LongTable with repeating header row so it never clips across pages."""
    head = [Paragraph(f"<b>{h}</b>", styles["cell"]) for h in headers]
    body = [[Paragraph(str(c) if c is not None else "—", styles["cell"]) for c in row] for row in rows]
    table = LongTable([head] + body, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0e4d4d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SOFT]),
    ]
    if risk_col is not None:
        for index, row in enumerate(rows, start=1):
            fill = RISK_FILL.get(str(row[risk_col]))
            if fill:
                style.append(("BACKGROUND", (risk_col, index), (risk_col, index), fill))
    table.setStyle(TableStyle(style))
    return table


def render_pdf(blocks, meta: dict[str, Any], path: Path, orientation: str = "portrait") -> None:
    """Render a flowable block list to PDF with repeating headers and page numbers."""
    pagesize = landscape(A4) if orientation == "landscape" else A4
    doc = SimpleDocTemplate(
        str(path), pagesize=pagesize,
        leftMargin=15 * mm, rightMargin=15 * mm, topMargin=20 * mm, bottomMargin=18 * mm,
        title=str(meta.get("report_title", "Report")), author="Ministry of Health, Uganda",
        subject="Malaria early warning research output",
    )
    furniture = _page_furniture(meta)
    doc.build(list(blocks), onFirstPage=furniture, onLaterPages=furniture)


def content_type(fmt: str) -> str:
    # Bare types: Flask appends the charset itself, so declaring it here too
    # would emit a duplicated "charset=utf-8; charset=utf-8" header.
    return {
        "pdf": "application/pdf",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "csv": "text/csv",
        "html": "text/html",
    }[fmt]


__all__ = [
    "sanitize_cell", "safe_filename", "temp_path", "render_csv", "render_xlsx",
    "render_pdf", "content_type", "_styles", "_kv_table", "_data_table",
    "Paragraph", "Spacer", "PageBreak", "KeepTogether", "REPORTS",
]
