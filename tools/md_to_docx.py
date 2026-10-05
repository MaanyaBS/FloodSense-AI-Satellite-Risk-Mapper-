"""Convert the IEEE paper draft from Markdown to a Word (.docx) file.

Usage:
    python tools/md_to_docx.py docs/paper_draft.md docs/paper_draft.docx

Formatting targets IEEE Access submission conventions:
    - Times New Roman 10pt body, 12pt title
    - two-column body layout, single-column title block
    - tables with a caption line above, in IEEE "TABLE I" style
    - justified body text, no first-line indent after a heading
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

BODY_FONT = "Times New Roman"
BODY_SIZE = Pt(10)
TITLE_SIZE = Pt(12)


# --------------------------------------------------------------------------- #
# low-level docx helpers
# --------------------------------------------------------------------------- #
def set_cell_borders(cell) -> None:
    """Apply the three horizontal rules IEEE uses under table captions."""
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge, size in (("top", 8), ("bottom", 8)):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(size))
        el.set(qn("w:color"), "000000")
        borders.append(el)
    tc_pr.append(borders)


def style_run(run, *, size=BODY_SIZE, bold=False, italic=False, mono=False) -> None:
    run.font.name = "Courier New" if mono else BODY_FONT
    run.font.size = Pt(8) if mono else size
    run.font.bold = bold
    run.font.italic = italic
    # east-asian font hint, otherwise Word substitutes a default on some systems
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), "Courier New" if mono else BODY_FONT)


SENTINEL = "\x01"


def add_inline(paragraph, text: str, *, base_size=BODY_SIZE) -> None:
    """Render markdown inline emphasis, strong, code and escaped characters."""
    tokens: list[tuple[str, bool]] = []  # (literal text, is_code)

    def _stash(value: str, is_code: bool) -> str:
        tokens.append((value, is_code))
        return f"{SENTINEL}{len(tokens) - 1}{SENTINEL}"

    # escape sequences first: an escaped \* must survive as a literal asterisk,
    # otherwise the corresponding-author marker is parsed as emphasis
    work = re.sub(r"\\([\\`*_{}\[\]()#+\-.!])", lambda m: _stash(m.group(1), False), text)
    # then inline code, whose contents must never be re-parsed
    work = re.sub(r"`([^`]+)`", lambda m: _stash(m.group(1), True), work)

    def emit(chunk: str, *, bold: bool = False, italic: bool = False) -> None:
        """Add one run per literal segment, honouring stashed code spans."""
        parts = re.split(f"{SENTINEL}(\\d+){SENTINEL}", chunk)
        for i, piece in enumerate(parts):
            if not piece:
                continue
            if i % 2 == 1:  # odd indices are stash indices
                value, is_code = tokens[int(piece)]
                run = paragraph.add_run(value)
                style_run(run, size=base_size, bold=bold, italic=italic, mono=is_code)
            else:
                run = paragraph.add_run(piece)
                style_run(run, size=base_size, bold=bold, italic=italic)

    pattern = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*)")
    for chunk in pattern.split(work):
        if not chunk:
            continue
        if chunk.startswith("**") and chunk.endswith("**"):
            emit(chunk[2:-2], bold=True)
        elif chunk.startswith("*") and chunk.endswith("*") and len(chunk) > 2:
            emit(chunk[1:-1], italic=True)
        else:
            emit(chunk)


def body_paragraph(doc: Document, text: str, *, justify: bool = True):
    p = doc.add_paragraph()
    fmt = p.paragraph_format
    fmt.space_after = Pt(0)
    fmt.space_before = Pt(0)
    fmt.line_spacing = 1.0
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        fmt.first_line_indent = Inches(0.2)
    add_inline(p, text)
    return p


def heading(doc: Document, text: str, level: int):
    p = doc.add_paragraph()
    fmt = p.paragraph_format
    fmt.space_before = Pt(10 if level == 1 else 6)
    fmt.space_after = Pt(2)
    fmt.keep_with_next = True
    # IEEE uses small caps / centred roman numerals for top-level sections
    if level == 1:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        style_run(run, size=Pt(10))
        run.font.small_caps = True
        run.font.bold = True
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(text)
        style_run(run, size=Pt(10))
        run.font.italic = True
    return p


def add_table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    ncols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=ncols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    for r_idx, row in enumerate(rows):
        for c_idx in range(ncols):
            cell = table.cell(r_idx, c_idx)
            cell.text = ""
            para = cell.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para.paragraph_format.space_after = Pt(0)
            raw = row[c_idx] if c_idx < len(row) else ""
            add_inline(para, raw)
            # bold the header row, as IEEE tables conventionally do
            for run in para.runs:
                if r_idx == 0:
                    run.font.bold = True
    # top rule above the header and bottom rule under the last row
    for c in range(ncols):
        set_cell_borders(table.cell(0, c))
        set_cell_borders(table.cell(len(rows) - 1, c))


def build(md_path: Path, docx_path: Path) -> None:
    raw = md_path.read_text(encoding="utf-8")

    # strip html comments: they are authoring notes, not manuscript content
    raw = re.sub(r"<!--[\s\S]*?-->", "", raw)

    doc = Document()

    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = BODY_SIZE

    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)

    lines = raw.split("\n")
    i = 0
    seen_title = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("```"):
            # code block: monospace, preserved verbatim
            block: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            for bline in block:
                p = doc.add_paragraph()
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.0
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                run = p.add_run(bline if bline.strip() else " ")
                style_run(run, mono=True)
            continue

        if stripped.startswith("|"):
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                # drop the |---|---| markdown alignment row
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            add_table(doc, rows)
            doc.add_paragraph().paragraph_format.space_after = Pt(4)
            continue

        if stripped.startswith("# "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run(stripped[2:])
            style_run(run, size=TITLE_SIZE)
            seen_title = True
            i += 1
            continue

        if stripped.startswith("### "):
            heading(doc, stripped[4:], level=2)
            i += 1
            continue

        if stripped.startswith("## "):
            heading(doc, stripped[3:], level=1)
            i += 1
            continue

        if stripped.startswith("> "):
            # footnote block: render once as a single-cell borderless table
            items: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                items.append(lines[i].strip().lstrip(">").strip())
                i += 1
            tbl = doc.add_table(rows=1, cols=1)
            cell = tbl.cell(0, 0)
            cell.text = ""
            first = True
            for it in items:
                if not it:
                    continue
                para = cell.paragraphs[0] if first else cell.add_paragraph()
                para.paragraph_format.space_after = Pt(3)
                para.paragraph_format.line_spacing = 1.0
                add_inline(para, it, base_size=Pt(8))
                first = False
            doc.add_paragraph().paragraph_format.space_after = Pt(4)
            continue

        if stripped.startswith("---"):
            i += 1
            continue

        if stripped.startswith("*") and stripped.endswith("*") and stripped.count("*") >= 2 \
                and not stripped.startswith("**"):
            # affiliation / caption line
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(4)
            add_inline(p, stripped.strip("*"), base_size=Pt(9))
            i += 1
            continue

        if stripped.startswith("**") and stripped.endswith("**"):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.keep_with_next = True
            add_inline(p, stripped, base_size=Pt(9))
            i += 1
            continue

        body_paragraph(doc, stripped)
        i += 1

    _apply_two_column_body(doc)

    doc.save(docx_path)


def _apply_two_column_body(doc: Document) -> None:
    """Keep title/abstract full width; switch to two columns at Section I.

    IEEE Access prints the title, author block, abstract and index terms across
    the full page width, and the body text in two columns. This inserts a
    continuous section break carrying the column definition on the last
    front-matter paragraph, so the body inherits two columns.
    """
    target = None
    for p in doc.paragraphs:
        if p.text.strip().upper().startswith("I. INTRODUCTION"):
            target = p
            break
    if target is None:
        return

    p_pr = target._p.get_or_add_pPr()
    sect = OxmlElement("w:sectPr")
    cols = OxmlElement("w:cols")
    cols.set(qn("w:num"), "2")
    cols.set(qn("w:space"), "288")   # 0.2 inch gutter
    cols.set(qn("w:sep"), "1")       # IEEE draws a rule between columns
    sect.append(cols)
    p_pr.append(sect)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    md = Path(sys.argv[1])
    out = Path(sys.argv[2])
    build(md, out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())