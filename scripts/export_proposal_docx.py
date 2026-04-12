#!/usr/bin/env python3
"""Export the full proposal markdown into a DOCX without fragile HTML import."""

from __future__ import annotations

import argparse
import html
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "proposal" / "proposal-final.md"
DEFAULT_OUTPUT = ROOT / "proposal" / "proposal-final.docx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render proposal markdown to DOCX.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input markdown file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output DOCX path.")
    return parser.parse_args()


def paragraph(text: str, *, style: str | None = None) -> str:
    style_xml = f'<w:pStyle w:val="{style}"/>' if style else ""
    return (
        f'<w:p><w:pPr>{style_xml}</w:pPr>'
        f'<w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'
    )


def blank() -> str:
    return "<w:p/>"


def list_paragraph(text: str, level: int = 0) -> str:
    left = 720 + level * 360
    return (
        "<w:p><w:pPr>"
        f'<w:ind w:left="{left}" w:hanging="360"/>'
        "</w:pPr>"
        f'<w:r><w:t xml:space="preserve">• {escape(text)}</w:t></w:r></w:p>'
    )


def table(rows: list[list[str]]) -> str:
    out = ['<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/></w:tblPr>']
    for row in rows:
        out.append("<w:tr>")
        for cell in row:
            out.append(
                "<w:tc><w:p>"
                f'<w:r><w:t xml:space="preserve">{escape(cell)}</w:t></w:r>'
                "</w:p></w:tc>"
            )
        out.append("</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def image_note(path: str) -> str:
    return paragraph(f"[Gambar: {path}]", style="IntenseQuote")


def markdown_to_docx_body(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    parts: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            parts.append(blank())
            i += 1
            continue

        if stripped.startswith("# "):
            parts.append(paragraph(stripped[2:], style="Title"))
            i += 1
            continue
        if stripped.startswith("## "):
            parts.append(paragraph(stripped[3:], style="Heading1"))
            i += 1
            continue
        if stripped.startswith("### "):
            parts.append(paragraph(stripped[4:], style="Heading2"))
            i += 1
            continue
        if stripped.startswith("#### "):
            parts.append(paragraph(stripped[5:], style="Heading3"))
            i += 1
            continue

        img_match = re.match(r"!\[[^\]]*\]\(([^)]+)\)", stripped)
        if img_match:
            parts.append(image_note(img_match.group(1)))
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = []
            for idx, row in enumerate(table_lines):
                cells = [c.strip() for c in row.strip("|").split("|")]
                if idx == 1 and all(set(c) <= {"-", ":"} for c in cells):
                    continue
                rows.append(cells)
            if rows:
                parts.append(table(rows))
            continue

        bullet_match = re.match(r"^[-*]\s+(.*)$", stripped)
        if bullet_match:
            parts.append(list_paragraph(bullet_match.group(1)))
            i += 1
            continue

        number_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if number_match:
            parts.append(list_paragraph(number_match.group(1)))
            i += 1
            continue

        if stripped == "---":
            parts.append(blank())
            i += 1
            continue

        parts.append(paragraph(stripped))
        i += 1

    return "".join(parts)


def build_document_xml(markdown_text: str) -> str:
    body = markdown_to_docx_body(markdown_text)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>
"""


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>
"""


RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""


STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="Heading 1"/></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="Heading 2"/></w:style>
  <w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="Heading 3"/></w:style>
  <w:style w:type="paragraph" w:styleId="IntenseQuote"><w:name w:val="Intense Quote"/></w:style>
</w:styles>
"""


def render_docx(input_md: Path, output_docx: Path) -> None:
    markdown_text = input_md.read_text(encoding="utf-8")
    document_xml = build_document_xml(markdown_text)
    output_docx.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_docx, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", CONTENT_TYPES)
        docx.writestr("_rels/.rels", RELS)
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/styles.xml", STYLES)


def main() -> int:
    args = parse_args()
    input_md = args.input.resolve()
    output_docx = args.output.resolve()
    if not input_md.exists():
        raise FileNotFoundError(f"Input markdown not found: {input_md}")
    render_docx(input_md, output_docx)
    print(f"Rendered proposal DOCX: {output_docx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
