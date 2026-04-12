#!/usr/bin/env python3
"""Export the full proposal markdown into a DOCX document via LibreOffice."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from export_proposal_pdf import DEFAULT_SUBTITLE, DEFAULT_TITLE, build_html


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "proposal" / "proposal-final.md"
DEFAULT_OUTPUT = ROOT / "proposal" / "proposal-final.docx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render proposal markdown to DOCX.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input markdown file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output DOCX path.")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="Document title.")
    parser.add_argument("--subtitle", default=DEFAULT_SUBTITLE, help="Document subtitle.")
    return parser.parse_args()


def render_docx(input_md: Path, output_docx: Path, *, title: str, subtitle: str) -> None:
    html_content = build_html(input_md.resolve(), title=title, subtitle=subtitle)
    output_docx = output_docx.resolve()
    output_docx.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="proposal-docx-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        html_path = temp_dir_path / "proposal-final.html"
        odt_path = temp_dir_path / "proposal-final.odt"
        docx_path = temp_dir_path / "proposal-final.docx"
        html_path.write_text(html_content, encoding="utf-8")

        subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to",
                "odt",
                "--outdir",
                str(temp_dir_path),
                str(html_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        if not odt_path.exists():
            raise FileNotFoundError(f"ODT intermediate was not created: {odt_path}")

        subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to",
                "docx",
                "--outdir",
                str(temp_dir_path),
                str(odt_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        if not docx_path.exists():
            raise FileNotFoundError(f"DOCX output was not created: {docx_path}")

        output_docx.write_bytes(docx_path.read_bytes())


def main() -> int:
    args = parse_args()
    input_md = args.input.resolve()
    output_docx = args.output.resolve()
    if not input_md.exists():
        raise FileNotFoundError(f"Input markdown not found: {input_md}")

    render_docx(input_md, output_docx, title=args.title, subtitle=args.subtitle)
    print(f"Rendered proposal DOCX: {output_docx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
