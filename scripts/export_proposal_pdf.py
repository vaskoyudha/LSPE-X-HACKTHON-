#!/usr/bin/env python3
"""Export the full proposal markdown into a reliable PDF.

This exporter intentionally uses a simpler HTML template because the previous
styled path truncated the document after early Bab 2 when printed to PDF.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "proposal" / "proposal-final.md"
DEFAULT_OUTPUT = ROOT / "proposal" / "proposal-final.pdf"
DEFAULT_TITLE = "Proposal LPSE-X — Find IT! 2026 Tahap 2"
DEFAULT_SUBTITLE = (
    "BismillahFirstTry-Phase2 · Track C · Single-model XGBoost · Offline Explainable Risk Screening"
)


HTML_TEMPLATE = """<!doctype html>
<html lang="id">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <base href="{base_href}" />
    <style>
      body {{
        font-family: Arial, Helvetica, sans-serif;
        font-size: 11pt;
        line-height: 1.5;
        color: #111827;
        margin: 24px;
      }}
      h1 {{ font-size: 20pt; margin-top: 28px; margin-bottom: 12px; }}
      h2 {{ font-size: 15pt; margin-top: 20px; margin-bottom: 8px; }}
      h3 {{ font-size: 12pt; margin-top: 14px; margin-bottom: 6px; }}
      p {{ margin: 8px 0; }}
      ul, ol {{ margin: 8px 0 8px 24px; }}
      li {{ margin: 4px 0; }}
      table {{
        border-collapse: collapse;
        width: 100%;
        margin: 12px 0;
        font-size: 10pt;
      }}
      th, td {{
        border: 1px solid #9ca3af;
        padding: 6px;
        vertical-align: top;
        text-align: left;
      }}
      th {{ background: #f3f4f6; }}
      img {{
        display: block;
        max-width: 100%;
        height: auto;
        margin: 12px auto;
      }}
      hr {{
        border: 0;
        border-top: 1px solid #9ca3af;
        margin: 20px 0;
      }}
      code {{
        font-family: Consolas, monospace;
        font-size: 0.92em;
        background: #f3f4f6;
        padding: 2px 4px;
      }}
      .cover {{
        border: 1px solid #d1d5db;
        padding: 18px;
        margin-bottom: 24px;
      }}
      .muted {{ color: #4b5563; }}
    </style>
  </head>
  <body>
    <section class="cover">
      <h1>{title}</h1>
      <p class="muted">{subtitle}</p>
      <p class="muted">Dokumen sumber: <strong>{source_name}</strong></p>
      <p class="muted">Bab pembuka: <strong>{heading}</strong></p>
    </section>
    {body}
  </body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render proposal markdown to PDF.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input markdown file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PDF path.")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="Document title.")
    parser.add_argument("--subtitle", default=DEFAULT_SUBTITLE, help="Document subtitle.")
    return parser.parse_args()


def require_binary(name: str, candidates: list[str] | None = None) -> str:
    for candidate in candidates or [name]:
        path = shutil.which(candidate)
        if path:
            return path
    raise FileNotFoundError(f"Required executable not found: {name}")


def extract_heading(markdown_text: str) -> str | None:
    match = re.search(r"^#\s+(.+)$", markdown_text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def markdown_to_html(markdown_text: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as handle:
        handle.write(markdown_text)
        temp_path = Path(handle.name)
    try:
        result = subprocess.run(
            ["npx", "--yes", "marked", "--gfm", "--input", str(temp_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout
    finally:
        temp_path.unlink(missing_ok=True)


def build_html_document(markdown_text: str, *, title: str, subtitle: str, source_name: str) -> str:
    heading = extract_heading(markdown_text) or source_name
    body = markdown_to_html(markdown_text)
    return HTML_TEMPLATE.format(
        title=html.escape(title),
        subtitle=html.escape(subtitle),
        source_name=html.escape(source_name),
        heading=html.escape(heading),
        base_href=(ROOT / "proposal").resolve().as_uri() + "/",
        body=body,
    )


def build_html(input_md: Path, *, title: str, subtitle: str) -> str:
    markdown_text = input_md.read_text(encoding="utf-8")
    return build_html_document(
        markdown_text,
        title=title,
        subtitle=subtitle,
        source_name=input_md.name,
    )


def print_pdf(html_content: str, output_pdf: Path) -> None:
    chrome = require_binary("chrome", ["google-chrome", "chromium", "chromium-browser", "brave-browser"])
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="proposal-render-") as temp_dir:
        html_path = Path(temp_dir) / "proposal-final.html"
        html_path.write_text(html_content, encoding="utf-8")
        subprocess.run(
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--allow-file-access-from-files",
                "--print-to-pdf-no-header",
                f"--print-to-pdf={output_pdf.resolve()}",
                html_path.as_uri(),
            ],
            check=True,
            capture_output=True,
            text=True,
        )


def main() -> int:
    args = parse_args()
    input_md = args.input.resolve()
    output_pdf = args.output.resolve()
    if not input_md.exists():
        raise FileNotFoundError(f"Input markdown not found: {input_md}")

    html_content = build_html(input_md, title=args.title, subtitle=args.subtitle)
    print_pdf(html_content, output_pdf)
    print(f"Rendered proposal PDF: {output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
