#!/usr/bin/env python3
"""Render the final proposal markdown to a professional PDF via local tools.

This renderer keeps the repo dependency-free by using:
- `npx marked` for GitHub-flavored markdown to HTML conversion
- `google-chrome` (or Chromium/Brave) headless printing for PDF export
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "proposal" / "proposal-final.md"
DEFAULT_OUTPUT = ROOT / "proposal" / "proposal-final.pdf"


HTML_TEMPLATE = """<!doctype html>
<html lang="id">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <base href="{base_href}" />
    <style>
      @page {{
        size: A4;
        margin: 18mm 16mm 18mm 16mm;
      }}

      :root {{
        color-scheme: light;
        --ink: #0f172a;
        --muted: #475569;
        --rule: #cbd5e1;
        --soft: #eef2ff;
        --soft-2: #f8fafc;
        --accent: #0f766e;
      }}

      * {{
        box-sizing: border-box;
      }}

      html {{
        background: white;
      }}

      body {{
        margin: 0 auto;
        color: var(--ink);
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
        font-size: 11pt;
        line-height: 1.62;
      }}

      h1, h2, h3, h4 {{
        color: var(--ink);
        font-weight: 700;
        line-height: 1.25;
        margin: 1.1em 0 0.45em;
        page-break-after: avoid;
      }}

      h1 {{
        font-size: 20pt;
        border-bottom: 2px solid var(--accent);
        padding-bottom: 0.22em;
        margin-top: 0;
      }}

      h2 {{
        font-size: 15pt;
        margin-top: 1.35em;
      }}

      h3 {{
        font-size: 12.5pt;
      }}

      h4 {{
        font-size: 11.5pt;
      }}

      p, li, blockquote, table {{
        page-break-inside: avoid;
      }}

      p {{
        margin: 0.55em 0;
      }}

      ul, ol {{
        padding-left: 1.3rem;
        margin: 0.55em 0;
      }}

      li + li {{
        margin-top: 0.18em;
      }}

      strong {{
        color: var(--ink);
      }}

      code {{
        font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
        font-size: 0.92em;
        background: #f1f5f9;
        border-radius: 4px;
        padding: 0.08rem 0.28rem;
      }}

      pre {{
        overflow-x: auto;
        border: 1px solid var(--rule);
        border-radius: 10px;
        padding: 0.9rem 1rem;
        background: #f8fafc;
        white-space: pre-wrap;
      }}

      blockquote {{
        margin: 0.9em 0;
        padding: 0.75rem 1rem;
        border-left: 4px solid var(--accent);
        background: var(--soft-2);
        color: var(--muted);
      }}

      img {{
        display: block;
        max-width: 100%;
        width: auto;
        margin: 0.95rem auto 0.75rem;
        border: 1px solid #dbeafe;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        page-break-inside: avoid;
      }}

      hr {{
        border: 0;
        border-top: 1.5px solid var(--rule);
        margin: 1.8rem 0;
        page-break-after: always;
      }}

      table {{
        width: 100%;
        border-collapse: collapse;
        margin: 0.85rem 0 1rem;
        font-size: 10pt;
      }}

      thead {{
        background: var(--soft);
      }}

      th, td {{
        border: 1px solid var(--rule);
        padding: 0.48rem 0.56rem;
        text-align: left;
        vertical-align: top;
      }}

      th {{
        font-weight: 700;
      }}

      tbody tr:nth-child(even) {{
        background: #f8fafc;
      }}

      .title-block {{
        margin-bottom: 1.4rem;
        padding: 1rem 1.15rem;
        border: 1px solid var(--rule);
        border-radius: 14px;
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
      }}

      .title-block p {{
        margin: 0.2rem 0;
        color: var(--muted);
      }}
    </style>
  </head>
  <body>
    <section class="title-block">
      <h1>{title}</h1>
      <p>Proposal final Tahap 2 untuk Find IT! 2026 Track C.</p>
      <p>Dokumen ini dirender secara lokal dari markdown sumber agar isi proposal dan PDF submission tetap konsisten.</p>
    </section>
    {body}
  </body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render proposal markdown to PDF.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input markdown file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PDF path.")
    return parser.parse_args()


def require_binary(name: str, candidates: list[str] | None = None) -> str:
    for candidate in candidates or [name]:
        path = shutil.which(candidate)
        if path:
            return path
    raise FileNotFoundError(f"Required executable not found: {name}")


def render_markdown_fragment(input_md: Path) -> str:
    command = [
        "npx",
        "--yes",
        "marked",
        "--gfm",
        "--input",
        str(input_md),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout


def build_html(input_md: Path) -> str:
    fragment = render_markdown_fragment(input_md)
    return HTML_TEMPLATE.format(
        title="Proposal LPSE-X — Find IT! 2026 Tahap 2",
        base_href=input_md.resolve().parent.as_uri() + "/",
        body=fragment,
    )


def print_pdf(html_content: str, output_pdf: Path) -> None:
    chrome = require_binary("chrome", ["google-chrome", "chromium", "chromium-browser", "brave-browser"])
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="proposal-render-") as temp_dir:
        html_path = Path(temp_dir) / "proposal-final.html"
        html_path.write_text(html_content, encoding="utf-8")

        command = [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--allow-file-access-from-files",
            "--disable-features=Translate,MediaRouter",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=15000",
            "--print-to-pdf-no-header",
            f"--print-to-pdf={output_pdf.resolve()}",
            html_path.as_uri(),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)


def main() -> int:
    args = parse_args()
    input_md = args.input.resolve()
    output_pdf = args.output.resolve()
    if not input_md.exists():
        raise FileNotFoundError(f"Input markdown not found: {input_md}")

    html_content = build_html(input_md)
    print_pdf(html_content, output_pdf)
    print(f"Rendered proposal PDF: {output_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
