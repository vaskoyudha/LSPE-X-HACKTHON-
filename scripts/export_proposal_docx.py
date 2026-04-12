#!/usr/bin/env python3
"""Wrapper to generate DOCX files via the installed docx-js skill path."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Render markdown to DOCX via the installed docx skill path.")
    parser.add_argument("--input", default="proposal/proposal-final.md", help="Input markdown path relative to repo root or absolute path.")
    parser.add_argument("--output", default="proposal/proposal-final.docx", help="Output DOCX path relative to repo root or absolute path.")
    args = parser.parse_args()

    script = ROOT / "scripts" / "generate_proposal_docx.js"
    subprocess.run(["node", str(script), args.input, args.output], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
