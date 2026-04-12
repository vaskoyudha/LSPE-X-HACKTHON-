#!/usr/bin/env python3
"""Wrapper to generate the proposal DOCX via the installed docx-js skill path."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    script = ROOT / "scripts" / "generate_proposal_docx.js"
    subprocess.run(["node", str(script)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
