#!/usr/bin/env python3
"""Assemble the Find IT! 2026 Tahap 2 clean-package submission bundle.

The builder copies the approved submission artifacts from the working repository into
one clean folder with the exact filenames expected for submission.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TEAM_NAME = "BismillahFirstTry-Phase2"
BUNDLE_NAME = f"{TEAM_NAME}_Tahap2_FindIT2026"
PROPOSAL_FILENAME = f"Proposal_{TEAM_NAME}_Tahap2_FindIT2026.pdf"
README_TEMPLATE = Path("submission") / BUNDLE_NAME / "README.md"
IGNORE_PATTERNS = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")


@dataclass(frozen=True)
class CopyRule:
    source: Path
    destination: Path
    is_directory: bool = False


FILE_RULES = (
    CopyRule(Path("proposal/proposal-final.pdf"), Path(PROPOSAL_FILENAME)),
    CopyRule(Path("proposal/proposal-final.md"), Path("proposal_preview.md")),
    CopyRule(Path("training.ipynb"), Path("training.ipynb")),
    CopyRule(Path("inference.ipynb"), Path("inference.ipynb")),
    CopyRule(Path("requirements.txt"), Path("requirements.txt")),
    CopyRule(Path("models/xgb_model.ubj"), Path("model_risk.ubj")),
    CopyRule(Path("models/xgb_model.onnx"), Path("model_risk.onnx")),
)

DIRECTORY_RULES = (
    CopyRule(Path("src"), Path("src"), is_directory=True),
    CopyRule(Path("train_data"), Path("train_data"), is_directory=True),
    CopyRule(Path("test_data"), Path("test_data"), is_directory=True),
    CopyRule(Path("proposal/figures"), Path("figures"), is_directory=True),
)

ALL_RULES = FILE_RULES + DIRECTORY_RULES
MANAGED_DESTINATIONS = tuple(rule.destination for rule in ALL_RULES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the clean-package Tahap 2 submission bundle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("submission") / BUNDLE_NAME,
        help="Destination folder for the assembled submission bundle.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove managed bundle contents in the output directory before copying.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_sources_exist(root: Path, rules: Iterable[CopyRule]) -> None:
    missing = [str(root / rule.source) for rule in rules if not (root / rule.source).exists()]
    missing_readme = root / README_TEMPLATE
    if not missing_readme.exists():
        missing.append(str(missing_readme))
    if missing:
        formatted = "\n - ".join(sorted(missing))
        raise FileNotFoundError(f"Missing required submission sources:\n - {formatted}")


def clear_managed_content(output_dir: Path) -> None:
    for relative_path in MANAGED_DESTINATIONS:
        target = output_dir / relative_path
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def copy_file(root: Path, output_dir: Path, rule: CopyRule) -> None:
    source = root / rule.source
    destination = output_dir / rule.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_directory(root: Path, output_dir: Path, rule: CopyRule) -> None:
    source = root / rule.source
    destination = output_dir / rule.destination
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=IGNORE_PATTERNS)


def install_readme(root: Path, output_dir: Path) -> None:
    template = root / README_TEMPLATE
    destination = output_dir / "README.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if template.resolve() == destination.resolve():
        return
    shutil.copy2(template, destination)


def build_bundle(output_dir: Path, clean: bool) -> list[Path]:
    root = repo_root()
    ensure_sources_exist(root, ALL_RULES)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if clean:
        clear_managed_content(output_dir)

    install_readme(root, output_dir)

    copied_paths: list[Path] = [output_dir / "README.md"]
    for rule in FILE_RULES:
        copy_file(root, output_dir, rule)
        copied_paths.append(output_dir / rule.destination)
    for rule in DIRECTORY_RULES:
        copy_directory(root, output_dir, rule)
        copied_paths.append(output_dir / rule.destination)

    return copied_paths


def main() -> int:
    args = parse_args()
    try:
        copied_paths = build_bundle(args.output_dir, clean=args.clean)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Built submission bundle: {args.output_dir.resolve()}")
    for path in copied_paths:
        print(f" - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
