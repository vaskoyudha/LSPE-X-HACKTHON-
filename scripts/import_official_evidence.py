"""Import official/public evidence records into normalized evidence + label tables."""

from __future__ import annotations

import argparse
import filecmp
import json
from datetime import UTC, datetime
from pathlib import Path
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evidence import evidence_to_label_record, normalize_evidence_record
from src.evidence_sources import SOURCE_TRANSFORMERS


DEFAULT_OUT_DIR = ROOT / "data" / "processed" / "evidence"
DEFAULT_RAW_ARCHIVE_DIR = ROOT / "data" / "raw" / "evidence"
PRIMARY_KEY_COLUMNS = ["source_name", "source_record_id"]


def _load_input(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return [payload]
        return list(payload)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False).to_dict(orient="records")
    raise ValueError(f"Unsupported input format: {path.suffix}")


def _archive_destination(input_path: Path, source: str, raw_archive_dir: Path) -> Path:
    source_dir = raw_archive_dir / source
    source_dir.mkdir(parents=True, exist_ok=True)
    destination = source_dir / input_path.name
    if input_path.resolve() == destination.resolve():
        return destination
    if not destination.exists():
        return destination
    try:
        if filecmp.cmp(input_path, destination, shallow=False):
            return destination
    except OSError:
        pass

    stem = destination.stem
    suffix = destination.suffix
    counter = 1
    while True:
        candidate = source_dir / f"{stem}--{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def _archive_raw_input(input_path: Path, source: str, raw_archive_dir: Path | None) -> Path | None:
    if raw_archive_dir is None:
        return None

    destination = _archive_destination(input_path, source, raw_archive_dir)

    if input_path.resolve() != destination.resolve():
        shutil.copy2(input_path, destination)
    return destination


def _transform_records(records: list[dict], source: str) -> list[dict]:
    try:
        transformer = SOURCE_TRANSFORMERS[source]
    except KeyError as exc:
        raise ValueError(f"Unsupported source mode: {source}") from exc
    return transformer(records)


def _merge_with_existing_output(
    output_path: Path,
    rows: list[dict],
    *,
    key_columns: list[str] = PRIMARY_KEY_COLUMNS,
) -> pd.DataFrame:
    new_df = pd.DataFrame(rows)
    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        combined = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df

    dedupe_columns = [column for column in key_columns if column in combined.columns]
    if dedupe_columns:
        combined = combined.drop_duplicates(subset=dedupe_columns, keep="last", ignore_index=True)
    else:
        combined = combined.reset_index(drop=True)
    return combined


def import_official_evidence(
    input_path: Path,
    output_dir: Path = DEFAULT_OUT_DIR,
    *,
    source: str = "generic",
    raw_archive_dir: Path | None = DEFAULT_RAW_ARCHIVE_DIR,
    access_date: str | None = None,
) -> dict[str, Path]:
    records = _load_input(input_path)
    transformed_records = _transform_records(records, source)

    archived_raw_path = _archive_raw_input(input_path, source, raw_archive_dir)
    access_date = access_date or datetime.now(UTC).date().isoformat()
    imported_at = datetime.now(UTC).isoformat()

    evidence_rows = [
        normalize_evidence_record(
            {
                **record,
                "raw_file_path": record.get("raw_file_path") or (str(archived_raw_path) if archived_raw_path else None),
                "access_date": record.get("access_date") or access_date,
                "imported_at": record.get("imported_at") or imported_at,
            }
        )
        for record in transformed_records
    ]
    label_rows = [
        evidence_to_label_record(
            evidence,
            ocid=evidence.get("matched_ocid"),
            confidence_score=evidence.get("match_confidence"),
            reviewer_needed=evidence.get("matched_ocid") is None,
        )
        for evidence in evidence_rows
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "evidence_records.parquet"
    label_path = output_dir / "label_records.parquet"
    evidence_df = _merge_with_existing_output(evidence_path, evidence_rows)
    label_df = _merge_with_existing_output(label_path, label_rows)
    evidence_df.to_parquet(evidence_path, index=False, engine="pyarrow")
    label_df.to_parquet(label_path, index=False, engine="pyarrow")
    return {"evidence_records": evidence_path, "label_records": label_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--source",
        choices=sorted(SOURCE_TRANSFORMERS.keys()),
        default="generic",
        help="Select a source-specific normalization mode.",
    )
    parser.add_argument(
        "--raw-archive-dir",
        type=Path,
        default=DEFAULT_RAW_ARCHIVE_DIR,
        help="Directory where the original source input should be copied for provenance.",
    )
    parser.add_argument(
        "--access-date",
        type=str,
        default=None,
        help="Override the access/import date stored on normalized evidence records.",
    )
    args = parser.parse_args()
    outputs = import_official_evidence(
        args.input_path,
        output_dir=args.output_dir,
        source=args.source,
        raw_archive_dir=args.raw_archive_dir,
        access_date=args.access_date,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
