"""Provenance for cached segmentation files.

JSONL has no portable file-level header.  Each output from
``scripts/segment_data.py`` therefore has a small ``.meta.json`` sidecar.  The
sidecar records the segmenter and the SHA-256 of the JSONL it describes, so a
metadata file cannot silently be reused after the cached segmentation changes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


METADATA_SCHEMA = "viword-segmentation-v1"
SEGMENTER_INFO = {
    "vncorenlp": {
        "implementation": "VnCoreNLP 1.2",
        "annotators": ["RDRSegmenter", "POS", "NER"],
    },
    "underthesea": {
        "implementation": "underthesea",
        "annotators": ["word segmentation", "POS", "NER"],
    },
    "pyvi": {
        "implementation": "pyvi",
        "annotators": ["word segmentation", "POS"],
    },
}


def metadata_path(output: str | Path) -> Path:
    """Return the sidecar path for a segmented JSONL file."""
    output = Path(output)
    return output.with_name(output.name + ".meta.json")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_metadata(
    output: str | Path,
    *,
    segmenter: str,
    input_path: str | None = None,
    field: str | None = None,
    rows: int | None = None,
    unique_texts: int | None = None,
    provenance: str = "scripts/segment_data.py",
) -> dict:
    """Build metadata for an existing segmented JSONL output."""
    if segmenter not in SEGMENTER_INFO:
        raise ValueError(f"unknown segmenter: {segmenter}")
    info = SEGMENTER_INFO[segmenter]
    metadata = {
        "schema": METADATA_SCHEMA,
        "segmenter": segmenter,
        "implementation": info["implementation"],
        "annotators": info["annotators"],
        "provenance": provenance,
        "output_sha256": sha256_path(output),
    }
    if input_path is not None:
        metadata["input"] = input_path
    if field is not None:
        metadata["field"] = field
    if rows is not None:
        metadata["rows"] = rows
    if unique_texts is not None:
        metadata["unique_texts"] = unique_texts
    return metadata


def write_metadata(output: str | Path, metadata: dict) -> Path:
    """Write a JSON sidecar next to ``output`` and return its path."""
    path = metadata_path(output)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


FRESH_PROVENANCE = "scripts/segment_data.py"


def needs_verification(metadata: dict) -> bool:
    """A sidecar written after the fact (not by segment_data.py) is only a claim until
    a VnCoreNLP re-segmentation sample confirms it (scripts/verify_segmentation.py)."""
    if metadata.get("provenance") == FRESH_PROVENANCE:
        return False
    check = metadata.get("verification") or {}
    return not (check.get("sample_size", 0) > 0 and check.get("matched") == check.get("sample_size"))


def verify_metadata(output: str | Path, expected_segmenter: str = "vncorenlp",
                    require_verified: bool = True) -> list[str]:
    """Return human-readable validation errors for one cached segmentation."""
    output = Path(output)
    sidecar = metadata_path(output)
    if not sidecar.exists():
        return [f"{output}: missing {sidecar.name}"]
    try:
        with open(sidecar, encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{sidecar}: cannot read metadata ({exc})"]

    errors = []
    if metadata.get("schema") != METADATA_SCHEMA:
        errors.append(f"{sidecar}: unsupported schema {metadata.get('schema')!r}")
    if metadata.get("segmenter") != expected_segmenter:
        errors.append(
            f"{output}: segmenter={metadata.get('segmenter')!r}, "
            f"expected {expected_segmenter!r}"
        )
    expected_info = SEGMENTER_INFO.get(expected_segmenter)
    if expected_info:
        if metadata.get("implementation") != expected_info["implementation"]:
            errors.append(f"{sidecar}: implementation is not {expected_info['implementation']!r}")
        if metadata.get("annotators") != expected_info["annotators"]:
            errors.append(f"{sidecar}: annotators must be {expected_info['annotators']!r}")
    expected_hash = metadata.get("output_sha256")
    if not expected_hash:
        errors.append(f"{sidecar}: missing output_sha256")
    elif expected_hash != sha256_path(output):
        errors.append(f"{output}: SHA-256 does not match {sidecar.name}")
    if require_verified and needs_verification(metadata):
        errors.append(f"{output}: provenance {metadata.get('provenance')!r} is unverified; run "
                      "scripts/verify_segmentation.py or re-segment with scripts/segment_data.py")
    return errors
