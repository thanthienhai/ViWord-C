import json
import tempfile
from pathlib import Path

from viword.segment_meta import build_metadata, metadata_path, verify_metadata, write_metadata


def test_vncorenlp_metadata_records_rdr_pos_ner_and_hash():
    with tempfile.TemporaryDirectory(dir=".") as directory:
        output = Path(directory) / "docs.jsonl"
        output.write_text('{"id": "x", "doc": []}\n', encoding="utf-8")

        metadata = build_metadata(
            output,
            segmenter="vncorenlp",
            input_path="data/tasks/example.jsonl",
            field="context",
            rows=1,
            unique_texts=1,
        )
        write_metadata(output, metadata)

        saved = json.loads(metadata_path(output).read_text(encoding="utf-8"))
        assert saved["segmenter"] == "vncorenlp"
        assert saved["annotators"] == ["RDRSegmenter", "POS", "NER"]
        assert verify_metadata(output) == []


def test_metadata_rejects_a_changed_jsonl():
    with tempfile.TemporaryDirectory(dir=".") as directory:
        output = Path(directory) / "docs.jsonl"
        output.write_text('{"id": "x", "doc": []}\n', encoding="utf-8")
        write_metadata(output, build_metadata(output, segmenter="vncorenlp"))
        output.write_text('{"id": "changed", "doc": []}\n', encoding="utf-8")

        assert any("SHA-256" in error for error in verify_metadata(output))
