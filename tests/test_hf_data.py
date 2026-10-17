import hashlib

import pytest

from scripts.hf_data import files_to_push


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))  # no newline translation, the checksum is of these bytes
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_only_checksummed_files_are_pushed_with_repo_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = _write(tmp_path / "data/segmented/vinli_dev.jsonl", "{}\n")
    b = _write(tmp_path / "results/teacher_test/vinli_test.jsonl", "{}\n{}\n")
    _write(tmp_path / "data/distill/distilled.jsonl", "not listed\n")
    # the second line is in the format sha256sum writes in Git Bash ("*" = binary mode)
    _write(tmp_path / "data/SHA256SUMS",
           f"{a}  data/segmented/vinli_dev.jsonl\r\n{b} *results/teacher_test/vinli_test.jsonl\r\n\r\n")

    assert files_to_push() == [("data/segmented/vinli_dev.jsonl", "segmented/vinli_dev.jsonl"),
                               ("results/teacher_test/vinli_test.jsonl", "teacher_test/vinli_test.jsonl")]


def test_missing_changed_or_foreign_files_stop_the_push(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "data/tasks/vinli_dev.jsonl", "edited\n")
    _write(tmp_path / "data/SHA256SUMS", "".join(f"{'0' * 64}  {p}\n" for p in (
        "data/tasks/vinli_dev.jsonl", "data/tasks/viquad_dev.jsonl", "data/annot/protect_sheet.jsonl")))

    with pytest.raises(SystemExit) as error:
        files_to_push()
    message = str(error.value)
    assert "data/tasks/vinli_dev.jsonl: differs" in message
    assert "data/tasks/viquad_dev.jsonl: missing" in message
    assert "data/annot/protect_sheet.jsonl: not under" in message
