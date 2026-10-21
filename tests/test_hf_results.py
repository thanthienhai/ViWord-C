import sys
import types

import pytest

from scripts import hf_results


def _package(tmp_path, data=b"tgz bytes"):
    path = tmp_path / "viword_part3_20261101_0900.tgz"
    path.write_bytes(data)
    return str(path)


def test_checksum_is_in_sha256sum_format_and_catches_a_broken_copy(tmp_path):
    path = _package(tmp_path)
    hf_results.write_checksum(path)
    line = open(path + ".sha256", encoding="utf-8").read()
    assert line == f"{hf_results.sha256(path)}  viword_part3_20261101_0900.tgz\n"
    hf_results.verify(path)

    open(path, "ab").write(b"!")  # a truncated or altered transfer
    with pytest.raises(SystemExit, match="differs"):
        hf_results.verify(path)


def test_a_package_without_its_checksum_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="sha256 missing"):
        hf_results.verify(_package(tmp_path))


def test_only_packages_with_a_checksum_are_listed():
    files = ["viword_part3_a.tgz", "viword_part3_a.tgz.sha256", "viword_part4_b.tgz", "README.md",
             "viword_part4_c.tgz", "viword_part4_c.tgz.sha256"]
    assert hf_results.packages(files) == ["viword_part3_a.tgz", "viword_part4_c.tgz"]
    assert hf_results.packages(files, "part4") == ["viword_part4_c.tgz"]


def test_push_uploads_the_package_and_its_checksum_in_one_private_commit(tmp_path, monkeypatch):
    calls = {}

    class Op:
        def __init__(self, path_in_repo, path_or_fileobj):
            self.path_in_repo, self.local = path_in_repo, path_or_fileobj

    class Api:
        def create_repo(self, repo, **kwargs):
            calls["repo"] = (repo, kwargs)

        def create_commit(self, repo, operations, **kwargs):
            calls["files"] = [op.path_in_repo for op in operations]
            return types.SimpleNamespace(oid="0123456789")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(CommitOperationAdd=Op, HfApi=Api))
    path = _package(tmp_path)
    hf_results.push("me/results", [path])

    assert calls["repo"] == ("me/results", {"repo_type": "dataset", "private": True, "exist_ok": True})
    assert calls["files"] == ["viword_part3_20261101_0900.tgz", "viword_part3_20261101_0900.tgz.sha256"]
    hf_results.verify(path)  # written by push when the package came without one

    open(path + ".sha256", "w").write("0" * 64 + "  x.tgz\n")  # a stale checksum stops the push
    with pytest.raises(SystemExit, match="differs"):
        hf_results.push("me/results", [path])
