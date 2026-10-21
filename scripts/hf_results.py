"""Bring the ops packages (.tgz) back from the GPU machine through a private Hugging Face dataset repo.

  python scripts/hf_results.py push viword_part3_<date>.tgz   # GPU machine (ops_full_run.sh does it)
  python scripts/hf_results.py list                            # project owner: what has come back
  python scripts/hf_results.py pull --match part3              # download into results/from_gpu/ and verify
  python scripts/hf_results.py verify viword_part3_<date>.tgz  # a package copied by hand, with its .sha256

Every package travels with "<name>.sha256" (sha256sum format, written by ops_full_run.sh or here), and
pull/verify refuse a package whose bytes differ from it. The results repo is not the data repo: the data
repo stays read-only for the GPU machine and keeps matching data/SHA256SUMS. Push needs HF_TOKEN with
write access to the results repo only (ops_full_run.sh passes HF_RESULTS_TOKEN as HF_TOKEN).
"""
import argparse
import hashlib
import os

REPO = os.environ.get("HF_RESULTS_REPO", "thanthienhai/viword-c-results")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_checksum(path: str) -> str:
    with open(path + ".sha256", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{sha256(path)}  {os.path.basename(path)}\n")
    return path + ".sha256"


def verify(path: str) -> None:
    """Raises SystemExit if `path` has no .sha256 next to it or differs from it."""
    try:
        with open(path + ".sha256", encoding="utf-8") as f:
            expected = f.read().split()[0]
    except (FileNotFoundError, IndexError):
        raise SystemExit(f"{path}.sha256 missing or empty: ask for the checksum printed by ops_full_run.sh")
    if sha256(path) != expected:
        raise SystemExit(f"{path} differs from {path}.sha256: the copy is broken, fetch it again")
    print(f"{path}: checksum OK")


def packages(files: list[str], match: str = "") -> list[str]:
    """Packages in the repo (with their checksum) whose name contains `match`."""
    have = set(files)
    return sorted(f for f in files if f.endswith(".tgz") and match in f and f + ".sha256" in have)


def push(repo: str, paths: list[str]) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    operations = []
    for path in paths:
        checksum = path + ".sha256"
        verify(path) if os.path.exists(checksum) else write_checksum(path)
        name = os.path.basename(path)
        operations += [CommitOperationAdd(path_in_repo=name, path_or_fileobj=path),
                       CommitOperationAdd(path_in_repo=name + ".sha256", path_or_fileobj=checksum)]
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.create_commit(repo, repo_type="dataset", operations=operations,
                             commit_message="ops: " + " ".join(os.path.basename(p) for p in paths))
    for path in paths:
        print(f"uploaded {os.path.basename(path)} (+ .sha256) -> {repo} @ {info.oid[:8]}")


def pull(repo: str, match: str, out: str) -> None:
    from huggingface_hub import HfApi, hf_hub_download

    found = packages(HfApi().list_repo_files(repo, repo_type="dataset"), match)
    if not found:
        raise SystemExit(f"no package matching {match!r} in {repo}")
    for name in found:
        for f in (name, name + ".sha256"):
            hf_hub_download(repo, f, repo_type="dataset", local_dir=out)
        verify(os.path.join(out, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "list", "pull", "verify"])
    ap.add_argument("paths", nargs="*", help="packages to push or verify")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--match", default="", help="pull: only packages whose name contains this")
    ap.add_argument("--out", default="results/from_gpu")
    args = ap.parse_args()
    if args.action in ("push", "verify") and not args.paths:
        ap.error(f"{args.action} needs at least one package")
    if args.action == "push":
        push(args.repo, args.paths)
    elif args.action == "verify":
        for path in args.paths:
            verify(path)
    elif args.action == "list":
        from huggingface_hub import HfApi
        print("\n".join(packages(HfApi().list_repo_files(args.repo, repo_type="dataset"))))
    else:
        pull(args.repo, args.match, args.out)


if __name__ == "__main__":
    main()
