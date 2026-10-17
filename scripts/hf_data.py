"""Share the data/ folder (not in git) through a private Hugging Face dataset repo.

  python scripts/hf_data.py push --repo <user>/viword-c-data     # upload the files in data/SHA256SUMS
  python scripts/hf_data.py pull --repo <user>/viword-c-data     # download into data/ and results/

Needs `huggingface-cli login` (or HF_TOKEN) with write access for push. The repo is created
private: it holds re-distributed third-party datasets and outputs of an internal teacher model.
Push uploads exactly the files listed in data/SHA256SUMS, in one commit, and refuses to run if
any of them is missing or differs from its checksum: the ops run checks every pulled file against
the same list, so nothing else is needed and a mismatch would only stop it later. Add a new file
to data/SHA256SUMS before pushing it. Every push is a new commit, so a result can always be tied
to the exact data revision.
"""
import argparse
import hashlib

FOLDERS = {"data/tasks": "tasks", "data/segmented": "segmented", "data/distill": "distill",
           "results/teacher_dev": "teacher_dev", "results/teacher_test": "teacher_test"}
SUMS = "data/SHA256SUMS"


def files_to_push(sums: str = SUMS) -> list[tuple[str, str]]:
    """(local path, path in the repo) for every file in the checksum list; raises SystemExit
    if one is missing, differs from its checksum or lies outside FOLDERS."""
    files, problems = [], []
    with open(sums, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            digest, path = line.split()
            path = path.removeprefix("*")  # binary-mode mark, the default of sha256sum in Git Bash
            folder = next((d for d in FOLDERS if path.startswith(d + "/")), None)
            if folder is None:
                problems.append(f"{path}: not under {', '.join(FOLDERS)}")
                continue
            try:
                actual = sha256(path)
            except FileNotFoundError:
                problems.append(f"{path}: missing")
                continue
            if actual != digest:
                problems.append(f"{path}: differs from {sums}")
                continue
            files.append((path, FOLDERS[folder] + path[len(folder):]))
    if problems:
        raise SystemExit("not pushed:\n  " + "\n  ".join(problems))
    return files


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def push(repo: str, message: str) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    files = files_to_push()
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    print(f"upload {len(files)} files of {SUMS} -> {repo}")
    info = api.create_commit(repo, repo_type="dataset", commit_message=message,
                             operations=[CommitOperationAdd(path_in_repo=remote, path_or_fileobj=local)
                                         for local, remote in files])
    print(f"{repo} @ {info.oid}")


def pull(repo: str, revision: str | None) -> None:
    import os
    import shutil

    from huggingface_hub import snapshot_download

    path = snapshot_download(repo, repo_type="dataset", revision=revision)
    for local, remote in FOLDERS.items():
        if os.path.isdir(os.path.join(path, remote)):
            shutil.copytree(os.path.join(path, remote), local, dirs_exist_ok=True)
            print(f"{repo}/{remote} -> {local}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", default=None, help="pull a specific commit/tag")
    ap.add_argument("--message", default="update data")
    args = ap.parse_args()
    push(args.repo, args.message) if args.action == "push" else pull(args.repo, args.revision)


if __name__ == "__main__":
    main()
