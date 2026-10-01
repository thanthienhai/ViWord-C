"""Share the data/ folder (not in git) through a private Hugging Face dataset repo.

  python scripts/hf_data.py push --repo <user>/viword-c-data     # upload data/ (+ teacher dev outputs)
  python scripts/hf_data.py pull --repo <user>/viword-c-data     # download into data/ and results/

Needs `huggingface-cli login` (or HF_TOKEN) with write access for push. The repo is created
private: it holds re-distributed third-party datasets and outputs of an internal teacher model.
Every push is a new commit, so a result can always be tied to the exact data revision.
"""
import argparse

FOLDERS = {"data/tasks": "tasks", "data/segmented": "segmented", "data/distill": "distill",
           "results/teacher_dev": "teacher_dev"}


def push(repo: str, message: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    for local, remote in FOLDERS.items():
        print(f"upload {local} -> {repo}/{remote}")
        api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=local, path_in_repo=remote,
                          allow_patterns=["*.jsonl", "*.json"], ignore_patterns=["smoke_*"],
                          commit_message=f"{message}: {remote}")


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
