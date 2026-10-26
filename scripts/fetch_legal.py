"""Download the legal sources of direction B (Vietnamese legal RAG) at pinned revisions.

  python scripts/fetch_legal.py                 # all sources into data/raw/<name>/
  python scripts/fetch_legal.py --only vlegal_bench phapdien

Each source lands in its own folder with a SOURCE.json (repo, revision, license, sha256 of every
file), so a corpus or a result can be tied to the exact upstream bytes. data/ is not versioned.
Licences (checked 2026-10-26, see the direction-B proposal):
  vlegal_bench  CC BY-NC-ND 4.0: evaluation only; compressed or translated contexts are derivatives
                and are never published, only scores.
  phapdien      CC BY 4.0 (the card asks to check the source site's terms before commercial use).
  vbpl          CC BY 4.0 for the compiled dataset; the statutory texts themselves are public.
The vbpl `legacy/` folder (3.5 GB) and the phapdien embeddings (450 MB) are not needed and skipped.
"""
import argparse
import hashlib
import json
import os

SOURCES = {
    "vlegal_bench": {
        "repo": "CMC-OPENAI/VLegal-Bench",
        "revision": "215f273d710a8848558c580ab6d9fc519b161551",
        "license": "CC BY-NC-ND 4.0",
        "allow": ["*/*.jsonl", "README.md", "LICENSE*"],
    },
    "phapdien": {
        "repo": "tmquan/phapdien-moj-gov-vn",
        "revision": "a1b6860c821b7142ffe2385178af4e4bb6f86aff",
        "license": "CC BY 4.0",
        "allow": ["documents-*.parquet", "subjects.parquet", "tree_nodes.parquet", "ontology_*.parquet",
                  "ontology.json", "README.md"],
    },
    "vbpl": {
        "repo": "th1nhng0/vietnamese-legal-documents",
        "revision": "8977887f17be2defae4c5171d55562e1cde7d695",
        "license": "CC BY 4.0 (compiled dataset)",
        "allow": ["data/*.parquet", "README.md"],
    },
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, root: str = "data/raw") -> str:
    from huggingface_hub import snapshot_download

    src = SOURCES[name]
    out = os.path.join(root, name)
    snapshot_download(src["repo"], repo_type="dataset", revision=src["revision"], allow_patterns=src["allow"],
                      local_dir=out)
    files = {}
    for folder, _, names in os.walk(out):
        if os.path.basename(folder) == ".cache":
            continue
        for n in names:
            path = os.path.join(folder, n)
            rel = os.path.relpath(path, out).replace(os.sep, "/")
            if rel != "SOURCE.json" and not rel.startswith(".cache/"):
                files[rel] = {"bytes": os.path.getsize(path), "sha256": sha256(path)}
    record = {**{k: src[k] for k in ("repo", "revision", "license")}, "files": dict(sorted(files.items()))}
    with open(os.path.join(out, "SOURCE.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=sorted(SOURCES), default=sorted(SOURCES))
    ap.add_argument("--root", default="data/raw")
    args = ap.parse_args()
    for name in args.only:
        out = fetch(name, args.root)
        n = len(json.load(open(os.path.join(out, "SOURCE.json"), encoding="utf-8"))["files"])
        print(f"{name}: {n} files in {out}")


if __name__ == "__main__":
    main()
