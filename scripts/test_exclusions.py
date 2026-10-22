"""Test rows that readers scored during development, and the Belebele dev set made of them (DATN §4.3,
amendment of 2026-10-22; viword/heldout.py).

Every development read below took the first rows of a test file (`--limit`, `head -n`), so a test file
loses its first N rows, N being the largest read. The list comes from an audit on 2026-10-22 of every row
file the project had: results/, run_reports/ and the packages on the HF results repo.

  python scripts/test_exclusions.py write          # config/test_exclusions.json from the list below
  python scripts/test_exclusions.py check          # the committed list against data/tasks
  python scripts/test_exclusions.py belebele-dev   # data/{tasks,segmented}/belebele_dev.jsonl

`write` is run once, when a new development read is found; the JSON is committed before the test run.
`belebele-dev` runs after every data pull (ops check_data, run_pipeline.sh segment): the dev set is
derived from the checksummed belebele.jsonl, so it needs no upload.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from viword.data import read_jsonl, write_jsonl
from viword.heldout import EXCLUSIONS, TEST_FILES
from viword.segment_meta import build_subset_metadata, sha256_path, write_metadata

REGISTERED = "2026-10-22"
SMOKE_SEALLM = ("none, no_context, truncation, tfidf_sent, llmlingua2_syl, lexprior, llmlingua2vi_s0, viword_s0, "
                "viword_pi_s0")
EXPOSURES = [
    {"date": "2026-09-29", "run": "local smoke, results/eval/smoke_belebele_qwen05b.jsonl",
     "reader": "Qwen/Qwen2.5-0.5B-Instruct", "rows": {"belebele": 60},
     "methods": "none, no_context, lead, llmlingua2_syl, llmlingua2_wordpool"},
    {"date": "2026-10-04", "run": "ops smoke, scripts/smoke_h100.sh (results/smoke, "
                                  "run_reports/SMOKE_REPORT_2026-10-04.md)",
     "reader": "Qwen/Qwen2.5-7B-Instruct", "rows": {"belebele": 200, "vinli_test": 300},
     "methods": "none, no_context, lead, truncation, selective_context, llmlingua2_syl, llmlingua2_wordpool, "
                "llmlingua2vi, llmlingua2vi_wordpool, viword_syl, viword, viword_pi, lexprior, probe_negation "
                "(ViNLI); smoke encoders (1 epoch, 1,500 paragraphs, segmentation before the 2026-10-12 fix); "
                "the H1 contrasts were tested with Holm"},
    {"date": "2026-10-18", "run": "Kaggle T4 trial, results/t4_kaggle/t4",
     "reader": "Qwen/Qwen2.5-1.5B-Instruct", "rows": {"belebele": 4}, "methods": "none, translate_llmlingua2"},
    {"date": "2026-10-20", "run": "Kaggle T4 trial of the teacher upper bound, results/t4_kaggle/t4b",
     "reader": "Qwen/Qwen2.5-1.5B-Instruct", "rows": {stem: 20 for stem in TEST_FILES},
     "methods": "none, truncation, teacher_upper"},
    {"date": "2026-10-20", "run": "ops smoke4 precheck (viword_smoke4_20261020_1714.tgz)",
     "reader": "Qwen/Qwen2.5-7B-Instruct", "rows": {"belebele": 20}, "methods": "none, translate_llmlingua2"},
    {"date": "2026-10-20", "run": "ops smoke4, second reader on 'dev' (viword_smoke4_20261020_1714.tgz)",
     "reader": "SeaLLMs/SeaLLM-7B-v2", "rows": {"belebele": 50}, "methods": SMOKE_SEALLM},
    {"date": "2026-10-21", "run": "Kaggle 2x T4 smoke, run_reports/kaggle_smoke_20261021",
     "reader": "Qwen/Qwen2.5-1.5B-Instruct, SeaLLMs/SeaLLMs-v3-1.5B-Chat", "rows": {"belebele": 50},
     "methods": "none, no_context, truncation, lead"},
    {"date": "2026-10-21", "run": "ops smoke5 precheck (viword_smoke5_20261021_1644.tgz)",
     "reader": "Qwen/Qwen2.5-7B-Instruct", "rows": {"belebele": 20}, "methods": "none, translate_llmlingua2"},
    {"date": "2026-10-21", "run": "ops smoke5, second reader on 'dev' (viword_smoke5_20261021_1644.tgz)",
     "reader": "SeaLLMs/SeaLLM-7B-v2", "rows": {"belebele": 50}, "methods": SMOKE_SEALLM},
]
RULE = ("A test row that a reader scored before the test run leaves the primary analysis (DATN §4.3); the "
        "full test sets are analysed as a sensitivity analysis. Rows, not clusters, are excluded: dropping "
        "the 203 ViNLI premises of the exposed rows would leave 252 negated-premise pairs, under the 300 of "
        "gate G2. The Belebele rows form belebele_dev (Belebele has no dev split), used only by smoke runs "
        "and prechecks, never for a decision.")


def exposed_rows() -> dict[str, int]:
    """Largest number of leading rows read from each test file."""
    rows = {}
    for exposure in EXPOSURES:
        for stem, n in exposure["rows"].items():
            rows[stem] = max(rows.get(stem, 0), n)
    return rows


def build(tasks_dir: Path) -> dict:
    sets = {}
    for stem, n in exposed_rows().items():
        path = tasks_dir / f"{stem}.jsonl"
        rows = read_jsonl(path)
        ids = [r["id"] for r in rows[:n]]
        sets[stem] = {"task": TEST_FILES[stem], "file": f"data/tasks/{stem}.jsonl", "file_sha256": sha256_path(path),
                      "rows": n, "clusters": len({r["cluster"] for r in rows[:n]}), "of_rows": len(rows),
                      "of_clusters": len({r["cluster"] for r in rows}), "ids": ids}
    return {"schema": 1, "registered": REGISTERED, "rule": RULE, "exposures": EXPOSURES, "sets": sets}


def check(config: dict, tasks_dir: Path) -> list[str]:
    """Errors if the committed list no longer matches the audit above or the data files."""
    errors = []
    if config.get("exposures") != EXPOSURES:
        errors.append("the exposures in the JSON differ from scripts/test_exclusions.py: rerun `write`")
    for stem, n in exposed_rows().items():
        entry = config["sets"].get(stem)
        if entry is None or entry["rows"] != n or len(entry["ids"]) != n:
            errors.append(f"{stem}: the JSON does not list the first {n} rows")
            continue
        path = tasks_dir / f"{stem}.jsonl"
        if not path.exists():
            errors.append(f"{path} is missing")
        elif sha256_path(path) != entry["file_sha256"]:
            errors.append(f"{path} differs from the file the exclusions were drawn from")
    return errors


def make_belebele_dev(config: dict, tasks_dir: Path, segmented_dir: Path) -> int:
    """belebele_dev = the Belebele rows readers have scored, copied line for line from the checksummed files."""
    entry = config["sets"]["belebele"]
    source = tasks_dir / "belebele.jsonl"
    if sha256_path(source) != entry["file_sha256"]:
        raise SystemExit(f"{source} differs from the file the exclusions were drawn from: pull the HF data")
    keep = set(entry["ids"])
    write_jsonl(tasks_dir / "belebele_dev.jsonl", [r for r in read_jsonl(source) if r["id"] in keep])
    seg_source, seg_out = segmented_dir / "belebele.jsonl", segmented_dir / "belebele_dev.jsonl"
    with open(seg_source, encoding="utf-8") as f, open(seg_out, "w", encoding="utf-8", newline="\n") as out:
        for line in f:
            if line.strip() and json.loads(line)["id"] in keep:
                out.write(line.rstrip("\r\n") + "\n")
    write_metadata(seg_out, build_subset_metadata(seg_out, seg_source))
    return len(keep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["write", "check", "belebele-dev"])
    ap.add_argument("--config", default=str(ROOT / EXCLUSIONS))
    ap.add_argument("--tasks-dir", default=str(ROOT / "data/tasks"))
    ap.add_argument("--segmented-dir", default=str(ROOT / "data/segmented"))
    args = ap.parse_args()
    tasks_dir, config_path = Path(args.tasks_dir), Path(args.config)
    if args.command == "write":
        config = build(tasks_dir)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        for stem, entry in config["sets"].items():
            print(f"{stem}: {entry['rows']}/{entry['of_rows']} rows, {entry['clusters']}/{entry['of_clusters']} "
                  "clusters excluded")
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.command == "check":
        errors = check(config, tasks_dir)
        if errors:
            raise SystemExit("test exclusions: " + "; ".join(errors))
        print(f"test exclusions OK: {sum(len(s['ids']) for s in config['sets'].values())} rows in "
              f"{len(config['sets'])} test files ({config_path})")
        return
    n = make_belebele_dev(config, tasks_dir, Path(args.segmented_dir))
    print(f"belebele_dev: {n} questions readers scored during development (not part of the test analysis)")


if __name__ == "__main__":
    main()
