import json

import pytest

from scripts.analyze import aggregate_seed_rows, main, paired, split_seed_method


def row(method, seed=None, example_id="x"):
    return {
        "task": "vimmrc",
        "reader": "reader",
        "method": method if seed is None else f"{method}_s{seed}",
        "ratio": 0.333,
        "id": example_id,
        "cluster": "passage",
    }


def test_split_seed_method_only_reads_stage_eval_suffix():
    assert split_seed_method("viword_s2") == ("viword", 2)
    assert split_seed_method("random:seed=2") == ("random:seed=2", None)


def test_aggregate_seed_rows_requires_and_normalizes_three_seeds():
    rows = [row("viword", seed, example_id=str(seed)) for seed in (0, 1, 2)]
    rows.append(row("truncation", example_id="shared"))

    normalized = aggregate_seed_rows(rows, [0, 1, 2])

    assert [r["method"] for r in normalized[:3]] == ["viword"] * 3
    assert [r["_seed"] for r in normalized[:3]] == [0, 1, 2]
    assert normalized[-1]["_seed"] is None

    with pytest.raises(ValueError, match="incomplete seed coverage"):
        aggregate_seed_rows(rows[:2], [0, 1, 2])


def test_paired_matches_same_seed_before_shared_baseline_fallback():
    left = aggregate_seed_rows([row("viword", seed, example_id="same") for seed in (0, 1, 2)], [0, 1, 2])
    right = aggregate_seed_rows([row("other", seed, example_id="same") for seed in (0, 1, 2)], [0, 1, 2])

    paired_left, paired_right = paired(left, right)

    assert [r["_seed"] for r in paired_left] == [0, 1, 2]
    assert [r["_seed"] for r in paired_right] == [0, 1, 2]


def test_analyze_aggregate_reports_seed_sd_and_stratified_test(tmp_path, monkeypatch, capsys):
    rows = []
    for seed in (0, 1, 2):
        for index, example_id in enumerate(("a", "b")):
            base = {
                "task": "vimmrc",
                "reader": "reader",
                "ratio": 0.5,
                "id": example_id,
                "cluster": example_id,
                "accuracy": float(index == 0),
                "tokens_per_budget": 1.0,
            }
            rows.append({**base, "method": f"viword_s{seed}", "accuracy": 1.0})
    for index, example_id in enumerate(("a", "b")):
        base = {
            "task": "vimmrc",
            "reader": "reader",
            "ratio": 0.5,
            "id": example_id,
            "cluster": example_id,
            "accuracy": float(index == 0),
            "tokens_per_budget": 1.0,
        }
        rows.append({**base, "method": "truncation"})
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr("sys.argv", [
        "analyze.py", "--rows", str(path), "--target", "viword", "--reference", "truncation",
        "--seed-mode", "aggregate", "--seeds", "0", "1", "2", "--primary-reader", "reader",
        "--n-boot", "100",
    ])

    main()
    output = capsys.readouterr().out

    assert "mean ± seed SD" in output
    assert "1.0000 ± 0.0000" in output
    assert "## viword vs truncation" in output


def test_analyze_h2_runs_mcnemar_and_rouge_noninferiority(tmp_path, monkeypatch, capsys):
    rows = []
    for seed in (0, 1, 2):
        for example_id, cluster in (("a", "premise-a"), ("b", "premise-b")):
            base = {
                "task": "vinli",
                "reader": "reader",
                "ratio": 0.333,
                "id": example_id,
                "cluster": cluster,
                "accuracy": 1.0,
                "prediction": "A",
                "tokens_per_budget": 1.0,
            }
            rows.append({**base, "method": f"viword_pi_s{seed}", "prediction": "A"})
            rows.append({**base, "method": f"viword_s{seed}", "prediction": "B" if example_id == "a" else "A"})
            rows.append({**base, "method": f"llmlingua2vi_force_s{seed}", "prediction": "B"})
    for example_id, cluster in (("a", "article-a"), ("b", "article-b")):
        base = {
            "task": "vietnews",
            "reader": "reader",
            "ratio": 0.333,
            "id": example_id,
            "cluster": cluster,
            "rougeL": 0.8,
            "tokens_per_budget": 1.0,
        }
        rows.append({**base, "method": "viword_pi_s0"})
        rows.append({**base, "method": "viword_pi_s1"})
        rows.append({**base, "method": "viword_pi_s2"})
        rows.append({**base, "method": "viword_s0", "rougeL": 0.79})
        rows.append({**base, "method": "viword_s1", "rougeL": 0.79})
        rows.append({**base, "method": "viword_s2", "rougeL": 0.79})
        rows.append({**base, "method": "llmlingua2vi_force_s0", "rougeL": 0.78})
        rows.append({**base, "method": "llmlingua2vi_force_s1", "rougeL": 0.78})
        rows.append({**base, "method": "llmlingua2vi_force_s2", "rougeL": 0.78})
    for task, cluster in (("vinli", "premise-a"), ("vinli", "premise-b")):
        rows.append({"task": task, "reader": "reader", "ratio": 1.0, "id": cluster[-1:],
                     "cluster": cluster, "accuracy": 1.0, "prediction": "A", "tokens_per_budget": 1.0,
                     "method": "none"})
    path = tmp_path / "h2_rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr("sys.argv", [
        "analyze.py", "--rows", str(path), "--target", "viword_pi",
        "--reference", "viword", "llmlingua2vi_force", "--seed-mode", "aggregate",
        "--seeds", "0", "1", "2", "--primary-reader", "reader", "--n-boot", "100",
        "--hypothesis", "H2",
    ])

    main()
    output = capsys.readouterr().out

    assert "H2 (active): test=nfr" in output
    assert "## viword_pi vs viword" in output
    assert "## viword_pi vs llmlingua2vi_force" in output
    assert "### NFR: viword_pi < viword (one-sided paired bootstrap over premises and seeds)" in output
    assert "### NFR: viword_pi < llmlingua2vi_force" in output
    assert "s0 0/1 p=0.500" in output  # per-seed McNemar is descriptive only
    # H2 is the active family, so the main-metric ViNLI tests are not Holm-adjusted as H1
    nfr_section = output.split("### NFR: viword_pi < viword")[1]
    assert "| 0.333 |" in nfr_section
    assert "VietNews non-inferiority vs viword" in output
    assert "VietNews non-inferiority vs llmlingua2_force" not in output
    assert "VietNews non-inferiority vs llmlingua2vi_force" in output
