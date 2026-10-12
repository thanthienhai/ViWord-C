import pytest

from scripts.mde import compute, render_report


def result(task, method, ratio, example_id, *, accuracy=1.0, rouge=0.8, prediction="A", seed=None):
    return {
        "task": task,
        "reader": "reader",
        "method": method if seed is None else f"{method}_s{seed}",
        "ratio": ratio,
        "id": example_id,
        "cluster": example_id,
        "accuracy": accuracy,
        "rougeL": rouge,
        "prediction": prediction,
        "tokens_per_budget": 1.0,
    }


def pilot_rows():
    rows = []
    for ratio in (0.5, 0.333, 0.2):
        for seed in (0, 1, 2):
            for example_id in ("a", "b"):
                rows.append(result("vimmrc", "viword", ratio, example_id, seed=seed, accuracy=1.0))
                rows.append(result("vimmrc", "llmlingua2vi", ratio, example_id, seed=seed, accuracy=0.5))
                rows.append(result("vietnews", "viword", ratio, example_id, seed=seed, rouge=0.8))
                rows.append(result("vietnews", "viword_pi", ratio, example_id, seed=seed, rouge=0.798))
                rows.append(result("vinli", "viword", ratio, example_id, seed=seed,
                                   prediction="B" if example_id == "a" else "A"))
                rows.append(result("vinli", "viword_pi", ratio, example_id, seed=seed, prediction="A"))
    for example_id in ("a", "b"):
        rows.append(result("vinli", "none", 1.0, example_id, prediction="A"))
    return rows


TEST_CLUSTERS = {"vimmrc": 2, "vietnews": 2, "vinli": 2}


def test_mde_uses_dev_rows_and_reports_margin_against_mde_and_seed_sd():
    result_data = compute(pilot_rows(), [0, 1, 2], n_boot=100, margin_points=0.5,
                          test_clusters=TEST_CLUSTERS)

    assert result_data["dev_only"] is True
    assert result_data["h1_dev_tasks"] == ["vimmrc"]
    assert result_data["h1_missing_dev_tasks"] == ["belebele"]
    assert len(result_data["h1"]) == 3
    assert len(result_data["h2"]["rougeL"]) == 3
    assert result_data["rouge_margin_check"]["margin_gt_all_mde"] is True
    assert result_data["rouge_margin_check"]["margin_gt_all_seed_sd"] is True
    assert result_data["passed"] is True
    assert "Not available as DEV: belebele" in render_report(result_data)
    assert result_data["h1_contrast"] == "viword − llmlingua2vi"
    assert all("ViWord-C − LLMLingua-2-vi" in row["label"] for row in result_data["h1"])


def test_mde_is_projected_to_the_test_size():
    small_test = compute(pilot_rows(), [0, 1, 2], n_boot=200, test_clusters=TEST_CLUSTERS)
    big_test = compute(pilot_rows(), [0, 1, 2], n_boot=200,
                       test_clusters={"vimmrc": 8, "vietnews": 8, "vinli": 8})
    for a, b in zip(small_test["h1"], big_test["h1"]):
        assert b["mde_test"] == pytest.approx(a["mde_test"] / 2)


def test_mde_requires_test_cluster_counts():
    with pytest.raises(ValueError, match="no test cluster count"):
        compute(pilot_rows(), [0, 1, 2], n_boot=50)


def test_verify_rejects_a_report_for_another_margin(tmp_path, monkeypatch):
    import json
    from scripts.mde import main
    report = compute(pilot_rows(), [0, 1, 2], n_boot=50, margin_points=0.5, test_clusters=TEST_CLUSTERS)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["mde.py", "--verify", str(path), "--margin-points", "1.0"])
    with pytest.raises(SystemExit, match="margin"):
        main()


def test_mde_fails_if_a_seed_is_missing():
    rows = [row for row in pilot_rows() if not row["method"].endswith("_s2")]
    with pytest.raises(ValueError, match="incomplete seed coverage"):
        compute(rows, [0, 1, 2], n_boot=50, test_clusters=TEST_CLUSTERS)
