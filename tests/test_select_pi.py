import json

import pytest

from scripts.select_pi import select, write_outputs


def result(task, method, ratio, example_id, *, rouge=0.8, prediction="A"):
    return {
        "task": task,
        "reader": "reader",
        "method": method,
        "ratio": ratio,
        "id": example_id,
        "cluster": example_id,
        "rougeL": rouge,
        "prediction": prediction,
    }


def selection_rows():
    rows = []
    for task in ("vinli", "vietnews"):
        for example_id in ("a", "b"):
            rows.append(result(task, "none", 1.0, example_id, prediction="A"))
    for ratio in (0.5, 0.333, 0.2):
        for example_id in ("a", "b"):
            rows.append(result("vietnews", "viword_s0", ratio, example_id, rouge=0.8))
            rows.append(result("vinli", "viword_s0", ratio, example_id, prediction="A"))

            # Soft: feasible but one of four ViNLI pairs flips.
            rows.append(result("vietnews", "viword_pi_soft_T1_d0p1_s0", ratio, example_id, rouge=0.796))
            rows.append(result("vinli", "viword_pi_soft_T1_d0p1_s0", ratio, example_id,
                               prediction="B" if ratio == 0.5 and example_id == "a" else "A"))

            # Hard: better NFR and still within the 0.5-point ROUGE margin.
            rows.append(result("vietnews", "viword_pi_hard_T1_d0_s0", ratio, example_id, rouge=0.798))
            rows.append(result("vinli", "viword_pi_hard_T1_d0_s0", ratio, example_id, prediction="A"))

            # Lower NFR but infeasible ROUGE loss (0.6 points).
            rows.append(result("vietnews", "viword_pi_soft_T1_T2_d0p2_s0", ratio, example_id, rouge=0.794))
            rows.append(result("vinli", "viword_pi_soft_T1_T2_d0p2_s0", ratio, example_id, prediction="A"))
    return rows


def test_select_pi_applies_rouge_constraint_before_nfr():
    decision = select(selection_rows())
    assert decision["selected"]["mode"] == "hard"
    assert decision["selected"]["tiers"] == ["T1"]
    assert decision["selected"]["delta"] == 0.0
    assert decision["selected"]["rouge_loss_points"] == pytest.approx(0.2)
    assert decision["selected"]["nfr"] == 0.0
    infeasible = next(row for row in decision["candidates"] if row["tiers"] == ["T1", "T2"])
    assert infeasible["feasible"] is False


def test_select_pi_rejects_a_large_loss_at_one_ratio_even_if_the_mean_is_small():
    rows = []
    for task in ("vinli", "vietnews"):
        for example_id in ("a", "b"):
            rows.append(result(task, "none", 1.0, example_id))
    for ratio in (0.5, 0.333, 0.2):
        for example_id in ("a", "b"):
            rows.append(result("vietnews", "viword_s0", ratio, example_id, rouge=0.8))
            rows.append(result("vinli", "viword_s0", ratio, example_id))
            # mean loss 0.4 points, but 1.2 points at 1/5
            rows.append(result("vietnews", "viword_pi_hard_T1_d0_s0", ratio, example_id,
                               rouge=0.788 if ratio == 0.2 else 0.8))
            rows.append(result("vinli", "viword_pi_hard_T1_d0_s0", ratio, example_id))
            rows.append(result("vietnews", "viword_pi_soft_T1_d0p1_s0", ratio, example_id, rouge=0.798))
            rows.append(result("vinli", "viword_pi_soft_T1_d0p1_s0", ratio, example_id,
                               prediction="B" if example_id == "a" else "A"))
    decision = select(rows)
    hard = next(row for row in decision["candidates"] if row["mode"] == "hard")
    assert hard["rouge_loss_points"] == pytest.approx(0.4)
    assert hard["feasible"] is False
    assert decision["selected"]["mode"] == "soft"


def test_select_pi_writes_json_report_and_env(tmp_path):
    decision = select(selection_rows())
    out_json, out_report, out_env = (tmp_path / name for name in ("selected.json", "report.md", "selected.env"))
    write_outputs(decision, str(out_json), str(out_report), str(out_env))
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["selected"]["mode"] == "hard"
    assert "PI_MODE=hard" in out_env.read_text(encoding="utf-8")
    assert "selected" in out_report.read_text(encoding="utf-8")


def test_select_pi_aggregates_the_same_grid_point_across_seeds():
    rows = selection_rows()
    for row in list(rows):
        if row["method"] == "none":
            continue
        clone = dict(row)
        clone["method"] = clone["method"].replace("_s0", "_s1")
        # Make seed 1 prefer soft; selection must still report one aggregate decision.
        if clone["method"].startswith("viword_pi_hard") and clone["task"] == "vinli":
            clone["prediction"] = "B" if clone["id"] == "a" else "A"
        if clone["method"].startswith("viword_pi_soft_T1_d0p1") and clone["task"] == "vinli":
            clone["prediction"] = "A"
        rows.append(clone)
    decision = select(rows)
    assert decision["selected"]["seeds"] == ["0", "1"]
    assert decision["selected"]["mode"] == "soft"
    assert decision["selected"]["nfr"] == pytest.approx(1 / 12)
    assert decision["selected"]["n_nfr_pairs"] == 12


def test_select_pi_fails_when_every_candidate_breaks_rouge_margin():
    rows = selection_rows()
    for row in rows:
        if row["method"].startswith("viword_pi_") and row["task"] == "vietnews":
            row["rougeL"] = 0.7
    with pytest.raises(ValueError, match="no π configuration"):
        select(rows)
