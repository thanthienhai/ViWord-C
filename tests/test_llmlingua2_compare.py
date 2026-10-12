from pathlib import Path

from scripts.compare_llmlingua2 import passes_thresholds, report_passes
from scripts.gate_report import choose_syllable_method, summarize_post_training_cbr


def _row(mask_exact_rate=0.95, cbr_diff_points=0.99):
    return {"mask_exact_rate": mask_exact_rate, "cbr_diff_points": cbr_diff_points}


def test_f2_thresholds_are_inclusive_for_mask_and_strict_for_cbr():
    assert passes_thresholds(_row())
    assert not passes_thresholds(_row(mask_exact_rate=0.949))
    assert not passes_thresholds(_row(cbr_diff_points=1.0))


def test_f2_failure_switches_g1_to_package_method():
    passing = {"task": "belebele", "passed": True, "per_ratio": {"0.333": _row()}}
    failing = {"task": "vinli", "passed": False, "per_ratio": {"0.333": _row(0.8, 3.0)}}
    assert choose_syllable_method("llmlingua2_syl", "llmlingua2", [passing]) == (
        "llmlingua2_syl", True
    )
    assert choose_syllable_method("llmlingua2_syl", "llmlingua2", [passing, failing]) == (
        "llmlingua2", True
    )
    assert report_passes({"per_ratio": {"0.333": _row()}})


def test_diagnose_wires_both_arms_and_gate_report():
    text = (Path(__file__).parents[1] / "scripts" / "run_pipeline.sh").read_text(encoding="utf-8")
    assert '"scored:model=$LL2,unit=syllable,name=llmlingua2_syl"' in text
    assert "llmlingua2" in text
    assert "scripts/compare_llmlingua2.py" in text
    assert "--llmlingua2-compare" in text
    assert 'name=llmlingua2vi_s$s' in text
    assert "--llmlingua2vi-reports" in text


def test_post_training_cbr_gate_pools_trained_seed_reports():
    reports = [
        {"task": "belebele", "methods": {"llmlingua2vi_s0": {
            "0.333": {"cbr": {"all": 0.04, "n_all": 100}}
        }}},
        {"task": "vinli", "methods": {"llmlingua2vi_s1": {
            "0.333": {"cbr": {"all": 0.06, "n_all": 100}}
        }}},
    ]
    summary = summarize_post_training_cbr(reports)
    assert summary["cbr"] == 0.05
    assert summary["passed"]
