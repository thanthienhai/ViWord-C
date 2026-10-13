"""The ops scripts must stop between the dev decisions and the test evaluation.

stage_eval refuses an uncommitted π (require_committed_pi), so select_pi and eval cannot run in
one go on the GPU machine: the project owner commits config/pi_selected.* in between.
"""
import re
from pathlib import Path

SCRIPTS = Path(__file__).parents[1] / "scripts"
FULL = (SCRIPTS / "full_h100.sh").read_text(encoding="utf-8")
OPS = (SCRIPTS / "ops_full_run.sh").read_text(encoding="utf-8")
HF = (SCRIPTS / "hf_data.py").read_text(encoding="utf-8")


def _body(text: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{[^\n]*\n(.*?)^\}}", text, re.S | re.M)
    assert match, name
    return match.group(1)


def test_select_pi_and_eval_are_in_different_phases():
    phase2, phase3 = _body(FULL, "phase2"), _body(FULL, "phase3")
    assert "select_pi" in phase2 and "mde" in phase2
    assert "run_stages eval" not in phase2
    assert "run_stages eval eval" in phase3 and "select_pi" not in phase3


def test_teacher_test_is_not_run_on_the_gpu_machine():
    assert "run_stages teacher_test" not in FULL
    assert '"results/teacher_test": "teacher_test"' in HF


def test_part3_takes_the_committed_pi_and_fresh_data():
    part3, update = _body(OPS, "part3"), _body(OPS, "update_code")
    assert part3.index("update_code") < part3.index("pull_data") < part3.index("check_data")
    assert "full_h100.sh phase3" in part3
    assert "--ff-only" in update
    assert "git ls-files --error-unmatch config/pi_selected.json config/pi_selected.env" in update


def test_gate_reports_are_collected_on_failure():
    run_stages = _body(FULL, "run_stages")
    failed = run_stages[:run_stages.index("exit 1")]
    assert '"collect_$name"' in failed
