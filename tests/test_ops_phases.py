"""The ops scripts must stop between the dev decisions and the test evaluation.

stage_eval refuses an uncommitted π (require_committed_pi), so select_pi and eval cannot run in
one go on the GPU machine: the project owner commits config/pi_selected.* in between.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

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
    assert "pull_code" in update and "--ff-only" in _body(OPS, "pull_code")
    assert "git ls-files --error-unmatch config/pi_selected.json config/pi_selected.env" in update


def test_a_ckpt_check_fail_ends_phase1_but_a_missing_report_stops_it():
    phase1 = _body(FULL, "phase1")
    assert "run_gate ckpt_check results/ckpt_check/report.json ckpt_check" in phase1
    assert phase1.index("run_gate ckpt_check") < phase1.index("Phase 1 done")
    gate = _body(FULL, "run_gate")
    assert 'elif [ -s "$report" ]' in gate and "FAILED_STEPS" in gate and "exit 1" in gate
    # part2 also accepts a part1 that ran older code and stopped at a ckpt_check FAIL with its report
    ended = _body(OPS, "phase1_ended_at_ckpt_check_gate")
    assert '"train done"' in ended and '"ckpt_check FAILED"' in ended and "results/ckpt_check/report.json" in ended
    assert "phase1_ended_at_ckpt_check_gate" in _body(OPS, "part2")


@pytest.mark.skipif(sys.platform == "win32" or not shutil.which("bash"), reason="needs a POSIX bash")
def test_run_gate_tells_a_gate_fail_from_a_broken_stage(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "results/full").mkdir(parents=True)
    # the fake stage exits with $STAGE_EXIT and writes the report when $WRITE_REPORT=1
    (tmp_path / "scripts/run_pipeline.sh").write_text(
        'mkdir -p results/ckpt_check; [ "$WRITE_REPORT" = 1 ] && echo "{}" > results/ckpt_check/report.json\n'
        'exit "$STAGE_EXIT"\n', encoding="utf-8")
    script = ('OUT=results/full\nlog() { echo "$*" >> "$OUT/full.log"; }\n' + "run_gate() {\n"
              + _body(FULL, "run_gate") + "}\nrun_gate ckpt_check results/ckpt_check/report.json ckpt_check\n"
              "echo reached\n")
    (tmp_path / "run.sh").write_text(script, encoding="utf-8")

    def run(stage_exit, write_report):
        for f in ("results/ckpt_check/report.json", "results/full/FAILED_STEPS"):
            (tmp_path / f).unlink(missing_ok=True)
        env = {**os.environ, "STAGE_EXIT": str(stage_exit), "WRITE_REPORT": str(write_report)}
        return subprocess.run(["bash", "run.sh"], cwd=tmp_path, env=env, capture_output=True, text=True)

    ok = run(0, 1)
    assert ok.returncode == 0 and "reached" in ok.stdout
    gate_fail = run(1, 1)
    assert gate_fail.returncode == 0 and "reached" in gate_fail.stdout
    assert "ckpt_check (gate FAIL)" in (tmp_path / "results/full/FAILED_STEPS").read_text()
    broken = run(1, 0)
    assert broken.returncode == 1 and "reached" not in broken.stdout


def test_part2_takes_the_latest_code_before_the_dev_decisions():
    # a dev rule fixed after part1 cloned the code (e.g. select_pi's fallback) must reach part2
    part2 = _body(OPS, "part2")
    assert '"$OUT/.done_update_code_part2" "$OUT/.done_pull_data_part2" "$OUT/.done_check_data_part2"' in part2
    # data/SHA256SUMS moves with the code, so the data is pulled again before it is checked
    assert part2.index("run update_code_part2 pull_code") < part2.index("run pull_data_part2 pull_data") \
        < part2.index("run check_data_part2 check_data") < part2.index("full_h100.sh phase2")


def test_the_test_run_uses_the_committed_ni_margin():
    pipeline = (SCRIPTS / "run_pipeline.sh").read_text(encoding="utf-8")
    assert "MDE_MARGIN=${MDE_MARGIN:-$NI_MARGIN_REGISTERED}" in pipeline
    assert _body(pipeline, "analyze_hypotheses").count('--margin "$MDE_MARGIN"') == 2  # H1 and H2 runs
    for stage in ("stage_eval", "stage_analyze"):
        assert "require_registered_margin" in _body(pipeline, stage)


def test_gate_reports_are_collected_on_failure():
    run_stages = _body(FULL, "run_stages")
    failed = run_stages[:run_stages.index("exit 1")]
    assert '"collect_$name"' in failed


def test_the_gpu_machine_needs_no_java():
    # BERTScore (VnCoreNLP needs Java) runs on the project owner's machine on the eval rows of part 3
    assert "bertscore" not in _body(FULL, "phase3") and "vncorenlp" not in _body(OPS, "part1")
    assert "command -v java" not in OPS
    assert 'cp results/eval/*.jsonl "$OUT/eval"/' in _body(FULL, "phase3")
    local = (SCRIPTS / "bertscore_local.sh").read_text(encoding="utf-8")
    assert "scripts/bertscore.py" in local and "VIETNEWS=${VIETNEWS:-data/tasks/vietnews_test.jsonl}" in local
    assert "export JAVA_HOME=${JAVA_HOME:-" in local  # a JRE alone: pyjnius otherwise looks for javac


def test_every_package_has_a_checksum_and_the_second_reader_comes_back():
    pack, part4 = _body(OPS, "pack"), _body(OPS, "part4")
    assert 'sha256sum "$(basename "$tgz")" > "$(basename "$tgz").sha256"' in pack
    assert "scripts/hf_results.py\" push" in pack and "Send back:" in pack
    assert 'grep -q "Phase 3 done"' in part4 and "update_code" not in part4  # same code and π as part3
    assert 'env READER="$READER2" bash scripts/run_pipeline.sh eval' in part4
    assert part4.index("run_pipeline.sh eval") < part4.index('"$OUT/eval"') < part4.index("pack part4")
