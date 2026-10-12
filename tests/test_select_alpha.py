import pytest

from scripts.select_alpha import select


def _rows(gains):
    """gains[(cost, alpha)] = metric gain over α = 0 on every task/ratio/seed."""
    rows = []
    for task, metric in (("vinli", "accuracy"), ("vietnews", "rougeL")):
        for ratio in (0.5, 0.333, 0.2):
            for s in (0, 1, 2):
                base = {"task": task, "reader": "r", "ratio": ratio, "id": "x", "cluster": "x"}
                rows.append({**base, "method": f"viword_pi_s{s}", metric: 0.5})
                for (cost, alpha), g in gains.items():
                    tag = f"{alpha:g}".replace(".", "p")
                    rows.append({**base, "method": f"viword_h3_a{tag}_c{cost}_s{s}", metric: 0.5 + g})
    return rows


def test_alpha_star_maximizes_dev_gain_per_cost():
    decision = select(_rows({("tokens", 0.25): 0.01, ("tokens", 0.5): 0.03,
                             ("syllables", 0.25): 0.02, ("syllables", 0.5): -0.01}))
    assert decision["alpha_tokens"] == 0.5 and decision["alpha_syllables"] == 0.25
    best = next(c for c in decision["candidates"] if c["cost"] == "tokens" and c["alpha"] == 0.5)
    assert best["gain_points"] == pytest.approx(3.0)


def test_alpha_zero_wins_when_no_candidate_helps():
    decision = select(_rows({("tokens", 0.5): -0.02, ("syllables", 0.5): 0.0}))
    assert decision["alpha_tokens"] == 0.0
    assert decision["alpha_syllables"] == 0.0  # tie with α = 0 goes to the smaller α


def test_missing_seed_is_an_error():
    rows = [r for r in _rows({("tokens", 0.5): 0.01}) if not r["method"].endswith("ctokens_s2")]
    with pytest.raises(ValueError, match="incomplete seeds"):
        select(rows)
