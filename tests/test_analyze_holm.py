from scripts.analyze import (
    DEFAULT_HOLM_FAMILY,
    adjust_for_holm_family,
    holm_family_members,
    parse_holm_family,
)


def test_default_h1_family_is_registered_for_primary_reader():
    family = parse_holm_family(DEFAULT_HOLM_FAMILY)

    assert family == {
        "name": "H1",
        "tasks": ("vimmrc", "belebele"),
        "ratios": (0.5, 0.333, 0.2),
        "reader": "primary",
        "test": "metric",
    }
    assert len(holm_family_members(
        family,
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    )) == 6


def test_holm_adjusts_only_registered_h1_cells_for_primary_reader():
    family = parse_holm_family(DEFAULT_HOLM_FAMILY)
    tests = {
        ("vimmrc", 0.5): {"p_one_sided": 0.01},
        ("vimmrc", 0.333): {"p_one_sided": 0.02},
        ("belebele", 0.2): {"p_one_sided": 0.03},
        ("vinli", 0.333): {"p_one_sided": 0.001},
        ("vietnews", 0.333): {"p_one_sided": 0.002},
    }

    adjusted, cells = adjust_for_holm_family(
        tests,
        family,
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    )

    assert set(adjusted) == {
        ("vimmrc", 0.5),
        ("vimmrc", 0.333),
        ("belebele", 0.2),
    }
    assert ("vinli", 0.333) not in cells
    assert ("vietnews", 0.333) not in adjusted


def test_non_primary_reader_has_no_h1_adjustment():
    family = parse_holm_family(DEFAULT_HOLM_FAMILY)
    tests = {("vimmrc", 0.5): {"p_one_sided": 0.01}}

    adjusted, cells = adjust_for_holm_family(
        tests,
        family,
        "other-reader",
        "main-reader",
    )

    assert adjusted == {}
    assert cells == set()


def test_h2_run_does_not_reuse_the_h1_family(tmp_path, monkeypatch, capsys):
    import json
    from scripts.analyze import main
    rows = []
    for method in ("viword_pi", "viword"):
        for i in range(4):
            rows.append({"task": "vimmrc", "reader": "r", "method": method, "ratio": 0.5,
                         "id": str(i), "cluster": f"c{i}", "accuracy": float(method == "viword_pi"),
                         "tokens_per_budget": 1.0})
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["analyze.py", "--rows", str(path), "--target", "viword_pi",
                                     "--reference", "viword", "--hypothesis", "H2", "--n-boot", "50"])
    main()
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("| vimmrc | 0.5 |"))
    assert line.split("|")[5].strip() == "—"  # no Holm adjustment outside the active family


def test_unknown_hypothesis_without_family_is_an_error(monkeypatch, tmp_path):
    import pytest
    from scripts.analyze import main
    path = tmp_path / "rows.jsonl"
    path.write_text("", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["analyze.py", "--rows", str(path), "--hypothesis", "H9"])
    with pytest.raises(SystemExit):
        main()
