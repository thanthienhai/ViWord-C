import json

from scripts.make_annotation_sheets import probe_sheet, protect_sheet


def _write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _fixture(tmp_path):
    seg, tasks = tmp_path / "segmented", tmp_path / "tasks"
    seg.mkdir()
    tasks.mkdir()
    for source in ("vinli_dev", "vinli_train"):
        task_rows, seg_rows = [], []
        for i in range(6):
            negated = i % 2 == 0
            words = [["Trời", "N", "O"], ["không" if negated else "rất", "R", "O"], ["mưa", "V", "O"]]
            task_rows.append({"id": f"{source}-{i}", "cluster": f"{source}-p{i}",
                              "premise": " ".join(w[0] for w in words)})
            seg_rows.append({"id": f"{source}-{i}", "doc": [words]})
        _write(tasks / f"{source}.jsonl", task_rows)
        _write(seg / f"{source}.jsonl", seg_rows)
    _write(tasks / "vietnews_dev.jsonl", [{"id": "n0", "cluster": "n0"}])
    _write(seg / "vietnews_dev.jsonl", [{"id": "n0", "doc": [[["Hà Nội", "Np", "B-LOC"]], [["đẹp", "A", "O"]]]}])
    return str(seg), str(tasks)


def test_protect_sheet_never_contains_detector_output(tmp_path):
    seg, tasks = _fixture(tmp_path)
    rows = protect_sheet(seg, tasks, n_per_source=1, seed=0)
    assert len(rows) == 2
    for row in rows:
        assert row["sheet"] == "blind-v1" and row["gold_tiers"] is None and row["annotator"] == ""
        assert all(len(w) == 4 for w in row["words"])  # index, text, POS, NER — no tier
    news = next(r for r in rows if r["source"] == "vietnews_dev")
    assert news["sentence"] in (0, 1) and len(news["words"]) == 1


def test_probe_sheet_mixes_edited_and_unedited_distinct_premises(tmp_path):
    seg, tasks = _fixture(tmp_path)
    rows = probe_sheet(seg, tasks, n_per_group=4, seed=0)
    assert len(rows) == 8 and len({r["cluster"] for r in rows}) == 8
    assert sum("không" in r["premise"] for r in rows) == 4
    assert all(r["gold_negation"] is None for r in rows)
