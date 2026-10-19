import json

import pytest

from scripts.teacher_compress import done_rows


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_resume_skips_rows_of_the_same_teacher(tmp_path):
    out = tmp_path / "teacher.jsonl"
    assert done_rows(str(out), "A") == set()  # no file yet
    _write(out, [{"id": "x", "ratio": 0.5, "compressed": "c", "model": "A"}])
    assert done_rows(str(out), "A") == {("x", 0.5)}


@pytest.mark.parametrize("other", [{"model": "B"}, {}])  # another teacher, or no teacher name
def test_resume_refuses_rows_of_another_teacher(tmp_path, other):
    out = tmp_path / "teacher.jsonl"
    _write(out, [{"id": "x", "ratio": 0.5, "compressed": "c", "model": "A"},
                 {"id": "y", "ratio": 0.5, "compressed": "c", **other}])
    with pytest.raises(SystemExit, match="another teacher"):
        done_rows(str(out), "A")


def test_dotenv_fills_only_what_the_environment_lacks(tmp_path, monkeypatch):
    from scripts.teacher_compress import load_dotenv
    env = tmp_path / ".env"
    env.write_text("# comment\nTEACHER_MODEL=GLM-5.2\nexport TEACHER_URL='https://x/v1'\n"
                   "TEACHER_API_KEY_1=abc==def\n", encoding="utf-8")
    monkeypatch.delenv("TEACHER_URL", raising=False)
    monkeypatch.delenv("TEACHER_API_KEY_1", raising=False)
    monkeypatch.setenv("TEACHER_MODEL", "from-shell")
    load_dotenv(str(env))
    import os
    assert os.environ["TEACHER_URL"] == "https://x/v1"
    assert os.environ["TEACHER_API_KEY_1"] == "abc==def"
    assert os.environ["TEACHER_MODEL"] == "from-shell"


def test_api_keys_drop_placeholders_and_duplicates():
    from scripts.teacher_compress import api_keys
    env = {"TEACHER_API_KEY_1": "k1", "TEACHER_API_KEY_2": "k1", "TEACHER_API_KEY_3": "<key-3>",
           "TEACHER_API_KEY_4": "k4"}
    assert api_keys(env) == ["k1", "k4"]
    assert api_keys({}) == ["EMPTY"]  # self-hosted vLLM


def test_truncated_teacher_output_is_an_error_not_a_short_compression():
    from types import SimpleNamespace as NS
    from viword.distill import TruncatedResponse, teacher_compress

    def client(finish_reason, content):
        create = lambda **kw: NS(choices=[NS(finish_reason=finish_reason, message=NS(content=content))])
        return NS(chat=NS(completions=NS(create=create)))

    assert teacher_compress(client("stop", "<think>x</think> một hai"), "m", "một hai ba", 0.5) == "một hai"
    with pytest.raises(TruncatedResponse):
        teacher_compress(client("length", "một"), "m", "một hai ba", 0.5)


def test_dotenv_keys_are_not_sent_to_another_endpoint(monkeypatch):
    from scripts.teacher_compress import api_keys, key_env
    dotenv = {"TEACHER_URL": "https://paid/v1", "TEACHER_API_KEY_1": "paid-key"}
    monkeypatch.setenv("TEACHER_API_KEY_1", "paid-key")
    monkeypatch.delenv("TEACHER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert "paid-key" in api_keys(key_env(dotenv, "https://paid/v1"))
    assert api_keys(key_env(dotenv, "http://self-hosted/v1")) == ["EMPTY"]
