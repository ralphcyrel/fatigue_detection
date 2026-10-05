"""Launcher logic without a display: outcome reporting and environment checks."""

import json
import subprocess
import sys

import pytest

pytest.importorskip("tkinter")
import launcher  # noqa: E402


@pytest.fixture
def session_files(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "ENROLL_RESULT", tmp_path / "enroll_result.json")
    monkeypatch.setattr(launcher, "SESSION_LOG", tmp_path / "session.log")
    return tmp_path


def finished(code: int) -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-c", f"import sys; sys.exit({code})"])
    proc.wait()
    return proc


def test_enroll_result_read_from_file(session_files):
    (session_files / "enroll_result.json").write_text(json.dumps(
        {"outcome": "dropped", "exit_code": 7, "message": "m"}))
    s = launcher.Session("Enroll X", [], enroll=True)
    s.proc = finished(7)
    assert s.enroll_result()["outcome"] == "dropped"


def test_enroll_result_falls_back_to_exit_code_and_log(session_files):
    (session_files / "session.log").write_text("ModuleNotFoundError: No module named 'x'\n")
    s = launcher.Session("Enroll X", [], enroll=True)
    s.proc = finished(1)
    result = s.enroll_result()
    assert result["outcome"] == "error" and result["exit_code"] == 1
    assert "ModuleNotFoundError" in result["reasons"]


def test_every_enroll_exit_code_has_text():
    for code in (0, 1, 2, 3, 5, 6, 7):
        assert code in launcher.EXIT_TEXT


def test_api_settings_problem(monkeypatch):
    monkeypatch.setenv("FATIGUE_API_BASE_URL", "http://x/api")
    monkeypatch.setenv("FATIGUE_API_TOKEN", "t")
    assert launcher.api_settings_problem() is None
    monkeypatch.delenv("FATIGUE_API_TOKEN")
    assert "FATIGUE_API_TOKEN" in launcher.api_settings_problem()
