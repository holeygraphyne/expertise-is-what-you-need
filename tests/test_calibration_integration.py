"""Integration smoke tests — exercise CLI wiring end-to-end."""
import subprocess
import sys


def test_cli_calibrate_judge_help_works():
    result = subprocess.run(
        [sys.executable, "-m", "traces", "calibrate", "judge", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "--run-id" in result.stdout
    assert "--starred-only" in result.stdout
    assert "--all" in result.stdout
    assert "--judge-model" in result.stdout


def test_cli_calibrate_recommend_help_works():
    result = subprocess.run(
        [sys.executable, "-m", "traces", "calibrate", "recommend", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "--run-id" in result.stdout
    assert "--proposer-model" in result.stdout


def test_cli_calibrate_help_lists_both_subcommands():
    result = subprocess.run(
        [sys.executable, "-m", "traces", "calibrate", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "judge" in result.stdout
    assert "recommend" in result.stdout


def test_cli_top_level_help_lists_calibrate():
    result = subprocess.run(
        [sys.executable, "-m", "traces", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "calibrate" in result.stdout
