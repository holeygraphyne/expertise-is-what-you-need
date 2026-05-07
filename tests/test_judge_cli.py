"""Smoke tests for judge argparse wiring."""
import sys

import pytest


def test_score_judge_help_runs(capsys, monkeypatch):
    from traces.__main__ import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["score", "judge", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--run-id" in out
    assert "--max-cost" in out
    assert "--sample" in out
    assert "--starred-only" in out
    assert "--verbose" in out
    assert "--debug" in out


def test_judge_is_help_runs(capsys):
    from traces.__main__ import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["judge", "is", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--run-id" in out
    assert "--max-cost" in out
    assert "--sample" in out


def test_score_subcommand_present(capsys):
    from traces.__main__ import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["score", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "judge" in out
    assert "promote-labels" in out


def test_top_level_help_lists_judge(capsys):
    from traces.__main__ import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "judge" in out


def test_score_judge_sample_help(capsys):
    """--sample and --sample-seed both appear in `traces score judge --help`."""
    from traces.__main__ import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["score", "judge", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--sample" in out
    assert "--sample-seed" in out


def test_score_judge_sweep_id_mutually_exclusive_with_run_id():
    """Passing both --sweep-id and --run-id to cmd_score_judge raises CliError."""
    from traces.__main__ import build_parser, cmd_score_judge
    from traces.cli_support import CliError

    parser = build_parser()
    args = parser.parse_args([
        "score", "judge",
        "--sweep-id", "my-sweep",
        "--run-id", "my-run",
    ])
    with pytest.raises(CliError, match="mutually exclusive"):
        cmd_score_judge(args)
