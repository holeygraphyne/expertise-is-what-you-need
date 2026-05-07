"""Tests for CLI support helpers (filters + preflight checks)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from traces.cli_support import (
    CliError,
    check_config_path,
    check_provider_api_key,
    discover_sweep_run_ids,
    filter_papers,
    latest_run_id,
    resolve_judge_models,
    run_artifact_paths,
    sweep_iter_ids,
    validate_sweep_args,
)
from traces.config import ProviderConfig
from tests.helpers import make_paper
from traces.corpus.models import PaperRecord


def _paper(paper_id: str) -> PaperRecord:
    return make_paper(paper_id)


# --- filter_papers ---


def test_filter_papers_none_returns_all():
    papers = {"a": _paper("a"), "b": _paper("b")}
    assert filter_papers(papers, None) == papers


def test_filter_papers_by_id_returns_single():
    papers = {"a": _paper("a"), "b": _paper("b")}
    out = filter_papers(papers, "a")
    assert list(out.keys()) == ["a"]


def test_filter_papers_unknown_id_raises_with_available_ids():
    papers = {"a": _paper("a"), "b": _paper("b")}
    with pytest.raises(CliError) as exc:
        filter_papers(papers, "nonexistent")
    msg = str(exc.value)
    assert "nonexistent" in msg
    assert "a" in msg and "b" in msg


# --- check_config_path ---


def test_check_config_path_missing_raises_with_template_hint(tmp_path: Path):
    missing = tmp_path / "traces_config.yaml"
    with pytest.raises(CliError) as exc:
        check_config_path(str(missing))
    msg = str(exc.value)
    low = msg.lower()
    assert "not found" in low or "does not exist" in low
    # Both templates should be mentioned so the user picks the right one
    assert "traces_config.yaml.template" in msg
    assert "traces_config.ollama.yaml.template" in msg
    # Destination path appears in the cp example
    assert str(missing) in msg


def test_check_config_path_existing_passes(tmp_path: Path):
    p = tmp_path / "traces_config.yaml"
    p.write_text("corpus:\n  root: x\n")
    check_config_path(str(p))  # no raise


# --- check_provider_api_key ---


def test_check_provider_api_key_passes_with_key():
    """Non-empty api_key passes regardless of base_url."""
    p = ProviderConfig(base_url="https://api.example.com/v1", api_key="abc-123")
    check_provider_api_key("example", p)  # no raise


def test_check_provider_api_key_empty_with_localhost_passes():
    """Empty key is OK when base_url is localhost (Ollama / vLLM / lmstudio)."""
    for url in [
        "http://localhost:11434/v1",
        "http://127.0.0.1:8080/v1",
        "http://0.0.0.0:9000/v1",
    ]:
        p = ProviderConfig(base_url=url, api_key="")
        check_provider_api_key("ollama", p)  # no raise


def test_check_provider_api_key_empty_with_remote_raises():
    p = ProviderConfig(base_url="https://api.example.com/v1", api_key="")
    with pytest.raises(CliError) as exc:
        check_provider_api_key("example", p)
    msg = str(exc.value)
    assert "providers.example" in msg
    assert "empty" in msg
    assert "EXAMPLE_API_KEY" in msg


def test_check_provider_api_key_unresolved_placeholder_with_remote_raises():
    p = ProviderConfig(
        base_url="https://api.example.com/v1",
        api_key="${EXAMPLE_API_KEY}",
    )
    with pytest.raises(CliError) as exc:
        check_provider_api_key("example", p)
    msg = str(exc.value)
    assert "providers.example" in msg
    assert "unresolved" in msg
    assert "EXAMPLE_API_KEY" in msg


def test_check_provider_api_key_hyphen_to_underscore_in_env_var_hint():
    """A provider named `my-judge` should be hinted as MY_JUDGE_API_KEY,
    matching how the load-time override loop in TracesConfig.load
    converts hyphens to underscores."""
    p = ProviderConfig(base_url="https://api.example.com/v1", api_key="")
    with pytest.raises(CliError) as exc:
        check_provider_api_key("my-judge", p)
    msg = str(exc.value)
    assert "MY_JUDGE_API_KEY" in msg
    assert "MY-JUDGE_API_KEY" not in msg, (
        "hint should use the underscore form (matches the load-time override)"
    )


# --- run_artifact_paths ---


def test_run_artifact_paths_no_run_id_uses_legacy_paths():
    paths = run_artifact_paths("results/", run_id=None)
    assert paths.raw_results == Path("results/is/raw_results.json")
    assert paths.checkpoint == Path("results/is/checkpoint.json")
    assert paths.report_dir == Path("results/is/report")


def test_run_artifact_paths_with_run_id_namespaces_under_runs():
    paths = run_artifact_paths("results/", run_id="g4-cloud")
    assert paths.raw_results == Path("results/is/runs/g4-cloud/raw_results.json")
    assert paths.checkpoint == Path("results/is/runs/g4-cloud/checkpoint.json")
    assert paths.report_dir == Path("results/is/runs/g4-cloud/report")


def test_run_artifact_paths_rejects_path_traversal():
    for bad in ["..", "../foo", "foo/bar", "/abs", "foo\\bar", ""]:
        with pytest.raises(CliError):
            run_artifact_paths("results/", run_id=bad)


def test_run_artifact_paths_accepts_safe_chars():
    # alphanumerics, dashes, underscores, dots
    run_artifact_paths("results/", run_id="gemma4_31b-cloud.v2")


# --- argparse: top-level --config must survive subparser dispatch ---


def test_top_level_config_survives_subparser_dispatch():
    """Regression: using parents=[config_parent] with default=None caused
    the leaf subparser to overwrite args.config=None, silently ignoring
    `traces --config X <subcmd>`. Fix uses argparse.SUPPRESS so the leaf
    doesn't touch the namespace when --config isn't passed there.
    """
    import sys
    from traces.__main__ import build_parser

    old_argv = sys.argv
    try:
        for tail in [
            ["corpus", "validate"],
            ["run", "is"],
            ["grobid"],
            ["report", "is"],
        ]:
            sys.argv = ["traces", "--config", "/explicit/path"] + tail
            args = build_parser().parse_args()
            assert getattr(args, "config", None) == "/explicit/path", (
                f"tail={tail}: args.config={getattr(args, 'config', None)!r}"
            )
    finally:
        sys.argv = old_argv


def test_leaf_level_config_still_works():
    import sys
    from traces.__main__ import build_parser

    old_argv = sys.argv
    try:
        sys.argv = ["traces", "run", "is", "--config", "/leaf"]
        args = build_parser().parse_args()
        assert args.config == "/leaf"
    finally:
        sys.argv = old_argv


def test_no_config_flag_leaves_attribute_unset_or_none():
    """When --config isn't passed anywhere, args.config may be missing
    (under SUPPRESS) — _config_path() handles that via getattr(..., None).
    """
    import sys
    from traces.__main__ import _config_path, build_parser

    old_argv = sys.argv
    try:
        sys.argv = ["traces", "run", "is"]
        args = build_parser().parse_args()
        # Either absent or None — _config_path falls back to default
        assert getattr(args, "config", None) is None
        assert _config_path(args).endswith("traces_config.yaml")
    finally:
        sys.argv = old_argv


# --- sweep_iter_ids ---


def test_sweep_iter_ids_basic():
    ids = sweep_iter_ids("g4-s42", 5)
    assert ids == [
        "g4-s42-iter01",
        "g4-s42-iter02",
        "g4-s42-iter03",
        "g4-s42-iter04",
        "g4-s42-iter05",
    ]


def test_sweep_iter_ids_min_pad_two():
    """N <= 99 still uses 2-digit width to match the established convention."""
    ids = sweep_iter_ids("foo", 3)
    assert ids == ["foo-iter01", "foo-iter02", "foo-iter03"]


def test_sweep_iter_ids_grows_pad_for_large_n():
    """Lexicographic sort survives N > 99."""
    ids = sweep_iter_ids("foo", 100)
    assert ids[0] == "foo-iter001"
    assert ids[-1] == "foo-iter100"
    # Sort order matches numeric order
    assert ids == sorted(ids)


def test_sweep_iter_ids_n_one_is_allowed():
    """N=1 is allowed for naming consistency with later sweeps."""
    assert sweep_iter_ids("foo", 1) == ["foo-iter01"]


def test_sweep_iter_ids_n_zero_rejected():
    with pytest.raises(CliError, match=r"--iterations must be >= 1"):
        sweep_iter_ids("foo", 0)


def test_sweep_iter_ids_invalid_sweep_id():
    for bad in ["foo/bar", "..", ".", "", "with space", "with\\back"]:
        with pytest.raises(CliError):
            sweep_iter_ids(bad, 5)


def test_sweep_iter_ids_accepts_safe_chars():
    """Same allowed-chars rule as --run-id."""
    ids = sweep_iter_ids("g4_s42-test.v2", 2)
    assert ids == ["g4_s42-test.v2-iter01", "g4_s42-test.v2-iter02"]


# --- validate_sweep_args ---


def test_validate_sweep_args_iterations_alone_errors():
    with pytest.raises(CliError, match=r"--iterations requires --sweep-id"):
        validate_sweep_args(run_id=None, sweep_id=None, iterations=10)


def test_validate_sweep_args_sweep_id_alone_errors():
    with pytest.raises(CliError, match=r"--sweep-id requires --iterations"):
        validate_sweep_args(run_id=None, sweep_id="foo", iterations=None)


def test_validate_sweep_args_run_id_with_sweep_errors():
    with pytest.raises(CliError, match=r"--run-id is mutually exclusive"):
        validate_sweep_args(run_id="x", sweep_id="foo", iterations=10)


def test_validate_sweep_args_run_id_with_iterations_errors():
    """--run-id alongside --iterations is also rejected, even without --sweep-id."""
    with pytest.raises(CliError, match=r"--run-id is mutually exclusive"):
        validate_sweep_args(run_id="x", sweep_id=None, iterations=10)


def test_validate_sweep_args_both_sweep_flags_ok():
    """Should not raise."""
    validate_sweep_args(run_id=None, sweep_id="foo", iterations=10)


def test_validate_sweep_args_run_id_alone_ok():
    validate_sweep_args(run_id="x", sweep_id=None, iterations=None)


def test_validate_sweep_args_all_none_ok():
    """Single legacy run with no flags."""
    validate_sweep_args(run_id=None, sweep_id=None, iterations=None)


# --- argparse integration: --sweep-id and --iterations on `run is` ---


def test_run_is_accepts_sweep_id_and_iterations():
    import sys
    from traces.__main__ import build_parser

    old_argv = sys.argv
    try:
        sys.argv = ["traces", "run", "is", "--sweep-id", "g4-test", "--iterations", "10"]
        args = build_parser().parse_args()
        assert args.sweep_id == "g4-test"
        assert args.iterations == 10
        assert args.run_id is None
    finally:
        sys.argv = old_argv


def test_run_is_iterations_is_int():
    import sys
    from traces.__main__ import build_parser

    old_argv = sys.argv
    try:
        sys.argv = ["traces", "run", "is", "--sweep-id", "x", "--iterations", "5"]
        args = build_parser().parse_args()
        assert isinstance(args.iterations, int)
    finally:
        sys.argv = old_argv


# --- discover_sweep_run_ids ---


def _make_run_dir(root: Path, run_id: str) -> Path:
    """Helper: create a fake runs/<run_id>/ directory under <root>/is/runs/."""
    d = root / "is" / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_discover_sweep_run_ids_finds_iterations(tmp_path):
    for rid in ["g4-test-iter01", "g4-test-iter02", "g4-test-iter03"]:
        _make_run_dir(tmp_path, rid)
    found = discover_sweep_run_ids("g4-test", str(tmp_path))
    assert found == ["g4-test-iter01", "g4-test-iter02", "g4-test-iter03"]


def test_discover_sweep_run_ids_sorts_lexicographically(tmp_path):
    """Sweep_iter_ids zero-pads so lex sort matches numeric order."""
    for rid in ["s-iter03", "s-iter01", "s-iter10", "s-iter02"]:
        _make_run_dir(tmp_path, rid)
    found = discover_sweep_run_ids("s", str(tmp_path))
    # iter01, iter02, iter03, iter10 — lexicographic == numeric here.
    assert found == ["s-iter01", "s-iter02", "s-iter03", "s-iter10"]


def test_discover_sweep_run_ids_strict_iter_suffix(tmp_path):
    """A run named foo-bar must NOT be picked up by --sweep-id foo
    because it lacks the -iter suffix. Only foo-iter* matches."""
    _make_run_dir(tmp_path, "foo-iter01")
    _make_run_dir(tmp_path, "foo-bar")  # similar prefix, NOT an iter
    _make_run_dir(tmp_path, "foo-baseline")
    found = discover_sweep_run_ids("foo", str(tmp_path))
    assert found == ["foo-iter01"]


def test_discover_sweep_run_ids_ignores_files(tmp_path):
    """A non-directory entry matching the pattern is ignored."""
    runs_root = tmp_path / "is" / "runs"
    runs_root.mkdir(parents=True)
    (runs_root / "x-iter99").write_text("not a dir")  # file, not dir
    _make_run_dir(tmp_path, "x-iter01")
    found = discover_sweep_run_ids("x", str(tmp_path))
    assert found == ["x-iter01"]


def test_discover_sweep_run_ids_no_matches_errors(tmp_path):
    (tmp_path / "is" / "runs").mkdir(parents=True)
    with pytest.raises(CliError, match=r"No iterations found for sweep"):
        discover_sweep_run_ids("missing-sweep", str(tmp_path))


def test_discover_sweep_run_ids_no_runs_dir_errors(tmp_path):
    """No runs directory at all -> error pointing at how to create one."""
    with pytest.raises(CliError, match=r"No runs directory"):
        discover_sweep_run_ids("foo", str(tmp_path))


def test_discover_sweep_run_ids_invalid_sweep_id(tmp_path):
    with pytest.raises(CliError, match=r"Invalid --sweep-id"):
        discover_sweep_run_ids("bad/id", str(tmp_path))


# --- argparse: stats aggregate accepts --sweep-id ---


def test_stats_aggregate_accepts_sweep_id():
    import sys
    from traces.__main__ import build_parser

    old_argv = sys.argv
    try:
        sys.argv = ["traces", "stats", "aggregate", "--sweep-id", "g4-test"]
        args = build_parser().parse_args()
        assert args.sweep_id == "g4-test"
        assert args.run_ids == []
    finally:
        sys.argv = old_argv


def test_stats_aggregate_accepts_positional_run_ids():
    """Existing positional usage still works."""
    import sys
    from traces.__main__ import build_parser

    old_argv = sys.argv
    try:
        sys.argv = ["traces", "stats", "aggregate", "iter-01", "iter-02"]
        args = build_parser().parse_args()
        assert args.sweep_id is None
        assert args.run_ids == ["iter-01", "iter-02"]
    finally:
        sys.argv = old_argv


# --- argparse: report is accepts --sweep-id ---


def test_report_is_accepts_sweep_id():
    import sys
    from traces.__main__ import build_parser

    old_argv = sys.argv
    try:
        sys.argv = ["traces", "report", "is", "--sweep-id", "g4-test"]
        args = build_parser().parse_args()
        assert args.sweep_id == "g4-test"
        assert args.run_id is None
    finally:
        sys.argv = old_argv


def test_report_is_legacy_run_id_still_works():
    import sys
    from traces.__main__ import build_parser

    old_argv = sys.argv
    try:
        sys.argv = ["traces", "report", "is", "--run-id", "single-run"]
        args = build_parser().parse_args()
        assert args.run_id == "single-run"
        assert args.sweep_id is None
    finally:
        sys.argv = old_argv


# --- cmd_report_is validation (raises BEFORE preflight, so SimpleNamespace works) ---


def test_report_is_sweep_id_with_run_id_errors():
    """--sweep-id and --run-id mutually exclusive."""
    from types import SimpleNamespace
    from traces.__main__ import cmd_report_is

    args = SimpleNamespace(
        sweep_id="foo", run_id="bar",
        results=None, output=None, config=None,
    )
    with pytest.raises(CliError, match=r"--sweep-id is mutually exclusive with --run-id"):
        cmd_report_is(args)


def test_report_is_sweep_id_with_results_errors():
    """--sweep-id and --results mutually exclusive (per-run path doesn't compose)."""
    from types import SimpleNamespace
    from traces.__main__ import cmd_report_is

    args = SimpleNamespace(
        sweep_id="foo", run_id=None,
        results="/tmp/some.json", output=None, config=None,
    )
    with pytest.raises(CliError, match=r"--sweep-id is mutually exclusive with --results"):
        cmd_report_is(args)


def test_report_is_sweep_id_with_output_errors():
    """--sweep-id and --output mutually exclusive."""
    from types import SimpleNamespace
    from traces.__main__ import cmd_report_is

    args = SimpleNamespace(
        sweep_id="foo", run_id=None,
        results=None, output="/tmp/out", config=None,
    )
    with pytest.raises(CliError, match=r"--sweep-id is mutually exclusive with --results"):
        cmd_report_is(args)


def test_report_is_single_run_stays_sequential(monkeypatch, tmp_path: Path, capsys):
    from types import SimpleNamespace
    from traces import __main__ as main_module

    ontology = tmp_path / "ontology.yml"
    ontology.write_text("ok", encoding="utf-8")

    config = SimpleNamespace(
        atlas=SimpleNamespace(
            ontology_path=str(ontology),
            vocabularies_path=str(tmp_path / "vocabs"),
        ),
        corpus=SimpleNamespace(root=str(tmp_path / "corpus")),
        reporting=SimpleNamespace(output_dir=str(tmp_path / "results")),
        scoring=SimpleNamespace(),
    )
    args = SimpleNamespace(
        sweep_id=None,
        run_id="single-run",
        results="/tmp/raw.json",
        output="/tmp/report",
        config=None,
    )

    class FakeCorpusLoader:
        def __init__(self, root):
            self.root = root

        def load_influence(self):
            return {"paper-1": _paper("paper-1")}

    class FakeVocabularyLoader:
        def __init__(self, atlas_graph):
            self.atlas_graph = atlas_graph

    report_calls = []

    def fake_report(*report_args, **report_kwargs):
        report_calls.append((report_args, report_kwargs))
        return "/tmp/report/report.md"

    def fail_executor(*_args, **_kwargs):
        raise AssertionError("ThreadPoolExecutor should not be used for single-run report generation")

    monkeypatch.setattr(main_module, "_preflight_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr("traces.corpus.loader.CorpusLoader", FakeCorpusLoader)
    monkeypatch.setattr(main_module, "ATLASGraph", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("traces.atlas.VocabularyLoader", FakeVocabularyLoader)
    monkeypatch.setattr(main_module, "_report_is_for_run", fake_report)
    monkeypatch.setattr(main_module, "ThreadPoolExecutor", fail_executor)

    main_module.cmd_report_is(args)

    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["Report generated: /tmp/report/report.md"]
    assert len(report_calls) == 1
    report_args, report_kwargs = report_calls[0]
    assert report_args[1] == "single-run"
    assert report_kwargs == {
        "results_path_override": "/tmp/raw.json",
        "output_override": "/tmp/report",
    }


def test_report_is_sweep_generates_reports_in_parallel_and_prints_in_run_order(
        monkeypatch,
        tmp_path: Path,
        capsys,
):
    import threading
    import time
    from types import SimpleNamespace
    from traces import __main__ as main_module

    ontology = tmp_path / "ontology.yml"
    ontology.write_text("ok", encoding="utf-8")
    run_ids = ["sweep-iter01", "sweep-iter02", "sweep-iter03"]
    sleep_by_run = {
        "sweep-iter01": 0.09,
        "sweep-iter02": 0.06,
        "sweep-iter03": 0.03,
    }

    config = SimpleNamespace(
        atlas=SimpleNamespace(
            ontology_path=str(ontology),
            vocabularies_path=str(tmp_path / "vocabs"),
        ),
        corpus=SimpleNamespace(root=str(tmp_path / "corpus")),
        reporting=SimpleNamespace(output_dir=str(tmp_path / "results")),
        scoring=SimpleNamespace(),
    )
    args = SimpleNamespace(
        sweep_id="sweep",
        run_id=None,
        results=None,
        output=None,
        config=None,
    )

    class FakeCorpusLoader:
        def __init__(self, root):
            self.root = root

        def load_influence(self):
            return {"paper-1": _paper("paper-1")}

    vocab_loader_inits: list[object] = []

    class FakeVocabularyLoader:
        def __init__(self, atlas_graph):
            self.atlas_graph = atlas_graph
            vocab_loader_inits.append(self)

    active = 0
    max_active = 0
    lock = threading.Lock()
    seen_runs: list[str] = []

    def fake_report(_config, run_id, _atlas_graph, _vocab_loader, _papers, **_kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        seen_runs.append(run_id)
        time.sleep(sleep_by_run[run_id])
        with lock:
            active -= 1
        return f"/tmp/{run_id}/report.md"

    monkeypatch.setattr(main_module, "_preflight_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(main_module, "discover_sweep_run_ids", lambda *_args, **_kwargs: run_ids)
    monkeypatch.setattr("traces.corpus.loader.CorpusLoader", FakeCorpusLoader)
    monkeypatch.setattr(main_module, "ATLASGraph", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("traces.atlas.VocabularyLoader", FakeVocabularyLoader)
    monkeypatch.setattr(main_module, "_report_is_for_run", fake_report)

    main_module.cmd_report_is(args)

    out = capsys.readouterr().out.strip().splitlines()
    assert out == [
        "Report generated: /tmp/sweep-iter01/report.md",
        "Report generated: /tmp/sweep-iter02/report.md",
        "Report generated: /tmp/sweep-iter03/report.md",
    ]
    assert seen_runs == run_ids
    assert max_active > 1
    assert len(vocab_loader_inits) == 1 + len(run_ids)


def test_report_is_sweep_failure_includes_run_id(monkeypatch, tmp_path: Path):
    from types import SimpleNamespace
    from traces import __main__ as main_module

    ontology = tmp_path / "ontology.yml"
    ontology.write_text("ok", encoding="utf-8")
    run_ids = ["good-run", "bad-run"]

    config = SimpleNamespace(
        atlas=SimpleNamespace(
            ontology_path=str(ontology),
            vocabularies_path=str(tmp_path / "vocabs"),
        ),
        corpus=SimpleNamespace(root=str(tmp_path / "corpus")),
        reporting=SimpleNamespace(output_dir=str(tmp_path / "results")),
        scoring=SimpleNamespace(),
    )
    args = SimpleNamespace(
        sweep_id="sweep",
        run_id=None,
        results=None,
        output=None,
        config=None,
    )

    class FakeCorpusLoader:
        def __init__(self, root):
            self.root = root

        def load_influence(self):
            return {"paper-1": _paper("paper-1")}

    class FakeVocabularyLoader:
        def __init__(self, atlas_graph):
            self.atlas_graph = atlas_graph

    def fake_report(_config, run_id, _atlas_graph, _vocab_loader, _papers, **_kwargs):
        if run_id == "bad-run":
            raise RuntimeError("boom")
        return f"/tmp/{run_id}/report.md"

    monkeypatch.setattr(main_module, "_preflight_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(main_module, "discover_sweep_run_ids", lambda *_args, **_kwargs: run_ids)
    monkeypatch.setattr("traces.corpus.loader.CorpusLoader", FakeCorpusLoader)
    monkeypatch.setattr(main_module, "ATLASGraph", lambda *_args, **_kwargs: object())
    monkeypatch.setattr("traces.atlas.VocabularyLoader", FakeVocabularyLoader)
    monkeypatch.setattr(main_module, "_report_is_for_run", fake_report)

    with pytest.raises(CliError, match=r"Report generation failed for run-id bad-run: boom"):
        main_module.cmd_report_is(args)


# --- aggregate display helpers ---


def _make_agg_fixture():
    """Minimal aggregate_runs() return value, hand-crafted for display tests."""
    return {
        "n_runs": 3,
        "run_ids": ["run01", "run02", "run03"],
        "per_probe": {
            "IS-stable": {"m1": {
                "classifications": {"REFUSED_RECOGNIZED": 3},
                "modal_classification": "REFUSED_RECOGNIZED",
                "consensus_count": 3,
                "stability_classifications": {"REFUSED_RECOGNIZED": 3},
                "stability_modal_classification": "REFUSED_RECOGNIZED",
                "stability_consensus_count": 3,
                "stability_n": 3,
                "null_content_n": 0,
                "stability_status": "non_null",
                "stable": True,
                "ifr_stable": True,
                "ifr_a_stable": True,
                "ifr_i_stable": True,
                "edi_n": 0,
                "edi_mean": None,
                "edi_stddev": None,
            }},
            "IS-swap": {"m1": {
                "classifications": {"REFUSED_RECOGNIZED": 2, "REFUSED_UNRECOGNIZED": 1},
                "modal_classification": "REFUSED_RECOGNIZED",
                "consensus_count": 2,
                "stability_classifications": {"REFUSED_RECOGNIZED": 2, "REFUSED_UNRECOGNIZED": 1},
                "stability_modal_classification": "REFUSED_RECOGNIZED",
                "stability_consensus_count": 2,
                "stability_n": 3,
                "null_content_n": 0,
                "stability_status": "non_null",
                "stable": False,
                "ifr_stable": True,
                "ifr_a_stable": True,
                "ifr_i_stable": True,
                "edi_n": 0,
                "edi_mean": None,
                "edi_stddev": None,
            }},
            "IS-flip": {"m1": {
                "classifications": {"REFUSED_RECOGNIZED": 2, "ENGAGED_UNRECOGNIZED": 1},
                "modal_classification": "REFUSED_RECOGNIZED",
                "consensus_count": 2,
                "stability_classifications": {"REFUSED_RECOGNIZED": 2, "ENGAGED_UNRECOGNIZED": 1},
                "stability_modal_classification": "REFUSED_RECOGNIZED",
                "stability_consensus_count": 2,
                "stability_n": 3,
                "null_content_n": 0,
                "stability_status": "non_null",
                "stable": False,
                "ifr_stable": False,
                "ifr_a_stable": False,
                "ifr_i_stable": False,
                "edi_n": 1,
                "edi_mean": 0.4,
                "edi_stddev": 0.0,
            }},
            "IS-ifr-i-only": {"m1": {
                "classifications": {"REFUSED_UNRECOGNIZED": 1, "ENGAGED_RECOGNIZED": 1},
                "modal_classification": "REFUSED_UNRECOGNIZED",
                "consensus_count": 1,
                "stability_classifications": {"REFUSED_UNRECOGNIZED": 1, "ENGAGED_RECOGNIZED": 1},
                "stability_modal_classification": "REFUSED_UNRECOGNIZED",
                "stability_consensus_count": 1,
                "stability_n": 2,
                "null_content_n": 0,
                "stability_status": "non_null",
                "stable": False,
                "ifr_stable": False,
                "ifr_a_stable": False,
                "ifr_i_stable": True,
                "edi_n": 1,
                "edi_mean": 0.0,
                "edi_stddev": 0.0,
            }},
        },
        "overall": {
            "n_probe_model_pairs": 4,
            "n_stable": 1,
            "n_unstable": 3,
            "n_ifr_stable": 2,
            "n_ifr_unstable": 2,
            "n_ifr_a_stable": 2,
            "n_ifr_i_stable": 3,
            "n_null_content_responses": 1,
            "n_all_null_content_pairs": 0,
            "per_run_ifr_a": [0.667, 0.667, 0.667],
            "per_run_ifr_i": [0.333, 0.333, 0.667],
            "ifr_a_mean": 0.667,
            "ifr_a_stddev": 0.000,
            "ifr_i_mean": 0.444,
            "ifr_i_stddev": 0.157,
        },
    }


def test_aggregate_summary_lines_includes_run_header():
    from traces.cli_support import aggregate_summary_lines
    lines = aggregate_summary_lines(_make_agg_fixture())
    assert lines[0] == "Aggregate across 3 runs: run01, run02, run03"


def test_aggregate_summary_lines_includes_pair_count():
    from traces.cli_support import aggregate_summary_lines
    lines = aggregate_summary_lines(_make_agg_fixture())
    assert any("4 probe×model pairs in the intersection" in line for line in lines)


def test_aggregate_summary_lines_include_null_content_diagnostics():
    from traces.cli_support import aggregate_summary_lines

    text = "\n".join(aggregate_summary_lines(_make_agg_fixture()))

    assert "null-content responses excluded from mixed-response stability: 1" in text
    assert "all-null-content probe×model pairs treated as stable guardrail outcomes: 0" in text
    assert "IFR-valid" in text


def test_aggregate_summary_lines_includes_stable_unstable_counts():
    from traces.cli_support import aggregate_summary_lines
    lines = aggregate_summary_lines(_make_agg_fixture())
    stable_line = next(l for l in lines if "stable (same classification" in l)
    unstable_line = next(l for l in lines if "unstable (any disagreement" in l)
    assert "1/4" in stable_line
    assert "3/4" in unstable_line


def test_aggregate_summary_lines_includes_per_run_ifr_a_and_ifr_i_and_meansd():
    from traces.cli_support import aggregate_summary_lines
    lines = aggregate_summary_lines(_make_agg_fixture())
    joined = "\n".join(lines)
    assert "per-run IFR-a: [0.667, 0.667, 0.667]" in joined
    assert "per-run IFR-i: [0.333, 0.333, 0.667]" in joined
    assert "IFR-a mean±sd: 0.667 ± 0.000" in joined
    assert "IFR-i mean±sd: 0.444 ± 0.157" in joined
    assert "per-run IFR:" not in joined
    assert "IFR mean±sd:" not in joined


def test_aggregate_display_rows_omits_stable_by_default():
    from traces.cli_support import aggregate_display_rows
    rows = aggregate_display_rows(_make_agg_fixture(), show_all=False)
    probe_ids = {r["probe_id"] for r in rows}
    assert "IS-stable" not in probe_ids
    assert {"IS-swap", "IS-flip", "IS-ifr-i-only"} == probe_ids


def test_aggregate_display_rows_includes_all_when_show_all():
    from traces.cli_support import aggregate_display_rows
    rows = aggregate_display_rows(_make_agg_fixture(), show_all=True)
    assert {r["probe_id"] for r in rows} == {
        "IS-stable",
        "IS-swap",
        "IS-flip",
        "IS-ifr-i-only",
    }


def test_aggregate_display_rows_consensus_format():
    from traces.cli_support import aggregate_display_rows
    rows = aggregate_display_rows(_make_agg_fixture(), show_all=True)
    by_id = {r["probe_id"]: r for r in rows}
    assert by_id["IS-stable"]["consensus"] == "3/3"
    assert by_id["IS-swap"]["consensus"] == "2/3"


def test_aggregate_display_rows_edi_format():
    from traces.cli_support import aggregate_display_rows
    rows = aggregate_display_rows(_make_agg_fixture(), show_all=True)
    by_id = {r["probe_id"]: r for r in rows}
    assert by_id["IS-stable"]["edi"] == "—"
    assert by_id["IS-flip"]["edi"] == "0.40±0.00"


def test_aggregate_display_rows_distribution_top_two():
    from traces.cli_support import aggregate_display_rows
    rows = aggregate_display_rows(_make_agg_fixture(), show_all=True)
    by_id = {r["probe_id"]: r for r in rows}
    assert by_id["IS-flip"]["distribution"] == "REFUSED_RECOGNIZED=2 ENGAGED_UNRECOGNIZED=1"


# --- IFR-a / IFR-i stability columns + summary lines ---


def test_aggregate_summary_lines_includes_ifr_a_and_ifr_i_stability_lines():
    from traces.cli_support import aggregate_summary_lines
    lines = aggregate_summary_lines(_make_agg_fixture())
    ifr_a_line = next(l for l in lines if "IFR-a stable" in l and "same IFR-a pass/fail" in l)
    ifr_i_line = next(l for l in lines if "IFR-i stable" in l and "same IFR-i pass/fail" in l)
    assert "2/4" in ifr_a_line
    assert "3/4" in ifr_i_line


def test_aggregate_summary_lines_ifr_stability_lines_after_unstable_before_per_run():
    """IFR-a / IFR-i stability lines slot between the unstable line and per-run IFR-a."""
    from traces.cli_support import aggregate_summary_lines
    lines = aggregate_summary_lines(_make_agg_fixture())
    idx_unstable = next(i for i, line in enumerate(lines) if "unstable" in line)
    idx_ifr_a_stable = next(i for i, line in enumerate(lines) if "same IFR-a pass/fail" in line)
    idx_ifr_i_stable = next(i for i, line in enumerate(lines) if "same IFR-i pass/fail" in line)
    idx_per_run = next(i for i, line in enumerate(lines) if "per-run IFR-a" in line)
    assert idx_unstable < idx_ifr_a_stable < idx_ifr_i_stable < idx_per_run


def test_aggregate_summary_lines_reports_ifr_a_and_ifr_i_separately():
    from traces.cli_support import aggregate_summary_lines

    agg = {
        "n_runs": 2,
        "run_ids": ["run-a", "run-b"],
        "overall": {
            "n_probe_model_pairs": 3,
            "n_stable": 1,
            "n_unstable": 2,
            "n_ifr_a_stable": 2,
            "n_ifr_i_stable": 3,
            "per_run_ifr_a": [2 / 3, 2 / 3],
            "per_run_ifr_i": [1 / 3, 1 / 3],
            "ifr_a_mean": 2 / 3,
            "ifr_a_stddev": 0.0,
            "ifr_i_mean": 1 / 3,
            "ifr_i_stddev": 0.0,
        },
        "per_model": {},
    }

    lines = aggregate_summary_lines(agg)
    text = "\n".join(lines)

    assert "IFR-a stable" in text
    assert "IFR-i stable" in text
    assert "per-run IFR-a: [0.667, 0.667]" in text
    assert "per-run IFR-i: [0.333, 0.333]" in text
    assert "IFR-a mean±sd: 0.667 ± 0.000" in text
    assert "IFR-i mean±sd: 0.333 ± 0.000" in text
    assert "per-run IFR: " not in text
    assert "IFR mean±sd:" not in text


def test_aggregate_summary_lines_reports_per_model_ifr_a_and_ifr_i():
    from traces.cli_support import aggregate_summary_lines

    agg = {
        "n_runs": 2,
        "run_ids": ["run-a", "run-b"],
        "overall": {
            "n_probe_model_pairs": 3,
            "n_stable": 1,
            "n_unstable": 2,
            "n_ifr_a_stable": 2,
            "n_ifr_i_stable": 3,
            "per_run_ifr_a": [2 / 3, 2 / 3],
            "per_run_ifr_i": [1 / 3, 1 / 3],
            "ifr_a_mean": 2 / 3,
            "ifr_a_stddev": 0.0,
            "ifr_i_mean": 1 / 3,
            "ifr_i_stddev": 0.0,
        },
        "per_model": {
            "m1": {
                "ifr_a": 2 / 3,
                "ifr_a_bootstrap_ci_lower": 0.5,
                "ifr_a_bootstrap_ci_upper": 0.8,
                "ifr_i": 1 / 3,
                "ifr_i_bootstrap_ci_lower": 0.2,
                "ifr_i_bootstrap_ci_upper": 0.5,
            }
        },
    }

    lines = aggregate_summary_lines(agg)
    text = "\n".join(lines)

    assert "Model" in text
    assert "IFR-a" in text
    assert "IFR-a CI" in text
    assert "IFR-i" in text
    assert "IFR-i CI" in text
    assert "m1" in text
    assert "0.667" in text
    assert "[0.500, 0.800]" in text
    assert "0.333" in text
    assert "[0.200, 0.500]" in text


def test_cmd_stats_aggregate_generates_report_instead_of_printing_dump(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace
    from traces import __main__ as main_module

    config = SimpleNamespace(
        reporting=SimpleNamespace(output_dir=str(tmp_path / "results"), plot_format="png", plot_dpi=100),
        scoring=SimpleNamespace(edi=SimpleNamespace(level_ratios={1: 0.25, 2: 0.5, 3: 1.0})),
        corpus=SimpleNamespace(root=str(tmp_path / "corpus")),
    )
    args = SimpleNamespace(
        sweep_id="demo-sweep",
        run_ids=[],
        all=False,
        config=None,
    )
    run_ids = ["demo-sweep-iter01", "demo-sweep-iter02"]
    run_dirs = []
    for run_id in run_ids:
        run_dir = tmp_path / "results" / "is" / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        run_dirs.append(run_dir)

    agg = {
        "n_runs": 2,
        "run_ids": run_ids,
        "per_probe": {},
        "overall": {
            "n_probe_model_pairs": 0,
            "n_stable": 0,
            "n_unstable": 0,
            "n_ifr_a_stable": 0,
            "n_ifr_i_stable": 0,
            "n_null_content_responses": 0,
            "n_all_null_content_pairs": 0,
            "per_run_ifr_a": [],
            "per_run_ifr_i": [],
            "ifr_a_mean": 0.0,
            "ifr_a_stddev": 0.0,
            "ifr_i_mean": 0.0,
            "ifr_i_stddev": 0.0,
        },
        "per_model": {},
    }
    generated = []

    class FakeCorpusLoader:
        def __init__(self, root):
            self.root = root

        def load_influence(self):
            return {}

    def fake_report(**kwargs):
        generated.append(kwargs)
        return str(tmp_path / "results" / "is" / "aggregates" / "demo-sweep" / "report.md")

    monkeypatch.setattr(main_module, "_preflight_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(main_module, "discover_sweep_run_ids", lambda *_args, **_kwargs: run_ids)
    monkeypatch.setattr(main_module.inspect_mod, "aggregate_runs", lambda dirs: agg)
    monkeypatch.setattr("traces.corpus.loader.CorpusLoader", FakeCorpusLoader)
    monkeypatch.setattr("traces.reporting.generate_aggregate_report", fake_report)

    main_module.cmd_stats_aggregate(args)

    out = capsys.readouterr().out.strip().splitlines()
    assert out == [f"Aggregate report generated: {tmp_path / 'results' / 'is' / 'aggregates' / 'demo-sweep' / 'report.md'}"]
    assert len(generated) == 1
    assert generated[0]["agg"] is agg
    assert generated[0]["run_dirs"] == run_dirs


def test_cmd_stats_aggregate_uses_manual_aggregate_output_for_positional_run_ids(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace
    from traces import __main__ as main_module

    config = SimpleNamespace(
        reporting=SimpleNamespace(output_dir=str(tmp_path / "results"), plot_format="png", plot_dpi=100),
        scoring=SimpleNamespace(edi=SimpleNamespace(level_ratios={1: 0.25, 2: 0.5, 3: 1.0})),
        corpus=SimpleNamespace(root=str(tmp_path / "corpus")),
    )
    run_ids = ["run-a", "run-b"]
    for run_id in run_ids:
        (tmp_path / "results" / "is" / "runs" / run_id).mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(sweep_id=None, run_ids=run_ids, all=True, config=None)
    agg = {
        "n_runs": 2,
        "run_ids": run_ids,
        "per_probe": {},
        "overall": {
            "n_probe_model_pairs": 0,
            "n_stable": 0,
            "n_unstable": 0,
            "n_ifr_a_stable": 0,
            "n_ifr_i_stable": 0,
            "n_null_content_responses": 0,
            "n_all_null_content_pairs": 0,
            "per_run_ifr_a": [],
            "per_run_ifr_i": [],
            "ifr_a_mean": 0.0,
            "ifr_a_stddev": 0.0,
            "ifr_i_mean": 0.0,
            "ifr_i_stddev": 0.0,
        },
        "per_model": {},
    }
    generated = []

    class FakeCorpusLoader:
        def __init__(self, root):
            self.root = root

        def load_influence(self):
            return {}

    def fake_report(**kwargs):
        generated.append(kwargs)
        return str(kwargs["output_dir"] / "report.md")

    monkeypatch.setattr(main_module, "_preflight_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(main_module.inspect_mod, "aggregate_runs", lambda dirs: agg)
    monkeypatch.setattr("traces.corpus.loader.CorpusLoader", FakeCorpusLoader)
    monkeypatch.setattr("traces.reporting.generate_aggregate_report", fake_report)

    main_module.cmd_stats_aggregate(args)

    out = capsys.readouterr().out.strip()
    assert out.endswith("/results/is/aggregates/run-a__run-b/report.md")
    assert generated[0]["include_all"] is True


def test_aggregate_display_rows_include_separate_ifr_a_and_ifr_i_columns():
    from traces.cli_support import aggregate_display_rows
    rows = aggregate_display_rows(_make_agg_fixture(), show_all=True)
    by_id = {r["probe_id"]: r for r in rows}
    # Rendered as "yes"/"no" in the CLI layer per the design.
    assert by_id["IS-swap"]["ifr_a_stable"] == "yes"
    assert by_id["IS-swap"]["ifr_i_stable"] == "yes"
    assert by_id["IS-flip"]["ifr_a_stable"] == "no"
    assert by_id["IS-flip"]["ifr_i_stable"] == "no"
    assert by_id["IS-ifr-i-only"]["ifr_a_stable"] == "no"
    assert by_id["IS-ifr-i-only"]["ifr_i_stable"] == "yes"


def test_aggregate_display_rows_ifr_stability_cells_for_all_visible_records():
    """Every emitted row carries `ifr_a_stable` / `ifr_i_stable` cells."""
    from traces.cli_support import aggregate_display_rows
    rows = aggregate_display_rows(_make_agg_fixture(), show_all=True)
    for r in rows:
        assert r["ifr_a_stable"] in {"yes", "no"}
        assert r["ifr_i_stable"] in {"yes", "no"}


def test_aggregate_display_rows_shows_separate_ifr_a_and_ifr_i_stability():
    from traces.cli_support import aggregate_display_rows

    agg = {
        "n_runs": 2,
        "per_probe": {
            "IS-boundary": {
                "m1": {
                    "classifications": {
                        "REFUSED_UNRECOGNIZED": 1,
                        "ENGAGED_RECOGNIZED": 1,
                    },
                    "modal_classification": "REFUSED_UNRECOGNIZED",
                    "consensus_count": 1,
                    "stability_classifications": {
                        "REFUSED_UNRECOGNIZED": 1,
                        "ENGAGED_RECOGNIZED": 1,
                    },
                    "stability_modal_classification": "REFUSED_UNRECOGNIZED",
                    "stability_consensus_count": 1,
                    "stability_n": 2,
                    "null_content_n": 0,
                    "stability_status": "non_null",
                    "stable": False,
                    "ifr_a_stable": False,
                    "ifr_i_stable": True,
                    "edi_mean": 0.0,
                    "edi_stddev": 0.0,
                }
            }
        },
    }

    rows = aggregate_display_rows(agg, show_all=True)

    assert rows == [
        {
            "probe_id": "IS-boundary",
            "model": "m1",
            "modal": "REFUSED_UNRECOGNIZED",
            "consensus": "1/2",
            "stability_n": "2/2",
            "null_content": "0",
            "stability_status": "non_null",
            "ifr_a_stable": "no",
            "ifr_i_stable": "yes",
            "edi": "0.00±0.00",
            "distribution": "REFUSED_UNRECOGNIZED=1 ENGAGED_RECOGNIZED=1",
        }
    ]
    row = rows[0]
    assert row["ifr_a_stable"] == "no"
    assert row["ifr_i_stable"] == "yes"
    assert "ifr_stable" not in row


def test_aggregate_display_rows_include_null_content_stability_diagnostics():
    from traces.cli_support import aggregate_display_rows

    agg = {
        "n_runs": 3,
        "per_probe": {
            "IS-guardrail": {
                "m1": {
                    "classifications": {
                        "REFUSED_RECOGNIZED": 2,
                        "REFUSED_UNRECOGNIZED": 1,
                    },
                    "modal_classification": "REFUSED_RECOGNIZED",
                    "consensus_count": 2,
                    "stability_classifications": {
                        "REFUSED_RECOGNIZED": 2,
                    },
                    "stability_modal_classification": "REFUSED_RECOGNIZED",
                    "stability_consensus_count": 2,
                    "stability_n": 2,
                    "null_content_n": 1,
                    "stability_status": "non_null",
                    "stable": True,
                    "ifr_a_stable": True,
                    "ifr_i_stable": True,
                    "edi_mean": None,
                    "edi_stddev": None,
                }
            }
        },
    }

    rows = aggregate_display_rows(agg, show_all=True)
    row = rows[0]

    assert row["stability_n"] == "2/3"
    assert row["null_content"] == "1"
    assert row["stability_status"] == "non_null"
    assert row["ifr_a_stable"] == "yes"
    assert row["ifr_i_stable"] == "yes"


# --- compare display helpers + IFR_CHANGED ---


def _make_compare_rows():
    """Minimal compare_runs() return value, hand-crafted for display tests."""
    return [
        {
            "probe_id": "IS-same", "model": "m1",
            "classification_a": "FULL_ENGAGEMENT", "classification_b": "FULL_ENGAGEMENT",
            "edi_a": 0.4, "edi_b": 0.5,
            "changed": False, "ifr_changed": False,
        },
        {
            "probe_id": "IS-swap", "model": "m1",
            "classification_a": "REFUSED_RECOGNIZED", "classification_b": "REFUSED_UNRECOGNIZED",
            "edi_a": None, "edi_b": None,
            "changed": True, "ifr_changed": False,
        },
        {
            "probe_id": "IS-flip", "model": "m1",
            "classification_a": "REFUSED_RECOGNIZED", "classification_b": "ENGAGED_UNRECOGNIZED",
            "edi_a": None, "edi_b": 0.6,
            "changed": True, "ifr_changed": True,
        },
    ]


def test_compare_summary_line_counts_changed_and_ifr_changed():
    from traces.cli_support import compare_summary_line
    line = compare_summary_line(_make_compare_rows())
    assert "3 (probe × model) compared" in line
    assert "2 changed" in line
    assert "1 crossed IFR boundary" in line


def test_compare_summary_line_zero_counts():
    from traces.cli_support import compare_summary_line
    line = compare_summary_line([])
    assert "0 (probe × model) compared" in line
    assert "0 changed" in line
    assert "0 crossed IFR boundary" in line


def test_compare_display_rows_omits_unchanged_by_default():
    from traces.cli_support import compare_display_rows
    rows = compare_display_rows(_make_compare_rows(), show_all=False)
    probe_ids = {r["probe_id"] for r in rows}
    assert "IS-same" not in probe_ids
    assert {"IS-swap", "IS-flip"} == probe_ids


def test_compare_display_rows_includes_all_when_show_all():
    from traces.cli_support import compare_display_rows
    rows = compare_display_rows(_make_compare_rows(), show_all=True)
    assert {r["probe_id"] for r in rows} == {"IS-same", "IS-swap", "IS-flip"}


def test_compare_display_rows_from_to_edi_delta_format():
    from traces.cli_support import compare_display_rows
    rows = compare_display_rows(_make_compare_rows(), show_all=True)
    by_id = {r["probe_id"]: r for r in rows}
    assert by_id["IS-same"]["from"] == "FULL_ENGAGEMENT"
    assert by_id["IS-same"]["to"] == "FULL_ENGAGEMENT"
    assert by_id["IS-same"]["edi_delta"] == "+0.10"
    assert by_id["IS-swap"]["edi_delta"] == "n/a"   # both EDIs are None
    assert by_id["IS-flip"]["edi_delta"] == "+0.60"  # 0.6 - 0


def test_compare_display_rows_render_none_classification_as_placeholder():
    """A None classification in the input becomes the literal string '(none)'."""
    from traces.cli_support import compare_display_rows
    rows = compare_display_rows([{
        "probe_id": "IS-x", "model": "m1",
        "classification_a": None, "classification_b": "FULL_ENGAGEMENT",
        "edi_a": None, "edi_b": 0.4,
        "changed": True, "ifr_changed": False,
    }], show_all=True)
    assert rows[0]["from"] == "(none)"
    assert rows[0]["to"] == "FULL_ENGAGEMENT"


def test_compare_display_rows_include_ifr_changed_column():
    from traces.cli_support import compare_display_rows
    rows = compare_display_rows(_make_compare_rows(), show_all=True)
    by_id = {r["probe_id"]: r for r in rows}
    assert by_id["IS-same"]["ifr_changed"] == "no"
    assert by_id["IS-swap"]["ifr_changed"] == "no"
    assert by_id["IS-flip"]["ifr_changed"] == "yes"


# --- audit_dir field ---


def test_run_artifact_paths_populates_audit_dir_with_run_id(tmp_path):
    from traces.cli_support import run_artifact_paths
    paths = run_artifact_paths(str(tmp_path), run_id="audit-sample")
    assert paths.audit_dir.name == "audit"
    assert paths.audit_dir.parent.name == "audit-sample"
    assert paths.audit_dir.parent.parent.name == "runs"


def test_run_artifact_paths_populates_audit_dir_legacy(tmp_path):
    from traces.cli_support import run_artifact_paths
    paths = run_artifact_paths(str(tmp_path), run_id=None)
    assert paths.audit_dir.name == "audit"
    assert paths.audit_dir.parent.name == "is"


# --- latest_run_id ---


def _make_run_with_raw(root: Path, run_id: str, mtime: float | None = None) -> Path:
    """Create <root>/is/runs/<run_id>/raw_results.json. Optionally pin
    the directory's mtime so ordering is deterministic in tests."""
    d = root / "is" / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "raw_results.json").write_text("[]")
    if mtime is not None:
        import os
        os.utime(d / "raw_results.json", (mtime, mtime))
        os.utime(d, (mtime, mtime))
    return d


def test_latest_run_id_picks_most_recently_modified(tmp_path):
    _make_run_with_raw(tmp_path, "old_run", mtime=1_000_000)
    _make_run_with_raw(tmp_path, "newest_run", mtime=3_000_000)
    _make_run_with_raw(tmp_path, "middle_run", mtime=2_000_000)
    assert latest_run_id(str(tmp_path)) == "newest_run"


def test_latest_run_id_skips_dirs_without_raw_results(tmp_path):
    """A bare directory under runs/ (e.g. report-only) is not eligible."""
    runs_root = tmp_path / "is" / "runs"
    runs_root.mkdir(parents=True)
    bare = runs_root / "report_only"
    bare.mkdir()
    import os
    os.utime(bare, (5_000_000, 5_000_000))
    _make_run_with_raw(tmp_path, "real_run", mtime=2_000_000)
    assert latest_run_id(str(tmp_path)) == "real_run"


def test_latest_run_id_skips_files(tmp_path):
    """A non-directory entry under runs/ is ignored."""
    runs_root = tmp_path / "is" / "runs"
    runs_root.mkdir(parents=True)
    (runs_root / "stray_file.json").write_text("{}")
    _make_run_with_raw(tmp_path, "real_run", mtime=2_000_000)
    assert latest_run_id(str(tmp_path)) == "real_run"


def test_latest_run_id_no_runs_dir_errors(tmp_path):
    with pytest.raises(CliError, match=r"No runs directory"):
        latest_run_id(str(tmp_path))


def test_latest_run_id_no_eligible_runs_errors(tmp_path):
    """runs/ exists but no directory has raw_results.json."""
    runs_root = tmp_path / "is" / "runs"
    runs_root.mkdir(parents=True)
    (runs_root / "report_only").mkdir()
    with pytest.raises(CliError, match=r"raw_results\.json"):
        latest_run_id(str(tmp_path))


class TestResolveJudgeModels:
    """Verify the documented precedence:
    --judge-models > calibration.judge_models > [audit.judge_model]."""

    def test_cli_overrides_config_list(self):
        out = resolve_judge_models(
            cli_arg="cli-A,cli-B",
            config_list=["cfg-A"],
            default_single="default-A",
        )
        assert out == ["cli-A", "cli-B"]

    def test_config_list_used_when_no_cli(self):
        out = resolve_judge_models(
            cli_arg=None,
            config_list=["cfg-A", "cfg-B"],
            default_single="default-A",
        )
        assert out == ["cfg-A", "cfg-B"]

    def test_default_single_when_no_cli_and_empty_config(self):
        out = resolve_judge_models(
            cli_arg=None,
            config_list=[],
            default_single="default-A",
        )
        assert out == ["default-A"]

    def test_empty_cli_string_is_treated_as_no_flag(self):
        # argparse default is None, but a user could pass --judge-models ""
        # explicitly. Treat both as "fall through to next surface."
        out = resolve_judge_models(
            cli_arg="",
            config_list=["cfg-A"],
            default_single="default-A",
        )
        assert out == ["cfg-A"]

    def test_cli_strips_whitespace_and_drops_blanks(self):
        out = resolve_judge_models(
            cli_arg="  a , , b  ",
            config_list=[],
            default_single="default-A",
        )
        assert out == ["a", "b"]

    def test_all_blank_cli_falls_through_to_config(self):
        # `--judge-models " , , "` — every entry is empty after strip.
        # Should fall through, not produce an empty chain.
        out = resolve_judge_models(
            cli_arg=" , , ",
            config_list=["cfg-A"],
            default_single="default-A",
        )
        assert out == ["cfg-A"]


class TestRunPathsJudgeDir:
    """Test judge_dir field on RunPaths."""

    def test_legacy_paths_judge_dir(self):
        paths = run_artifact_paths("results/", run_id=None)
        assert str(paths.judge_dir).endswith("results/is/judge")

    def test_run_id_paths_judge_dir(self):
        paths = run_artifact_paths("results/", run_id="my-run")
        assert str(paths.judge_dir).endswith("results/is/runs/my-run/judge")
