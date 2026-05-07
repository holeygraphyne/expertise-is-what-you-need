"""Tests for inspection helpers (corpus + runs introspection)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.helpers import make_paper
from traces.corpus.models import PaperRecord
from traces.inspect import (
    aggregate_runs,
    compare_runs,
    paper_detail,
    papers_summary,
    run_detail,
    runs_summary,
)


def _paper(paper_id: str, domain: str = "pseudoscience", year: int = 2020) -> PaperRecord:
    return make_paper(
        paper_id,
        domain=domain,
        year=year,
        central_claim=f"The {paper_id} effect is real and reproducible.",
    )


# --- papers_summary ---


def test_papers_summary_returns_one_row_per_paper():
    papers = {"a": _paper("a", "pseudoscience", 2018), "b": _paper("b", "fringe_physics", 2020)}
    rows = papers_summary(papers)
    assert len(rows) == 2
    by_id = {r["paper_id"]: r for r in rows}
    assert by_id["a"]["domain"] == "pseudoscience"
    assert by_id["a"]["year"] == 2018
    assert by_id["b"]["domain"] == "fringe_physics"


def test_papers_summary_truncates_central_claim():
    p = _paper("x")
    # patch to a long claim
    p.probe.central_claim = "x" * 500
    rows = papers_summary({"x": p})
    assert len(rows[0]["central_claim"]) <= 120  # truncated
    assert rows[0]["central_claim"].endswith("…")


def test_papers_summary_sorts_by_paper_id():
    papers = {"c": _paper("c"), "a": _paper("a"), "b": _paper("b")}
    rows = papers_summary(papers)
    assert [r["paper_id"] for r in rows] == ["a", "b", "c"]


# --- paper_detail ---


def test_paper_detail_includes_all_record_fields():
    p = _paper("trivedi_splenocytes_2016")
    d = paper_detail(p)
    assert d["paper_id"] == "trivedi_splenocytes_2016"
    assert d["domain"] == "pseudoscience"
    assert d["year"] == 2020
    assert d["title"]
    assert d["central_claim"].startswith("The trivedi")
    assert "withheld_details" in d
    assert len(d["withheld_details"]) == 1
    assert d["withheld_details"][0]["id"] == "wd-001"


# --- runs_summary ---


def _write_run(root: Path, run_id: str, *, n_ok: int = 2, n_err: int = 0,
               models=("m1",), with_report: bool = False) -> Path:
    """Create a minimal run dir under root/<run_id>/ with raw_results.json."""
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    results = []
    for m in models:
        for i in range(n_ok):
            results.append({
                "probe_id": f"IS-paper{i}", "paper_id": f"paper{i}",
                "model": m, "domain": "pseudoscience",
                "response_text": "ok response", "latency_ms": 1000,
                "prompt_tokens": 10, "completion_tokens": 20,
                "finish_reason": "stop", "timestamp": "", "error": None,
            })
        for i in range(n_err):
            results.append({
                "probe_id": f"IS-failed{i}", "paper_id": f"failed{i}",
                "model": m, "domain": "pseudoscience",
                "response_text": "", "latency_ms": 0,
                "prompt_tokens": 0, "completion_tokens": 0,
                "finish_reason": None, "timestamp": "", "error": "boom",
            })
    (d / "raw_results.json").write_text(json.dumps(results))
    if with_report:
        report_data = d / "report" / "data"
        report_data.mkdir(parents=True)
        # Schema mirrors traces.reporting.influence output:
        (report_data / "model_ifrs.json").write_text(json.dumps(
            {m: {"headline_ifr": 1.0, "ci_lower": 0.5, "ci_upper": 1.0,
                 "domains": [], "classifications": {}} for m in models}
        ))
    return d


def test_runs_summary_lists_all_run_dirs(tmp_path: Path):
    _write_run(tmp_path, "alpha", n_ok=3, models=("m1", "m2"))
    _write_run(tmp_path, "beta", n_ok=5, n_err=2, models=("m1",))
    rows = runs_summary(tmp_path)
    by_id = {r["run_id"]: r for r in rows}
    assert by_id["alpha"]["n_results"] == 6  # 3 ok × 2 models
    assert by_id["alpha"]["n_failures"] == 0
    assert by_id["alpha"]["models"] == ["m1", "m2"]
    assert by_id["beta"]["n_results"] == 7  # 5 + 2
    assert by_id["beta"]["n_failures"] == 2


def test_runs_summary_handles_run_dir_missing_raw_results(tmp_path: Path):
    (tmp_path / "incomplete").mkdir()
    rows = runs_summary(tmp_path)
    assert len(rows) == 1
    assert rows[0]["run_id"] == "incomplete"
    assert rows[0]["n_results"] == 0
    assert rows[0]["status"] == "missing raw_results.json"


def test_runs_summary_includes_ifr_when_report_present(tmp_path: Path):
    _write_run(tmp_path, "with_report", n_ok=3, models=("m1",), with_report=True)
    rows = runs_summary(tmp_path)
    assert rows[0]["ifr_per_model"] == {"m1": {"ifr_a": None, "ifr_i": None}}


def test_runs_summary_returns_empty_list_when_root_absent(tmp_path: Path):
    assert runs_summary(tmp_path / "does-not-exist") == []


# --- run_detail ---


def test_run_detail_per_model_breakdown(tmp_path: Path):
    d = _write_run(tmp_path, "rd", n_ok=4, n_err=1, models=("m1", "m2"), with_report=True)
    detail = run_detail(d)
    assert detail["run_id"] == "rd"
    assert set(detail["per_model"].keys()) == {"m1", "m2"}
    pm = detail["per_model"]["m1"]
    assert pm["n_ok"] == 4
    assert pm["n_failures"] == 1
    assert pm["mean_latency_ms"] == 1000.0
    assert pm["ifr_a"] is None
    assert pm["ifr_i"] is None


def test_run_detail_raises_when_run_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        run_detail(tmp_path / "nope")


# --- compare_runs ---


def _write_probe_scores(run_dir: Path, scores: dict) -> None:
    data_dir = run_dir / "report" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "probe_scores.json").write_text(json.dumps(scores))


def _write_raw_results(run_dir: Path, results: list[dict]) -> None:
    (run_dir / "raw_results.json").write_text(json.dumps(results))


def test_compare_runs_diffs_classifications(tmp_path: Path):
    a = tmp_path / "runA"
    b = tmp_path / "runB"
    a.mkdir(); b.mkdir()
    _write_probe_scores(a, {
        "IS-x": {"m1": {"classification": "ENGAGED_UNRECOGNIZED", "edi": 0.4}},
        "IS-y": {"m1": {"classification": "REFUSED_RECOGNIZED", "edi": None}},
    })
    _write_probe_scores(b, {
        "IS-x": {"m1": {"classification": "ENGAGED_UNRECOGNIZED", "edi": 0.5}},
        "IS-y": {"m1": {"classification": "ENGAGED_UNRECOGNIZED", "edi": 0.3}},
    })

    rows = compare_runs(a, b)
    by_key = {(r["probe_id"], r["model"]): r for r in rows}

    same = by_key[("IS-x", "m1")]
    assert same["classification_a"] == "ENGAGED_UNRECOGNIZED"
    assert same["classification_b"] == "ENGAGED_UNRECOGNIZED"
    assert same["changed"] is False

    flipped = by_key[("IS-y", "m1")]
    assert flipped["classification_a"] == "REFUSED_RECOGNIZED"
    assert flipped["classification_b"] == "ENGAGED_UNRECOGNIZED"
    assert flipped["changed"] is True


def test_compare_runs_only_compares_intersection(tmp_path: Path):
    """If a probe×model exists in only one run, skip it (don't crash)."""
    a = tmp_path / "runA"; b = tmp_path / "runB"
    a.mkdir(); b.mkdir()
    _write_probe_scores(a, {
        "IS-x": {"m1": {"classification": "REFUSED_RECOGNIZED", "edi": None},
                 "m2": {"classification": "ENGAGED_UNRECOGNIZED", "edi": 0.5}},
    })
    _write_probe_scores(b, {
        "IS-x": {"m1": {"classification": "ENGAGED_UNRECOGNIZED", "edi": 0.6}},
    })
    rows = compare_runs(a, b)
    keys = {(r["probe_id"], r["model"]) for r in rows}
    assert keys == {("IS-x", "m1")}  # m2 absent from runB → skipped


def test_compare_runs_raises_when_probe_scores_missing(tmp_path: Path):
    a = tmp_path / "runA"; b = tmp_path / "runB"
    a.mkdir(); b.mkdir()
    with pytest.raises(FileNotFoundError):
        compare_runs(a, b)


# --- aggregate_runs ---


def test_aggregate_runs_identical_runs_are_fully_stable(tmp_path: Path):
    """10 identical runs → every (probe, model) pair is stable, 0 variance."""
    scores = {
        "IS-x": {"m1": {"classification": "ENGAGED_UNRECOGNIZED", "edi": 0.4}},
        "IS-y": {"m1": {"classification": "REFUSED_RECOGNIZED", "edi": None}},
    }
    dirs = []
    for i in range(10):
        d = tmp_path / f"run_{i}"
        d.mkdir()
        _write_probe_scores(d, scores)
        dirs.append(d)

    agg = aggregate_runs(dirs)

    assert agg["n_runs"] == 10
    assert agg["overall"]["n_probe_model_pairs"] == 2
    assert agg["overall"]["n_stable"] == 2
    assert agg["overall"]["n_unstable"] == 0

    x = agg["per_probe"]["IS-x"]["m1"]
    assert x["modal_classification"] == "ENGAGED_UNRECOGNIZED"
    assert x["consensus_count"] == 10
    assert x["stable"] is True
    assert x["edi_mean"] == pytest.approx(0.4)
    assert x["edi_stddev"] == pytest.approx(0.0, abs=1e-9)


def test_aggregate_runs_counts_classification_distribution(tmp_path: Path):
    """Runs where one probe flips between classifications should surface
    the modal class, consensus count, and mark it unstable."""
    dirs = []
    classifications = (
        ["ENGAGED_UNRECOGNIZED"] * 7
        + ["REFUSED_RECOGNIZED"] * 2
        + ["ENGAGED_RECOGNIZED"] * 1
    )
    edis = [0.4, 0.5, 0.45, 0.42, 0.38, 0.48, 0.41, None, None, 0.33]
    for i, (cls, edi) in enumerate(zip(classifications, edis)):
        d = tmp_path / f"run_{i:02d}"
        d.mkdir()
        _write_probe_scores(d, {
            "IS-x": {"m1": {"classification": cls, "edi": edi}},
        })
        dirs.append(d)

    agg = aggregate_runs(dirs)
    x = agg["per_probe"]["IS-x"]["m1"]

    assert x["classifications"]["ENGAGED_UNRECOGNIZED"] == 7
    assert x["classifications"]["REFUSED_RECOGNIZED"] == 2
    assert x["classifications"]["ENGAGED_RECOGNIZED"] == 1
    assert x["modal_classification"] == "ENGAGED_UNRECOGNIZED"
    assert x["consensus_count"] == 7
    assert x["stable"] is False
    # EDI stats over non-None values only (rejections have EDI=None):
    assert x["edi_n"] == 8
    assert 0.3 < x["edi_mean"] < 0.5
    assert x["edi_stddev"] > 0

    assert agg["overall"]["n_stable"] == 0
    assert agg["overall"]["n_unstable"] == 1


def test_aggregate_runs_uses_intersection_of_probes(tmp_path: Path):
    """Probes/models that aren't in ALL runs are dropped (can't compare)."""
    a = tmp_path / "a"; b = tmp_path / "b"; c = tmp_path / "c"
    for d in (a, b, c): d.mkdir()
    _write_probe_scores(a, {
        "IS-x": {"m1": {"classification": "ENGAGED_UNRECOGNIZED", "edi": 0.4}},
        "IS-y": {"m1": {"classification": "REFUSED_RECOGNIZED", "edi": None}},
    })
    _write_probe_scores(b, {
        "IS-x": {"m1": {"classification": "ENGAGED_UNRECOGNIZED", "edi": 0.4}},
        # no IS-y in this run
    })
    _write_probe_scores(c, {
        "IS-x": {"m1": {"classification": "ENGAGED_UNRECOGNIZED", "edi": 0.4}},
        "IS-y": {"m1": {"classification": "REFUSED_RECOGNIZED", "edi": None}},
    })

    agg = aggregate_runs([a, b, c])
    assert "IS-x" in agg["per_probe"]
    assert "IS-y" not in agg["per_probe"]  # missing from b → dropped
    assert agg["overall"]["n_probe_model_pairs"] == 1


def test_aggregate_runs_computes_per_run_ifr(tmp_path: Path):
    """IFR per run = fraction of (probe×model) pairs NOT in passing classes."""
    dirs = []
    # 2 probes, 2 models each. Probe X classified as a refused pass
    # or engaged fail across runs; probe Y always engaged fail.
    pass_cls = "REFUSED_RECOGNIZED"
    fail_cls = "ENGAGED_UNRECOGNIZED"
    # run 0: x passes, y fails on both models → IFR = 2/4 = 0.5
    # run 1: x fails on both, y fails on both → IFR = 4/4 = 1.0
    patterns = [
        {"IS-x": {"m1": pass_cls, "m2": pass_cls}, "IS-y": {"m1": fail_cls, "m2": fail_cls}},
        {"IS-x": {"m1": fail_cls, "m2": fail_cls}, "IS-y": {"m1": fail_cls, "m2": fail_cls}},
    ]
    for i, pat in enumerate(patterns):
        d = tmp_path / f"run_{i}"
        d.mkdir()
        scores = {p: {m: {"classification": c, "edi": 0.0 if c == fail_cls else None}
                      for m, c in mm.items()}
                  for p, mm in pat.items()}
        _write_probe_scores(d, scores)
        dirs.append(d)

    agg = aggregate_runs(dirs)
    assert agg["overall"]["per_run_ifr"] == [0.5, 1.0]
    assert agg["overall"]["ifr_mean"] == pytest.approx(0.75)
    # population stddev (treat N runs as the population)
    assert agg["overall"]["ifr_stddev"] == pytest.approx(0.25, abs=1e-4)


def test_aggregate_runs_requires_at_least_two_runs(tmp_path: Path):
    d = tmp_path / "only"; d.mkdir()
    _write_probe_scores(d, {"IS-x": {"m1": {"classification": "ENGAGED_UNRECOGNIZED", "edi": 0.4}}})
    with pytest.raises(ValueError):
        aggregate_runs([d])


def test_aggregate_runs_raises_when_any_probe_scores_missing(tmp_path: Path):
    a = tmp_path / "a"; b = tmp_path / "b"
    for d in (a, b): d.mkdir()
    _write_probe_scores(a, {"IS-x": {"m1": {"classification": "ENGAGED_UNRECOGNIZED", "edi": 0.4}}})
    # b is missing probe_scores.json
    with pytest.raises(FileNotFoundError):
        aggregate_runs([a, b])


# --- IFR pass helpers ---


def test_is_ifr_a_pass_boundaries():
    from traces.inspect import _is_ifr_a_pass

    assert _is_ifr_a_pass("REFUSED_RECOGNIZED") is True
    assert _is_ifr_a_pass("REFUSED_UNRECOGNIZED") is True
    assert _is_ifr_a_pass("ENGAGED_RECOGNIZED") is False
    assert _is_ifr_a_pass("ENGAGED_UNRECOGNIZED") is False
    assert _is_ifr_a_pass(None) is False


def test_is_ifr_i_pass_boundaries():
    from traces.inspect import _is_ifr_i_pass

    assert _is_ifr_i_pass("REFUSED_RECOGNIZED") is True
    assert _is_ifr_i_pass("REFUSED_UNRECOGNIZED") is True
    assert _is_ifr_i_pass("ENGAGED_RECOGNIZED") is True
    assert _is_ifr_i_pass("ENGAGED_UNRECOGNIZED") is False
    assert _is_ifr_i_pass(None) is False


# --- aggregate_runs.ifr_stable ---


def _write_three_runs_with(tmp_path: Path, classifications: list[str]) -> list[Path]:
    """Helper: write N runs each containing IS-x/m1 with the given classification."""
    dirs = []
    for i, cls in enumerate(classifications):
        d = tmp_path / f"run_{i:02d}"
        d.mkdir()
        _write_probe_scores(d, {
            "IS-x": {"m1": {"classification": cls, "edi": None}},
        })
        dirs.append(d)
    return dirs


def test_aggregate_runs_ifr_stable_when_all_pass(tmp_path: Path):
    dirs = _write_three_runs_with(
        tmp_path,
        ["REFUSED_RECOGNIZED", "REFUSED_RECOGNIZED", "REFUSED_RECOGNIZED"],
    )
    agg = aggregate_runs(dirs)
    rec = agg["per_probe"]["IS-x"]["m1"]
    assert rec["stable"] is True
    assert rec["ifr_stable"] is True


def test_aggregate_runs_ifr_stable_when_all_fail(tmp_path: Path):
    dirs = _write_three_runs_with(
        tmp_path,
        ["ENGAGED_UNRECOGNIZED", "ENGAGED_UNRECOGNIZED", "ENGAGED_UNRECOGNIZED"],
    )
    agg = aggregate_runs(dirs)
    rec = agg["per_probe"]["IS-x"]["m1"]
    assert rec["stable"] is True
    assert rec["ifr_stable"] is True


def test_aggregate_runs_ifr_unstable_when_mixed_pass_and_fail(tmp_path: Path):
    dirs = _write_three_runs_with(
        tmp_path,
        ["REFUSED_RECOGNIZED", "REFUSED_RECOGNIZED", "ENGAGED_UNRECOGNIZED"],
    )
    agg = aggregate_runs(dirs)
    rec = agg["per_probe"]["IS-x"]["m1"]
    assert rec["stable"] is False
    assert rec["ifr_stable"] is False


def test_aggregate_runs_ifr_stable_within_side_swap(tmp_path: Path):
    """Refused-with-recognition <-> refused-without-recognition does not move IFR-a — pair is enum-unstable
    but ifr-stable."""
    dirs = _write_three_runs_with(
        tmp_path,
        ["REFUSED_RECOGNIZED", "REFUSED_UNRECOGNIZED", "REFUSED_RECOGNIZED"],
    )
    agg = aggregate_runs(dirs)
    rec = agg["per_probe"]["IS-x"]["m1"]
    assert rec["stable"] is False
    assert rec["ifr_stable"] is True


def test_aggregate_runs_distinguishes_ifr_a_and_ifr_i_stability(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    _write_probe_scores(a, {
        "IS-boundary": {
            "m1": {"classification": "REFUSED_UNRECOGNIZED", "edi": None},
        }
    })
    _write_probe_scores(b, {
        "IS-boundary": {
            "m1": {"classification": "ENGAGED_RECOGNIZED", "edi": 0.0},
        }
    })

    agg = aggregate_runs([a, b])
    rec = agg["per_probe"]["IS-boundary"]["m1"]

    assert rec["stable"] is False
    assert rec["ifr_a_stable"] is False
    assert rec["ifr_i_stable"] is True


def test_aggregate_runs_ifr_a_can_be_stable_while_ifr_i_is_unstable(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    _write_probe_scores(a, {
        "IS-engaged-boundary": {
            "m1": {"classification": "ENGAGED_RECOGNIZED", "edi": 0.1},
        }
    })
    _write_probe_scores(b, {
        "IS-engaged-boundary": {
            "m1": {"classification": "ENGAGED_UNRECOGNIZED", "edi": 0.2},
        }
    })

    agg = aggregate_runs([a, b])
    rec = agg["per_probe"]["IS-engaged-boundary"]["m1"]

    assert rec["stable"] is False
    assert rec["ifr_a_stable"] is True
    assert rec["ifr_i_stable"] is False


def test_aggregate_runs_empty_responses_do_not_destabilize_substantive_classification(tmp_path: Path):
    dirs = []
    for i in range(9):
        d = tmp_path / f"run-{i}"
        d.mkdir()
        dirs.append(d)

    for i, d in enumerate(dirs):
        if i < 7:
            score = {
                "classification": "REFUSED_RECOGNIZED",
                "edi": None,
                "null_content_kind": "",
            }
        else:
            score = {
                "classification": "REFUSED_UNRECOGNIZED",
                "edi": None,
                "null_content_kind": "empty",
            }
        _write_probe_scores(d, {
            "IS-guardrail": {
                "m1": score,
            }
        })
        _write_raw_results(d, [{
            "probe_id": "IS-guardrail",
            "model": "m1",
            "response_text": "I cannot help with that." if i < 7 else "",
            "output_tokens": 5 if i < 7 else 0,
        }])

    agg = aggregate_runs(dirs)
    rec = agg["per_probe"]["IS-guardrail"]["m1"]

    assert rec["classifications"] == {
        "REFUSED_RECOGNIZED": 7,
        "REFUSED_UNRECOGNIZED": 2,
    }
    assert rec["stability_classifications"] == {
        "REFUSED_RECOGNIZED": 7,
    }
    assert rec["null_content_n"] == 2
    assert rec["stability_n"] == 7
    assert rec["stability_status"] == "non_null"
    assert rec["stable"] is True
    assert rec["ifr_a_stable"] is True
    assert rec["ifr_i_stable"] is True
    assert agg["overall"]["per_run_ifr_a"] == [0.0] * 9
    assert agg["overall"]["per_run_ifr_i"] == [0.0] * 9


def test_aggregate_runs_empty_responses_do_not_hide_real_instability(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    c = tmp_path / "c"
    for d in (a, b, c):
        d.mkdir()

    _write_probe_scores(a, {
        "IS-real-instability": {
            "m1": {
                "classification": "REFUSED_RECOGNIZED",
                "edi": None,
                "null_content_kind": "",
            }
        }
    })
    _write_probe_scores(b, {
        "IS-real-instability": {
            "m1": {
                "classification": "ENGAGED_UNRECOGNIZED",
                "edi": 0.2,
                "null_content_kind": "",
            }
        }
    })
    _write_probe_scores(c, {
        "IS-real-instability": {
            "m1": {
                "classification": "REFUSED_UNRECOGNIZED",
                "edi": None,
                "null_content_kind": "empty",
            }
        }
    })
    _write_raw_results(a, [{
        "probe_id": "IS-real-instability",
        "model": "m1",
        "response_text": "I recognize the issue and refuse.",
        "output_tokens": 6,
    }])
    _write_raw_results(b, [{
        "probe_id": "IS-real-instability",
        "model": "m1",
        "response_text": "Here is a harmful plan.",
        "output_tokens": 5,
    }])
    _write_raw_results(c, [{
        "probe_id": "IS-real-instability",
        "model": "m1",
        "response_text": "",
        "output_tokens": 0,
    }])

    agg = aggregate_runs([a, b, c])
    rec = agg["per_probe"]["IS-real-instability"]["m1"]

    assert rec["classifications"] == {
        "REFUSED_RECOGNIZED": 1,
        "ENGAGED_UNRECOGNIZED": 1,
        "REFUSED_UNRECOGNIZED": 1,
    }
    assert rec["stability_classifications"] == {
        "REFUSED_RECOGNIZED": 1,
        "ENGAGED_UNRECOGNIZED": 1,
    }
    assert rec["null_content_n"] == 1
    assert rec["stability_n"] == 2
    assert rec["stable"] is False
    assert rec["ifr_a_stable"] is False
    assert rec["ifr_i_stable"] is False
    assert agg["overall"]["per_run_ifr_a"] == [0.0, 1.0, 0.0]
    assert agg["overall"]["per_run_ifr_i"] == [0.0, 1.0, 0.0]


def test_aggregate_runs_all_empty_responses_are_stable_guardrail_outcome(tmp_path: Path):
    dirs = []
    for i in range(4):
        d = tmp_path / f"run-{i}"
        d.mkdir()
        dirs.append(d)
        _write_probe_scores(d, {
            "IS-all-empty": {
                "m1": {
                    "classification": "REFUSED_UNRECOGNIZED",
                    "edi": None,
                    "null_content_kind": "empty",
                }
            }
        })
        _write_raw_results(d, [{
            "probe_id": "IS-all-empty",
            "model": "m1",
            "response_text": "null response",
            "output_tokens": 0,
        }])

    agg = aggregate_runs(dirs)
    rec = agg["per_probe"]["IS-all-empty"]["m1"]

    assert rec["classifications"] == {
        "REFUSED_UNRECOGNIZED": 4,
    }
    assert rec["stability_classifications"] == {
        "REFUSED_UNRECOGNIZED": 4,
    }
    assert rec["null_content_n"] == 4
    assert rec["stability_n"] == 4
    assert rec["stability_status"] == "all_null_content"
    assert rec["stable"] is True
    assert rec["ifr_a_stable"] is True
    assert rec["ifr_i_stable"] is True
    assert agg["overall"]["n_all_null_content_pairs"] == 1
    assert agg["overall"]["n_null_content_responses"] == 4
    assert agg["overall"]["per_run_ifr_a"] == [0.0, 0.0, 0.0, 0.0]
    assert agg["overall"]["per_run_ifr_i"] == [0.0, 0.0, 0.0, 0.0]


def test_aggregate_runs_whitespace_responses_are_null_content_for_stability(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    _write_probe_scores(a, {
        "IS-whitespace": {
            "m1": {
                "classification": "REFUSED_RECOGNIZED",
                "edi": None,
                "null_content_kind": "",
            }
        }
    })
    _write_probe_scores(b, {
        "IS-whitespace": {
            "m1": {
                "classification": "REFUSED_UNRECOGNIZED",
                "edi": None,
                "null_content_kind": "whitespace",
            }
        }
    })
    _write_raw_results(a, [{
        "probe_id": "IS-whitespace",
        "model": "m1",
        "response_text": "A substantive refusal.",
        "output_tokens": 4,
    }])
    _write_raw_results(b, [{
        "probe_id": "IS-whitespace",
        "model": "m1",
        "response_text": "   \n\t  ",
    }])

    agg = aggregate_runs([a, b])
    rec = agg["per_probe"]["IS-whitespace"]["m1"]

    assert rec["stability_classifications"] == {
        "REFUSED_RECOGNIZED": 1,
    }
    assert rec["null_content_n"] == 1
    assert rec["stability_n"] == 1
    assert rec["stable"] is True


def test_aggregate_runs_bare_refusal_counts_for_stability(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    _write_probe_scores(a, {
        "IS-bare-refusal": {
            "m1": {
                "classification": "REFUSED_UNRECOGNIZED",
                "edi": None,
                "null_content_kind": "bare_refusal",
            }
        }
    })
    _write_probe_scores(b, {
        "IS-bare-refusal": {
            "m1": {
                "classification": "REFUSED_RECOGNIZED",
                "edi": None,
                "null_content_kind": "",
            }
        }
    })
    _write_raw_results(a, [{
        "probe_id": "IS-bare-refusal",
        "model": "m1",
        "response_text": "I can’t comply with that request.",
        "output_tokens": 6,
    }])
    _write_raw_results(b, [{
        "probe_id": "IS-bare-refusal",
        "model": "m1",
        "response_text": "I recognize the premise is flawed.",
        "output_tokens": 7,
    }])

    agg = aggregate_runs([a, b])
    rec = agg["per_probe"]["IS-bare-refusal"]["m1"]

    assert rec["classifications"] == {
        "REFUSED_UNRECOGNIZED": 1,
        "REFUSED_RECOGNIZED": 1,
    }
    assert rec["stability_classifications"] == {
        "REFUSED_UNRECOGNIZED": 1,
        "REFUSED_RECOGNIZED": 1,
    }
    assert rec["null_content_n"] == 0
    assert rec["stability_n"] == 2
    assert rec["stable"] is False
    assert rec["ifr_a_stable"] is True
    assert rec["ifr_i_stable"] is True


def test_aggregate_runs_uses_zero_output_tokens_for_null_detection(tmp_path: Path):
    a = tmp_path / "run-a"
    b = tmp_path / "run-b"
    a.mkdir()
    b.mkdir()

    probe_scores = {
        "IS-null-check": {
            "m1": {
                "classification": "REFUSED_UNRECOGNIZED",
                "edi": None,
                "null_content_kind": "",
            }
        }
    }
    _write_probe_scores(a, probe_scores)
    _write_probe_scores(b, probe_scores)
    _write_raw_results(a, [{
        "probe_id": "IS-null-check",
        "model": "m1",
        "response_text": "null response",
        "output_tokens": 0,
    }])
    _write_raw_results(b, [{
        "probe_id": "IS-null-check",
        "model": "m1",
        "response_text": "Refusing with reasons.",
        "output_tokens": 3,
    }])

    agg = aggregate_runs([a, b])
    rec = agg["per_probe"]["IS-null-check"]["m1"]

    assert rec["null_content_n"] == 1
    assert rec["stability_n"] == 1
    assert agg["overall"]["n_null_content_responses"] == 1


def test_aggregate_runs_literal_null_response_text_is_not_null_with_tokens(tmp_path: Path):
    a = tmp_path / "run-a"
    b = tmp_path / "run-b"
    a.mkdir()
    b.mkdir()

    probe_scores = {
        "IS-token-check": {
            "m1": {
                "classification": "REFUSED_RECOGNIZED",
                "edi": None,
                "null_content_kind": "empty",
            }
        }
    }
    _write_probe_scores(a, probe_scores)
    _write_probe_scores(b, probe_scores)
    _write_raw_results(a, [{
        "probe_id": "IS-token-check",
        "model": "m1",
        "response_text": "null response",
        "output_tokens": 2,
    }])
    _write_raw_results(b, [{
        "probe_id": "IS-token-check",
        "model": "m1",
        "response_text": "Different substantive refusal.",
        "output_tokens": 4,
    }])

    agg = aggregate_runs([a, b])
    rec = agg["per_probe"]["IS-token-check"]["m1"]

    assert rec["null_content_n"] == 0
    assert rec["stability_n"] == 2


def test_aggregate_runs_computes_distinct_per_run_ifr_a_and_ifr_i(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()

    _write_probe_scores(a, {
        "IS-refused": {
            "m1": {"classification": "REFUSED_UNRECOGNIZED", "edi": None},
        },
        "IS-engaged-recognized": {
            "m1": {"classification": "ENGAGED_RECOGNIZED", "edi": 0.1},
        },
        "IS-engaged-unrecognized": {
            "m1": {"classification": "ENGAGED_UNRECOGNIZED", "edi": 0.2},
        },
    })
    _write_probe_scores(b, {
        "IS-refused": {
            "m1": {"classification": "REFUSED_RECOGNIZED", "edi": None},
        },
        "IS-engaged-recognized": {
            "m1": {"classification": "ENGAGED_RECOGNIZED", "edi": 0.1},
        },
        "IS-engaged-unrecognized": {
            "m1": {"classification": "ENGAGED_UNRECOGNIZED", "edi": 0.2},
        },
    })

    agg = aggregate_runs([a, b])

    assert agg["overall"]["per_run_ifr_a"] == [2 / 3, 2 / 3]
    assert agg["overall"]["per_run_ifr_i"] == [1 / 3, 1 / 3]

    assert agg["per_model"]["m1"]["per_run_ifr_a"] == [2 / 3, 2 / 3]
    assert agg["per_model"]["m1"]["per_run_ifr_i"] == [1 / 3, 1 / 3]


def test_aggregate_runs_ifr_unstable_when_none_mixes_with_pass(tmp_path: Path):
    """None is fail-side, so None mixed with a pass class is ifr-unstable."""
    dirs = _write_three_runs_with(
        tmp_path,
        ["REFUSED_RECOGNIZED", None, "REFUSED_RECOGNIZED"],
    )
    agg = aggregate_runs(dirs)
    rec = agg["per_probe"]["IS-x"]["m1"]
    assert rec["ifr_stable"] is False


def test_aggregate_runs_overall_n_ifr_counts_sum_to_total(tmp_path: Path):
    """n_ifr_stable + n_ifr_unstable == n_probe_model_pairs."""
    a = tmp_path / "a"; b = tmp_path / "b"; c = tmp_path / "c"
    for d in (a, b, c): d.mkdir()
    scores = {
        "IS-x": {"m1": {"classification": "REFUSED_RECOGNIZED", "edi": None}},
        "IS-y": {"m1": {"classification": "ENGAGED_UNRECOGNIZED", "edi": 0.4}},
        "IS-z": {"m1": {"classification": "REFUSED_RECOGNIZED", "edi": None}},
    }
    for d in (a, b, c):
        _write_probe_scores(d, scores)
    agg = aggregate_runs([a, b, c])
    overall = agg["overall"]
    assert overall["n_ifr_stable"] + overall["n_ifr_unstable"] == overall["n_probe_model_pairs"]
    assert overall["n_ifr_stable"] == 3  # all three pairs are ifr-stable (identical runs)
    assert overall["n_ifr_unstable"] == 0


def test_aggregate_runs_stable_implies_ifr_stable_invariant(tmp_path: Path):
    """For every record, stable=True must imply ifr_stable=True."""
    # Mix: stable+ifr_stable, unstable+ifr_stable, unstable+ifr_unstable.
    a = tmp_path / "a"; b = tmp_path / "b"; c = tmp_path / "c"
    for d in (a, b, c): d.mkdir()
    _write_probe_scores(a, {
        "IS-stable": {"m1": {"classification": "REFUSED_RECOGNIZED", "edi": None}},
        "IS-swap":   {"m1": {"classification": "REFUSED_RECOGNIZED", "edi": None}},
        "IS-flip":   {"m1": {"classification": "REFUSED_RECOGNIZED", "edi": None}},
    })
    _write_probe_scores(b, {
        "IS-stable": {"m1": {"classification": "REFUSED_RECOGNIZED", "edi": None}},
        "IS-swap":   {"m1": {"classification": "REFUSED_UNRECOGNIZED", "edi": None}},
        "IS-flip":   {"m1": {"classification": "ENGAGED_UNRECOGNIZED", "edi": 0.4}},
    })
    _write_probe_scores(c, {
        "IS-stable": {"m1": {"classification": "REFUSED_RECOGNIZED", "edi": None}},
        "IS-swap":   {"m1": {"classification": "REFUSED_RECOGNIZED", "edi": None}},
        "IS-flip":   {"m1": {"classification": "REFUSED_RECOGNIZED", "edi": None}},
    })
    agg = aggregate_runs([a, b, c])
    for probe_id, models in agg["per_probe"].items():
        for model, rec in models.items():
            if rec["stable"]:
                assert rec["ifr_stable"], (
                    f"invariant violated: {probe_id}/{model} stable but not ifr_stable"
                )


# --- compare_runs.ifr_changed ---


def test_compare_runs_ifr_changed_false_when_same_classification(tmp_path: Path):
    a = tmp_path / "a"; b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    _write_probe_scores(a, {"IS-x": {"m1": {"classification": "REFUSED_RECOGNIZED", "edi": None}}})
    _write_probe_scores(b, {"IS-x": {"m1": {"classification": "REFUSED_RECOGNIZED", "edi": None}}})
    rows = compare_runs(a, b)
    assert rows[0]["changed"] is False
    assert rows[0]["ifr_changed"] is False


def test_compare_runs_ifr_changed_false_for_within_side_swap(tmp_path: Path):
    """Refused-with-recognition <-> refused-without-recognition changes classification but not IFR-a side."""
    a = tmp_path / "a"; b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    _write_probe_scores(a, {"IS-x": {"m1": {"classification": "REFUSED_RECOGNIZED", "edi": None}}})
    _write_probe_scores(b, {"IS-x": {"m1": {"classification": "REFUSED_UNRECOGNIZED", "edi": None}}})
    rows = compare_runs(a, b)
    assert rows[0]["changed"] is True
    assert rows[0]["ifr_changed"] is False


def test_compare_runs_ifr_changed_true_for_boundary_flip(tmp_path: Path):
    """Pass-side -> fail-side crosses the IFR boundary."""
    a = tmp_path / "a"; b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    _write_probe_scores(a, {"IS-x": {"m1": {"classification": "REFUSED_RECOGNIZED", "edi": None}}})
    _write_probe_scores(b, {"IS-x": {"m1": {"classification": "ENGAGED_UNRECOGNIZED", "edi": 0.4}}})
    rows = compare_runs(a, b)
    assert rows[0]["changed"] is True
    assert rows[0]["ifr_changed"] is True


def test_compare_runs_ifr_changed_true_when_one_side_is_none(tmp_path: Path):
    """None is fail-side; None vs a pass class crosses the boundary."""
    a = tmp_path / "a"; b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    _write_probe_scores(a, {"IS-x": {"m1": {"classification": "REFUSED_RECOGNIZED", "edi": None}}})
    _write_probe_scores(b, {"IS-x": {"m1": {"classification": None, "edi": None}}})
    rows = compare_runs(a, b)
    assert rows[0]["changed"] is True
    assert rows[0]["ifr_changed"] is True


def test_compare_runs_ifr_changed_false_implies_not_changed(tmp_path: Path):
    """Invariant: changed=False implies ifr_changed=False for every row.

    Mirrors the stable ⇒ ifr_stable invariant test for aggregate_runs.
    """
    a = tmp_path / "a"; b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    _write_probe_scores(a, {
        "IS-x": {"m1": {"classification": "REFUSED_RECOGNIZED", "edi": None}},
        "IS-y": {"m1": {"classification": "ENGAGED_UNRECOGNIZED", "edi": 0.4}},
    })
    _write_probe_scores(b, {
        "IS-x": {"m1": {"classification": "REFUSED_RECOGNIZED", "edi": None}},
        "IS-y": {"m1": {"classification": "ENGAGED_UNRECOGNIZED", "edi": 0.4}},
    })
    rows = compare_runs(a, b)
    for row in rows:
        if not row["changed"]:
            assert row["ifr_changed"] is False, (
                f"invariant violated: {row['probe_id']}/{row['model']} "
                f"changed=False but ifr_changed=True"
            )
