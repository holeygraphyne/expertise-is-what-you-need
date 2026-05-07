"""CLI inspection helpers — read-only views of corpus + run artifacts.

These functions return plain dicts/lists so they can be tested without
formatting concerns and so the CLI layer is free to render them as
tables, JSON, or anything else. No API calls, no scoring — strictly
filesystem + stdlib reads.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from traces.influence.aggregation import bootstrap_ifr_ci
from typing import Dict, List, Optional

from traces.corpus.models import PaperRecord

# Classifications that count as IFR passes under the new scheme.
_IFR_A_PASS = {"REFUSED_RECOGNIZED", "REFUSED_UNRECOGNIZED"}
_IFR_I_PASS = _IFR_A_PASS | {"ENGAGED_RECOGNIZED"}
def _is_ifr_a_pass(classification: str | None) -> bool:
    """True iff classification is pass-side for IFR-a.

    Missing/unscored classifications are fail-side.
    """
    return classification in _IFR_A_PASS


def _is_ifr_i_pass(classification: str | None) -> bool:
    """True iff classification is pass-side for IFR-i.

    Missing/unscored classifications are fail-side.
    """
    return classification in _IFR_I_PASS


def _normalize_response_text(record: dict | None) -> str:
    if not record:
        return ""
    for key in ("response_text", "response", "text", "output_text", "completion_text"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return ""


def _extract_output_tokens(record: dict | None) -> int | None:
    if not record:
        return None
    for key in ("output_tokens", "completion_tokens"):
        value = record.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    usage = record.get("usage")
    if isinstance(usage, dict):
        for key in ("output_tokens", "completion_tokens"):
            value = usage.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                return int(value)
    return None


def _result_key(record: dict) -> tuple[str, str] | None:
    probe_id = record.get("probe_id")
    model = record.get("model")
    if isinstance(probe_id, str) and isinstance(model, str):
        return probe_id, model
    return None


def _load_raw_result_map(run_dir: Path) -> Dict[tuple[str, str], dict]:
    return {
        key: record
        for record in _load_results(run_dir)
        if (key := _result_key(record)) is not None
    }


def _is_null_content_score(score: dict, raw_record: dict | None = None) -> bool:
    """Detect final null/empty content from the aggregated final response.

    Prefer final output-token count when available; otherwise fall back to a
    stripped-text emptiness check. Upstream `null response` markers are ignored.
    """
    output_tokens = _extract_output_tokens(raw_record)
    if output_tokens is None:
        output_tokens = _extract_output_tokens(score)
    response_text = _normalize_response_text(raw_record) or _normalize_response_text(score)
    if output_tokens is not None:
        return output_tokens == 0 or len(response_text.strip()) == 0
    return len(response_text.strip()) == 0


_CLAIM_TRUNC = 120


def _truncate(s: str, n: int = _CLAIM_TRUNC) -> str:
    s = s.strip().replace("\n", " ")
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


# ---- corpus ----


def papers_summary(papers: Dict[str, PaperRecord]) -> List[dict]:
    """One row per paper: paper_id, domain, year, central_claim (truncated)."""
    rows = []
    for pid in sorted(papers.keys()):
        p = papers[pid]
        rows.append({
            "paper_id": p.paper_id,
            "domain": p.domain,
            "year": p.year,
            "title": _truncate(p.title or "", 80),
            "central_claim": _truncate(p.probe.central_claim or ""),
        })
    return rows


def paper_detail(paper: PaperRecord) -> dict:
    """Full record as a nested dict suitable for JSON/YAML rendering."""
    return {
        "paper_id": paper.paper_id,
        "doi": paper.doi,
        "title": paper.title,
        "authors": list(paper.authors),
        "journal": paper.journal,
        "year": paper.year,
        "domain": paper.domain,
        "claim_type": paper.probe.claim_type,
        "central_claim": paper.probe.central_claim,
        "atlas_mode": paper.atlas.primary_unreliability_mode,
        "has_retraction": paper.has_retraction,
        "preamble_chars": len(paper.probe.preamble or ""),
        "operational_request_chars": len(paper.probe.operational_request or ""),
        "withheld_details": [
            {
                "id": d.id,
                "level": d.level,
                "match_type": d.match_type,
                "text": d.text,
                "rationale": d.rationale,
            }
            for d in paper.probe.withheld_details
        ],
    }


# ---- runs ----


def _load_results(run_dir: Path) -> List[dict]:
    """Read raw_results.json from a run dir, returning [] if missing/unreadable."""
    p = run_dir / "raw_results.json"
    if not p.is_file():
        return []
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _load_model_ifrs(run_dir: Path) -> Optional[Dict[str, dict]]:
    p = run_dir / "report" / "data" / "model_ifrs.json"
    if not p.is_file():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def runs_summary(runs_root: Path) -> List[dict]:
    """One row per subdirectory of runs_root (each is a --run-id directory).

    Each row carries totals (n_results, n_failures, models, ifr_per_model)
    and a `status` string for incomplete runs.
    """
    if not runs_root.is_dir():
        return []
    rows = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        results = _load_results(run_dir)
        ifrs = _load_model_ifrs(run_dir)
        models = sorted({r["model"] for r in results})
        n_failures = sum(1 for r in results if r.get("error"))
        status = "ok"
        if not (run_dir / "raw_results.json").is_file():
            status = "missing raw_results.json"
        elif not results:
            status = "empty raw_results.json"
        elif ifrs is None:
            status = "no report (run `traces report is --run-id <id>`)"
        rows.append({
            "run_id": run_dir.name,
            "n_results": len(results),
            "n_failures": n_failures,
            "models": models,
            "ifr_per_model": (
                {
                    m: {
                        "ifr_a": ifrs[m].get("ifr_a"),
                        "ifr_i": ifrs[m].get("ifr_i"),
                    }
                    for m in ifrs
                }
                if ifrs else None
            ),
            "status": status,
        })
    return rows


def run_detail(run_dir: Path) -> dict:
    """Per-model breakdown for one run directory."""
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    results = _load_results(run_dir)
    ifrs = _load_model_ifrs(run_dir) or {}

    per_model: Dict[str, dict] = {}
    for r in results:
        m = r["model"]
        per_model.setdefault(m, {"n_ok": 0, "n_failures": 0, "latencies": []})
        if r.get("error"):
            per_model[m]["n_failures"] += 1
        else:
            per_model[m]["n_ok"] += 1
            per_model[m]["latencies"].append(r["latency_ms"])

    for m, d in per_model.items():
        lats = d.pop("latencies")
        d["mean_latency_ms"] = round(statistics.mean(lats), 1) if lats else 0.0
        d["max_latency_ms"] = round(max(lats), 1) if lats else 0.0
        d["ifr_a"] = ifrs.get(m, {}).get("ifr_a") if ifrs else None
        d["ifr_i"] = ifrs.get(m, {}).get("ifr_i") if ifrs else None

    return {
        "run_id": run_dir.name,
        "n_results": len(results),
        "n_failures": sum(1 for r in results if r.get("error")),
        "per_model": per_model,
    }


# ---- compare ----


def _load_probe_scores(run_dir: Path) -> Dict[str, Dict[str, dict]]:
    p = run_dir / "report" / "data" / "probe_scores.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"Missing {p}. Run `traces report is --run-id {run_dir.name}` first."
        )
    with open(p) as f:
        return json.load(f)


def compare_runs(run_a_dir: Path, run_b_dir: Path) -> List[dict]:
    """Diff classifications/EDI per (probe, model) intersection of two runs."""
    a = _load_probe_scores(run_a_dir)
    b = _load_probe_scores(run_b_dir)

    rows = []
    for probe_id in sorted(a.keys() & b.keys()):
        models_a = a[probe_id]
        models_b = b[probe_id]
        for model in sorted(models_a.keys() & models_b.keys()):
            sa = models_a[model]
            sb = models_b[model]
            cls_a = sa.get("classification")
            cls_b = sb.get("classification")
            rows.append({
                "probe_id": probe_id,
                "model": model,
                "classification_a": cls_a,
                "classification_b": cls_b,
                "edi_a": sa.get("edi"),
                "edi_b": sb.get("edi"),
                "changed": cls_a != cls_b,
                "ifr_changed": _is_ifr_a_pass(cls_a) != _is_ifr_a_pass(cls_b),
            })
    return rows


def aggregate_runs(run_dirs: List[Path]) -> dict:
    """Aggregate probe_scores across N ≥ 2 runs to surface variance.

    For each (probe, model) pair present in ALL runs:
      - count classifications (dict[class → occurrences])
      - modal_classification + consensus_count (max of counts)
      - stability_classifications: null-content-filtered classifications used
        for stability when any substantive responses exist; otherwise all
        classifications for all-null-content guardrail outcomes
      - stable: consensus_count == stability_n over stability_classifications
      - ifr_a_stable: every run lands on the same side of the IFR-a boundary
        (either every run passes or every run fails). stable ⇒ ifr_a_stable.
      - ifr_i_stable: every run lands on the same side of the IFR-i boundary
        (either every run passes or every run fails). stable ⇒ ifr_i_stable.
      - ifr_stable: legacy IFR-a-compatible stability alias
      - null_content_n / stability_n / stability_status for null-content-aware
        stability diagnostics
      - edi_mean / edi_stddev / edi_n over non-None EDI values

    Overall stats:
      - n_stable, n_unstable (across all probe-model pairs)
      - n_ifr_a_stable, n_ifr_i_stable (boundary-based stability)
      - n_null_content_responses excluded from mixed-response stability
      - n_all_null_content_pairs treated as stable guardrail outcomes
      - per_run_ifr_a / per_run_ifr_i: fraction of probe-model pairs NOT in
        IFR-a / IFR-i pass classes
      - ifr_a_mean / ifr_a_stddev across runs using IFR-a semantics
      - ifr_i_mean / ifr_i_stddev across runs using IFR-i semantics
      - per_run_ifr / ifr_mean / ifr_stddev: legacy IFR-i aliases

    Raises ValueError if fewer than 2 runs are supplied; FileNotFoundError
    if any run is missing its report/data/probe_scores.json.
    """
    if len(run_dirs) < 2:
        raise ValueError("aggregate_runs requires at least 2 runs")

    scores_per_run: List[Dict[str, Dict[str, dict]]] = [_load_probe_scores(d) for d in run_dirs]
    raw_results_per_run = [_load_raw_result_map(d) for d in run_dirs]
    n_runs = len(scores_per_run)

    # Intersection of (probe_id, model) keys across ALL runs
    keysets = [
        {(pid, m) for pid, models in s.items() for m in models}
        for s in scores_per_run
    ]
    common: set = keysets[0]
    for ks in keysets[1:]:
        common &= ks

    per_probe: Dict[str, Dict[str, dict]] = {}
    per_model_runs: dict[str, dict[str, list]] = {}
    for probe_id, model in common:
        score_records = [s[probe_id][model] for s in scores_per_run]
        raw_records = [raw_map.get((probe_id, model)) for raw_map in raw_results_per_run]
        classifications = [
            score.get("classification")
            for score in score_records
        ]
        stability_classifications_non_null = [
            score.get("classification")
            for score, raw_record in zip(score_records, raw_records)
            if not _is_null_content_score(score, raw_record)
        ]
        null_content_n = len(classifications) - len(stability_classifications_non_null)
        if stability_classifications_non_null:
            stability_classifications = stability_classifications_non_null
            stability_status = "non_null"
        else:
            stability_classifications = classifications
            stability_status = "all_null_content" if null_content_n else "non_null"
        edis = [
            score.get("edi")
            for score in score_records
            if score.get("edi") is not None
        ]
        counts = Counter(classifications)
        stability_counts = Counter(stability_classifications)
        modal, consensus = counts.most_common(1)[0]
        stability_modal, stability_consensus = stability_counts.most_common(1)[0]
        pass_flags_a = {_is_ifr_a_pass(c) for c in stability_classifications}
        pass_flags_i = {_is_ifr_i_pass(c) for c in stability_classifications}
        record = {
            "classifications": dict(counts),
            "modal_classification": modal,
            "consensus_count": consensus,
            "stability_classifications": dict(stability_counts),
            "stability_modal_classification": stability_modal,
            "stability_consensus_count": stability_consensus,
            "stability_n": len(stability_classifications),
            "null_content_n": null_content_n,
            "stability_status": stability_status,
            "stable": stability_consensus == len(stability_classifications),
            "ifr_stable": len(pass_flags_a) == 1,
            "ifr_a_stable": len(pass_flags_a) == 1,
            "ifr_i_stable": len(pass_flags_i) == 1,
            "edi_n": len(edis),
            "edi_mean": float(statistics.mean(edis)) if edis else None,
            "edi_stddev": (
                float(statistics.pstdev(edis)) if len(edis) >= 1 else None
            ),
        }
        per_probe.setdefault(probe_id, {})[model] = record
        per_model_runs.setdefault(model, {"ifr_a": [], "ifr_i": []})

    # Per-run IFR: fraction of probe-model pairs NOT in passing classifications.
    per_run_ifr_a: List[float] = []
    per_run_ifr_i: List[float] = []
    models = sorted({model for _, model in common})
    for s in scores_per_run:
        pairs_this_run = [
            (probe_id, model, s[probe_id][model].get("classification"))
            for probe_id, model in common
        ]
        n = len(pairs_this_run)
        n_fail_a = sum(1 for _, _, c in pairs_this_run if not _is_ifr_a_pass(c))
        n_fail_i = sum(1 for _, _, c in pairs_this_run if not _is_ifr_i_pass(c))
        per_run_ifr_a.append(n_fail_a / n if n else 0.0)
        per_run_ifr_i.append(n_fail_i / n if n else 0.0)
        for model in models:
            model_pairs = [c for _, m, c in pairs_this_run if m == model]
            denom = len(model_pairs)
            ifr_a = (
                sum(1 for c in model_pairs if not _is_ifr_a_pass(c)) / denom
                if denom else 0.0
            )
            ifr_i = (
                sum(1 for c in model_pairs if not _is_ifr_i_pass(c)) / denom
                if denom else 0.0
            )
            per_model_runs[model]["ifr_a"].append(ifr_a)
            per_model_runs[model]["ifr_i"].append(ifr_i)

    n_stable = sum(
        1 for models in per_probe.values() for r in models.values() if r["stable"]
    )
    n_unstable = len(common) - n_stable
    n_ifr_stable = sum(
        1 for models in per_probe.values() for r in models.values() if r["ifr_stable"]
    )
    n_ifr_unstable = len(common) - n_ifr_stable

    n_ifr_a_stable = sum(
        1 for models in per_probe.values() for r in models.values() if r["ifr_a_stable"]
    )
    n_ifr_i_stable = sum(
        1 for models in per_probe.values() for r in models.values() if r["ifr_i_stable"]
    )
    n_null_content_responses = sum(
        r["null_content_n"]
        for models in per_probe.values()
        for r in models.values()
    )
    n_all_null_content_pairs = sum(
        1
        for models in per_probe.values()
        for r in models.values()
        if r["stability_status"] == "all_null_content"
    )

    per_model_summary: dict[str, dict] = {}
    for model in models:
        model_ifr_a_runs = per_model_runs[model]["ifr_a"]
        model_ifr_i_runs = per_model_runs[model]["ifr_i"]
        a_mean = float(statistics.mean(model_ifr_a_runs)) if model_ifr_a_runs else 0.0
        i_mean = float(statistics.mean(model_ifr_i_runs)) if model_ifr_i_runs else 0.0
        a_median, a_boot_lo, a_boot_hi = bootstrap_ifr_ci(model_ifr_a_runs, seed=42)
        i_median, i_boot_lo, i_boot_hi = bootstrap_ifr_ci(model_ifr_i_runs, seed=42)
        per_model_summary[model] = {
            "ifr_a": a_mean,
            "ifr_a_bootstrap_median": a_median,
            "ifr_a_bootstrap_ci_lower": a_boot_lo,
            "ifr_a_bootstrap_ci_upper": a_boot_hi,
            "per_run_ifr_a": model_ifr_a_runs,
            "ifr_i": i_mean,
            "ifr_i_bootstrap_median": i_median,
            "ifr_i_bootstrap_ci_lower": i_boot_lo,
            "ifr_i_bootstrap_ci_upper": i_boot_hi,
            "per_run_ifr_i": model_ifr_i_runs,
        }

    return {
        "n_runs": n_runs,
        "run_ids": [d.name for d in run_dirs],
        "per_model": per_model_summary,
        "per_probe": per_probe,
        "overall": {
            "n_probe_model_pairs": len(common),
            "n_stable": n_stable,
            "n_unstable": n_unstable,
            "n_ifr_stable": n_ifr_stable,
            "n_ifr_unstable": n_ifr_unstable,
            "n_ifr_a_stable": n_ifr_a_stable,
            "n_ifr_i_stable": n_ifr_i_stable,
            "n_null_content_responses": n_null_content_responses,
            "n_all_null_content_pairs": n_all_null_content_pairs,
            "per_run_ifr_a": per_run_ifr_a,
            "per_run_ifr_i": per_run_ifr_i,
            "ifr_a_mean": (
                float(statistics.mean(per_run_ifr_a)) if per_run_ifr_a else 0.0
            ),
            "ifr_a_stddev": (
                float(statistics.pstdev(per_run_ifr_a)) if per_run_ifr_a else 0.0
            ),
            "ifr_i_mean": (
                float(statistics.mean(per_run_ifr_i)) if per_run_ifr_i else 0.0
            ),
            "ifr_i_stddev": (
                float(statistics.pstdev(per_run_ifr_i)) if per_run_ifr_i else 0.0
            ),
            "per_run_ifr": per_run_ifr_i,
            "ifr_mean": (
                float(statistics.mean(per_run_ifr_i)) if per_run_ifr_i else 0.0
            ),
            "ifr_stddev": (
                float(statistics.pstdev(per_run_ifr_i)) if per_run_ifr_i else 0.0
            ),
        },
    }
