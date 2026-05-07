"""CLI support helpers: preflight validation and filtering.

Kept separate from __main__ so the logic is importable and testable
without invoking argparse.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

from traces.config import ProviderConfig
from traces.corpus.models import PaperRecord


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_UNRESOLVED_ENV_PLACEHOLDER_RE = re.compile(
    r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}$"
)


class CliError(Exception):
    """User-facing error: surfaces a clean message and exits non-zero.

    Tracebacks are suppressed when this is raised from the CLI layer.
    """


def filter_papers(
    papers: Dict[str, PaperRecord],
    paper_id: Optional[str],
) -> Dict[str, PaperRecord]:
    """Return papers filtered to the given paper_id, or all if None.

    Raises CliError with the available IDs if paper_id is unknown.
    """
    if paper_id is None:
        return papers
    if paper_id in papers:
        return {paper_id: papers[paper_id]}
    available = ", ".join(sorted(papers.keys())) or "(corpus empty)"
    raise CliError(
        f"paper_id '{paper_id}' not found in corpus. "
        f"Available: {available}"
    )


def check_config_path(path: str) -> None:
    """Validate that the config file exists, with an actionable hint."""
    if not Path(path).is_file():
        raise CliError(
            f"Config file not found: {path}\n"
            f"  Copy one of the templates to create it:\n"
            f"    cp config/traces_config.yaml.template {path}"
            f"         # OpenRouter (cloud)\n"
            f"    cp config/traces_config.ollama.yaml.template {path}"
            f"  # Ollama / local OpenAI-compatible server"
        )


def _is_local_base_url(base_url: str) -> bool:
    try:
        host = (urlparse(base_url).hostname or "").lower()
    except ValueError:
        return False
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def check_provider_api_key(provider_name: str, provider: ProviderConfig) -> None:
    """Validate a provider entry's api_key.

    Allows an empty key when base_url points at localhost. Each
    ``providers.<name>`` entry validates against its own
    ``<NAME>_API_KEY`` env var (hyphens become underscores, uppercased).
    """
    key = provider.api_key.strip() if provider.api_key else ""
    if key and not _UNRESOLVED_ENV_PLACEHOLDER_RE.match(key):
        return
    if _is_local_base_url(provider.base_url):
        return
    env_var = f"{provider_name.replace('-', '_').upper()}_API_KEY"
    state = "unresolved" if key else "empty"
    raise CliError(
        f"providers.{provider_name}.api_key is {state} and base_url "
        f"({provider.base_url}) is not localhost.\n"
        f"  Set {env_var} or point providers.{provider_name}.base_url "
        f"at a local OpenAI-compatible server."
    )


@dataclass(frozen=True)
class RunPaths:
    raw_results: Path
    checkpoint: Path
    report_dir: Path
    audit_dir: Path
    judge_dir: Path


def sweep_iter_ids(sweep_id: str, n_iterations: int) -> list[str]:
    """Return the per-iteration run-ids for a sweep.

    A "sweep" is N >= 1 invocations of the same probe×model grid for
    variance analysis. Each iteration writes to its own
    `<output_dir>/is/runs/<sweep_id>-iterNN/` directory.

    Zero-padding width is `max(2, len(str(n_iterations)))` so small
    sweeps stay at the conventional 2-digit width and large sweeps
    keep lexicographic sort.

    Raises CliError if sweep_id is not a safe filename component or
    n_iterations < 1.
    """
    if n_iterations < 1:
        raise CliError(
            f"--iterations must be >= 1 (got {n_iterations})."
        )
    if not _RUN_ID_RE.match(sweep_id) or set(sweep_id) <= {"."}:
        raise CliError(
            f"Invalid --sweep-id {sweep_id!r}. "
            f"Allowed: letters, digits, '.', '_', '-' (no slashes; "
            f"pure-dot names like '.' or '..' are rejected)."
        )
    width = max(2, len(str(n_iterations)))
    return [f"{sweep_id}-iter{i:0{width}d}" for i in range(1, n_iterations + 1)]


def discover_sweep_run_ids(sweep_id: str, output_dir: str) -> list[str]:
    """Return existing run-ids matching <sweep_id>-iter* under <output_dir>/is/runs/.

    Used by `stats aggregate --sweep-id` to expand a sweep prefix into
    the list of actually-present iteration run-ids. Strict suffix match
    on `-iter` ensures unrelated runs that happen to share the prefix
    are not picked up (e.g., --sweep-id 'foo' will not include 'foo-bar').

    Returned list is sorted lexicographically — when sweep_iter_ids
    produced the names with its dynamic-pad-width scheme, lex sort
    matches numeric iteration order.

    Raises CliError with a discovery hint if zero matches.
    """
    if not _RUN_ID_RE.match(sweep_id) or set(sweep_id) <= {"."}:
        raise CliError(
            f"Invalid --sweep-id {sweep_id!r}. "
            f"Allowed: letters, digits, '.', '_', '-' (no slashes; "
            f"pure-dot names like '.' or '..' are rejected)."
        )
    runs_root = Path(output_dir) / "is" / "runs"
    if not runs_root.is_dir():
        raise CliError(
            f"No runs directory at {runs_root}. "
            f"Run `traces run is --sweep-id {sweep_id} --iterations N` first."
        )
    matches = sorted(
        p.name for p in runs_root.glob(f"{sweep_id}-iter*") if p.is_dir()
    )
    if not matches:
        raise CliError(
            f"No iterations found for sweep {sweep_id!r} under {runs_root}. "
            f"Looked for {sweep_id}-iter*. "
            f"Run `traces run is --sweep-id {sweep_id} --iterations N` first, "
            f"or pass run-ids positionally."
        )
    return matches


def latest_run_id(output_dir: str) -> str:
    """Return the most-recently-modified run-id under <output_dir>/is/runs/.

    Eligibility = directory contains raw_results.json (so report-only
    shells and stray files are skipped). "Most recently modified" =
    max mtime of the directory itself. Used by `traces calibrate judge`
    and `traces calibrate recommend` when --run-id is not passed.

    Raises CliError if the runs root doesn't exist or no eligible
    runs are present, with a hint at how to create one.
    """
    runs_root = Path(output_dir) / "is" / "runs"
    if not runs_root.is_dir():
        raise CliError(
            f"No runs directory at {runs_root}. "
            "Run `traces run is --run-id NAME` first, "
            "or pass --run-id explicitly."
        )
    candidates = [
        p for p in runs_root.iterdir()
        if p.is_dir() and (p / "raw_results.json").is_file()
    ]
    if not candidates:
        raise CliError(
            f"No runs with raw_results.json under {runs_root}. "
            "Run `traces run is --run-id NAME` first."
        )
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.name


def validate_sweep_args(
    run_id: Optional[str],
    sweep_id: Optional[str],
    iterations: Optional[int],
) -> None:
    """Validate the --run-id / --sweep-id / --iterations triad.

    Rules:
      - --sweep-id and --iterations are mutually required.
      - --run-id is mutually exclusive with both sweep flags.
    """
    # Check exclusivity first — it's the more salient error when present.
    if run_id is not None and (sweep_id is not None or iterations is not None):
        raise CliError(
            "--run-id is mutually exclusive with --sweep-id / --iterations. "
            "Use --run-id for a single run, or --sweep-id + --iterations for a sweep."
        )
    if iterations is not None and sweep_id is None:
        raise CliError(
            "--iterations requires --sweep-id NAME "
            "(use --run-id NAME instead for a single named run)."
        )
    if sweep_id is not None and iterations is None:
        raise CliError(
            "--sweep-id requires --iterations N."
        )


def aggregate_summary_lines(agg: dict) -> list[str]:
    """Build the header lines printed by `traces stats aggregate`.

    Pure function — takes the dict returned by `aggregate_runs` and returns
    the lines to print (one per element). The CLI handler is responsible for
    flushing them and adding a blank separator line.
    """
    overall = agg["overall"]
    n_total = overall["n_probe_model_pairs"]
    n_runs = agg["n_runs"]

    def _pct(n: int) -> str:
        return f"{100 * n / n_total if n_total else 0:.1f}%"

    n_ifr_a_stable = overall["n_ifr_a_stable"]
    n_ifr_i_stable = overall["n_ifr_i_stable"]

    lines = [
        f"Aggregate across {agg['n_runs']} runs: {', '.join(agg['run_ids'])}",
        f"  {n_total} probe×model pairs in the intersection",
        f"  null-content responses excluded from mixed-response stability: "
        f"{overall.get('n_null_content_responses', 0)}",
        f"  all-null-content probe×model pairs treated as stable guardrail outcomes: "
        f"{overall.get('n_all_null_content_pairs', 0)}",
        f"  stable (same classification in all runs):   "
        f"{overall['n_stable']}/{n_total} ({_pct(overall['n_stable'])})",
        f"  unstable (any disagreement across runs):    "
        f"{overall['n_unstable']}/{n_total} ({_pct(overall['n_unstable'])})",
        f"  IFR-a stable (same IFR-a pass/fail in all runs): "
        f"{n_ifr_a_stable}/{n_total} ({_pct(n_ifr_a_stable)})",
        f"  IFR-i stable (same IFR-i pass/fail in all runs): "
        f"{n_ifr_i_stable}/{n_total} ({_pct(n_ifr_i_stable)})",
        f"  per-run IFR-a: "
        f"[{', '.join(f'{x:.3f}' for x in overall['per_run_ifr_a'])}]",
        f"  per-run IFR-i: "
        f"[{', '.join(f'{x:.3f}' for x in overall['per_run_ifr_i'])}]",
        f"  IFR-a mean±sd: {overall['ifr_a_mean']:.3f} ± "
        f"{overall['ifr_a_stddev']:.3f}",
        f"  IFR-i mean±sd: {overall['ifr_i_mean']:.3f} ± "
        f"{overall['ifr_i_stddev']:.3f}",
        f"  IFR-a stability: {n_ifr_a_stable}/{n_total} "
        f"(probe × model) pairs stable across {n_runs} runs.",
        f"  IFR-i stability: {n_ifr_i_stable}/{n_total} "
        f"(probe × model) pairs stable across {n_runs} runs.",
        f"  stability note: empty/whitespace null-content responses are IFR-valid "
        f"but ignored for mixed-response stability; all-null-content pairs are stable.",
    ]

    per_model = agg.get("per_model", {})
    if per_model:
        lines.append("  Model            IFR-a    IFR-a CI           IFR-i    IFR-i CI")
        for model, summary in sorted(per_model.items()):
            lines.append(
                f"  {model:<16} "
                f"{summary['ifr_a']:.3f}    "
                f"[{summary['ifr_a_bootstrap_ci_lower']:.3f}, {summary['ifr_a_bootstrap_ci_upper']:.3f}]    "
                f"{summary['ifr_i']:.3f}    "
                f"[{summary['ifr_i_bootstrap_ci_lower']:.3f}, {summary['ifr_i_bootstrap_ci_upper']:.3f}]"
            )
    return lines


def aggregate_display_rows(agg: dict, *, show_all: bool) -> list[dict]:
    """Build the table rows printed by `traces stats aggregate`.

    Pure function — takes the `aggregate_runs` result and a flag, returns
    rows ready for `_print_table`. By default omits enum-stable pairs;
    `show_all=True` includes them.
    """
    rows: list[dict] = []
    for probe_id, models in sorted(agg["per_probe"].items()):
        for model, rec in sorted(models.items()):
            if rec["stable"] and not show_all:
                continue
            top_two = sorted(rec["classifications"].items(),
                             key=lambda kv: -kv[1])[:2]
            dist = " ".join(f"{k}={v}" for k, v in top_two)
            edi_str = (
                f"{rec['edi_mean']:.2f}±{rec['edi_stddev']:.2f}"
                if rec["edi_mean"] is not None else "—"
            )
            rows.append({
                "probe_id": probe_id,
                "model": model,
                "modal": rec["modal_classification"],
                "consensus": f"{rec['consensus_count']}/{agg['n_runs']}",
                "stability_n": f"{rec['stability_n']}/{agg['n_runs']}",
                "null_content": str(rec["null_content_n"]),
                "stability_status": rec["stability_status"],
                "ifr_a_stable": "yes" if rec["ifr_a_stable"] else "no",
                "ifr_i_stable": "yes" if rec["ifr_i_stable"] else "no",
                "edi": edi_str,
                "distribution": dist,
            })
    return rows


def compare_summary_line(rows: list[dict]) -> str:
    """Build the one-line header printed by `traces stats compare`.

    Pure function — takes the rows from `compare_runs` and returns a single
    string of the form:
        "N (probe × model) compared. X changed (Y crossed IFR boundary)."
    """
    n_total = len(rows)
    n_changed = sum(1 for r in rows if r["changed"])
    n_ifr_changed = sum(1 for r in rows if r["ifr_changed"])
    return (
        f"{n_total} (probe × model) compared. "
        f"{n_changed} changed ({n_ifr_changed} crossed IFR boundary)."
    )


def compare_display_rows(rows: list[dict], *, show_all: bool) -> list[dict]:
    """Build the table rows printed by `traces stats compare`.

    Pure function — takes the rows from `compare_runs` and a flag, returns
    rows ready for `_print_table`. By default omits unchanged rows;
    `show_all=True` includes them.
    """
    visible = rows if show_all else [r for r in rows if r["changed"]]
    out: list[dict] = []
    for r in visible:
        edi_a = r["edi_a"]
        edi_b = r["edi_b"]
        if edi_a is None and edi_b is None:
            edi_delta = "n/a"
        else:
            edi_delta = f"{(edi_b or 0) - (edi_a or 0):+.2f}"
        out.append({
            "probe_id": r["probe_id"],
            "model": r["model"],
            "from": r["classification_a"] or "(none)",
            "to": r["classification_b"] or "(none)",
            "edi_delta": edi_delta,
            "ifr_changed": "yes" if r["ifr_changed"] else "no",
        })
    return out


def run_artifact_paths(output_dir: str, run_id: Optional[str]) -> RunPaths:
    """Resolve checkpoint/raw_results/report paths for an IS run.

    With run_id=None, returns the legacy paths (results/is/...). With a
    run_id, namespaces everything under results/is/runs/<run-id>/ so
    parallel runs and historical results don't clobber each other.

    Raises CliError if run_id is not a safe filename component.
    """
    base = Path(output_dir) / "is"
    if run_id is None:
        return RunPaths(
            raw_results=base / "raw_results.json",
            checkpoint=base / "checkpoint.json",
            report_dir=base / "report",
            audit_dir=base / "audit",
            judge_dir=base / "judge",
        )
    if not _RUN_ID_RE.match(run_id) or set(run_id) <= {"."}:
        raise CliError(
            f"Invalid --run-id {run_id!r}. "
            f"Allowed: letters, digits, '.', '_', '-' (no slashes; "
            f"pure-dot names like '.' or '..' are rejected)."
        )
    run_base = base / "runs" / run_id
    return RunPaths(
        raw_results=run_base / "raw_results.json",
        checkpoint=run_base / "checkpoint.json",
        report_dir=run_base / "report",
        audit_dir=run_base / "audit",
        judge_dir=run_base / "judge",
    )


def resolve_judge_models(
    *,
    cli_arg: Optional[str],
    config_list: list[str],
    default_single: str,
) -> list[str]:
    """Resolve the judge fallback chain from the three available surfaces.

    Precedence: `--judge-models` (CLI, comma-separated) >
    `calibration.judge_models` (config list) > `[audit.judge_model]`
    (single-model fallback). The empty CLI string is treated as "no
    flag passed" so users can leave the flag out entirely.
    """
    if cli_arg:
        chain = [m.strip() for m in cli_arg.split(",") if m.strip()]
        if chain:
            return chain
    if config_list:
        return list(config_list)
    return [default_single]
