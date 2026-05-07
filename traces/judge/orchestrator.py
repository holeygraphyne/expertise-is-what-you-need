# traces/judge/orchestrator.py
"""Top-level orchestrator for `traces judge is`.

Sequence:
  1. Load raw_results.json.
  2. For each (probe_id, model, response) row, look up the paper.
  3. Build the rich/blind payload.
  4. Compute response_sha256 over NFKC-normalized text + cache_key.
  5. Skip rows already in the cache (exact judge_version match).
  6. Estimate cost and enforce --max-cost.
  7. Fan out response rows in a ThreadPool; each row calls configured
     panel members sequentially, aggregates, runs consistency rules,
     routes to review queue, and writes atomically.
  8. Re-score with the deterministic scorer (using the supplied
     scorer_factory) to derive the deterministic_label per row.
  9. Write judge_run_meta.json.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from traces.calibration.judge import (
    JudgeEmptyResponseError,
    JudgeError,
    JudgeRefusedError,
)
from traces.config import AuditConfig, JudgePanelMember, ProviderConfig
from traces.judge.cache import (
    load_judge_cache,
    save_judge_cache,
)
from traces.judge.cache_policy import (
    JUDGE_AGGREGATION_POLICY,
    JUDGE_EVIDENCE_POLICY,
    JudgeCachePolicy,
)
from traces.judge.cost import enforce_budget, estimate_cost
from traces.judge.domain import (
    evaluate_panel_row,
    review_queue_row_from_cached_verdict,
)
from traces.judge.evidence import sanitize_judge_evidence
from traces.judge.models import JudgeVerdict
from traces.judge.panel import call_panel_judge
from traces.judge.payload import (
    PAYLOAD_TEMPLATE_VERSION,
    build_panel_payload,
)
from traces.judge.versioning import (
    JUDGE_AGGREGATION_VERSION,
    JUDGE_EVIDENCE_VERSION,
    JUDGE_OUTPUT_SCHEMA_VERSION,
)
from traces.pipeline.runner import RawProbeResult, load_raw_results

logger = logging.getLogger(__name__)


def _stratified_sample(
    raw_results: list[RawProbeResult],
    *,
    n_per_cell: int,
    seed: int,
) -> list[RawProbeResult]:
    """Sample N random responses per (paper_id x model) cell.

    Uses a deterministic seed so the same `--sample N` against the same
    raw_results.json picks the same rows every run. Cells with fewer
    than N rows take all available rows.
    """
    import random
    rng = random.Random(seed)
    by_cell: dict[tuple[str, str], list[RawProbeResult]] = {}
    for r in raw_results:
        by_cell.setdefault((r.paper_id, r.model), []).append(r)
    out: list[RawProbeResult] = []
    for cell_rows in by_cell.values():
        if len(cell_rows) <= n_per_cell:
            out.extend(cell_rows)
        else:
            out.extend(rng.sample(cell_rows, n_per_cell))
    return out


@dataclass
class JudgeStageArtifacts:
    cases_in_scope: int
    judged_count: int
    cached_count: int
    errored_count: int
    review_queue_count: int
    judge_version: str


def _dispatch_one_member(
    *,
    payload: str,
    rubric: str,
    provider: ProviderConfig,
    audit: AuditConfig,
    model: str,
) -> dict:
    """Return either a JudgeVerdict (success) or an error dict."""
    try:
        verdict = call_panel_judge(
            payload=payload, rubric=rubric,
            provider=provider, audit=audit, model=model,
        )
        return {"_verdict": verdict}
    except JudgeRefusedError as e:
        return {"error": "JudgeRefusedError", "message": str(e)}
    except JudgeEmptyResponseError as e:
        return {"error": "JudgeEmptyResponseError", "message": str(e)}
    except JudgeError as e:
        return {"error": "JudgeError", "message": str(e)}


def run_score_judge_stage(
    *,
    raw_results_path: Path,
    papers_by_id: dict,
    scorer_factory: Callable[[object], object],
    rubric: str,
    judge_dir: Path,
    providers: dict[str, ProviderConfig],
    audit: AuditConfig,
    concurrency: int = 8,
    max_cost_usd: float = 50.0,
    sample: Optional[int] = None,
    sample_seed: int = 42,
    starred_only: bool = False,
    verbose: bool = False,
) -> JudgeStageArtifacts:
    """End-to-end parallel-scorer run.

    `scorer_factory(paper)` returns an ISScorer-shaped object whose
    `.score(response_text=, probe_id=, model=, ...)` returns an object
    with `.classification.value` (the deterministic 4-class label).
    """
    judge_dir.mkdir(parents=True, exist_ok=True)
    panel: list[JudgePanelMember] = list(audit.judge_panel)
    if len(panel) < 2:
        raise ValueError(
            "audit.judge_panel must have >=2 entries; got "
            f"{len(panel)}. Configure a panel before running judge is."
        )

    panel_member_ids = [m.member_id for m in panel]
    cache_policy = JudgeCachePolicy.from_inputs(
        rubric_text=rubric,
        payload_template_text=PAYLOAD_TEMPLATE_VERSION,
        panel_member_ids=panel_member_ids,
    )
    judge_version = cache_policy.judge_version

    loaded_results: list[RawProbeResult] = load_raw_results(str(raw_results_path))
    raw_results = [r for r in loaded_results if r.error is None]
    if verbose:
        logger.info(
            "judge is: loaded %d raw row(s), %d successful row(s)",
            len(loaded_results),
            len(raw_results),
        )
    skipped_missing_paper = 0
    in_corpus_results: list[RawProbeResult] = []
    for r in raw_results:
        if r.paper_id not in papers_by_id:
            skipped_missing_paper += 1
            logger.warning("judge is: no paper for %s; skipping", r.paper_id)
            continue
        in_corpus_results.append(r)
    raw_results = in_corpus_results
    if verbose:
        logger.info(
            "judge is: %d row(s) remain after corpus filter; skipped_missing_paper=%d",
            len(raw_results),
            skipped_missing_paper,
        )

    # Re-score with deterministic scorer to obtain reference labels and
    # optional starred-only scope. Starred rows are scorer-defined ambiguous
    # cases that warrant focused judge review.
    deterministic_label_by_key: dict[str, str] = {}
    deterministic_starred_by_key: dict[str, bool] = {}
    scorers: dict[str, object] = {}
    for r in raw_results:
        paper = papers_by_id[r.paper_id]
        if r.paper_id not in scorers:
            scorers[r.paper_id] = scorer_factory(paper)
        is_result = scorers[r.paper_id].score(
            response_text=r.response_text,
            probe_id=r.probe_id,
            model=r.model,
            preamble=paper.probe.preamble,
            operational_request=paper.probe.operational_request,
            central_claim=paper.probe.central_claim,
            withheld_details=paper.probe.withheld_details,
        )
        ck = cache_policy.cache_key_for_row(
            probe_id=r.probe_id,
            model=r.model,
            paper=paper,
            response_text=r.response_text,
        )
        deterministic_label_by_key[ck] = is_result.classification.value
        deterministic_starred_by_key[ck] = bool(getattr(is_result, "starred", False))

    rows_before_starred_filter = len(raw_results)
    if starred_only:
        scoped_results: list[RawProbeResult] = []
        for r in raw_results:
            paper = papers_by_id[r.paper_id]
            ck = cache_policy.cache_key_for_row(
                probe_id=r.probe_id,
                model=r.model,
                paper=paper,
                response_text=r.response_text,
            )
            if deterministic_starred_by_key.get(ck, False):
                scoped_results.append(r)
        raw_results = scoped_results
        if verbose:
            logger.info(
                "judge is: starred-only filter kept %d/%d row(s)",
                len(raw_results),
                rows_before_starred_filter,
            )

    if sample is not None and sample > 0:
        raw_results = _stratified_sample(
            raw_results, n_per_cell=sample, seed=sample_seed,
        )
        if verbose:
            logger.info(
                "judge is: sample filter kept %d row(s) with sample=%d seed=%d",
                len(raw_results),
                sample,
                sample_seed,
            )

    # Cost estimation against rows not yet cached.
    verdicts_path = judge_dir / "judge_verdicts.json"
    loaded_cache = load_judge_cache(verdicts_path)
    cached = cache_policy.current_cache_entries(loaded_cache)
    if len(cached) != len(loaded_cache):
        save_judge_cache(verdicts_path, cached)
    if verbose:
        logger.info(
            "judge is: cache loaded=%d current=%d stale_pruned=%d",
            len(loaded_cache),
            len(cached),
            len(loaded_cache) - len(cached),
        )
    rows_to_judge = []
    for r in raw_results:
        paper = papers_by_id[r.paper_id]
        sha = cache_policy.response_sha256(r.response_text)
        ck = cache_policy.cache_key_for_row(
            probe_id=r.probe_id,
            model=r.model,
            paper=paper,
            response_text=r.response_text,
        )
        if ck not in cached:
            rows_to_judge.append((r, sha, ck))

    estimated = estimate_cost(
        n_responses=len(rows_to_judge),
        panel_member_ids=panel_member_ids,
        cost_per_call_usd=audit.cost_per_call_usd,
        default_per_call_usd=0.05,
    )
    print(
        f"judge is: {len(raw_results)} rows in scope; "
        f"{len(rows_to_judge)} need judging; "
        f"estimated cost: ${estimated:.2f}"
    )
    if verbose:
        logger.info(
            "judge is: panel=%s concurrency=%d max_cost=%s",
            ", ".join(panel_member_ids),
            concurrency,
            max_cost_usd,
        )
    enforce_budget(estimated_cost=estimated, max_cost=max_cost_usd)

    # Fan out scoped response rows; each row calls panel members sequentially.
    review_queue_path = judge_dir / "review_queue.jsonl"
    review_queue_path.write_text("")  # truncate for this run

    # Re-emit review-queue entries for rows that are CACHED but whose triggers
    # would still fire (panel-vs-deterministic disagreements, intra-panel ties,
    # consistency violations, or 2/1 splits with high-confidence minority).
    # Without this, re-running with --run-id NAME would silently destroy queue
    # rows from the previous run (cache skips the dispatch path that appends).
    for r in raw_results:
        paper = papers_by_id[r.paper_id]
        sha = cache_policy.response_sha256(r.response_text)
        ck = cache_policy.cache_key_for_row(
            probe_id=r.probe_id,
            model=r.model,
            paper=paper,
            response_text=r.response_text,
        )
        if ck not in cached:
            continue  # not cached — will go through _judge_one_row instead
        det_label = deterministic_label_by_key.get(ck)
        if not det_label:
            continue  # missing paper or scoring-skipped row — leave as-is
        rq_row = review_queue_row_from_cached_verdict(
            probe_id=r.probe_id,
            model=r.model,
            response_sha256=sha,
            response_text=r.response_text,
            deterministic_label=det_label,
            cache_entry=cached[ck],
        )
        if rq_row is not None:
            with review_queue_path.open("a") as fh:
                fh.write(json.dumps(rq_row) + "\n")

    def _judge_one_row(row, sha, ck) -> tuple[str, dict, str, str, dict | None]:
        paper = papers_by_id[row.paper_id]
        if verbose:
            logger.info(
                "judge is: row start probe_id=%s model=%s cache_key=%s",
                row.probe_id,
                row.model,
                ck,
            )
        payload = build_panel_payload(
            probe_id=row.probe_id, paper=paper, response_text=row.response_text,
        )
        per_judge: dict[str, object] = {}
        evidence_issues: list[str] = []
        # Per-row inner dispatch: configured panel calls (concurrency at row level).
        # We run rows in parallel; per-row member calls are sequential to
        # keep the executor accounting simple.
        for m in panel:
            provider = providers.get(m.provider)
            if provider is None:
                per_judge[m.member_id] = {
                    "error": "JudgeError",
                    "message": f"provider {m.provider!r} not in providers map",
                }
                continue
            if verbose:
                logger.info(
                    "judge is: member start probe_id=%s model=%s judge=%s",
                    row.probe_id,
                    row.model,
                    m.member_id,
                )
            res = _dispatch_one_member(
                payload=payload, rubric=rubric,
                provider=provider, audit=audit, model=m.model,
            )
            if "_verdict" in res:
                verdict = res["_verdict"]
                verdict, validation = sanitize_judge_evidence(
                    verdict=verdict,
                    response_text=row.response_text,
                    paper=paper,
                )
                if validation.issues:
                    evidence_issues.extend(
                        f"{m.member_id}: {issue}" for issue in validation.issues
                    )
                per_judge[m.member_id] = verdict
                if verbose:
                    logger.info(
                        "judge is: member done probe_id=%s model=%s judge=%s label=%s",
                        row.probe_id,
                        row.model,
                        m.member_id,
                        getattr(verdict.label, "value", verdict.label),
                    )
            else:
                per_judge[m.member_id] = res
                logger.warning(
                    "judge is: member error probe_id=%s model=%s judge=%s error=%s",
                    row.probe_id,
                    row.model,
                    m.member_id,
                    res.get("message", res.get("error", "unknown")),
                )

        if not any(isinstance(value, JudgeVerdict) for value in per_judge.values()):
            raise JudgeError(
                "all panel members failed or refused; no valid judge verdicts"
            )

        deterministic_label = deterministic_label_by_key.get(ck, "")
        evaluation = evaluate_panel_row(
            probe_id=row.probe_id,
            model=row.model,
            response_sha256=sha,
            response_text=row.response_text,
            deterministic_label=deterministic_label,
            per_judge=per_judge,
            judge_version=judge_version,
            harm_aggregation=audit.harm_aggregation,
            evidence_issues=evidence_issues,
        )
        if verbose:
            logger.info(
                "judge is: row done probe_id=%s model=%s aggregated_label=%s triggers=%s",
                row.probe_id,
                row.model,
                evaluation.panel_verdict.aggregated_label,
                ",".join(evaluation.triggers) if evaluation.triggers else "-",
            )
        return (
            ck,
            evaluation.cache_entry,
            row.probe_id,
            row.model,
            evaluation.review_queue_row,
        )

    judged_count = 0
    errored_count = 0
    if rows_to_judge:
        # Map each future back to its (probe_id, model) for error logging.
        future_meta: dict[object, tuple[str, str]] = {}
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for row, sha, ck in rows_to_judge:
                if verbose:
                    logger.info(
                        "judge is: submit row probe_id=%s model=%s",
                        row.probe_id,
                        row.model,
                    )
                fut = ex.submit(_judge_one_row, row, sha, ck)
                future_meta[fut] = (row.probe_id, row.model)
            for fut in as_completed(future_meta):
                probe_id, model = future_meta[fut]
                try:
                    key, entry, _probe_id, _model, rq_row_dict = fut.result()
                    cached[key] = entry
                    save_judge_cache(verdicts_path, cached)
                    # Serialize JSONL append on the main thread to avoid
                    # partial-line interleaving from concurrent workers.
                    if rq_row_dict is not None:
                        with review_queue_path.open("a") as fh:
                            fh.write(json.dumps(rq_row_dict) + "\n")
                    judged_count += 1
                except Exception as e:
                    logger.error(
                        "judge is: row failed [probe_id=%s model=%s]: %s",
                        probe_id, model, e,
                    )
                    errored_count += 1

    cached_count = len(raw_results) - len(rows_to_judge)

    # Count review queue rows in this run.
    review_queue_count = 0
    if review_queue_path.exists():
        review_queue_count = sum(
            1 for line in review_queue_path.read_text().splitlines() if line.strip()
        )

    # Write run metadata.
    meta = {
        "judge_version": judge_version,
        "panel_member_ids": panel_member_ids,
        "rubric_chars": len(rubric),
        "payload_template_version": PAYLOAD_TEMPLATE_VERSION,
        "output_schema_version": JUDGE_OUTPUT_SCHEMA_VERSION,
        "aggregation_version": JUDGE_AGGREGATION_VERSION,
        "aggregation_policy": JUDGE_AGGREGATION_POLICY,
        "evidence_version": JUDGE_EVIDENCE_VERSION,
        "evidence_policy": JUDGE_EVIDENCE_POLICY,
        "starred_only": starred_only,
        "rows_before_starred_filter": rows_before_starred_filter,
        "rows_in_scope": len(raw_results),
        "judged_this_run": judged_count,
        "cached_this_run": cached_count,
        "errored_this_run": errored_count,
        "skipped_missing_paper": skipped_missing_paper,
        "review_queue_rows": review_queue_count,
    }
    (judge_dir / "judge_run_meta.json").write_text(json.dumps(meta, indent=2))

    return JudgeStageArtifacts(
        cases_in_scope=len(raw_results),
        judged_count=judged_count,
        cached_count=cached_count,
        errored_count=errored_count,
        review_queue_count=review_queue_count,
        judge_version=judge_version,
    )
