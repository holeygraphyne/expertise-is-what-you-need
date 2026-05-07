"""Judge stage orchestrator (Stage 1 of the auditor).

Sequence:
  1. Re-score raw_results with the current ISScorer (per-paper vocab).
  2. Project to StarredCase[]; filter by `only_starred` + `models`.
  3. Load judge_labels.json cache.
  4. ThreadPool: for each uncached case, build payload + call_judge.
  5. Persist verdicts atomically after each completion.
  6. Build disagreements + clusters + rule_gap aggregates.
  7. Write judge_report.md.

Stage 2 (recommender) lives in `recommender.py` and reads the
artifacts written here.
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Callable, Iterable, Optional

from traces.calibration import diff
from traces.calibration.judge import JudgeError, JudgeRefusedError, call_judge
from traces.calibration.models import Disagreement, JudgeLabel
from traces.calibration.payload import build_judge_payload
from traces.calibration.rescoring import ScoredResponse, rescore_responses, PaperLike
from traces.calibration.starred_selection import select_starred_cases
from traces.config import AuditConfig, ProviderConfig
from traces.influence.scorer import ISScorer

logger = logging.getLogger(__name__)


@dataclass
class JudgeArtifacts:
    cases_in_scope: int
    judged_count: int
    errored_count: int
    disagreement_count: int
    # Agreement rates by starredness, plus an extrapolated error rate.
    # `agreement_starred` / `agreement_unstarred` are None when there are
    # no audited cases of that kind (avoids division-by-zero). The
    # `*_audited_count` fields are the denominators of those rates;
    # `*_corpus_count` are the totals across the whole run (used to
    # weight `implied_error_rate`).
    agreement_starred: Optional[float] = None
    agreement_unstarred: Optional[float] = None
    implied_error_rate: float = 0.0
    starred_audited_count: int = 0
    unstarred_audited_count: int = 0
    starred_corpus_count: int = 0
    unstarred_corpus_count: int = 0
    # Optional: which judge actually answered each successful case
    # (only populated when judge_models has more than one entry).
    judge_dispatch_counts: Optional[dict[str, int]] = None


def _dispatch_with_fallback(
    *,
    judge_models: list[str],
    payload: str,
    rubric: str,
    provider: ProviderConfig,
    audit: AuditConfig,
) -> tuple[str, JudgeLabel]:
    """Run the judge call through a fallback chain. Returns (judge_used, label).

    Semantics:
    - Iterate `judge_models` in order. For each, call `call_judge` once
      (which internally retries up to `provider.max_retries` for transport
      failures and `audit.parse_retries` for parse failures).
    - On `JudgeRefusedError` (deterministic refusal — safety classifier
      output): fall through to the next model in the chain.
    - On any other `JudgeError` (transport, parse, schema): also fall
      through. The next model is unlikely to share the same failure
      mode. (Upstream's old design retried the same model again here;
      we don't because `call_judge` already did its own retries.)
    - On success: return immediately.
    - On chain exhaustion: raise `JudgeError` with the last underlying
      exception's message.
    """
    last_error: Exception | None = None
    for judge_model in judge_models:
        try:
            label = call_judge(
                payload=payload,
                rubric=rubric,
                provider=provider,
                audit=audit,
                model=judge_model,
            )
        except JudgeRefusedError as exc:
            logger.info(
                "judge %r refused; falling through to next model in chain",
                judge_model,
            )
            last_error = exc
            continue
        except JudgeError as exc:
            logger.warning(
                "judge %r raised %s; falling through to next model in chain",
                judge_model, type(exc).__name__,
            )
            last_error = exc
            continue
        return judge_model, label
    raise JudgeError(
        f"all {len(judge_models)} judge model(s) in the fallback chain "
        f"exhausted: {last_error}"
    )


def _load_verdicts(path: Path) -> dict[str, dict]:
    """Load the on-disk verdict cache, filtering out transient errors.

    Errored entries (`{"error": "..."}`) get filtered on load so a
    transient HTTP 500 from a previous run is retried this run. This
    matches `ISRunner._load_checkpoint`'s behavior, which also filters
    failed entries so a network blip doesn't become a permanent gap.

    Successful verdicts are retained; the filter is conservative
    (only the explicit `"error"` shape is dropped).
    """
    if not path.exists():
        return {}
    try:
        all_entries: dict[str, dict] = json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("judge_labels.json was malformed; starting fresh: %s", path)
        return {}
    fresh: dict[str, dict] = {}
    dropped = 0
    for key, raw in all_entries.items():
        if isinstance(raw, dict) and "error" in raw:
            dropped += 1
            continue
        fresh[key] = raw
    if dropped:
        logger.info(
            "judge_labels.json: dropped %d errored entr%s on load (will retry)",
            dropped, "y" if dropped == 1 else "ies",
        )
    return fresh


def _atomic_write_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def _fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _render_judge_report_md(
    *,
    cases_in_scope: int,
    judged_count: int,
    errored_keys: list[str],
    disagreements: list[Disagreement],
    clusters: dict[tuple[str, str], list[Disagreement]],
    rule_gap_aggregates: dict[str, dict[str, int]],
    agreement_starred: Optional[float] = None,
    agreement_unstarred: Optional[float] = None,
    implied_error_rate: float = 0.0,
    starred_audited_count: int = 0,
    unstarred_audited_count: int = 0,
    starred_corpus_count: int = 0,
    unstarred_corpus_count: int = 0,
    judge_dispatch_counts: Optional[dict[str, int]] = None,
) -> str:
    """Render the judge stage's report. Recommender writes a separate findings.md."""
    buf = StringIO()
    buf.write("# Judge Stage Report\n\n")
    buf.write("## Headline\n\n")
    buf.write(f"- Cases in scope: **{cases_in_scope}**\n")
    buf.write(f"- Successfully judged: **{judged_count}**\n")
    buf.write(f"- Errored: **{len(errored_keys)}**\n")
    buf.write(f"- Disagreements: **{len(disagreements)}**\n\n")

    buf.write("## Agreement metrics\n\n")
    buf.write(
        f"- Starred agreement: **{_fmt_pct(agreement_starred)}** "
        f"({starred_audited_count}/{starred_corpus_count} audited)\n"
    )
    buf.write(
        f"- Unstarred agreement: **{_fmt_pct(agreement_unstarred)}** "
        f"({unstarred_audited_count}/{unstarred_corpus_count} audited)\n"
    )
    buf.write(
        f"- Implied corpus-wide scorer error rate: **{implied_error_rate:.1%}**\n\n"
    )

    if judge_dispatch_counts:
        buf.write("## Judge dispatch (fallback chain)\n\n")
        for jm, n in sorted(
            judge_dispatch_counts.items(), key=lambda kv: -kv[1]
        ):
            buf.write(f"- {jm}: **{n}**\n")
        buf.write("\n")

    buf.write("## `rule_gap` aggregate\n\n")
    if not rule_gap_aggregates:
        buf.write("_No rule_gap entries (all judgements agreed with scorer)._\n\n")
    else:
        for gap, per_dir in sorted(
            rule_gap_aggregates.items(),
            key=lambda kv: -sum(kv[1].values()),
        ):
            total = sum(per_dir.values())
            buf.write(f"- **{gap}** (total {total}): {per_dir}\n")
        buf.write("\n")

    buf.write("## Cluster summary\n\n")
    if not clusters:
        buf.write("_No clusters._\n\n")
    else:
        for (s_cls, j_cls), rows in sorted(
            clusters.items(), key=lambda kv: -len(kv[1])
        ):
            buf.write(f"### {s_cls} → {j_cls} ({len(rows)} rows)\n\n")
            for d in rows[:3]:
                buf.write(f"- **{d.probe_id}** / {d.model} (rule_gap: {d.judge_rule_gap})\n")
                buf.write(f"  - reason: {d.judge_reason}\n")
            if len(rows) > 3:
                buf.write(f"  (+ {len(rows) - 3} more)\n")
            buf.write("\n")

    return buf.getvalue()


def run_judge_stage(
    *,
    raw_results: Iterable,
    papers_by_id: dict,
    scorer_factory: Callable[[PaperLike], ISScorer],
    rubric: str,
    audit_dir: Path,
    provider: ProviderConfig,
    audit: AuditConfig,
    only_starred: bool = True,
    models: Optional[set[str]] = None,
    concurrency: int = 8,
    judge_models: Optional[list[str]] = None,
) -> JudgeArtifacts:
    """Run the judge stage end-to-end. See module docstring.

    `judge_models` is a fallback chain: each case is dispatched to the
    first model in the list that doesn't refuse or error. If `None` or
    empty, the chain collapses to `[audit.judge_model]` (single-model
    fast path — no fallback overhead).
    """
    audit_dir.mkdir(parents=True, exist_ok=True)

    models_chain: list[str] = (
        list(judge_models) if judge_models else [audit.judge_model]
    )
    multi_judge = len(models_chain) > 1

    raws = list(raw_results)
    scored_by_key = rescore_responses(
        raw_results=raws, papers_by_id=papers_by_id,
        scorer_factory=scorer_factory,
    )
    cases = select_starred_cases(
        scored_by_key, only_starred=only_starred, models=models,
    )
    cases_by_key = {c.cache_key: c for c in cases}
    logger.info("judge stage: %d cases in scope", len(cases_by_key))

    verdicts_path = audit_dir / "judge_labels.json"
    cached = _load_verdicts(verdicts_path)
    to_judge = [c for c in cases if c.cache_key not in cached]

    def _one(case) -> tuple[str, dict]:
        sr = scored_by_key[case.cache_key]
        payload = build_judge_payload(
            probe_id=case.probe_id,
            paper=sr.paper,
            response_text=case.response_text,
            is_result=sr.is_result,
        )
        try:
            if multi_judge:
                judge_used, label = _dispatch_with_fallback(
                    judge_models=models_chain,
                    payload=payload,
                    rubric=rubric,
                    provider=provider,
                    audit=audit,
                )
            else:
                judge_used = models_chain[0]
                label = call_judge(
                    payload=payload,
                    rubric=rubric,
                    provider=provider,
                    audit=audit,
                    model=judge_used,
                )
            result = label.model_dump(by_alias=True)
            result["_judge_used"] = judge_used
            return case.cache_key, result
        except JudgeError as e:
            logger.warning("judge error for %s: %s", case.cache_key, e)
            return case.cache_key, {"error": str(e)}

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_one, c) for c in to_judge]
        for fut in as_completed(futures):
            key, result = fut.result()
            cached[key] = result
            _atomic_write_json(verdicts_path, cached)

    # Partition cached → judged vs errored, scoped to current cases only.
    judged_by_key: dict[str, JudgeLabel] = {}
    errored_keys: list[str] = []
    judge_dispatch_counts: dict[str, int] = {}
    for key, raw in cached.items():
        if key not in cases_by_key:
            continue  # stale cache entry from a previous broader run
        if "error" in raw:
            errored_keys.append(key)
            continue
        # `_judge_used` is sidecar metadata — strip before validating
        # against JudgeLabel's schema. Older cache entries (written
        # before the multi-judge feature) won't have this key.
        raw_for_label = {k: v for k, v in raw.items() if k != "_judge_used"}
        judge_used = raw.get("_judge_used")
        if judge_used:
            judge_dispatch_counts[judge_used] = (
                judge_dispatch_counts.get(judge_used, 0) + 1
            )
        judged_by_key[key] = JudgeLabel.model_validate(raw_for_label)

    # Diff against scorer
    scoped_scored = {k: scored_by_key[k] for k in cases_by_key}
    disagreements = diff.build_disagreements(
        scoped_scored, judged_by_key, errored_keys,
    )
    clusters = diff.cluster_disagreements(disagreements)
    rule_gap_agg = diff.aggregate_rule_gaps(disagreements)

    _atomic_write_json(
        audit_dir / "disagreements.json",
        [d.model_dump() for d in disagreements],
    )

    # Agreement metrics — computed against the FULL rescored corpus
    # (scored_by_key), not just cases_by_key, so we can extrapolate
    # per-class agreement back over starred/unstarred populations even
    # when only starred cases were judged.
    starred_corpus_count = sum(
        1 for sr in scored_by_key.values()
        if getattr(sr.is_result, "starred", False)
    )
    unstarred_corpus_count = len(scored_by_key) - starred_corpus_count
    starred_agree = starred_total = 0
    unstarred_agree = unstarred_total = 0
    for key, sr in scored_by_key.items():
        label = judged_by_key.get(key)
        if label is None:
            continue
        agreed = label.classification == sr.is_result.classification.value
        if getattr(sr.is_result, "starred", False):
            starred_total += 1
            starred_agree += int(agreed)
        else:
            unstarred_total += 1
            unstarred_agree += int(agreed)
    agreement_starred = (starred_agree / starred_total) if starred_total else None
    agreement_unstarred = (unstarred_agree / unstarred_total) if unstarred_total else None
    implied_numer = 0.0
    if agreement_starred is not None:
        implied_numer += starred_corpus_count * (1 - agreement_starred)
    if agreement_unstarred is not None:
        implied_numer += unstarred_corpus_count * (1 - agreement_unstarred)
    corpus_total = len(scored_by_key)
    implied_error_rate = implied_numer / corpus_total if corpus_total else 0.0

    report_md = _render_judge_report_md(
        cases_in_scope=len(cases_by_key),
        judged_count=len(judged_by_key),
        errored_keys=errored_keys,
        disagreements=disagreements,
        clusters=clusters,
        rule_gap_aggregates=rule_gap_agg,
        agreement_starred=agreement_starred,
        agreement_unstarred=agreement_unstarred,
        implied_error_rate=implied_error_rate,
        starred_audited_count=starred_total,
        unstarred_audited_count=unstarred_total,
        starred_corpus_count=starred_corpus_count,
        unstarred_corpus_count=unstarred_corpus_count,
        judge_dispatch_counts=judge_dispatch_counts if multi_judge else None,
    )
    (audit_dir / "judge_report.md").write_text(report_md)

    return JudgeArtifacts(
        cases_in_scope=len(cases_by_key),
        judged_count=len(judged_by_key),
        errored_count=len(errored_keys),
        disagreement_count=len(disagreements),
        agreement_starred=agreement_starred,
        agreement_unstarred=agreement_unstarred,
        implied_error_rate=implied_error_rate,
        starred_audited_count=starred_total,
        unstarred_audited_count=unstarred_total,
        starred_corpus_count=starred_corpus_count,
        unstarred_corpus_count=unstarred_corpus_count,
        judge_dispatch_counts=judge_dispatch_counts if multi_judge else None,
    )
