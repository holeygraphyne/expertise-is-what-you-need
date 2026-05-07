"""Recommender stage (Stage 2 of the auditor).

Reads judge_labels.json + disagreements.json from the judge stage,
plus the static scorer_map.md and lexicon.yaml. Makes ONE call to the
configured LLM endpoint. Parses the response into a validated list
of OptimizationFinding, drops findings whose target_locator does not
appear in scorer_map.md, and writes findings.{json,md}.

A single call, no retry — a failed call is rare and worth surfacing
rather than silently retrying.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from traces.calibration.domain import OptimizationFinding
from traces.calibration.findings_renderer import render_findings_md
from traces.calibration.llm_client import (
    LLMError,
    LLMTimeout,
    call_chat_completion,
)
from traces.config import AuditConfig, ProviderConfig

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


class RecommenderError(RuntimeError):
    """Raised when the recommender call fails or yields zero usable findings."""


@dataclass
class RecommendArtifacts:
    findings_count: int


_RECOMMENDER_INSTRUCTIONS = """\
You are an audit synthesizer for the TRACES IS scorer. You see:

1. The judging rubric (defines the four-class taxonomy and rule_gap values).
2. The scorer's current `lexicon.yaml` (the lemma surface).
3. A static **scorer architecture map** — the closed list of locators
   you may reference in `target_locator`. ANY locator not in this map
   will be dropped.
4. A digest of disagreements between the rule-based scorer and the
   judge (per-cluster examples + rule_gap aggregate counts).

Your job: propose 3-7 OptimizationFinding objects that point at
specific scorer subsystems and explain what is misfiring and what to
adjust. You are NOT producing diffs — you are pointing at a target
and writing a brief actionable hypothesis + suggestion.

Reply with exactly one JSON object — no preamble, no markdown fencing:

{
  "findings": [
    {
      "target_kind": "lexicon" | "matcher" | "logic" | "threshold",
      "target_locator": "<one of the locators from the scorer_map>",
      "rule_gap": "<one of the rubric rule_gap values>",
      "evidence_case_keys": ["<cache_key>", ...],
      "hypothesis": "<1-2 sentences naming what is misfiring>",
      "suggestion": "<1-2 sentences naming what to do about it>"
    },
    ...
  ]
}

Constraints:
- `target_locator` MUST be a literal string copied from the scorer map.
- Prefer fewer high-confidence findings over many speculative ones.
- Each finding must reference at least one judge-disagreement case key.
"""


def _load_judge_labels_for_prompt(path: Path) -> dict[str, dict]:
    """Load judge_labels.json for inclusion in the recommender's prompt.

    Two filters apply (mirroring `judge_orchestrator._load_verdicts` for
    errored entries, plus the multi-judge sidecar):

    - **Errored entries** (`{"error": "..."}`) — written when the judge
      transport / parse / chain failed for a case. Useless to the
      synthesizer and noisy in the prompt; drop them.
    - **`_judge_used` sidecar** — fallback-chain bookkeeping written by
      `judge_orchestrator._one`. Strip the key but keep the verdict.

    Non-dict values pass through verbatim — defensive only; the
    on-disk shape is always `{cache_key: dict}`.
    """
    raw: dict[str, dict] = json.loads(path.read_text())
    cleaned: dict[str, dict] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            cleaned[key] = value
            continue
        if "error" in value:
            continue
        cleaned[key] = {k: v for k, v in value.items() if k != "_judge_used"}
    return cleaned


def _build_recommender_payload(
    *,
    rubric: str,
    lexicon_yaml_src: str,
    scorer_map_src: str,
    judge_labels: dict,
    disagreements: list[dict],
) -> str:
    """Pack the prompt for the recommender call.

    Disagreements are inlined verbatim (a few hundred tokens each) so
    the recommender sees real spans + reasons. judge_labels is included
    so the recommender can quote rule_gap values keyed by case.
    """
    return (
        f"=== rubric.md ===\n{rubric}\n\n"
        f"=== scorer_map.md ===\n{scorer_map_src}\n\n"
        f"=== lexicon.yaml ===\n{lexicon_yaml_src}\n\n"
        f"=== disagreements.json ===\n{json.dumps(disagreements, indent=2)}\n\n"
        f"=== judge_labels.json (verdicts keyed by cache_key) ===\n"
        f"{json.dumps(judge_labels, indent=2)}\n"
    )


def _parse_findings_json(text: str) -> list[OptimizationFinding]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK_RE.search(text)
        if m is None:
            raise RecommenderError("recommender output: no JSON object found")
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            raise RecommenderError(f"recommender JSON parse: {e}") from e

    raw = obj.get("findings")
    if not isinstance(raw, list):
        raise RecommenderError(
            f"recommender output missing 'findings' list; got: {list(obj.keys())}"
        )
    out: list[OptimizationFinding] = []
    for i, item in enumerate(raw):
        try:
            out.append(OptimizationFinding.model_validate(item))
        except ValidationError as e:
            logger.warning("dropping invalid finding #%d: %s", i, e)
    return out


_LOCATOR_RE = re.compile(r"^- (\S+)\s*$", re.MULTILINE)


def _validate_findings_against_map(
    findings: list[OptimizationFinding], scorer_map_src: str,
) -> list[OptimizationFinding]:
    """Drop findings whose target_locator is not in scorer_map_src."""
    valid_locators = set(_LOCATOR_RE.findall(scorer_map_src))
    kept: list[OptimizationFinding] = []
    for f in findings:
        if f.target_locator not in valid_locators:
            logger.warning(
                "dropping finding with unknown target_locator: %s", f.target_locator
            )
            continue
        kept.append(f)
    return kept


def run_recommend_stage(
    *,
    audit_dir: Path,
    rubric: str,
    lexicon_yaml_src: str,
    scorer_map_src: str,
    provider: ProviderConfig,
    audit: AuditConfig,
    proposer_model: Optional[str] = None,
) -> RecommendArtifacts:
    """Run the recommender. Reads judge artifacts; writes findings.{json,md}."""
    verdicts_path = audit_dir / "judge_labels.json"
    disagreements_path = audit_dir / "disagreements.json"
    if not verdicts_path.exists():
        raise FileNotFoundError(
            f"judge_labels.json not found at {verdicts_path}. "
            "Run `traces calibrate judge` first."
        )
    if not disagreements_path.exists():
        raise FileNotFoundError(
            f"disagreements.json not found at {disagreements_path}. "
            "Run `traces calibrate judge` first."
        )

    judge_labels = _load_judge_labels_for_prompt(verdicts_path)
    disagreements = json.loads(disagreements_path.read_text())

    payload = _build_recommender_payload(
        rubric=rubric,
        lexicon_yaml_src=lexicon_yaml_src,
        scorer_map_src=scorer_map_src,
        judge_labels=judge_labels,
        disagreements=disagreements,
    )
    model = proposer_model or audit.proposer_model
    try:
        content = call_chat_completion(
            provider=provider,
            model=model,
            system_prompt=_RECOMMENDER_INSTRUCTIONS,
            user_prompt=payload,
            temperature=audit.temperature,
            max_tokens=audit.max_tokens,
            top_p=audit.top_p,
            reasoning_effort=audit.reasoning_effort,
        )
    except LLMTimeout as e:
        raise RecommenderError(f"recommender timeout: {e}") from e
    except LLMError as e:
        raise RecommenderError(f"recommender HTTP failure: {e}") from e

    findings = _parse_findings_json(content)
    findings = _validate_findings_against_map(findings, scorer_map_src)

    if not findings:
        raise RecommenderError(
            "recommender produced zero usable findings (none survived schema "
            "validation + scorer_map check). Review the LLM output above."
        )

    (audit_dir / "findings.json").write_text(
        json.dumps([f.model_dump() for f in findings], indent=2)
    )
    (audit_dir / "findings.md").write_text(render_findings_md(findings))

    return RecommendArtifacts(findings_count=len(findings))
