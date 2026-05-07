"""Pure markdown rendering for OptimizationFinding lists."""
from __future__ import annotations

from io import StringIO

from traces.calibration.domain import OptimizationFinding


def aggregate_findings_by_target(
    findings: list[OptimizationFinding],
) -> list[dict]:
    """Group by `target_locator`, summing evidence counts.

    Returns a list of dicts ordered by evidence count desc:
      [{
        target_kind, target_locator, count,
        evidence_case_keys, hypotheses, suggestions, rule_gaps
      }, ...]
    """
    bucket: dict[str, dict] = {}
    for f in findings:
        if f.target_locator not in bucket:
            bucket[f.target_locator] = {
                "target_kind": f.target_kind,
                "target_locator": f.target_locator,
                "evidence_case_keys": list(f.evidence_case_keys),
                "hypotheses": [f.hypothesis],
                "suggestions": [f.suggestion],
                "rule_gaps": [f.rule_gap],
            }
        else:
            entry = bucket[f.target_locator]
            entry["evidence_case_keys"].extend(f.evidence_case_keys)
            entry["hypotheses"].append(f.hypothesis)
            entry["suggestions"].append(f.suggestion)
            entry["rule_gaps"].append(f.rule_gap)
    rows = []
    for entry in bucket.values():
        unique_keys = list(dict.fromkeys(entry["evidence_case_keys"]))
        rows.append({
            "target_kind": entry["target_kind"],
            "target_locator": entry["target_locator"],
            "count": len(unique_keys),
            "evidence_case_keys": unique_keys,
            "hypotheses": entry["hypotheses"],
            "suggestions": entry["suggestions"],
            "rule_gaps": list(dict.fromkeys(entry["rule_gaps"])),
        })
    rows.sort(key=lambda r: -r["count"])
    return rows


def render_findings_md(findings: list[OptimizationFinding]) -> str:
    buf = StringIO()
    buf.write("# Optimization Findings\n\n")
    if not findings:
        buf.write("_None._\n")
        return buf.getvalue()
    grouped = aggregate_findings_by_target(findings)
    buf.write(
        f"{len(findings)} raw finding(s) → {len(grouped)} unique target(s), "
        f"sorted by evidence count.\n\n"
    )
    for idx, row in enumerate(grouped, start=1):
        buf.write(
            f"## {idx}. `{row['target_locator']}`  "
            f"(kind: {row['target_kind']}, evidence: {row['count']})\n\n"
        )
        buf.write(f"- **rule_gap(s) addressed:** {', '.join(row['rule_gaps'])}\n")
        buf.write(
            f"- **evidence cases:** "
            f"{', '.join(row['evidence_case_keys']) or '_(unspecified)_'}\n\n"
        )
        for hyp, sug in zip(row["hypotheses"], row["suggestions"]):
            buf.write(f"**Hypothesis:** {hyp}\n\n")
            buf.write(f"**Suggestion:** {sug}\n\n")
    return buf.getvalue()
