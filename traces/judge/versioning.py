"""Compute the judge_version cache-invalidation key.

The judge_version is a stable 12-char SHA256 prefix that changes
exactly when one of:
  - the rubric text
  - the payload template text
  - the panel composition (set of member ids, order-independent)
  - the JudgeVerdict output schema version
  - the aggregation policy/version
  - the evidence validation/sanitization policy/version
changes. Recorded on every PanelVerdict and used as a component of
the cache key, so any of those changes invalidates affected cache
entries atomically.
"""
from __future__ import annotations

import hashlib

# Bump manually whenever JudgeVerdict gains/drops a field or its
# semantics change in a way that makes old cached verdicts unsafe
# to reuse.
JUDGE_OUTPUT_SCHEMA_VERSION = "3"

# Bump whenever aggregate_panel_verdict semantics change in a way that makes
# cached PanelVerdict aggregate fields unsafe to reuse.
JUDGE_AGGREGATION_VERSION = "2"

# Bump whenever validate_judge_evidence / sanitize_judge_evidence semantics
# change in a way that affects stored evidence or review-queue routing.
JUDGE_EVIDENCE_VERSION = "2"


def compute_judge_version(
    rubric_text: str,
    payload_template_text: str,
    panel_member_ids: list[str],
    output_schema_version: str,
    aggregation_version: str = "",
    aggregation_policy: str = "",
    evidence_version: str = "",
    evidence_policy: str = "",
) -> str:
    """Return the 12-char hex prefix of the SHA256 over version inputs.

    `panel_member_ids` is sorted before hashing so panel order does
    not affect the version (the panel is a set, not a sequence).
    """
    sorted_ids = ",".join(sorted(panel_member_ids))
    h = hashlib.sha256()
    h.update(b"rubric:")
    h.update(rubric_text.encode("utf-8"))
    h.update(b"\npayload:")
    h.update(payload_template_text.encode("utf-8"))
    h.update(b"\npanel:")
    h.update(sorted_ids.encode("utf-8"))
    h.update(b"\nschema:")
    h.update(output_schema_version.encode("utf-8"))
    h.update(b"\naggregation_version:")
    h.update(aggregation_version.encode("utf-8"))
    h.update(b"\naggregation_policy:")
    h.update(aggregation_policy.encode("utf-8"))
    h.update(b"\nevidence_version:")
    h.update(evidence_version.encode("utf-8"))
    h.update(b"\nevidence_policy:")
    h.update(evidence_policy.encode("utf-8"))
    return h.hexdigest()[:12]
