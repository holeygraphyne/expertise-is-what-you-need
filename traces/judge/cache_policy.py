"""Cache-key and judge-version policy for the blind benchmark judge."""
from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from typing import Callable

from traces.corpus.models import PaperRecord
from traces.judge.cache import cache_key_for, cache_key_has_case_hash
from traces.judge.payload import build_panel_payload
from traces.judge.versioning import (
    JUDGE_AGGREGATION_VERSION,
    JUDGE_EVIDENCE_VERSION,
    JUDGE_OUTPUT_SCHEMA_VERSION,
    compute_judge_version,
)


JUDGE_AGGREGATION_POLICY = "native_boolean_majority;sanewashing=derived"
JUDGE_EVIDENCE_POLICY = "verbatim;source_reference=metadata_or_paper_id_alias"


def normalized_sha256(text: str) -> str:
    """Hash text using the normalization policy shared by judge cache keys."""
    normalized = unicodedata.normalize("NFKC", text or "").replace("’", "'")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class JudgeCachePolicy:
    """Single source of truth for current judge cache/version semantics."""

    judge_version: str
    payload_builder: Callable[..., str] = build_panel_payload

    @classmethod
    def from_inputs(
        cls,
        *,
        rubric_text: str,
        payload_template_text: str,
        panel_member_ids: list[str],
        output_schema_version: str = JUDGE_OUTPUT_SCHEMA_VERSION,
        aggregation_version: str = JUDGE_AGGREGATION_VERSION,
        aggregation_policy: str = JUDGE_AGGREGATION_POLICY,
        evidence_version: str = JUDGE_EVIDENCE_VERSION,
        evidence_policy: str = JUDGE_EVIDENCE_POLICY,
        payload_builder: Callable[..., str] = build_panel_payload,
    ) -> "JudgeCachePolicy":
        return cls(
            judge_version=compute_judge_version(
                rubric_text=rubric_text,
                payload_template_text=payload_template_text,
                panel_member_ids=panel_member_ids,
                output_schema_version=output_schema_version,
                aggregation_version=aggregation_version,
                aggregation_policy=aggregation_policy,
                evidence_version=evidence_version,
                evidence_policy=evidence_policy,
            ),
            payload_builder=payload_builder,
        )

    def response_sha256(self, response_text: str) -> str:
        return normalized_sha256(response_text)

    def case_sha256(
        self,
        *,
        probe_id: str,
        paper: PaperRecord,
        response_text: str,
    ) -> str:
        payload = self.payload_builder(
            probe_id=probe_id,
            paper=paper,
            response_text=response_text,
        )
        return normalized_sha256(payload)

    def cache_key_for_row(
        self,
        *,
        probe_id: str,
        model: str,
        paper: PaperRecord,
        response_text: str,
    ) -> str:
        return cache_key_for(
            probe_id=probe_id,
            model=model,
            response_sha256=self.response_sha256(response_text),
            case_sha256=self.case_sha256(
                probe_id=probe_id,
                paper=paper,
                response_text=response_text,
            ),
            judge_version=self.judge_version,
        )

    def current_cache_entries(self, loaded_cache: dict[str, dict]) -> dict[str, dict]:
        """Keep only entries valid for the current cache-key/version policy."""
        return {
            key: entry
            for key, entry in loaded_cache.items()
            if cache_key_has_case_hash(key)
            and (
                entry.get("_judge_version")
                or entry.get("panel_verdict", {}).get("judge_version")
            ) == self.judge_version
        }
