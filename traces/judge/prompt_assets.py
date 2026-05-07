"""Prompt asset loading for the blind benchmark judge."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from traces.corpus.models import PaperRecord
from traces.judge.payload import PAYLOAD_TEMPLATE_VERSION, build_panel_payload


DEFAULT_RUBRIC_PATH = Path(__file__).with_name("rubric.xml")


@dataclass(frozen=True)
class JudgePromptAssets:
    """Versioned prompt assets used by `traces judge is`."""

    rubric: str
    rubric_path: Path
    payload_template_version: str = PAYLOAD_TEMPLATE_VERSION

    def build_payload(
        self,
        *,
        probe_id: str,
        paper: PaperRecord,
        response_text: str,
    ) -> str:
        return build_panel_payload(
            probe_id=probe_id,
            paper=paper,
            response_text=response_text,
        )


def load_judge_prompt_assets(
    rubric_path: Path | None = None,
) -> JudgePromptAssets:
    """Load the judge rubric and payload-template version as one asset set."""
    path = rubric_path or DEFAULT_RUBRIC_PATH
    rubric = path.read_text()
    if not rubric.strip():
        raise ValueError(
            f"Rubric at {path} is empty or whitespace-only. "
            "The judge panel cannot operate without a system prompt."
        )
    return JudgePromptAssets(
        rubric=rubric,
        rubric_path=path,
        payload_template_version=PAYLOAD_TEMPLATE_VERSION,
    )
