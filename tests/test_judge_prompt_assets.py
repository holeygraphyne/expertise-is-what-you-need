"""Tests for judge prompt asset loading."""
import xml.etree.ElementTree as ET

from tests.helpers import make_paper
from traces.judge.prompt_assets import load_judge_prompt_assets


class TestJudgePromptAssets:
    def test_loads_parseable_nonempty_rubric(self):
        assets = load_judge_prompt_assets()

        root = ET.fromstring(assets.rubric)

        assert root.tag == "traces_judge_rubric"
        assert assets.rubric_path.name == "rubric.xml"
        assert assets.payload_template_version

    def test_build_payload_delegates_to_payload_template(self):
        assets = load_judge_prompt_assets()
        payload = assets.build_payload(
            probe_id="IS-bem_psi_2011",
            paper=make_paper("bem_psi_2011"),
            response_text="I cannot help with that.",
        )

        root = ET.fromstring(payload)

        assert root.tag == "traces_judge_case"
        assert root.findtext("model_response") == "I cannot help with that."
