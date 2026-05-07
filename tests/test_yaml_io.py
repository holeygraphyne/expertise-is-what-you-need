"""Tests for shared paper YAML serialization."""
import yaml

from traces.corpus.yaml_io import dump_paper_yaml


def test_dump_paper_yaml_preserves_block_scalar_for_multiline():
    data = {
        "probe": {
            "central_claim": "Line one.\nLine two.\nLine three.",
            "preamble": "single line",
        }
    }
    text = dump_paper_yaml(data)

    assert "central_claim: |-" in text or "central_claim: |" in text
    assert "preamble: single line" in text


def test_dump_paper_yaml_round_trips_simple_dict():
    data = {"paper_id": "x", "year": 2026, "authors": ["A. Author"]}
    text = dump_paper_yaml(data)
    assert yaml.safe_load(text) == data


def test_dump_paper_yaml_does_not_sort_keys():
    data = {"z_first": 1, "a_second": 2}
    text = dump_paper_yaml(data)
    assert text.index("z_first") < text.index("a_second")
