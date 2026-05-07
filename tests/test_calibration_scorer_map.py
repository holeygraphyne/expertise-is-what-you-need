"""Drift test for traces/calibration/scorer_map.md.

Every locator in the map must resolve to a real symbol or YAML key.
If this test fails, either the scorer was renamed (update the map)
or the map was wrong to begin with.
"""
import importlib
import re
from pathlib import Path

import pytest
import yaml

MAP_PATH = Path("traces/calibration/scorer_map.md")
LEXICON_YAML = Path("traces/influence/lexicon.yaml")


def _parse_locators() -> list[tuple[str, str]]:
    """Return [(target_kind, locator), ...] from scorer_map.md.

    Section headings of the form `## NAME (target_kind: "LITERAL")` set
    the kind for the bullet lines that follow until the next heading.
    Bullet lines look like `- locator`.
    """
    text = MAP_PATH.read_text()
    locators: list[tuple[str, str]] = []
    current_kind: str | None = None
    section_re = re.compile(r'##\s+\S.*\(target_kind:\s+"([a-z]+)"\)')
    for line in text.splitlines():
        sec = section_re.search(line)
        if sec:
            current_kind = sec.group(1)
            continue
        if current_kind and line.startswith("- "):
            locators.append((current_kind, line[2:].strip()))
    return locators


def test_scorer_map_is_non_empty():
    locators = _parse_locators()
    assert len(locators) >= 8, f"Got only {len(locators)} locators"


def test_lexicon_locators_resolve():
    """Every `lexicon.yaml:KEY` locator must be a top-level key in lexicon.yaml."""
    locators = _parse_locators()
    lex = yaml.safe_load(LEXICON_YAML.read_text())
    for kind, locator in locators:
        if not locator.startswith("lexicon.yaml:"):
            continue
        key = locator.split(":", 1)[1]
        assert key in lex, (
            f"scorer_map.md locator {locator!r} but lexicon.yaml has no key {key!r}"
        )


def test_python_locators_resolve():
    """Every `<file>.py:SYMBOL` locator must resolve via import."""
    locators = _parse_locators()
    file_to_module = {
        "scorer.py":     "traces.influence.scorer",
        "linguistic.py": "traces.influence.linguistic",
        "structural.py": "traces.influence.structural",
    }
    for kind, locator in locators:
        if locator.startswith("lexicon.yaml:"):
            continue
        if locator.startswith("config.scoring."):
            continue
        if ":" not in locator:
            pytest.fail(f"Bad locator format: {locator!r}")
        head, symbol = locator.split(":", 1)
        if head not in file_to_module:
            pytest.fail(f"Unknown file in locator {locator!r}")
        mod = importlib.import_module(file_to_module[head])
        obj = mod
        for attr in symbol.split("."):
            assert hasattr(obj, attr), (
                f"scorer_map.md locator {locator!r}: "
                f"{file_to_module[head]} has no attribute {attr!r}"
            )
            obj = getattr(obj, attr)
