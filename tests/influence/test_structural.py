from __future__ import annotations

from traces.influence.structural import DISCLAIMER_HEADER_RE


def test_disclaimer_header_regex_does_not_match_bare_note() -> None:
    assert DISCLAIMER_HEADER_RE.search("Note: This is a hypothetical study design.") is None


def test_disclaimer_header_regex_matches_markdown_header_branch() -> None:
    match = DISCLAIMER_HEADER_RE.search("# Important Scientific Context\nText")

    assert match is not None


def test_disclaimer_header_regex_matches_qualified_plain_headers() -> None:
    assert DISCLAIMER_HEADER_RE.search("Important Note: read this first") is not None
    assert DISCLAIMER_HEADER_RE.search("Critical Caveat: read this first") is not None
    assert DISCLAIMER_HEADER_RE.search("Scientific Disclaimer: read this first") is not None


def test_disclaimer_header_regex_rejects_non_disclaimer_preamble() -> None:
    assert DISCLAIMER_HEADER_RE.search("Just a heads-up: read this first") is None