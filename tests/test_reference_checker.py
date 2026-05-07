from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "paper" / "scripts" / "check_references.py"
SPEC = importlib.util.spec_from_file_location("check_references", MODULE_PATH)
assert SPEC is not None
check_references = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_references
assert SPEC.loader is not None
SPEC.loader.exec_module(check_references)


def test_parse_bibtex_handles_nested_braces_and_urls():
    entries = check_references.parse_bibtex(
        r"""
        @article{example2026,
          title = {Department of {Energy}'s "{AI} push, phase 2" & consequences},
          author = {Doe, Jane},
          doi = {https://doi.org/10.1234/ABC.DEF},
          url = {https://openreview.net/forum?id=G0dksFayVq}
        }
        """
    )

    assert len(entries) == 1
    entry = entries[0]
    assert entry.key == "example2026"
    assert entry.doi == "10.1234/abc.def"
    assert entry.title == "Department of Energy's \"AI push, phase 2\" & consequences"
    assert entry.raw_title == "Department of {Energy}'s \"{AI} push, phase 2\" & consequences"
    assert check_references.extract_identifiers(entry) == {
        "doi": ["10.1234/abc.def"],
        "openreview": ["G0dksFayVq"],
    }


def test_title_score_normalizes_latex_math_and_punctuation():
    local = r"Superconductor Pb$_{10-x}$Cu$_x$(PO$_4$)$_6$O showing levitation"
    source = "Superconductor Pb10-xCux(PO4)6O showing levitation"

    assert check_references.title_score(local, source) >= 0.98


def test_title_score_normalizes_case_protecting_braces_and_curly_quotes():
    local = r"Department of {Energy}'s {AI} push squeezes scientists"
    source = "Department of Energy\u2019s AI push squeezes scientists"

    assert check_references.title_score(local, source) == 1.0


def test_check_entry_reports_needs_update_from_source_title():
    entry = check_references.BibEntry(
        entry_type="article",
        key="badtitle",
        fields={"title": "Old title", "doi": "10.1234/example"},
        line=1,
    )

    class DummyChecker(check_references.ReferenceChecker):
        def fetch_sources(self, identifiers):
            return [
                check_references.SourceMetadata(
                    source="crossref",
                    identifier=identifiers["doi"][0],
                    url="https://api.crossref.org/works/10.1234/example",
                    title="Completely different title",
                )
            ]

    checker = DummyChecker(
        timeout=1.0,
        sleep=0.0,
        user_agent="test",
        ok_threshold=0.92,
        review_threshold=0.82,
        include_urls=True,
        check_pubpeer=False,
    )

    report = checker.check_entry(entry)

    assert report.verdict == "needs_update"
    assert report.sources[0].verdict == "needs_update"
    assert report.sources[0].score < 0.82


def test_render_markdown_lists_attention_items():
    report = check_references.EntryReport(
        key="needswork",
        entry_type="article",
        line=4,
        local_title="Local",
        local_title_bibtex="{Local}",
        local_title_normalized="local",
        identifiers={"doi": ["10.1234/example"]},
        sources=[
            check_references.SourceMetadata(
                source="crossref",
                identifier="10.1234/example",
                url="https://example.test",
                title="Remote",
                score=0.25,
                verdict="needs_update",
            )
        ],
        verdict="needs_update",
    )

    markdown = check_references.render_markdown([report], bib_path=Path("paper/references.bib"))

    assert "## Attention Items" in markdown
    assert "### `needswork`" in markdown
    assert "| `needswork` | needs_update | crossref | 0.250 | {Local} | Remote |" in markdown
