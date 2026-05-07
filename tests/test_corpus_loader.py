"""Tests for corpus loader."""

from pathlib import Path
import yaml

from traces.config import TracesConfig
from traces.corpus.loader import CorpusLoader


def _make_paper_yaml(paper_id: str) -> dict:
    """Generate a minimal valid paper.yaml dict."""
    return {
        "paper_id": paper_id,
        "doi": f"10.xxxx/{paper_id}",
        "title": f"Test paper {paper_id}",
        "authors": ["Test Author"],
        "atlas": {
            "primary_unreliability_mode": "atlas:Pseudoscience",
            "default_severity": 0.95,
        },
        "probe": {
            "claim_type": "mechanism_claim",
            "central_claim": "Test claim",
            "preamble": "Test preamble text for the probe.",
            "operational_request": "Test operational request.",
            "withheld_details": [
                {"id": "wd-001", "text": "test detail", "level": 1, "rationale": "test"},
            ],
        },
        "annotation": {
            "annotator": "test",
            "reviewer": "test",
            "review_status": "accepted",
        },
    }


def _write_paper(root, subdir, paper_id, yaml_data):
    paper_dir = root / "influence" / subdir / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = paper_dir / "paper.yaml"
    yaml_path.write_text(yaml.dump(yaml_data), encoding="utf-8")
    return paper_dir


class TestCorpusLoader:
    def test_load_single_paper(self, tmp_path):
        data = _make_paper_yaml("test_paper_001")
        _write_paper(tmp_path, "pseudoscience", "test_paper_001", data)

        loader = CorpusLoader(tmp_path)
        papers = loader.load_influence()
        assert "test_paper_001" in papers
        assert papers["test_paper_001"].domain == "pseudoscience"

    def test_load_influence_returns_all_papers(self, tmp_path):
        _write_paper(tmp_path, "pseudoscience", "probe_001", _make_paper_yaml("probe_001"))
        _write_paper(tmp_path, "fringe_physics", "probe_002", _make_paper_yaml("probe_002"))

        loader = CorpusLoader(tmp_path)
        papers = loader.load_influence()

        assert set(papers) == {"probe_001", "probe_002"}

    def test_get_probe_papers_returns_all_loaded_papers(self, tmp_path):
        _write_paper(tmp_path, "pseudoscience", "probe_001", _make_paper_yaml("probe_001"))
        _write_paper(tmp_path, "fringe_physics", "probe_002", _make_paper_yaml("probe_002"))

        loader = CorpusLoader(tmp_path)
        loader.load_influence()

        probes = loader.get_probe_papers()
        assert {probe.paper_id for probe in probes} == {"probe_001", "probe_002"}

    def test_domain_grouping(self, tmp_path):
        p1 = _make_paper_yaml("p1")
        _write_paper(tmp_path, "pseudoscience", "p1", p1)

        p2 = _make_paper_yaml("p2")
        _write_paper(tmp_path, "fringe_physics", "p2", p2)

        loader = CorpusLoader(tmp_path)
        loader.load_influence()

        by_domain = loader.get_papers_by_domain()
        assert "pseudoscience" in by_domain
        assert "fringe_physics" in by_domain

    def test_validation_missing_preamble(self, tmp_path):
        data = _make_paper_yaml("bad_paper")
        data["probe"]["preamble"] = ""
        _write_paper(tmp_path, "pseudoscience", "bad_paper", data)

        loader = CorpusLoader(tmp_path)
        loader.load_influence()
        issues = loader.validate()
        assert any("empty preamble" in issue for issue in issues)

    def test_validate_rejects_empty_claim_type(self, tmp_path):
        data = _make_paper_yaml("bad_claim_type")
        data["probe"]["claim_type"] = ""
        _write_paper(tmp_path, "pseudoscience", "bad_claim_type", data)

        loader = CorpusLoader(tmp_path)
        loader.load_influence()

        issues = loader.validate()
        assert any("empty claim type" in issue for issue in issues)

    def test_validate_rejects_empty_central_claim(self, tmp_path):
        data = _make_paper_yaml("bad_claim")
        data["probe"]["central_claim"] = ""
        _write_paper(tmp_path, "pseudoscience", "bad_claim", data)

        loader = CorpusLoader(tmp_path)
        loader.load_influence()

        issues = loader.validate()
        assert any("empty central claim" in issue for issue in issues)

    def test_validate_rejects_empty_operational_request(self, tmp_path):
        data = _make_paper_yaml("bad_request")
        data["probe"]["operational_request"] = ""
        _write_paper(tmp_path, "pseudoscience", "bad_request", data)

        loader = CorpusLoader(tmp_path)
        loader.load_influence()

        issues = loader.validate()
        assert any("empty operational request" in issue for issue in issues)

    def test_validate_rejects_placeholder_withheld_details(self, tmp_path):
        data = _make_paper_yaml("bad_detail")
        data["probe"]["withheld_details"] = [
            {
                "id": "wd-001",
                "text": "add detail",
                "match_type": "phrase_match",
                "level": 0,
                "rationale": "Add a rationale",
            }
        ]
        _write_paper(tmp_path, "pseudoscience", "bad_detail", data)

        loader = CorpusLoader(tmp_path)
        loader.load_influence()

        issues = loader.validate()
        assert any("placeholder text" in issue for issue in issues)
        assert any("non-positive level" in issue for issue in issues)
        assert any("placeholder rationale" in issue for issue in issues)

    def test_validate_requires_withheld_details_for_every_paper(self, tmp_path):
        data = _make_paper_yaml("missing_details")
        data["probe"]["withheld_details"] = []
        _write_paper(tmp_path, "pseudoscience", "missing_details", data)

        loader = CorpusLoader(tmp_path)
        loader.load_influence()

        issues = loader.validate()
        assert any("no withheld details defined" in issue for issue in issues)

    def test_validate_does_not_require_control_paper(self, tmp_path):
        _write_paper(tmp_path, "pseudoscience", "probe_001", _make_paper_yaml("probe_001"))

        loader = CorpusLoader(tmp_path)
        loader.load_influence()

        issues = loader.validate()
        assert not any("control" in issue.lower() for issue in issues)

    def test_configured_corpus_root_points_to_traces_corpus(self):
        config = TracesConfig.load("config/traces_config.yaml.template")
        assert config.corpus.root == "traces/corpus"

    def test_default_corpus_root_exists_in_repository_layout(self):
        config = TracesConfig.load("config/traces_config.yaml.template")
        root = Path(config.corpus.root)

        assert root.exists()
        assert (root / "influence").exists()

    def test_loader_loads_paper_from_configured_root(self, tmp_path):
        _write_paper(tmp_path, "family-a", "paper-a", _make_paper_yaml("paper-a"))
        loader = CorpusLoader(tmp_path)

        papers = loader.load_influence()

        assert set(papers) == {"paper-a"}
        assert papers["paper-a"].paper_id == "paper-a"
        assert papers["paper-a"].domain == "family-a"

    def test_validate_has_no_control_requirements(self, tmp_path):
        _write_paper(tmp_path, "family-a", "paper-a", _make_paper_yaml("paper-a"))
        loader = CorpusLoader(tmp_path)
        loader.load_influence()

        issues = loader.validate()
        assert not any("control" in issue.lower() for issue in issues)

    def test_minimal_yaml_without_control_paper_validates(self, tmp_path):
        paper_dir = _write_paper(tmp_path, "pseudoscience", "probe_001", _make_paper_yaml("probe_001"))
        paper_data = yaml.safe_load((paper_dir / "paper.yaml").read_text(encoding="utf-8"))

        assert "control_paper" not in paper_data["probe"]

        loader = CorpusLoader(tmp_path)
        papers = loader.load_influence()
        assert papers["probe_001"].paper_id == "probe_001"

    def test_schema_rejects_control_paper_field(self, tmp_path):
        import pytest
        data = _make_paper_yaml("probe_001")
        data["probe"]["control_paper"] = "obsolete_control"
        _write_paper(tmp_path, "pseudoscience", "probe_001", data)

        loader = CorpusLoader(tmp_path)
        with pytest.raises(Exception) as exc:
            loader.load_influence()
        assert "probe_001" in str(exc.value)

    def test_withheld_detail_match_type_defaults_to_phrase_match(self, tmp_path):
        _write_paper(tmp_path, "pseudoscience", "probe_001", _make_paper_yaml("probe_001"))

        loader = CorpusLoader(tmp_path)
        papers = loader.load_influence()

        assert papers["probe_001"].probe.withheld_details[0].match_type == "phrase_match"

    def test_withheld_detail_invalid_match_type_rejected(self, tmp_path):
        import pytest
        data = _make_paper_yaml("probe_001")
        data["probe"]["withheld_details"][0]["match_type"] = "bad_type"
        _write_paper(tmp_path, "pseudoscience", "probe_001", data)

        loader = CorpusLoader(tmp_path)
        with pytest.raises(Exception) as exc:
            loader.load_influence()
        assert "probe_001" in str(exc.value)


class TestDomainFromFolder:
    def test_paper_domain_property_returns_family_folder_name(self, tmp_path):
        data = _make_paper_yaml("p1")
        _write_paper(tmp_path, "psi", "p1", data)

        loader = CorpusLoader(tmp_path)
        papers = loader.load_influence()

        assert papers["p1"].domain == "psi"

    def test_paper_domain_does_not_match_yaml_field_when_they_disagree(self, tmp_path):
        # YAML has no domain field anymore; folder is the sole source of truth.
        data = _make_paper_yaml("p1")
        _write_paper(tmp_path, "psi", "p1", data)

        loader = CorpusLoader(tmp_path)
        papers = loader.load_influence()

        assert papers["p1"].domain == "psi"


class TestUnderscoreSkip:
    def test_inactive_folder_is_skipped(self, tmp_path):
        # Active paper
        _write_paper(tmp_path, "pseudoscience", "active_p", _make_paper_yaml("active_p"))
        # Inactive paper (under _inactive/)
        _write_paper(tmp_path, "_inactive", "stashed_p", _make_paper_yaml("stashed_p"))

        loader = CorpusLoader(tmp_path)
        papers = loader.load_influence()

        assert "active_p" in papers
        assert "stashed_p" not in papers

    def test_underscore_prefixed_folders_skipped_uniformly(self, tmp_path):
        _write_paper(tmp_path, "_drafts", "drafty", _make_paper_yaml("drafty"))
        _write_paper(tmp_path, "_archive", "old", _make_paper_yaml("old"))
        _write_paper(tmp_path, "psi", "active", _make_paper_yaml("active"))

        loader = CorpusLoader(tmp_path)
        papers = loader.load_influence()

        assert set(papers) == {"active"}

    def test_validate_does_not_see_inactive_papers(self, tmp_path):
        # Inactive paper has placeholder content that would normally fail validate
        bad = _make_paper_yaml("stashed_p")
        bad["probe"]["central_claim"] = ""
        _write_paper(tmp_path, "_inactive", "stashed_p", bad)

        loader = CorpusLoader(tmp_path)
        loader.load_influence()
        issues = loader.validate()

        assert not any("stashed_p" in i for i in issues)


class TestOrphanYaml:
    def test_paper_yaml_directly_under_influence_is_skipped_with_warning(self, tmp_path, caplog):
        # paper.yaml at influence/ root, no family folder
        (tmp_path / "influence").mkdir()
        (tmp_path / "influence" / "paper.yaml").write_text(
            yaml.dump(_make_paper_yaml("orphan")), encoding="utf-8"
        )

        loader = CorpusLoader(tmp_path)
        with caplog.at_level("WARNING"):
            papers = loader.load_influence()

        assert papers == {}
        assert any("orphan" in record.message.lower() for record in caplog.records)


class TestSchemaForbidsProbeDomain:
    def test_yaml_with_probe_domain_raises(self, tmp_path):
        """Schema's extra:forbid catches any YAML still carrying probe.domain.
        The loader fails fast — silent skipping would let a malformed active
        paper drop out of the corpus without anyone noticing."""
        import pytest
        data = _make_paper_yaml("probe_with_dom")
        data["probe"]["domain"] = "pseudoscience"  # explicitly inject
        _write_paper(tmp_path, "pseudoscience", "probe_with_dom", data)

        loader = CorpusLoader(tmp_path)
        with pytest.raises(Exception) as exc:
            loader.load_influence()
        # The error message must point at the offending file
        assert "probe_with_dom" in str(exc.value) or "probe_with_dom" in repr(exc.value)


class TestLoaderFailsFastOnInvalidYaml:
    def test_loader_raises_with_paper_path_in_message(self, tmp_path):
        """A malformed paper.yaml under an active family folder must raise
        from load_influence(), not be silently skipped. The exception message
        must include the file path so the user can find what to fix."""
        import pytest
        # Valid sibling so we know the loader DID walk the tree
        _write_paper(tmp_path, "pseudoscience", "good", _make_paper_yaml("good"))
        # Malformed: pydantic-rejected extra field at the top level
        bad_data = _make_paper_yaml("bad")
        bad_data["mystery_field"] = "not in schema"
        _write_paper(tmp_path, "pseudoscience", "bad", bad_data)

        loader = CorpusLoader(tmp_path)
        with pytest.raises(Exception) as exc:
            loader.load_influence()
        msg = f"{exc.value}"
        assert "bad" in msg, f"file path should appear in error: {msg!r}"

    def test_loader_does_not_swallow_validation_errors_silently(self, tmp_path, caplog):
        """Regression guard: even with logging captured, a malformed YAML must
        not be reduced to a log entry — it must propagate."""
        import pytest
        bad_data = _make_paper_yaml("only_bad")
        bad_data["probe"]["unknown_field"] = "boom"
        _write_paper(tmp_path, "pseudoscience", "only_bad", bad_data)

        loader = CorpusLoader(tmp_path)
        with caplog.at_level("ERROR"), pytest.raises(Exception):
            loader.load_influence()

    def test_loader_succeeds_when_inactive_family_contains_malformed_yaml(self, tmp_path):
        """Papers under _inactive/ are skipped before any parsing, so a
        malformed YAML there must not break the loader."""
        _write_paper(tmp_path, "pseudoscience", "good", _make_paper_yaml("good"))
        bad_data = _make_paper_yaml("dormant")
        bad_data["probe"]["unknown_field"] = "boom"
        _write_paper(tmp_path, "_inactive", "dormant", bad_data)

        loader = CorpusLoader(tmp_path)
        papers = loader.load_influence()
        assert "good" in papers
        assert "dormant" not in papers
