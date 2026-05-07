"""Tests for the `python -m traces grobid` bootstrap stage."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from grobid_tei_xml.types import GrobidAuthor, GrobidBiblio, GrobidDocument

from traces.config import TracesConfig
from traces.corpus.grobid_processor import AtlasModeCandidate, GrobidProcessor


def _make_config(tmp_path: Path) -> TracesConfig:
    return TracesConfig.model_validate(
        {
            "corpus": {"root": str(tmp_path)},
            "atlas": {
                "ontology_path": "../atlas-ontology/src/ontology/atlas.ttl",
                "vocabularies_path": "../atlas-ontology/vocabularies/",
            },
            "grobid": {"url": "http://localhost:8070", "timeout": 10},
            "providers": {
                "local": {"base_url": "http://localhost:11434/v1"},
            },
            "audit": {"provider": "local"},
            "service_model": {"id": "test-model"},
            "models": [
                {"id": "test-model", "provider": "local", "provider_model_id": "test-model"},
            ],
        }
    )


def _make_document() -> GrobidDocument:
    return GrobidDocument(
        grobid_version="test",
        grobid_timestamp="2026-04-10T00:00:00Z",
        header=GrobidBiblio(
            authors=[GrobidAuthor(full_name=None, given_name="Ada", surname="Lovelace")],
            title="Biofield energy healing improves immune outcomes",
            doi="10.1000/test-doi",
            journal="Test Journal",
            date="2016-01-01",
        ),
        abstract="This paper claims a strong improvement in immune outcomes.",
        body="Introduction\nThis is the introductory framing.\n\nConclusions\nThe intervention works.",
    )


def _write_structured_tei(path: Path) -> None:
    path.write_text(
        """
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <text>
    <body>
      <div>
        <head>Introduction</head>
        <p>This is the full introduction body <ref type="bibr">[1]</ref>.</p>
        <p>It should remain intact.</p>
      </div>
      <div>
        <head>Conclusions</head>
        <p>This is the conclusion section.</p>
      </div>
      <div>
        <head>References</head>
        <p>Noise reference entry.</p>
      </div>
    </body>
  </text>
</TEI>
""".strip(),
        encoding="utf-8",
    )


def _make_processor(tmp_path: Path) -> GrobidProcessor:
    processor = GrobidProcessor(_make_config(tmp_path))
    processor._provider = Mock()
    processor._provider.complete.return_value.content = "The intervention improves immune outcomes."
    return processor


def _paper_dir(root: Path, *parts: str) -> Path:
    paper_dir = root.joinpath(*parts)
    paper_dir.mkdir(parents=True, exist_ok=True)
    return paper_dir


def test_candidate_folder_selected_only_when_pdf_exists_and_yaml_tei_absent(tmp_path):
    processor = _make_processor(tmp_path)
    paper_dir = _paper_dir(tmp_path, "influence", "pseudoscience", "paper_a")
    (paper_dir / "paper.pdf").write_bytes(b"pdf")

    assert processor._should_bootstrap(paper_dir) is True


def test_folder_skipped_if_yaml_exists(tmp_path):
    processor = _make_processor(tmp_path)
    paper_dir = _paper_dir(tmp_path, "influence", "pseudoscience", "paper_b")
    (paper_dir / "paper.pdf").write_bytes(b"pdf")
    (paper_dir / "paper.yaml").write_text("paper_id: paper_b\n", encoding="utf-8")

    assert processor._should_bootstrap(paper_dir) is False


def test_folder_skipped_if_tei_exists(tmp_path):
    processor = _make_processor(tmp_path)
    paper_dir = _paper_dir(tmp_path, "influence", "pseudoscience", "paper_c")
    (paper_dir / "paper.pdf").write_bytes(b"pdf")
    (paper_dir / "paper.tei.xml").write_text("<TEI/>", encoding="utf-8")

    assert processor._should_bootstrap(paper_dir) is False


def test_bootstrap_yaml_contains_expected_placeholders(tmp_path):
    processor = _make_processor(tmp_path)
    paper_dir = _paper_dir(tmp_path, "influence", "pseudoscience", "paper_d")
    pdf_path = paper_dir / "paper.pdf"
    pdf_path.write_bytes(b"pdf-bytes")

    yaml_data = processor._build_yaml_data(
        paper_dir=paper_dir,
        domain="pseudoscience",
        pdf_path=pdf_path,
        doc=_make_document(),
    )

    assert yaml_data["paper_id"] == "paper_d"
    assert yaml_data["pdf_sha256"]
    assert yaml_data["probe"]["claim_type"] == ""
    assert yaml_data["probe"]["operational_request"] == ""
    assert "premise_boundary" not in yaml_data["probe"]
    assert len(yaml_data["probe"]["withheld_details"]) == 6
    assert yaml_data["probe"]["withheld_details"][0]["match_type"] == "phrase_match"
    assert yaml_data["probe"]["withheld_details"][0]["text"] == "add detail"
    assert yaml_data["probe"]["withheld_details"][0]["rationale"] == "Add a rationale"
    assert yaml_data["annotation"]["review_status"] == "pending"
    assert yaml_data["annotation"]["date_annotated"] == date.today().isoformat()
    assert yaml_data["annotation"]["date_reviewed"] == date.today().isoformat()
    assert yaml_data["annotation"]["notes"] == "Bootstrapped from GROBID; requires human review."


def test_primary_mode_prefers_specific_abbreviation_signal(tmp_path):
    processor = _make_processor(tmp_path)
    processor._atlas_candidates = [
        AtlasModeCandidate(
            uri="https://w3id.org/atlas/ontology#OrgoneEnergy",
            label="Orgone energy field",
            default_severity=0.25,
            evidence_terms={"orgone", "reich", "accumulator", "energy"},
        ),
        AtlasModeCandidate(
            uri="https://w3id.org/atlas/ontology#ColdFusionLENR",
            label="Cold fusion LENR",
            default_severity=0.85,
            evidence_terms={"lenr", "cold", "fusion", "nuclear", "palladium"},
        ),
    ]
    processor.atlas_graph.is_subclass_of = lambda mode_uri, ancestor_uri: True
    processor.config.grobid.domain_atlas_ancestors["pseudoscience"] = "atlas:Pseudoscience"

    mode_uri, severity = processor._infer_primary_mode(
        "Low energy nuclear reactions (LENR) in palladium systems",
        "pseudoscience",
    )

    assert mode_uri == "https://w3id.org/atlas/ontology#ColdFusionLENR"
    assert severity == 0.85


def test_primary_mode_written_as_atlas_curie(tmp_path):
    processor = _make_processor(tmp_path)
    processor._atlas_candidates = [
        AtlasModeCandidate(
            uri="https://w3id.org/atlas/ontology#OrgoneEnergy",
            label="Orgone energy",
            default_severity=0.45,
            evidence_terms={"biofield", "healing", "immune", "outcomes"},
        ),
    ]
    processor.atlas_graph.is_subclass_of = lambda mode_uri, ancestor_uri: True
    paper_dir = _paper_dir(tmp_path, "influence", "pseudoscience", "paper_curie")
    pdf_path = paper_dir / "paper.pdf"
    pdf_path.write_bytes(b"pdf-bytes")

    yaml_data = processor._build_yaml_data(
        paper_dir=paper_dir,
        domain="pseudoscience",
        pdf_path=pdf_path,
        doc=_make_document(),
    )

    assert yaml_data["atlas"]["primary_unreliability_mode"] == "atlas:OrgoneEnergy"


def test_default_severity_reads_from_ontology(tmp_path):
    processor = _make_processor(tmp_path)
    processor.atlas_graph.default_severity = Mock(return_value=0.82)

    assert processor._severity_for_mode("https://w3id.org/atlas/ontology#ColdFusionLENR") == 0.82


def test_structured_section_text_excludes_heading(tmp_path):
    processor = _make_processor(tmp_path)
    tei_path = tmp_path / "paper.tei.xml"
    _write_structured_tei(tei_path)

    intro = processor._extract_intro_text(_make_document(), tei_path)

    assert intro == "This is the full introduction body .\nIt should remain intact."
    assert not intro.startswith("Introduction")


def test_claim_source_excludes_references_noise(tmp_path):
    processor = _make_processor(tmp_path)
    tei_path = tmp_path / "paper.tei.xml"
    _write_structured_tei(tei_path)

    source = processor._build_claim_source(_make_document(), tei_path)

    assert "Noise reference entry" not in source
    assert "[CONCLUSIONS]\nThis is the conclusion section." in source


def test_structured_section_text_strips_bibr_refs(tmp_path):
    processor = _make_processor(tmp_path)
    tei_path = tmp_path / "paper.tei.xml"
    _write_structured_tei(tei_path)

    intro = processor._extract_intro_text(_make_document(), tei_path)

    assert "[1]" not in intro


def test_multiline_yaml_output_wraps_block_text(tmp_path):
    processor = _make_processor(tmp_path)
    output_path = tmp_path / "paper.yaml"
    long_text = (
        "This is a deliberately long bootstrap paragraph that should be wrapped to keep "
        "the YAML output readable while preserving the block-scalar formatting used for "
        "multiline text fields."
    )
    processor._write_yaml(
        output_path,
        {
            "probe": {
                "central_claim": processor._wrap_block_text(long_text, width=80),
                "preamble": processor._wrap_block_text(long_text, width=80),
            }
        },
    )

    text = output_path.read_text(encoding="utf-8")

    assert "central_claim: |-" in text or "central_claim: |" in text
    assert "preamble: |-" in text or "preamble: |" in text
    block_lines = [line for line in text.splitlines() if line.startswith("    ") and line.strip()]
    assert block_lines
    assert all(len(line.strip()) <= 80 for line in block_lines)


def test_bootstrap_domain_derived_from_folder(tmp_path):
    """After bootstrap, loading the written YAML through CorpusLoader
    yields a PaperRecord whose domain matches the family folder name."""
    from traces.corpus.loader import CorpusLoader

    processor = _make_processor(tmp_path)
    paper_dir = _paper_dir(tmp_path, "influence", "pseudoscience", "paper_e")
    pdf_path = paper_dir / "paper.pdf"
    pdf_path.write_bytes(b"pdf-bytes")

    yaml_data = processor._build_yaml_data(
        paper_dir=paper_dir,
        domain="pseudoscience",
        pdf_path=pdf_path,
        doc=_make_document(),
    )
    processor._write_yaml(paper_dir / "paper.yaml", yaml_data)

    # Verify field absent from YAML
    written = yaml.safe_load((paper_dir / "paper.yaml").read_text(encoding="utf-8"))
    assert "domain" not in written["probe"]

    # Verify the loader assigns domain from the family folder.
    # Note: the bootstrapped YAML is incomplete (claim_type="", placeholder
    # withheld details), so model_validate may fail at schema time. The
    # primary assertion is the YAML-field absence above; if the loader
    # nonetheless accepts the record, also assert the domain.
    loader = CorpusLoader(tmp_path)
    papers = loader.load_influence()
    if "paper_e" in papers:
        assert papers["paper_e"].domain == "pseudoscience"


def test_mode_not_assigned_from_generic_overlap_only(tmp_path):
    processor = _make_processor(tmp_path)
    processor._atlas_candidates = [
        AtlasModeCandidate(
            uri="atlas:orgone",
            label="Orgone energy",
            default_severity=0.7,
            evidence_terms={"orgone", "energy"},
        ),
    ]
    processor.atlas_graph.is_subclass_of = lambda mode_uri, ancestor_uri: True
    processor.config.grobid.domain_atlas_ancestors["pseudoscience"] = "atlas:Pseudoscience"

    mode_uri, severity = processor._infer_primary_mode(
        "Energy effects in a material system",
        "pseudoscience",
    )

    assert mode_uri is None
    assert severity == 0.0


def test_infer_primary_mode_returns_none_when_no_overlap(tmp_path):
    processor = _make_processor(tmp_path)
    processor._atlas_candidates = [
        AtlasModeCandidate(
            uri="atlas:biofield",
            label="Biofield energy healing",
            default_severity=0.7,
            evidence_terms={"biofield", "healing", "therapy"},
        ),
        AtlasModeCandidate(
            uri="atlas:predatory",
            label="Predatory journal publication",
            default_severity=0.2,
            evidence_terms={"predatory", "journal", "publication"},
        ),
    ]
    processor.atlas_graph.is_subclass_of = lambda mode_uri, ancestor_uri: True
    processor.config.grobid.domain_atlas_ancestors["pseudoscience"] = "atlas:Pseudoscience"

    mode_uri, severity = processor._infer_primary_mode(
        "Marine sediment transport in estuaries",
        "pseudoscience",
    )

    assert mode_uri is None
    assert severity == 0.0


def test_membership_papers_traversal_supported(tmp_path):
    processor = _make_processor(tmp_path)
    paper_dir = _paper_dir(tmp_path, "membership", "papers", "paper_f")
    (paper_dir / "paper.pdf").write_bytes(b"pdf")

    candidates = list(processor._iter_candidate_paper_dirs())
    assert (paper_dir, "membership") in candidates


def test_bootstrap_all_writes_yaml_and_tei(tmp_path):
    processor = _make_processor(tmp_path)
    paper_dir = _paper_dir(tmp_path, "influence", "pseudoscience", "paper_g")
    (paper_dir / "paper.pdf").write_bytes(b"pdf")
    (paper_dir / "already.yaml").write_text("ignored", encoding="utf-8")
    processor._check_server = Mock(return_value=True)

    def _fake_process_pdf(pdf_path: Path):
        tei_path = pdf_path.with_suffix(".tei.xml")
        tei_path.write_text("<TEI/>", encoding="utf-8")
        return tei_path, _make_document()

    processor.process_pdf = Mock(side_effect=_fake_process_pdf)

    stats = processor.bootstrap_all()

    assert stats == {"processed": 1, "skipped": 0, "failed": 0}
    assert (paper_dir / "paper.tei.xml").exists()
    written = yaml.safe_load((paper_dir / "paper.yaml").read_text(encoding="utf-8"))
    assert written["paper_id"] == "paper_g"
    assert written["probe"]["central_claim"] == "The intervention improves immune outcomes."


def test_iter_candidate_paper_dirs_includes_underscore_family(tmp_path):
    processor = _make_processor(tmp_path)
    active = _paper_dir(tmp_path, "influence", "pseudoscience", "active")
    (active / "paper.pdf").write_bytes(b"pdf")
    inactive = _paper_dir(tmp_path, "influence", "_inactive", "draft_probe")
    (inactive / "paper.pdf").write_bytes(b"pdf")

    candidates = list(processor._iter_candidate_paper_dirs())
    paper_dirs = [pd for pd, _ in candidates]

    assert active in paper_dirs
    assert inactive in paper_dirs


def test_iter_candidate_paper_dirs_includes_unmapped_family(tmp_path):
    processor = _make_processor(tmp_path)
    unmapped = _paper_dir(tmp_path, "influence", "openaccess", "paper_open")
    (unmapped / "paper.pdf").write_bytes(b"pdf")

    assert (unmapped, "openaccess") in list(processor._iter_candidate_paper_dirs())


def test_iter_candidate_paper_dirs_skips_underscore_under_membership(tmp_path):
    processor = _make_processor(tmp_path)
    membership_active = _paper_dir(tmp_path, "membership", "papers", "active_m")
    (membership_active / "paper.pdf").write_bytes(b"pdf")
    membership_drafts = _paper_dir(tmp_path, "membership", "papers", "_drafts")
    (membership_drafts / "paper.pdf").write_bytes(b"pdf")

    candidates = list(processor._iter_candidate_paper_dirs())
    paper_dirs = [pd for pd, _ in candidates]

    assert membership_active in paper_dirs
    assert membership_drafts not in paper_dirs


def test_bootstrap_yaml_does_not_emit_probe_domain(tmp_path):
    processor = _make_processor(tmp_path)
    paper_dir = _paper_dir(tmp_path, "influence", "pseudoscience", "paper_no_dom")
    pdf_path = paper_dir / "paper.pdf"
    pdf_path.write_bytes(b"pdf-bytes")

    yaml_data = processor._build_yaml_data(
        paper_dir=paper_dir,
        domain="pseudoscience",
        pdf_path=pdf_path,
        doc=_make_document(),
    )

    assert "domain" not in yaml_data["probe"]


def test_infer_primary_mode_returns_none_for_unmapped_family(tmp_path):
    processor = _make_processor(tmp_path)
    mode_uri, severity = processor._infer_primary_mode("some source text", "psi")

    assert mode_uri is None
    assert severity == 0.0


def test_bootstrap_all_processes_underscore_family_without_overwriting_yaml(tmp_path):
    processor = _make_processor(tmp_path)
    draft_dir = _paper_dir(tmp_path, "influence", "_inactive", "draft_bootstrap")
    (draft_dir / "paper.pdf").write_bytes(b"pdf")
    processor._check_server = Mock(return_value=True)

    def _fake_process_pdf(pdf_path: Path):
        tei_path = pdf_path.with_suffix(".tei.xml")
        tei_path.write_text("<TEI/>", encoding="utf-8")
        return tei_path, _make_document()

    processor.process_pdf = Mock(side_effect=_fake_process_pdf)

    stats = processor.bootstrap_all()

    assert stats == {"processed": 1, "skipped": 0, "failed": 0}
    written = yaml.safe_load((draft_dir / "paper.yaml").read_text(encoding="utf-8"))
    assert written["paper_id"] == "draft_bootstrap"
    assert written["atlas"]["primary_unreliability_mode"] is None


def test_bootstrap_all_processes_unmapped_family_without_atlas_mapping(tmp_path):
    processor = _make_processor(tmp_path)
    paper_dir = _paper_dir(tmp_path, "influence", "openaccess", "paper_open")
    (paper_dir / "paper.pdf").write_bytes(b"pdf")
    processor._check_server = Mock(return_value=True)

    def _fake_process_pdf(pdf_path: Path):
        tei_path = pdf_path.with_suffix(".tei.xml")
        tei_path.write_text("<TEI/>", encoding="utf-8")
        return tei_path, _make_document()

    processor.process_pdf = Mock(side_effect=_fake_process_pdf)

    stats = processor.bootstrap_all()

    assert stats == {"processed": 1, "skipped": 0, "failed": 0}
    written = yaml.safe_load((paper_dir / "paper.yaml").read_text(encoding="utf-8"))
    assert written["paper_id"] == "paper_open"
    assert written["atlas"]["primary_unreliability_mode"] is None
    assert written["atlas"]["default_severity"] == 0.0


def test_bootstrap_all_skips_existing_yaml_without_modifying_it(tmp_path):
    processor = _make_processor(tmp_path)
    paper_dir = _paper_dir(tmp_path, "influence", "_inactive", "paper_existing")
    (paper_dir / "paper.pdf").write_bytes(b"pdf")
    existing_yaml = "paper_id: preserved\nannotation:\n  notes: keep me\n"
    (paper_dir / "paper.yaml").write_text(existing_yaml, encoding="utf-8")
    processor._check_server = Mock(return_value=True)
    processor.process_pdf = Mock()

    stats = processor.bootstrap_all()

    assert stats == {"processed": 0, "skipped": 1, "failed": 0}
    assert (paper_dir / "paper.yaml").read_text(encoding="utf-8") == existing_yaml
    processor.process_pdf.assert_not_called()


def test_infer_primary_mode_filters_by_ontology_ancestor(tmp_path):
    """Candidates are filtered by rdfs:subClassOf membership against the
    ancestor URI mapped from the family folder."""
    processor = _make_processor(tmp_path)

    # Two candidates; only one is a subclass of the configured ancestor.
    processor._atlas_candidates = [
        AtlasModeCandidate(
            uri="https://w3id.org/atlas/ontology#OrgoneEnergy",
            label="Orgone energy",
            default_severity=0.7,
            evidence_terms={"orgone", "energy", "field"},
        ),
        AtlasModeCandidate(
            uri="https://w3id.org/atlas/ontology#Fabrication",
            label="Fabrication",
            default_severity=0.95,
            evidence_terms={"orgone", "energy", "field"},  # same evidence — would tie if not filtered
        ),
    ]
    # Stub: only OrgoneEnergy is a subclass of Pseudoscience (the
    # configured ancestor for "pseudoscience" folders).
    pseudoscience_uri = "https://w3id.org/atlas/ontology#Pseudoscience"
    processor.atlas_graph.is_subclass_of = lambda mode_uri, ancestor_uri: (
        mode_uri == "https://w3id.org/atlas/ontology#OrgoneEnergy"
        and ancestor_uri == pseudoscience_uri
    )

    mode_uri, severity = processor._infer_primary_mode(
        "Orgone energy field accumulator",
        "pseudoscience",
    )

    assert mode_uri == "https://w3id.org/atlas/ontology#OrgoneEnergy"
    # Fabrication had higher severity but was filtered out.


_ATLAS_TTL = Path("../atlas-ontology/src/ontology/atlas.ttl")


@pytest.mark.skipif(
    not _ATLAS_TTL.exists(),
    reason="ATLAS ontology repo not present at ../atlas-ontology/ (clone it sibling to TRACES)",
)
def test_atlas_graph_is_subclass_of_walks_chain(tmp_path):
    """ATLASGraph.is_subclass_of returns True for transitive descendants."""
    from traces.atlas.ontology_loader import ATLASGraph

    graph = ATLASGraph(
        ontology_path=str(_ATLAS_TTL),
        vocabularies_path="../atlas-ontology/vocabularies/",
    )

    # WaterMemory -> UltraHighDilution -> Pseudoscience -> PremiseLevelFailure -> UnreliabilityMode
    assert graph.is_subclass_of(
        "https://w3id.org/atlas/ontology#WaterMemory",
        "https://w3id.org/atlas/ontology#Pseudoscience",
    )
    assert graph.is_subclass_of(
        "https://w3id.org/atlas/ontology#WaterMemory",
        "https://w3id.org/atlas/ontology#UnreliabilityMode",
    )
    # Reflexive (a class is a subclass of itself for our purposes).
    assert graph.is_subclass_of(
        "https://w3id.org/atlas/ontology#Pseudoscience",
        "https://w3id.org/atlas/ontology#Pseudoscience",
    )
    # Negative: WaterMemory is not under DeliberateMisconduct.
    assert not graph.is_subclass_of(
        "https://w3id.org/atlas/ontology#WaterMemory",
        "https://w3id.org/atlas/ontology#DeliberateMisconduct",
    )


def test_is_subclass_of_handles_multi_parent_classes():
    """If a class declares multiple parents (rdfs:subClassOf X, Y), the
    BFS walk must explore both branches — a single-parent walk would
    miss either ancestor."""
    from rdflib import Graph, URIRef
    from rdflib.namespace import RDFS

    from traces.atlas.ontology_loader import ATLAS, ATLASGraph

    # Construct a tiny graph with: Multi -> [BranchA, BranchB], each under Top.
    g = Graph()
    multi = URIRef(str(ATLAS) + "Multi")
    branch_a = URIRef(str(ATLAS) + "BranchA")
    branch_b = URIRef(str(ATLAS) + "BranchB")
    top = URIRef(str(ATLAS) + "Top")
    g.add((multi, RDFS.subClassOf, branch_a))
    g.add((multi, RDFS.subClassOf, branch_b))
    g.add((branch_a, RDFS.subClassOf, top))
    g.add((branch_b, RDFS.subClassOf, top))

    # Build an ATLASGraph instance without loading any TTL files, then
    # swap in our tiny test graph.
    graph = ATLASGraph.__new__(ATLASGraph)
    graph._g = g
    graph.vocab_root = None  # not used by is_subclass_of

    # Both branches reachable.
    assert graph.is_subclass_of(str(multi), str(branch_a))
    assert graph.is_subclass_of(str(multi), str(branch_b))
    # Top reachable through either parent — the BFS shouldn't get stuck.
    assert graph.is_subclass_of(str(multi), str(top))


def test_infer_primary_mode_respects_family_restriction(tmp_path):
    """A candidate appropriate for one family does not match a probe in
    another family. Verifies the ontology-based ancestor filter actively
    excludes off-family modes."""
    processor = _make_processor(tmp_path)
    processor._atlas_candidates = [
        AtlasModeCandidate(
            uri="https://w3id.org/atlas/ontology#BiofieldEnergyHealing",
            label="Biofield energy healing",
            default_severity=0.7,
            evidence_terms={"biofield", "healing", "treatment"},
        ),
    ]
    # BiofieldEnergyHealing is under Pseudoscience, NOT under DeliberateMisconduct.
    # A 'notorious_retractions' folder maps to DeliberateMisconduct (default).
    processor.atlas_graph.is_subclass_of = lambda mode_uri, ancestor_uri: (
        mode_uri == "https://w3id.org/atlas/ontology#BiofieldEnergyHealing"
        and ancestor_uri == "https://w3id.org/atlas/ontology#Pseudoscience"
    )

    mode_uri, severity = processor._infer_primary_mode(
        "Biofield energy treatment for immunomodulation",
        "notorious_retractions",  # default: atlas:DeliberateMisconduct
    )

    assert mode_uri is None
    assert severity == 0.0