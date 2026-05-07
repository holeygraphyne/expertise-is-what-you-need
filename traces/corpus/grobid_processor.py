"""Bootstrap missing corpus paper metadata with GROBID outputs."""

from __future__ import annotations

import textwrap
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import requests
import spacy
import yaml
from grobid_tei_xml import parse_document_xml
from grobid_tei_xml.types import GrobidDocument
from rdflib.namespace import RDFS

from traces.atlas.ontology_loader import ATLAS, ATLASGraph
from traces.config import TracesConfig
from traces.corpus.utils import compute_sha256
from traces.corpus.yaml_io import dump_paper_yaml
from traces.pipeline import ProviderClient

logger = logging.getLogger(__name__)

_PLACEHOLDER_WITHHELD_DETAILS = [
    {
        "id": f"wd-{index:03d}",
        "text": "add detail",
        "match_type": "phrase_match",
        "level": 0,
        "rationale": "Add a rationale",
    }
    for index in range(1, 7)
]
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_BIBLIOGRAPHY_RE = re.compile(
    r"(?is)(?:^|\n)\s*(references|bibliography|works cited|literature cited)\s*(?:\n|$).*$"
)
_TEXT_BREAK_RE = re.compile(r"\n{3,}")
_GENERIC_MODE_TERMS = {
    "effect",
    "effects",
    "study",
    "studies",
    "paper",
    "result",
    "results",
    "energy",
    "reaction",
    "reactions",
    "system",
    "potential",
    "role",
    "roles",
    "high",
    "low",
    "treatment",
    "method",
    "methods",
}
_SECTION_STOP_HEADINGS = {"references", "bibliography", "works cited", "literature cited"}
_INTRO_HEADINGS = {"introduction", "background", "overview"}
_CONCLUSION_HEADINGS = {"conclusion", "conclusions", "discussion and conclusions"}
_TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


@dataclass(frozen=True)
class AtlasModeCandidate:
    uri: str
    label: str
    default_severity: float
    evidence_terms: set[str]


class GrobidProcessor:
    """Process unprepared corpus folders through GROBID and create bootstrap YAML."""

    def __init__(self, config: TracesConfig):
        self.config = config
        self.root = Path(config.corpus.root)
        self.base_url = config.grobid.url.rstrip("/")
        self._nlp = spacy.load("en_core_web_sm")
        self.atlas_graph = ATLASGraph(
            config.atlas.ontology_path,
            config.atlas.vocabularies_path,
        )
        # Resolve the service model: its id must appear in config.models, and
        # that model's `provider` must reference an entry in config.providers.
        # The API-visible model name is provider_model_id (falls back to id).
        self._service_model_id, self._provider = self._build_service_provider_client(config)
        self._atlas_candidates = self._load_atlas_candidates()

    @staticmethod
    def _build_service_provider_client(
        config: TracesConfig,
    ) -> tuple[str, ProviderClient | None]:
        """Resolve service_model.id to a (api_model_name, ProviderClient).

        Returns ``(api_model_name, None)`` when no service model is configured
        (empty service_model.id) — central-claim summarization will be a no-op
        in that case. Raises ValueError if a service_model.id is set but its
        config entry is missing or has no provider.
        """
        sm_id = config.service_model.id
        if not sm_id:
            return "", None
        sm = next((m for m in config.models if m.id == sm_id), None)
        if sm is None or sm.provider is None:
            raise ValueError(
                f"service_model.id={sm_id!r} not found in models, "
                f"or model has no provider"
            )
        provider_cfg = config.providers.get(sm.provider)
        if provider_cfg is None:
            raise ValueError(
                f"service_model {sm_id!r} references provider {sm.provider!r} "
                f"which is not in config.providers"
            )
        return (sm.provider_model_id or sm.id), ProviderClient(provider_cfg)

    @classmethod
    def from_config_path(cls, config_path: str | Path) -> "GrobidProcessor":
        return cls(TracesConfig.load(config_path))

    def _check_server(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/isalive", timeout=10)
        except requests.RequestException:
            return False
        return response.status_code == 200

    def process_pdf(self, pdf_path: Path) -> tuple[Path, GrobidDocument]:
        with open(pdf_path, "rb") as handle:
            response = requests.post(
                f"{self.base_url}/api/processFulltextDocument",
                files={"input": (pdf_path.name, handle, "application/pdf")},
                data={
                    "consolidateHeader": str(self.config.grobid.consolidate_header),
                    "consolidateCitations": str(self.config.grobid.consolidate_citations),
                    "includeRawCitations": "1",
                    "includeRawAffiliations": "1",
                },
                timeout=self.config.grobid.timeout,
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"GROBID returned {response.status_code} for {pdf_path.name}: "
                f"{response.text[:200]}"
            )

        tei_path = pdf_path.with_suffix(".tei.xml")
        tei_path.write_text(response.text, encoding="utf-8")
        return tei_path, parse_document_xml(response.text)

    def bootstrap_all(self) -> dict[str, int]:
        if not self._check_server():
            raise ConnectionError(
                f"GROBID server not reachable at {self.base_url}. "
                "Start it with: docker run --rm -p 8070:8070 lfoppiano/grobid:latest"
            )

        stats = {"processed": 0, "skipped": 0, "failed": 0}
        for paper_dir, domain in self._iter_candidate_paper_dirs():
            if not self._should_bootstrap(paper_dir):
                stats["skipped"] += 1
                continue

            try:
                pdf_path = paper_dir / "paper.pdf"
                tei_path, doc = self.process_pdf(pdf_path)
                yaml_data = self._build_yaml_data(
                    paper_dir=paper_dir,
                    domain=domain,
                    pdf_path=pdf_path,
                    doc=doc,
                    tei_path=tei_path,
                )
                self._write_yaml(paper_dir / "paper.yaml", yaml_data)
                stats["processed"] += 1
            except Exception:
                stats["failed"] += 1
                logger.exception("Failed to bootstrap %s", paper_dir)

        return stats

    def _iter_candidate_paper_dirs(self) -> Iterable[tuple[Path, str]]:
        influence_root = self.root / "influence"
        if influence_root.exists():
            for family_dir in sorted(path for path in influence_root.iterdir() if path.is_dir()):
                for paper_dir in sorted(path for path in family_dir.iterdir() if path.is_dir()):
                    if paper_dir.name.startswith("_"):
                        continue
                    yield paper_dir, family_dir.name

        membership_root = self.root / "membership" / "papers"
        if membership_root.exists():
            for paper_dir in sorted(path for path in membership_root.iterdir() if path.is_dir()):
                if paper_dir.name.startswith("_"):
                    continue
                yield paper_dir, "membership"

    @staticmethod
    def _should_bootstrap(paper_dir: Path) -> bool:
        return (
            (paper_dir / "paper.pdf").exists()
            and not (paper_dir / "paper.yaml").exists()
            and not (paper_dir / "paper.tei.xml").exists()
        )

    def _build_yaml_data(
        self,
        paper_dir: Path,
        domain: str,
        pdf_path: Path,
        doc: GrobidDocument,
        tei_path: Path | None = None,
    ) -> dict[str, Any]:
        header = doc.header
        today = date.today().isoformat()
        title = (header.title or "").strip() if header else ""
        claim_source = self._build_claim_source(doc, tei_path)
        central_claim = self._wrap_block_text(self._summarize_central_claim(claim_source))
        mode_source = self._build_mode_inference_source(doc, central_claim, tei_path)
        primary_mode, severity = self._infer_primary_mode(mode_source, domain)
        preamble = self._wrap_block_text(self._build_preamble(doc, tei_path))

        return {
            "paper_id": paper_dir.name,
            "doi": (header.doi or None) if header else None,
            "title": title or None,
            "authors": self._extract_authors(doc),
            "journal": (header.journal or None) if header else None,
            "year": self._extract_year(header.date if header else None),
            "pdf_sha256": compute_sha256(pdf_path),
            "atlas": {
                "primary_unreliability_mode": self.atlas_graph.curie(primary_mode) if primary_mode else None,
                "secondary_unreliability_modes": [],
                "claimed_domain": None,
                "detection_markers": [],
                "default_severity": severity,
                "retraction": None,
            },
            "probe": {
                "claim_type": "",
                "central_claim": central_claim,
                "preamble": preamble,
                "operational_request": "",
                "withheld_details": [dict(detail) for detail in _PLACEHOLDER_WITHHELD_DETAILS],
            },
            "annotation": {
                "annotator": "",
                "reviewer": "",
                "review_status": "pending",
                "date_annotated": today,
                "date_reviewed": today,
                "notes": self._wrap_block_text("Bootstrapped from GROBID; requires human review."),
            },
        }

    def _load_atlas_candidates(self) -> list[AtlasModeCandidate]:
        """All ATLAS mode candidates as a flat list. Filtering by domain
        happens in _infer_primary_mode using the config's
        domain_atlas_ancestors map plus the rdfs:subClassOf chain."""
        candidates: list[AtlasModeCandidate] = []
        graph = getattr(self.atlas_graph, "_g")
        # set() dedupes subjects that have multiple rdfs:label triples
        # (e.g., language-tagged labels for i18n). graph.subjects() yields
        # one row per matching triple, not one per subject.
        for subject in set(graph.subjects(RDFS.label, None)):
            if not str(subject).startswith(str(ATLAS)):
                continue
            label = graph.value(subject, RDFS.label)
            if label is None:
                continue
            uri = str(subject)
            candidates.append(AtlasModeCandidate(
                uri=uri,
                label=str(label),
                default_severity=self._severity_for_mode(uri),
                evidence_terms=self._candidate_evidence_terms(uri, str(label)),
            ))
        return candidates

    def _build_mode_inference_source(
        self,
        doc: GrobidDocument,
        central_claim: str,
        tei_path: Path | None = None,
    ) -> str:
        parts: list[str] = []
        if doc.header and doc.header.title:
            parts.append(doc.header.title.strip())
        if doc.abstract:
            parts.append(self._normalize_text_block(doc.abstract))
        if central_claim.strip():
            parts.append(central_claim.strip())
        body = self._extract_body_text(doc, tei_path)
        if body:
            parts.append(self._fallback_top_10_percent(body))
        return "\n\n".join(part for part in parts if part).strip()

    def _infer_primary_mode(
        self, source_text: str, family_folder: str
    ) -> tuple[str | None, float]:
        """Score source_text against ATLAS modes restricted to subclasses of
        the ancestor class configured for `family_folder`. Returns the
        highest-scoring (uri, severity), or (None, 0.0) if nothing matches.

        If family_folder is absent from config.grobid.domain_atlas_ancestors,
        bootstrap remains permissive and emits no inferred mode rather than
        failing. This keeps GROBID bootstrap decoupled from active-corpus
        family registration.
        """
        if not source_text.strip():
            return None, 0.0

        ancestors_map = self.config.grobid.domain_atlas_ancestors
        if family_folder not in ancestors_map:
            logger.info(
                "No GROBID ATLAS ancestor mapping configured for family %s; "
                "leaving bootstrap mode unset",
                family_folder,
            )
            return None, 0.0
        ancestor_curie = ancestors_map[family_folder]
        # Expand CURIE to full URI: "atlas:Foo" -> "https://w3id.org/atlas/ontology#Foo"
        if ancestor_curie.startswith("atlas:"):
            ancestor_uri = str(ATLAS) + ancestor_curie.removeprefix("atlas:")
        else:
            ancestor_uri = ancestor_curie  # already full URI

        source_terms = self._lemma_terms(source_text)
        if not source_terms:
            return None, 0.0

        best_candidate: AtlasModeCandidate | None = None
        best_score = 0.0
        for candidate in self._atlas_candidates:
            if not self.atlas_graph.is_subclass_of(candidate.uri, ancestor_uri):
                continue
            score = self._mode_match_score(source_terms, candidate.evidence_terms)
            if score > best_score:
                best_candidate = candidate
                best_score = score
        if best_candidate is None:
            return None, 0.0
        return best_candidate.uri, best_candidate.default_severity

    def _severity_for_mode(self, mode_uri: str) -> float:
        return self.atlas_graph.default_severity(mode_uri) or 0.0

    def _candidate_evidence_terms(self, mode_uri: str, label: str) -> set[str]:
        terms = set(self._lemma_terms(label))
        definition = self.atlas_graph.definition(mode_uri)
        if definition:
            terms.update(self._lemma_terms(definition))
        for lexicon_file in self.atlas_graph.resolve_lexicon_files(mode_uri):
            if not lexicon_file.exists():
                continue
            data = yaml.safe_load(lexicon_file.read_text(encoding="utf-8")) or {}
            for entry in data.get("terms", []):
                if not isinstance(entry, dict):
                    continue
                term_text = (entry.get("term") or "").strip()
                if term_text:
                    terms.update(self._lemma_terms(term_text))
        return {term for term in terms if term}

    def _lemma_terms(self, text: str) -> set[str]:
        terms: set[str] = set()
        for token in self._nlp(text):
            if token.is_space or token.is_punct or token.is_stop:
                continue
            term = self._normalize_mode_token(token)
            if term:
                terms.add(term)
        return terms

    @staticmethod
    def _normalize_mode_token(token: Any) -> str:
        text = token.text.strip()
        if not text:
            return ""
        lowered_text = token.lower_.strip()
        if any(ch.isdigit() for ch in text) or text.isupper() or (len(text) <= 5 and any(ch.isupper() for ch in text)):
            return lowered_text
        lemma = (token.lemma_ or "").strip().lower()
        if lemma and lemma not in {"-pron-", lowered_text}:
            return lemma
        return lowered_text

    @staticmethod
    def _mode_match_score(source_terms: set[str], candidate_terms: set[str]) -> float:
        if not source_terms or not candidate_terms:
            return 0.0
        overlap = source_terms & candidate_terms
        if not overlap:
            return 0.0
        distinctive = overlap - _GENERIC_MODE_TERMS
        if not distinctive:
            return 0.0
        base = len(distinctive) / len(candidate_terms)
        short_technical = sum(
            1 for term in distinctive if any(ch.isdigit() for ch in term) or len(term) <= 5
        )
        return base + (0.2 * short_technical)

    @staticmethod
    def _extract_authors(doc: GrobidDocument) -> list[str]:
        authors = []
        header = doc.header
        if header and header.authors:
            for author in header.authors:
                name = f"{author.given_name or ''} {author.surname or ''}".strip()
                if name:
                    authors.append(name)
        return authors

    @staticmethod
    def _extract_year(header_date: str | None) -> int | None:
        if not header_date:
            return None
        match = _YEAR_RE.search(header_date)
        return int(match.group(0)) if match else None

    @staticmethod
    def _normalize_text_block(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        cleaned = "\n".join(line for line in lines if line)
        return _TEXT_BREAK_RE.sub("\n\n", cleaned).strip()

    def _extract_paragraph_text_without_bibr_refs(self, paragraph_elem: ET.Element) -> str:
        parts: list[str] = []

        def visit(elem: ET.Element) -> None:
            if elem.tag.endswith("ref") and elem.attrib.get("type") == "bibr":
                return

            if elem.text:
                parts.append(elem.text)

            for child in elem:
                visit(child)
                if child.tail:
                    parts.append(child.tail)

        visit(paragraph_elem)
        return self._normalize_text_block("".join(parts))

    def _extract_structured_sections(self, tei_path: Path | None) -> list[tuple[str, str]]:
        if tei_path is None or not tei_path.exists():
            return []
        try:
            root = ET.fromstring(tei_path.read_text(encoding="utf-8"))
        except ET.ParseError:
            return []
        sections: list[tuple[str, str]] = []
        for div in root.findall(".//tei:text/tei:body//tei:div", _TEI_NS):
            head = " ".join(div.findtext("tei:head", default="", namespaces=_TEI_NS).split())
            if not head:
                continue
            paragraphs = []
            for paragraph in div.findall("./tei:p", _TEI_NS):
                text = self._extract_paragraph_text_without_bibr_refs(paragraph)
                if text:
                    paragraphs.append(text)
            body = self._normalize_text_block("\n\n".join(paragraphs))
            if body:
                sections.append((head, body))
        return sections

    def _clean_body_text(self, text: str) -> str:
        return self._normalize_text_block(_BIBLIOGRAPHY_RE.sub("", text or ""))

    def _extract_body_text(self, doc: GrobidDocument, tei_path: Path | None = None) -> str:
        sections = self._extract_structured_sections(tei_path)
        if sections:
            kept = [body for head, body in sections if head.strip().lower() not in _SECTION_STOP_HEADINGS]
            return self._clean_body_text("\n\n".join(kept))
        return self._clean_body_text(doc.body or "")

    def _section_text(
        self,
        tei_path: Path | None,
        headings: set[str],
    ) -> str:
        for head, body in self._extract_structured_sections(tei_path):
            normalized_head = self._normalize_mode_heading(head)
            if normalized_head in headings:
                return self._normalize_text_block(body)
        return ""

    def _extract_intro_text(self, doc: GrobidDocument, tei_path: Path | None = None) -> str:
        intro = self._section_text(tei_path, _INTRO_HEADINGS)
        if intro:
            return intro
        return ""

    @staticmethod
    def _fallback_top_10_percent(text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        limit = max(1, int(len(text) * 0.10))
        return text[:limit].strip()

    @staticmethod
    def _fallback_last_10_percent(text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        limit = max(1, int(len(text) * 0.10))
        return text[-limit:].strip()

    def _build_preamble(self, doc: GrobidDocument, tei_path: Path | None = None) -> str:
        intro = self._extract_intro_text(doc, tei_path)
        if intro:
            return intro
        return self._fallback_top_10_percent(self._extract_body_text(doc, tei_path))

    @staticmethod
    def _normalize_mode_heading(heading: str) -> str:
        return " ".join(heading.lower().split())

    def _extract_conclusion_text(self, doc: GrobidDocument, tei_path: Path | None = None) -> str:
        conclusion = self._section_text(tei_path, _CONCLUSION_HEADINGS)
        if conclusion:
            return conclusion
        return ""

    def _build_claim_source(self, doc: GrobidDocument, tei_path: Path | None = None) -> str:
        parts = []
        if doc.header and doc.header.title:
            parts.append(f"[TITLE]\n{doc.header.title.strip()}")
        if doc.abstract:
            parts.append(f"[ABSTRACT]\n{self._normalize_text_block(doc.abstract)}")
        conclusion = self._extract_conclusion_text(doc, tei_path)
        if conclusion:
            parts.append(f"[CONCLUSIONS]\n{conclusion}")
        body = self._extract_body_text(doc, tei_path)
        if body and not conclusion:
            top = self._fallback_top_10_percent(body)
            tail = self._fallback_last_10_percent(body)
            parts.append(f"[EXCERPTS]\n{top}\n\n{tail}".strip())
        return "\n\n".join(part for part in parts if part).strip()

    @staticmethod
    def _wrap_block_text(text: str, width: int = 80) -> str:
        text = text.strip()
        if not text:
            return ""

        paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
        wrapped = [
            textwrap.fill(
                paragraph,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
            for paragraph in paragraphs
        ]
        return "\n\n".join(wrapped)

    def _summarize_central_claim(self, text: str) -> str:
        if not text.strip() or self._provider is None:
            return ""
        prompt = (
            "Summarize the central claim of the following scientific paper material.\n\n"
            "Requirements:\n"
            "- Output only the claim.\n"
            "- Maximum 4 lines assuming an 80-character column.\n"
            "- Prefer one compact paragraph.\n"
            "- Do not include preamble, caveats, or explanation.\n"
            "- Focus on what the paper claims is true or effective.\n\n"
            f"Material:\n{text}"
        )
        try:
            response = self._provider.complete(
                model=self._service_model_id,
                user_prompt=prompt,
                max_tokens=256,
                temperature=0.0,
            )
        except Exception:
            logger.exception("Central-claim summarization failed")
            return ""
        return response.content.strip()

    @staticmethod
    def _write_yaml(path: Path, data: dict[str, Any]) -> None:
        path.write_text(dump_paper_yaml(data), encoding="utf-8")