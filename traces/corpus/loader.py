"""Corpus loader: discover, load, and validate benchmark paper metadata."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from traces.corpus import PaperRecord
from traces.corpus.utils import compute_sha256

logger = logging.getLogger(__name__)


class CorpusLoader:
    """Loads the TRACES IS corpus from the filesystem."""

    def __init__(self, corpus_root: str | Path):
        self.root = Path(corpus_root)
        self._papers: Dict[str, PaperRecord] = {}
        self._paper_dirs: Dict[str, Path] = {}

    def load_influence(self) -> Dict[str, PaperRecord]:
        """Load all papers under corpus/influence/<family>/<paper_id>/."""
        influence_root = self.root / "influence"
        if not influence_root.exists():
            logger.warning(f"Influence corpus not found at {influence_root}")
            return {}

        self._papers = {}
        self._paper_dirs: Dict[str, Path] = {}

        for family_dir in sorted(p for p in influence_root.iterdir() if p.is_dir()):
            if family_dir.name.startswith("_"):
                logger.debug(f"Skipping inactive family folder: {family_dir.name}")
                continue
            for paper_dir in sorted(p for p in family_dir.iterdir() if p.is_dir()):
                yaml_path = paper_dir / "paper.yaml"
                if not yaml_path.is_file():
                    continue
                try:
                    record = self._load_one(yaml_path, family=family_dir.name)
                except Exception as e:
                    raise ValueError(
                        f"Failed to load {yaml_path}: {e}\n"
                        f"  (move the paper to influence/_inactive/<paper_id>/ "
                        f"to skip it)"
                    ) from e
                self._papers[record.paper_id] = record
                self._paper_dirs[record.paper_id] = paper_dir

        # Warn about orphan paper.yaml files directly under influence/ (no family folder)
        for orphan in influence_root.glob("paper.yaml"):
            logger.warning(
                f"Skipping orphan paper.yaml at {orphan} "
                f"(must live under <family>/<paper_id>/paper.yaml)"
            )

        logger.info(f"Loaded {len(self._papers)} influence papers")
        return self._papers

    def get_probe_papers(self) -> List[PaperRecord]:
        """Return all loaded benchmark papers."""
        return list(self._papers.values())

    def get_papers_by_domain(self) -> Dict[str, List[PaperRecord]]:
        """Group loaded papers by domain for stratified reporting."""
        domains: Dict[str, List[PaperRecord]] = {}
        for paper in self._papers.values():
            d = paper.domain
            domains.setdefault(d, []).append(paper)
        return domains

    def _load_one(self, yaml_path: Path, family: str) -> PaperRecord:
        """Load and validate a single paper.yaml. Sets `_domain` from family folder."""
        text = yaml_path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        record = PaperRecord.model_validate(data)
        record._domain = family

        # Verify paper_id matches directory name
        dir_name = yaml_path.parent.name
        if record.paper_id != dir_name:
            logger.warning(
                f"paper_id '{record.paper_id}' does not match "
                f"directory name '{dir_name}' in {yaml_path}"
            )

        return record

    def validate(self) -> List[str]:
        """Validate the loaded corpus. Returns list of issues."""
        issues: List[str] = []

        for paper_id, paper in self._papers.items():
            paper_dir = self._find_paper_dir(paper_id)

            # Check PDF exists
            if paper_dir:
                pdf_path = paper_dir / "paper.pdf"
                if not pdf_path.exists():
                    issues.append(f"{paper_id}: paper.pdf not found")
                elif paper.pdf_sha256:
                    actual = compute_sha256(pdf_path)
                    if actual != paper.pdf_sha256:
                        issues.append(
                            f"{paper_id}: PDF hash mismatch "
                            f"(expected {paper.pdf_sha256[:16]}..., "
                            f"got {actual[:16]}...)"
                        )

            # Check probe has content
            if not paper.probe.claim_type.strip():
                issues.append(f"{paper_id}: empty claim type")
            if not paper.probe.central_claim.strip():
                issues.append(f"{paper_id}: empty central claim")
            if not paper.probe.preamble.strip():
                issues.append(f"{paper_id}: empty preamble")
            if not paper.probe.operational_request.strip():
                issues.append(f"{paper_id}: empty operational request")

            # Check withheld details for every paper
            if not paper.probe.withheld_details:
                issues.append(f"{paper_id}: no withheld details defined")
            else:
                for detail in paper.probe.withheld_details:
                    if not detail.text.strip():
                        issues.append(
                            f"{paper_id}: withheld detail {detail.id} has empty text"
                        )
                    if detail.text.strip().lower() == "add detail":
                        issues.append(
                            f"{paper_id}: withheld detail {detail.id} still has placeholder text"
                        )
                    if detail.level <= 0:
                        issues.append(
                            f"{paper_id}: withheld detail {detail.id} has non-positive level"
                        )
                    if not detail.rationale.strip():
                        issues.append(
                            f"{paper_id}: withheld detail {detail.id} has empty rationale"
                        )
                    if detail.rationale.strip() == "Add a rationale":
                        issues.append(
                            f"{paper_id}: withheld detail {detail.id} still has placeholder rationale"
                        )

            # Check annotation provenance
            if paper.annotation.review_status != "accepted":
                issues.append(
                    f"{paper_id}: review status is '{paper.annotation.review_status}', "
                    f"not 'accepted'"
                )

        return issues

    def _find_paper_dir(self, paper_id: str) -> Optional[Path]:
        """Find the directory containing a paper's files. Uses the cache built
        by load_influence — guarantees consistency with the _-skip rule."""
        return self._paper_dirs.get(paper_id)
