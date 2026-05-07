"""
ATLAS vocabulary loader.

Loads rejection and engagement YAML vocabularies and resolves
inheritance for a given unreliability mode.
"""
from __future__ import annotations
from traces.atlas.ontology_loader import ATLASGraph

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)


class VocabularyTerm:
    """A single term from a rejection or engagement vocabulary."""

    __slots__ = ("term", "classification", "rationale")

    def __init__(self, term: str, classification: str, rationale: str = ""):
        self.term = term.lower()
        self.classification = classification
        self.rationale = rationale

    def __repr__(self) -> str:
        return f"VocabularyTerm({self.term!r}, {self.classification!r})"


class Vocabulary:
    """Collection of terms for one vocabulary file."""

    def __init__(self, domain: str, terms: List[VocabularyTerm]):
        self.domain = domain
        self.terms = terms
        self._by_classification: Dict[str, List[VocabularyTerm]] = {}
        for t in terms:
            self._by_classification.setdefault(t.classification, []).append(t)

    def terms_for(self, classification: str) -> List[VocabularyTerm]:
        return self._by_classification.get(classification, [])

    @property
    def all_terms(self) -> List[str]:
        return [t.term for t in self.terms]

    def match_text(self, text: str) -> List[VocabularyTerm]:
        """Find all vocabulary terms present in the given text."""
        text_lower = text.lower()
        matches = []
        # Sort by length descending to prefer longer matches
        for term in sorted(self.terms, key=lambda t: len(t.term), reverse=True):
            if term.term in text_lower:
                matches.append(term)
        return matches


class VocabularyLoader:
    def __init__(self, atlas_graph: ATLASGraph):
        self.atlas = atlas_graph
        self._cache: dict[str, Vocabulary] = {}

    def load_rejection_vocabulary(
            self, mode: str | None = None, has_retraction: bool = False
    ) -> Vocabulary:
        files: list[Path] = []
        seen: set[Path] = set()

        if mode:
            for path in self.atlas.resolve_lexicon_files(mode):
                if path not in seen:
                    files.append(path)
                    seen.add(path)

        if has_retraction:
            retraction_file = self.atlas.vocab_root / "rejection/retracted-literature-base.yaml"
            if retraction_file not in seen:
                files.append(retraction_file)

        if not files:
            files = [self.atlas.vocab_root / "rejection/pseudoscience-base.yaml"]

        return self._merge(files)

    def _merge(self, paths: list[Path]) -> Vocabulary:
        all_terms: dict[str, VocabularyTerm] = {}
        for path in reversed(paths):   # root-first, children override
            vocab = self._load_file(path)
            if vocab:
                for t in vocab.terms:
                    all_terms[t.term] = t
        domain = str(paths[0]) if paths else "default"
        return Vocabulary(domain=domain, terms=list(all_terms.values()))

    def _load_file(self, path: Path) -> Vocabulary | None:
        key = str(path)
        if key in self._cache:
            return self._cache[key]
        if not path.exists():
            logger.warning(f"Vocabulary file not found: {path}")
            return None
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            terms = [
                VocabularyTerm(
                    term=e["term"],
                    classification=e["classification"],
                    rationale=e.get("rationale", ""),
                )
                for e in data.get("terms", [])
            ]
            vocab = Vocabulary(domain=data.get("domain", ""), terms=terms)
            self._cache[key] = vocab
            return vocab
        except Exception as e:
            logger.error(f"Failed to load vocabulary {path}: {e}")
            return None