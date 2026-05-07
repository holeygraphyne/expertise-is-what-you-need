"""Lemma vocabularies for IFR signal detection. Lowercase, lemmatized form."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

import yaml


def _load_lexicon() -> Mapping[str, Any]:
    lexicon_path = Path(__file__).with_name("lexicon.yaml")
    with lexicon_path.open(encoding="utf-8") as handle:
        return cast(Mapping[str, Any], yaml.safe_load(handle))


_LEXICON = _load_lexicon()


def _load_set(name: str) -> set[str]:
    return set(cast(list[str], _LEXICON[name]))


LEXICON_VERSION: str = str(_LEXICON["version"])
INABILITY_PREDICATES: set[str] = _load_set("inability_predicates")
STUDY_PRODUCTION_VERBS: set[str] = _load_set("study_production_verbs")
STUDY_NOUNS: set[str] = _load_set("study_nouns")
DOMAIN_VIOLATION_LEMMAS: set[str] = _load_set("domain_violation_lemmas")
NEGATIVE_VALENCE_LEMMAS: set[str] = _load_set("negative_valence_lemmas")
EVIDENCE_AUTHORITIES: set[str] = _load_set("evidence_authorities")
EVIDENTIAL_NOUNS: set[str] = _load_set("evidential_nouns")
EPISTEMIC_OPENER_VERBS: set[str] = _load_set("epistemic_opener_verbs")
HEDGE_TOKENS: set[str] = _load_set("hedge_tokens")
MUNDANE_MECHANISM_NOUNS: set[str] = _load_set("mundane_mechanism_nouns")
SOURCE_REFERENCE_NOUNS: set[str] = _load_set("source_reference_nouns")
SOURCE_RECOGNITION_VERBS: set[str] = _load_set("source_recognition_verbs")
RETRACTION_LEMMAS: set[str] = _load_set("retraction_lemmas")
REPLICATION_FAILURE_LEMMAS: set[str] = _load_set("replication_failure_lemmas")
SCIENCE_DOMAIN_NOUNS: set[str] = _load_set("science_domain_nouns")
OPERATIONAL_REQUEST_SCAFFOLDING: set[str] = _load_set("operational_request_scaffolding")
GENERIC_INTRO_CONCEPT_TERMS: set[str] = _load_set("generic_intro_concept_terms")
ADDITIONAL_INTRO_CONCEPT_BOUNDARY_TERMS: set[str] = _load_set(
    "additional_intro_concept_boundary_terms"
)
INTRO_CONCEPT_BOUNDARY_TERMS: set[str] = (
        GENERIC_INTRO_CONCEPT_TERMS | ADDITIONAL_INTRO_CONCEPT_BOUNDARY_TERMS
)
CANNED_DISCLAIMERS: set[str] = _load_set("canned_disclaimers")
GENERIC_DISCLAIMER_LEMMAS: set[str] = _load_set("generic_disclaimer_lemmas")
GENERIC_DISCLAIMER_PHRASES: set[str] = _load_set("generic_disclaimer_phrases")
CONTINUATION_CUE_PHRASES: set[str] = _load_set("continuation_cue_phrases")