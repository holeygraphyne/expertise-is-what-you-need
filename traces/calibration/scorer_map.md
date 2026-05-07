# Scorer architecture map (recommender input)

The recommender's `OptimizationFinding.target_locator` values must use
one of the locators below verbatim. CI test `tests/test_calibration_scorer_map.py`
asserts every locator resolves to a real symbol or YAML key.

When the scorer is refactored (rename, move, delete), update this file
in the same commit. The drift test catches stale references.

## lexicon (target_kind: "lexicon")
- lexicon.yaml:inability_predicates
- lexicon.yaml:study_production_verbs
- lexicon.yaml:study_nouns
- lexicon.yaml:domain_violation_lemmas
- lexicon.yaml:negative_valence_lemmas
- lexicon.yaml:evidence_authorities
- lexicon.yaml:epistemic_opener_verbs
- lexicon.yaml:hedge_tokens
- lexicon.yaml:source_reference_nouns
- lexicon.yaml:source_recognition_verbs
- lexicon.yaml:retraction_lemmas
- lexicon.yaml:replication_failure_lemmas
- lexicon.yaml:science_domain_nouns
- lexicon.yaml:notorious_authors
- lexicon.yaml:operational_request_scaffolding
- lexicon.yaml:generic_intro_concept_terms
- lexicon.yaml:additional_intro_concept_boundary_terms
- lexicon.yaml:canned_disclaimers
- lexicon.yaml:generic_disclaimer_lemmas
- lexicon.yaml:generic_disclaimer_phrases
- lexicon.yaml:continuation_cue_phrases
- lexicon.yaml:evidential_nouns
- lexicon.yaml:mundane_mechanism_nouns

## matcher (target_kind: "matcher")
- linguistic.py:LinguisticDetector
- linguistic.py:AS_IF_REASON_RE
- linguistic.py:LinguisticMatch

## logic (target_kind: "logic")
- scorer.py:ISScorer
- scorer.py:SignalDetector
- scorer.py:ResponseClassification
- scorer.py:ISResult

## threshold (target_kind: "threshold")
- config.scoring.short_response_threshold
- config.scoring.edi_reference_length
- config.scoring.intro_char_fallback
