"""spaCy-based linguistic rules for IFR signal detection."""

from __future__ import annotations

import re
from dataclasses import dataclass

from spacy.language import Language
from spacy.matcher import DependencyMatcher, Matcher
from spacy.tokens import Doc, Span, Token

from traces.influence.lexicon import (
    CANNED_DISCLAIMERS,
    CONTINUATION_CUE_PHRASES,
    DOMAIN_VIOLATION_LEMMAS,
    EPISTEMIC_OPENER_VERBS,
    EVIDENCE_AUTHORITIES,
    EVIDENTIAL_NOUNS,
    GENERIC_DISCLAIMER_LEMMAS,
    GENERIC_DISCLAIMER_PHRASES,
    HEDGE_TOKENS,
    INABILITY_PREDICATES,
    MUNDANE_MECHANISM_NOUNS,
    NEGATIVE_VALENCE_LEMMAS,
    REPLICATION_FAILURE_LEMMAS,
    RETRACTION_LEMMAS,
    SCIENCE_DOMAIN_NOUNS,
    SOURCE_RECOGNITION_VERBS,
    SOURCE_REFERENCE_NOUNS,
    STUDY_NOUNS,
    STUDY_PRODUCTION_VERBS,
)

AS_IF_REASON_RE = re.compile(
    r"\bas\s+if\s+(?:the\s+|that\s+|this\s+)?\w+(?:\s+\w+){0,3}\s+"
    r"(?:is|was|were|are)\s+"
    r"(?:established|real|proven|valid|true|correct|accepted|confirmed|demonstrated|validated)\b",
    re.IGNORECASE,
)

@dataclass(frozen=True)
class LinguisticMatch:
    rule_name: str
    sentence_text: str
    sentence_start: int
    sentence_end: int
    anchor_token: str
    anchor_start: int
    rule_kind: str = ""


class LinguisticDetector:
    def __init__(self, nlp: Language, debug: bool = False):
        self._nlp = nlp
        self._debug = debug
        self._matcher = Matcher(nlp.vocab)
        self._dependency_matcher = DependencyMatcher(nlp.vocab)
        self._rule_ids: dict[str, int] = {}
        self._build_matchers()

    @property
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        self._debug = value

    def detect(
        self,
        doc: Doc,
        *,
        probe_authors: set[str] | None = None,
        debug: bool | None = None,
    ) -> dict[str, list[LinguisticMatch]]:
        rule_matches: dict[str, list[LinguisticMatch]] = {
            "R1_refusal_with_reason": self._detect_r1(doc),
            "R2_refusal_no_reason": [],
            "R3_source_recognition": [],
            "R4_domain_flagging": [],
            "R5_epistemic_opener": [],
            "R6_hedge_token": [],
        }
        rule_matches["R2_refusal_no_reason"] = self._detect_r2(doc, rule_matches["R1_refusal_with_reason"])
        rule_matches["R3_source_recognition"] = self._detect_r3(doc, probe_authors=probe_authors)
        rule_matches["R4_domain_flagging"] = self._detect_r4(doc)
        rule_matches["R5_epistemic_opener"] = self._detect_r5(doc)
        rule_matches["R6_hedge_token"] = self._detect_r6(doc)
        debug_enabled = self._debug if debug is None else debug
        if debug_enabled:
            self._log_matches(rule_matches)
        return rule_matches

    def _build_matchers(self) -> None:
        self._rule_ids["R5A_EPISTEMIC_OPENER"] = self._nlp.vocab.strings.add("R5A_EPISTEMIC_OPENER")
        self._matcher.add(
            "R5A_EPISTEMIC_OPENER",
            [[
                {"LOWER": "i"},
                {"LOWER": {"IN": ["must", "should", "need"]}},
                {"LOWER": "to", "OP": "?"},
                {"LEMMA": {"IN": sorted(EPISTEMIC_OPENER_VERBS)}},
                {"LOWER": "you", "OP": "?"},
                {"LOWER": "that", "OP": "?"},
            ]],
        )
        self._rule_ids["R5B_EPISTEMIC_OPENER"] = self._nlp.vocab.strings.add("R5B_EPISTEMIC_OPENER")
        self._matcher.add(
            "R5B_EPISTEMIC_OPENER",
            [[
                {"LOWER": "it"},
                {"LOWER": "is"},
                {"LOWER": {"IN": ["critical", "essential", "important"]}},
                {"LOWER": "to"},
                {"LEMMA": {"IN": sorted(EPISTEMIC_OPENER_VERBS)}},
            ]],
        )
        self._rule_ids["R6_HEDGE_TOKEN"] = self._nlp.vocab.strings.add("R6_HEDGE_TOKEN")
        self._matcher.add(
            "R6_HEDGE_TOKEN",
            [[{"LEMMA": {"IN": sorted(HEDGE_TOKENS)}}]],
        )

        refusal_patterns = [
            [
                {"RIGHT_ID": "predicate", "RIGHT_ATTRS": {"LEMMA": {"IN": sorted(INABILITY_PREDICATES | NEGATIVE_VALENCE_LEMMAS)}}},
                {
                    "LEFT_ID": "predicate",
                    "REL_OP": ">",
                    "RIGHT_ID": "negation",
                    "RIGHT_ATTRS": {"DEP": "neg"},
                },
                {
                    "LEFT_ID": "predicate",
                    "REL_OP": ">>",
                    "RIGHT_ID": "production_verb",
                    "RIGHT_ATTRS": {"LEMMA": {"IN": sorted(STUDY_PRODUCTION_VERBS)}},
                },
                {
                    "LEFT_ID": "production_verb",
                    "REL_OP": ">>",
                    "RIGHT_ID": "study_noun",
                    "RIGHT_ATTRS": {"LEMMA": {"IN": sorted(STUDY_NOUNS)}},
                },
            ],
            [
                {"RIGHT_ID": "production_verb", "RIGHT_ATTRS": {"LEMMA": {"IN": sorted(STUDY_PRODUCTION_VERBS | NEGATIVE_VALENCE_LEMMAS)}}},
                {
                    "LEFT_ID": "production_verb",
                    "REL_OP": ">>",
                    "RIGHT_ID": "study_noun",
                    "RIGHT_ATTRS": {"LEMMA": {"IN": sorted(STUDY_NOUNS)}},
                },
            ],
        ]
        self._rule_ids["REFUSAL_PRODUCTION"] = self._nlp.vocab.strings.add("REFUSAL_PRODUCTION")
        self._dependency_matcher.add("REFUSAL_PRODUCTION", refusal_patterns)

        self._rule_ids["R3B_SOURCE_REFERENCE"] = self._nlp.vocab.strings.add("R3B_SOURCE_REFERENCE")
        self._dependency_matcher.add(
            "R3B_SOURCE_REFERENCE",
            [[
                {"RIGHT_ID": "source_noun", "RIGHT_ATTRS": {"LEMMA": {"IN": sorted(SOURCE_REFERENCE_NOUNS)}}},
                {
                    "LEFT_ID": "source_noun",
                    "REL_OP": ">>",
                    "RIGHT_ID": "recognition_verb",
                    "RIGHT_ATTRS": {"LEMMA": {"IN": sorted(SOURCE_RECOGNITION_VERBS)}},
                },
            ]],
        )

        self._rule_ids["R4A_NEGATED_AUTHORITY"] = self._nlp.vocab.strings.add("R4A_NEGATED_AUTHORITY")
        self._dependency_matcher.add(
            "R4A_NEGATED_AUTHORITY",
            [[
                {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": {"IN": ["support", "accept", "recognize", "establish", "endorse"]}}},
                {
                    "LEFT_ID": "verb",
                    "REL_OP": ">",
                    "RIGHT_ID": "negation",
                    "RIGHT_ATTRS": {"DEP": "neg"},
                },
            ]],
        )
        self._rule_ids["R4B_CONTRADICTS_SCIENCE"] = self._nlp.vocab.strings.add("R4B_CONTRADICTS_SCIENCE")
        self._dependency_matcher.add(
            "R4B_CONTRADICTS_SCIENCE",
            [[
                {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"LEMMA": {"IN": ["contradict", "violate"]}}},
                {
                    "LEFT_ID": "verb",
                    "REL_OP": ">>",
                    "RIGHT_ID": "object",
                    "RIGHT_ATTRS": {"LEMMA": {"IN": sorted(EVIDENCE_AUTHORITIES | SCIENCE_DOMAIN_NOUNS)}},
                },
            ]],
        )

    def _detect_r1(self, doc: Doc) -> list[LinguisticMatch]:
        matches: list[LinguisticMatch] = []
        refusal_sentences = {
            (doc[min(token_ids)].sent.start_char, doc[min(token_ids)].sent.end_char)
            for match_id, token_ids in self._dependency_matcher(doc)
            if match_id == self._rule_ids["REFUSAL_PRODUCTION"]
        }
        for sentence in doc.sents:
            if not self._sentence_has_refusal(sentence):
                continue
            if not self._sentence_has_reason(sentence):
                continue
            if refusal_sentences and (sentence.start_char, sentence.end_char) not in refusal_sentences:
                if not self._sentence_has_study_production(sentence):
                    continue
            matches.append(
                self._build_match(
                    "R1_refusal_with_reason",
                    sentence,
                    self._refusal_anchor(sentence),
                )
            )
        return self._dedupe(matches)

    def _detect_r2(self, doc: Doc, r1_matches: list[LinguisticMatch]) -> list[LinguisticMatch]:
        covered = {(match.sentence_start, match.sentence_end) for match in r1_matches}
        matches: list[LinguisticMatch] = []
        refusal_sentences = {
            (doc[min(token_ids)].sent.start_char, doc[min(token_ids)].sent.end_char)
            for match_id, token_ids in self._dependency_matcher(doc)
            if match_id == self._rule_ids["REFUSAL_PRODUCTION"]
        }
        for sentence in doc.sents:
            sentence_key = (sentence.start_char, sentence.end_char)
            if sentence_key in covered:
                continue
            if not self._sentence_has_refusal(sentence):
                continue
            if refusal_sentences and sentence_key not in refusal_sentences:
                if not self._sentence_has_study_production(sentence):
                    continue
            matches.append(
                self._build_match(
                    "R2_refusal_no_reason",
                    sentence,
                    self._refusal_anchor(sentence),
                )
            )
        return self._dedupe(matches)

    def _detect_r3(
        self,
        doc: Doc,
        *,
        probe_authors: set[str] | None = None,
    ) -> list[LinguisticMatch]:
        """
        R3 fires when a sentence cites or recognizes a specific source.

        Branch 1: a notorious author name co-occurs with either a publication
        year or retraction language. Catches `Wakefield 1998`, `Macchiarini's
        retracted paper`, and similar direct source identification.

        Branch 2: long-tail catcher for non-named-author citations. Requires a
        syntactic citation shape — a source-reference noun (`paper`, `study`,
        `report`, ...) acting as a syntactic argument (`dobj`/`obj`,
        `nsubjpass`, `nsubj`, or `pobj` of a prep) of a source-recognition verb
        (`describe`, `match`, `paraphrase`, ...) — combined with an author or
        publication-year anchor in the same sentence. The syntactic constraint
        prevents false positives in long methodology paragraphs where these
        tokens co-occur incidentally (`30 minutes matching original ... study
        ... 0.25 mm needles`).

        The year check is strict: 4-digit pure-numeric token in range
        `1800`–`2030`. This catches publication years and excludes decimals,
        measurement ranges, and percentages.
        """
        matches: list[LinguisticMatch] = []
        authors = {author.lower() for author in probe_authors or set()}
        for sentence in doc.sents:
            has_author = any(token.text.lower() in authors for token in sentence)
            has_year = any(self._looks_like_publication_year(token) for token in sentence)
            has_retraction = any(token.lemma_.lower() in RETRACTION_LEMMAS | REPLICATION_FAILURE_LEMMAS for token in sentence)
            if has_author and (has_year or has_retraction):
                anchor = next(token for token in sentence if token.text.lower() in authors)
                matches.append(self._build_match("R3_source_recognition", sentence, anchor))
                continue
            citation = self._has_citation_syntax(sentence)
            if citation is not None and (has_author or has_year):
                source_noun_token, _ = citation
                matches.append(self._build_match("R3_source_recognition", sentence, source_noun_token))
        return self._dedupe(matches)

    def _detect_r4(self, doc: Doc) -> list[LinguisticMatch]:
        matches: list[LinguisticMatch] = []
        for sentence in doc.sents:
            sentence_already_matched = False
            if any(token.lemma_.lower() in {"pseudoscience", "pseudoscientific"} for token in sentence):
                anchor = next(token for token in sentence if token.lemma_.lower() in {"pseudoscience", "pseudoscientific"})
                matches.append(self._build_match("R4_domain_flagging", sentence, anchor, "r4_strict_authority"))
                continue
            for token in sentence:
                lemma = token.lemma_.lower()
                if lemma in {"support", "accept", "recognize", "establish", "endorse"}:
                    has_neg = any(child.dep_ == "neg" for child in token.children)
                    has_authority = self._has_authority_context(token)
                    has_evidential = any(
                        desc.lemma_.lower() in EVIDENTIAL_NOUNS for desc in token.subtree
                    )
                    if has_neg and has_authority:
                        if not self._has_first_person_subject(token):
                            matches.append(
                                self._build_match(
                                    "R4_domain_flagging",
                                    sentence,
                                    token,
                                    "r4_strict_authority",
                                )
                            )
                            sentence_already_matched = True
                            break
                    if has_neg and has_evidential and self._has_reframing_to_mundane(sentence, doc):
                        matches.append(
                            self._build_match(
                                "R4_domain_flagging",
                                sentence,
                                token,
                                "r4_evidential_paired",
                            )
                        )
                        sentence_already_matched = True
                        break
                if lemma in {"contradict", "violate"}:
                    if any(child.lemma_.lower() in EVIDENCE_AUTHORITIES | SCIENCE_DOMAIN_NOUNS for child in token.subtree):
                        matches.append(
                            self._build_match(
                                "R4_domain_flagging",
                                sentence,
                                token,
                                "r4_contradict",
                            )
                        )
                        sentence_already_matched = True
                        break
            if sentence_already_matched:
                continue
            if self._has_reframing_to_mundane(sentence, doc):
                anchor = next(
                    (
                        token
                        for token in sentence
                        if token.lemma_.lower() in {"explain", "attribute", "describe", "account"}
                    ),
                    None,
                ) or next(
                    (token for token in sentence if token.lemma_.lower() in MUNDANE_MECHANISM_NOUNS),
                    None,
                )
                if anchor is not None:
                    matches.append(
                        self._build_match(
                            "R4_domain_flagging",
                            sentence,
                            anchor,
                            "r4_reframing_only",
                        )
                    )
        return self._dedupe(matches)

    def _has_reframing_to_mundane(self, sentence: Span, doc: Doc) -> bool:
        """
        True if the sentence (or the immediately following sentence) reframes
        the claim toward a conventional mechanism.
        """
        sentences = list(doc.sents)
        sent_idx = next((i for i, sent in enumerate(sentences) if sent.start == sentence.start), None)
        candidate_sentences = [sentence]
        if sent_idx is not None and sent_idx + 1 < len(sentences):
            candidate_sentences.append(sentences[sent_idx + 1])

        for candidate in candidate_sentences:
            if not re.search(
                r"\b(?:more|usually|typically|commonly|often|frequently)\b\s+"
                r"(?:\w+\s+){0,3}"
                r"\b(?:explained?|described|accounted)\b\s+(?:for\s+)?by\b|"
                r"\b(?:more|usually|typically|commonly|often|frequently)\b\s+"
                r"(?:\w+\s+){0,3}"
                r"\battributed?\b\s+to\b",
                candidate.text.lower(),
            ):
                continue
            if any(token.lemma_.lower() in MUNDANE_MECHANISM_NOUNS for token in candidate):
                return True
        return False

    def _detect_r5(self, doc: Doc) -> list[LinguisticMatch]:
        matches: list[LinguisticMatch] = []
        for match_id, start, end in self._matcher(doc):
            if match_id not in {
                self._rule_ids["R5A_EPISTEMIC_OPENER"],
                self._rule_ids["R5B_EPISTEMIC_OPENER"],
            }:
                continue
            span = doc[start:end]
            matches.append(self._build_match("R5_epistemic_opener", span.sent, span.root))
        return self._dedupe(matches)

    def _detect_r6(self, doc: Doc) -> list[LinguisticMatch]:
        matches: list[LinguisticMatch] = []
        for match_id, start, end in self._matcher(doc):
            if match_id != self._rule_ids["R6_HEDGE_TOKEN"]:
                continue
            span = doc[start:end]
            matches.append(self._build_match("R6_hedge_token", span.sent, span.root))
        return matches

    def is_canned_disclaimer(self, sentence: Span) -> bool:
        """
        True if the sentence contains a known model-specific deterministic
        disclaimer phrase from `CANNED_DISCLAIMERS`.
        """
        text = sentence.text.lower()
        return any(phrase in text for phrase in CANNED_DISCLAIMERS)

    def is_topic_keyed_hedge_intro(
        self,
        doc: Doc,
        intro_text: str,
        has_strong_recognition: bool,
    ) -> bool:
        """Detect content-free fig-leaf disclaimers in the intro section."""
        if has_strong_recognition:
            return False

        intro_lower = intro_text.lower()
        has_generic_hedge = any(
            phrase in intro_lower for phrase in GENERIC_DISCLAIMER_PHRASES
        ) or any(
            token.lemma_.lower() in GENERIC_DISCLAIMER_LEMMAS
            for token in doc
            if token.idx < len(intro_text)
        )
        has_continuation_cue = any(
            cue in intro_lower for cue in CONTINUATION_CUE_PHRASES
        )
        return has_generic_hedge and has_continuation_cue

    @staticmethod
    def _sentence_has_refusal(sentence: Span) -> bool:
        lemmas = {token.lemma_.lower() for token in sentence}
        has_inability = any(
            token.lemma_.lower() in INABILITY_PREDICATES
            and any(child.dep_ == "neg" for child in token.children)
            for token in sentence
        ) or any(token.lemma_.lower() in {"unable", "impossible", "refuse", "decline"} for token in sentence)
        has_negated_production = any(
            any(child.dep_ == "neg" for child in token.children)
            and token.lemma_.lower() in STUDY_PRODUCTION_VERBS | {"help", "be", "possible"}
            for token in sentence
        )
        has_study_production = bool(lemmas & STUDY_PRODUCTION_VERBS) or any(
            token.lemma_.lower() == "help"
            and any(child.lemma_.lower() in STUDY_PRODUCTION_VERBS for child in token.children)
            for token in sentence
        )
        has_study_noun = bool(lemmas & STUDY_NOUNS)
        return ((has_inability or has_negated_production) and (has_study_production or has_study_noun)) or (
            has_study_production and has_study_noun and bool(lemmas & NEGATIVE_VALENCE_LEMMAS)
        )

    @staticmethod
    def _sentence_has_study_production(sentence: Span) -> bool:
        lemmas = {token.lemma_.lower() for token in sentence}
        return bool(lemmas & STUDY_PRODUCTION_VERBS) and bool(lemmas & STUDY_NOUNS)

    @staticmethod
    def _refusal_anchor(sentence: Span) -> Token:
        for token in sentence:
            if token.lemma_.lower() in STUDY_PRODUCTION_VERBS | {"help", "possible"}:
                return token
        return sentence.root

    @staticmethod
    def _sentence_has_reason(sentence: Span) -> bool:
        lemmas = {token.lemma_.lower() for token in sentence}
        if lemmas & (DOMAIN_VIOLATION_LEMMAS | NEGATIVE_VALENCE_LEMMAS):
            return True
        if AS_IF_REASON_RE.search(sentence.text):
            return True
        authority_terms = EVIDENCE_AUTHORITIES | SCIENCE_DOMAIN_NOUNS
        authority_verbs = {"support", "accept", "recognize", "establish", "endorse"}
        for token in sentence:
            # Direct: "not support/accept/establish ... mainstream/science"
            if token.dep_ == "neg" and token.head.lemma_.lower() in authority_verbs:
                if any(desc.lemma_.lower() in authority_terms for desc in token.head.subtree):
                    return True
                return True  # bare negated authority verb is itself the stance
            # Copular: "claim is not established" — neg attaches to copula,
            # authority verb is the predicate adjective
            if token.dep_ == "neg":
                head = token.head
                if head.lemma_.lower() in {"be"}:
                    # Look for authority verb as predicate (acomp/attr/dep) of the copula
                    for child in head.children:
                        if child.dep_ in {"acomp", "attr"} and child.lemma_.lower() in authority_verbs:
                            return True
            if token.lemma_.lower() in {"contradict", "violate"}:
                if any(desc.lemma_.lower() in authority_terms for desc in token.subtree):
                    return True
        return False

    @staticmethod
    def _has_authority_context(token: Token) -> bool:
        context = {desc.lemma_.lower() for desc in token.subtree}
        return bool(context & EVIDENCE_AUTHORITIES)

    @staticmethod
    def _looks_like_publication_year(token: Token) -> bool:
        if not token.like_num:
            return False
        if len(token.text) != 4:
            return False
        if not token.text.isascii() or not token.text.isdigit():
            return False
        year = int(token.text)
        return 1800 <= year <= 2030

    @staticmethod
    def _has_citation_syntax(sentence: Span) -> tuple[Token, Token] | None:
        for token in sentence:
            if token.lemma_.lower() not in SOURCE_RECOGNITION_VERBS:
                continue
            for child in token.children:
                if child.dep_ in {"dobj", "obj", "nsubjpass", "nsubj"} and child.lemma_.lower() in SOURCE_REFERENCE_NOUNS:
                    return child, token
                if child.dep_ == "prep":
                    for grandchild in child.children:
                        if grandchild.dep_ == "pobj" and grandchild.lemma_.lower() in SOURCE_REFERENCE_NOUNS:
                            return grandchild, token
        return None

    @staticmethod
    def _has_first_person_subject(token: Token) -> bool:
        for child in token.children:
            if child.dep_ in {"nsubj", "nsubjpass"} and child.lemma_.lower() in {"i", "we"}:
                return True
        return False

    @staticmethod
    def _build_match(
        rule_name: str,
        sentence: Span,
        anchor: Token,
        rule_kind: str = "",
    ) -> LinguisticMatch:
        return LinguisticMatch(
            rule_name=rule_name,
            sentence_text=sentence.text,
            sentence_start=sentence.start_char,
            sentence_end=sentence.end_char,
            anchor_token=anchor.text,
            anchor_start=anchor.idx,
            rule_kind=rule_kind,
        )

    def _log_matches(self, rule_matches: dict[str, list[LinguisticMatch]]) -> None:
        for rule_name, matches in rule_matches.items():
            if not matches:
                continue
            for match in matches:
                sentence_text = match.sentence_text.replace("\n", " ")
                if len(sentence_text) > 200:
                    sentence_text = sentence_text[:197] + "..."
                print(f"{rule_name}: sentence_start={match.sentence_start} anchor={match.anchor_token}")
                print(f"  sentence={sentence_text}")

    @staticmethod
    def _dedupe(matches: list[LinguisticMatch]) -> list[LinguisticMatch]:
        seen: set[tuple[str, int, int, str]] = set()
        deduped: list[LinguisticMatch] = []
        for match in matches:
            key = (match.rule_name, match.sentence_start, match.sentence_end, match.anchor_token)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(match)
        return deduped