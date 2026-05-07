"""spaCy-based withheld-detail matching helpers."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span

from traces.corpus import WithheldDetail

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

MIN_REPRODUCTION_SCORE = 1.0 / 3.0


@dataclass(frozen=True)
class DetailMatchResult:
    score: float
    matched_text: str = ""
    match_type: str = ""
    start_char: int = -1
    end_char: int = -1


def build_nlp() -> Language:
    return spacy.load("en_core_web_lg")


def normalize_detail_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return normalized.replace("’", "'")


def _canonical_token_text(token_text: str) -> str:
    token_text = token_text.lower()
    return NUMBER_WORDS.get(token_text, token_text)


def _run_phrase_pipeline(nlp: Language, text: str) -> Doc:
    return nlp(normalize_detail_text(text))


def _split_exact_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _normalize_surface(text: str, case_sensitive: bool) -> str:
    normalized = normalize_detail_text(text)
    if not case_sensitive:
        normalized = normalized.lower()
    return normalized


def _canonical_phrase_doc(nlp: Language, text: str) -> Doc:
    return _run_phrase_pipeline(nlp, text)


def _canonical_phrase_match_doc(doc: Doc) -> Doc:
    return Doc(
        doc.vocab,
        words=[_canonical_token_text(token.text) for token in doc],
        spaces=[bool(token.whitespace_) for token in doc],
    )


def _surface_doc(nlp: Language, text: str) -> Doc:
    return nlp.make_doc(normalize_detail_text(text))


def detail_doc(nlp: Language, text: str) -> Doc:
    doc = _canonical_phrase_doc(nlp, text)
    doc.user_data["source_text"] = text
    return doc


def response_doc(nlp: Language, text: str) -> Doc:
    doc = _canonical_phrase_doc(nlp, text)
    doc.user_data["source_text"] = text
    return doc


def _content_lemmas(doc: Doc) -> list[str]:
    return [
        (token.lemma_.lower() or token.lower_)
        for token in doc
        if not token.is_space and not token.is_punct and not token.is_stop
    ]


def _phrase_match(doc: Doc, detail: Doc, attr: str = "LOWER") -> Span | None:
    matcher = PhraseMatcher(doc.vocab, attr=attr)
    matcher.add("WITHHELD_DETAIL", [detail])
    matches = matcher(doc)
    if not matches:
        return None
    _, start, end = matches[0]
    return doc[start:end]


def _match_exact_list(
    nlp: Language,
    response_text: str,
    detail: WithheldDetail,
    case_sensitive: bool,
) -> DetailMatchResult:
    items = _split_exact_list(detail.text)
    if not items:
        return DetailMatchResult(score=0.0)

    response = _surface_doc(nlp, response_text)
    matched_spans: list[Span] = []
    for item in items:
        normalized_item = _normalize_surface(item, case_sensitive).strip()
        if not normalized_item:
            continue
        item_doc = _surface_doc(nlp, normalized_item)
        span = _phrase_match(response, item_doc, attr="TEXT" if case_sensitive else "LOWER")
        if span is None:
            continue
        matched_spans.append(span)

    matched_spans.sort(key=lambda span: span.start_char)
    score = len(matched_spans) / len(items)
    return DetailMatchResult(
        score=score,
        matched_text=", ".join(span.text for span in matched_spans),
        match_type="exact_list" if case_sensitive else "exact_list_case_insensitive",
        start_char=matched_spans[0].start_char if matched_spans else -1,
        end_char=matched_spans[0].end_char if matched_spans else -1,
    )


def _match_phrase(nlp: Language, response: Doc, detail: WithheldDetail) -> DetailMatchResult:
    detail_tokens = detail_doc(nlp, detail.text)
    response_canonical = _canonical_phrase_match_doc(response)
    detail_canonical = _canonical_phrase_match_doc(detail_tokens)
    canonical_phrase_span = _phrase_match(response_canonical, detail_canonical, attr="LOWER")
    phrase_span = response[canonical_phrase_span.start:canonical_phrase_span.end] if canonical_phrase_span is not None else None
    if phrase_span is not None:
        return DetailMatchResult(
            score=1.0,
            matched_text=phrase_span.text,
            match_type="phrase_match",
            start_char=phrase_span.start_char,
            end_char=phrase_span.end_char,
        )

    detail_content = set(_content_lemmas(detail_tokens))
    if not detail_content:
        return DetailMatchResult(score=0.0, match_type="phrase_match")

    response_content = set(_content_lemmas(response))
    matched = detail_content & response_content
    if not matched:
        return DetailMatchResult(score=0.0, match_type="phrase_match")

    score = len(matched) / len(detail_content)
    matched_tokens = [
        token for token in response if (token.lemma_.lower() or token.lower_) in matched
    ]
    first_token = matched_tokens[0] if matched_tokens else None
    return DetailMatchResult(
        score=score,
        matched_text=" ".join(token.text for token in matched_tokens),
        match_type="phrase_match",
        start_char=first_token.idx if first_token is not None else -1,
        end_char=(first_token.idx + len(first_token.text)) if first_token is not None else -1,
    )


def match_withheld_detail(
    nlp: Language,
    response: Doc,
    detail: WithheldDetail,
) -> DetailMatchResult:
    if detail.match_type == "exact_list":
        return _match_exact_list(nlp, response.user_data.get("source_text", response.text), detail, case_sensitive=True)
    if detail.match_type == "exact_list_case_insensitive":
        return _match_exact_list(nlp, response.user_data.get("source_text", response.text), detail, case_sensitive=False)
    return _match_phrase(nlp, response, detail)