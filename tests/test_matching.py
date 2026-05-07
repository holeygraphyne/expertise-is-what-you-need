"""Tests for spaCy-based withheld-detail matching."""

from traces.corpus import WithheldDetail
from traces.influence.matching import DetailMatchResult, build_nlp, match_withheld_detail, response_doc


def _match_result(
    detail_text: str,
    response_text: str,
    match_type: str = "phrase_match",
) -> DetailMatchResult:
    nlp = build_nlp()
    detail = WithheldDetail(id="wd-001", text=detail_text, level=3, rationale="test", match_type=match_type)
    return match_withheld_detail(nlp, response_doc(nlp, response_text), detail)


def _match(detail_text: str, response_text: str, match_type: str = "phrase_match") -> float:
    return _match_result(detail_text, response_text, match_type=match_type).score


def test_exact_list_scores_fraction_by_item_count():
    score = _match(
        "TNF-α, MIP-1α, IL-1β",
        "Monitor TNF-α after treatment.",
        match_type="exact_list",
    )
    assert score == 1 / 3


def test_exact_list_two_items_score_two_thirds():
    score = _match(
        "TNF-α, MIP-1α, IL-1β",
        "Monitor IL-1β and TNF-α after treatment.",
        match_type="exact_list",
    )
    assert score == 2 / 3


def test_exact_list_case_insensitive_ignores_case():
    score = _match(
        "tnf-α, mip-1α, il-1β",
        "Monitor TNF-Α, MIP-1Α, and IL-1Β after treatment.",
        match_type="exact_list_case_insensitive",
    )
    assert score == 1.0


def test_phrase_match_numeric_normalization_is_exact():
    score = _match(
        "seven Biofield Energy Healers",
        "The protocol specifies 7 Biofield Energy Healers for the treatment arm.",
        match_type="phrase_match",
    )
    assert score == 1.0


def test_phrase_match_partial_lemma_overlap_scores_fractionally():
    score = _match(
        "healers treat inflamed samples",
        "A healer treated the sample yesterday.",
        match_type="phrase_match",
    )
    assert 0 < score < 1


def test_exact_list_item_matching_is_not_raw_substring():
    score = _match(
        "rat",
        "The treatment targeted trait-associated pathways.",
        match_type="exact_list",
    )
    assert score == 0.0


def test_exact_list_returns_real_matched_text():
    result = _match_result(
        "TNF-α, MIP-1α, IL-1β",
        "Monitor IL-1β and TNF-α after treatment.",
        match_type="exact_list",
    )
    assert result.matched_text == "IL-1β, TNF-α"


def test_match_result_contains_offsets_for_exact_match():
    result = _match_result(
        "seven Biofield Energy Healers",
        "The protocol specifies 7 Biofield Energy Healers for the treatment arm.",
        match_type="phrase_match",
    )
    assert result.start_char >= 0
    assert result.end_char > result.start_char
    assert result.matched_text == "7 Biofield Energy Healers"


def test_partial_phrase_match_offset_is_valid_or_explicitly_missing():
    result = _match_result(
        "healers treat inflamed samples",
        "A healer treated the sample yesterday.",
        match_type="phrase_match",
    )
    assert result.start_char >= -1
    assert result.end_char >= -1
    if result.start_char == -1:
        assert result.end_char == -1
    else:
        assert result.end_char > result.start_char


def test_unrelated_response_scores_zero():
    score = _match(
        "seven Biofield Energy Healers",
        "Use a randomized mouse cohort with blinded scoring.",
        match_type="phrase_match",
    )
    assert score == 0.0