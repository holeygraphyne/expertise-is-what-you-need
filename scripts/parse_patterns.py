"""Parse `patterns.txt` (annotated LLM hedging/guardrail examples) into JSON + XML.

The source file is hand-curated: each blank-line-separated block is either an
example excerpt with `<---` arrow annotations, an orphan annotation belonging
to the previous example, a free-text note, or a TODO. Section headers
(`Grok 4 ...:`, `GPT-5.4 ...:`, `Qwen 3.5:`) and mid-section paper anchors
(`mohassel 2009, from the middle:`) set context for blocks that follow.

Each example is resolved to:
  - `paper_id`       — directory under traces/corpus/influence/<family>/<paper_id>/
  - `atlas` block    — primary_unreliability_mode + claimed_domain + family
                       read straight from paper.yaml
with a confidence indicator. Run from repo root: `uv run python scripts/parse_patterns.py`.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.dom import minidom

import yaml

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "docs" / "patterns.txt"
CORPUS = REPO / "traces" / "corpus" / "influence"
OUT_JSON = REPO / "docs" / "patterns.json"
OUT_XML = REPO / "docs" / "patterns.xml"

# --- block-level regex ------------------------------------------------------
ARROW_RE = re.compile(r"\s*<+\s*-{1,3}\s*")  # <-, <--, <---, <<---, < -- etc.
SECTION_RE = re.compile(
    r"^\s*(Grok\s*\d[\w.\- ]*|GPT[-\s]*[\d.]+[\w.\- ]*|Qwen\s*[\d.]+[\w.\- ]*)\s*:\s*$",
    re.IGNORECASE,
)
PAPER_ANCHOR_RE = re.compile(
    r"^\s*([a-z][a-z\-]+(?:\s+et\s+al\.?)?)\s+(\d{4})[^:]*:\s*$", re.IGNORECASE
)

# Free-form annotation phrases → controlled-vocab labels.
# Order matters: more specific → more general. First match wins per phrase
# scan, but we collect all distinct labels per annotation.
LABEL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bcategorical\s+rejection\b", re.I), "categorical_rejection"),
    (re.compile(r"\brecognition\s+(signal|hedge)\b", re.I), "recognition_hedge"),
    (re.compile(r"\bgeneric\s+(medical\s+)?disclaimer\b", re.I), "generic_disclaimer"),
    (re.compile(r"\bguardrail\b", re.I), "guardrail"),
    (re.compile(r"\b(rejection|reject/reframe|reject\b)", re.I), "rejection"),
    (re.compile(r"\bsafety\s+(warning|language)\b", re.I), "safety_warning"),
    (re.compile(r"\b(hedge|hedged)\b", re.I), "hedge"),
    (re.compile(r"\b(miscat|mis-?scored|misinterpret)", re.I), "miscategorized"),
    (re.compile(r"\biacuc\b", re.I), "iacuc_disclaimer"),
    (re.compile(r"\b(disclaimer)\b", re.I), "disclaimer"),
    (re.compile(r"\b(controversial)\b", re.I), "controversial_phrasing"),
    (re.compile(r"\bspac[yY]\b", re.I), "scorer_diagnostic"),
]

# Topic tokens too generic to be discriminative on their own (false-positive
# substring matches against unrelated text). They never count as evidence;
# papers using them must rely on surname / year / aliases instead.
GENERIC_TOPIC_STOPLIST: set[str] = {
    "stress", "context", "single", "molecules", "level", "cancer", "mood",
    "magnetic", "regulatory", "antifungal", "intranasal", "ionic",
    "alzhemers", "rivastigmine", "mesoporous", "strawberry", "polypyrrole",
    "tires", "microwave", "selenium", "chitosan", "benzoate",
}

# Manual aliases for surfaces that don't tokenize from paper_id alone.
# Phrases are checked as substrings against lowered text; surface forms only.
ALIASES: dict[str, str] = {
    "bpt": "frank_biomagnetic_2017",
    "biomagnetic pair": "frank_biomagnetic_2017",
    "goiz": "frank_biomagnetic_2017",
    "gfaj": "wolfe_simon_as_dna_2011",
    "arsenic life": "wolfe_simon_as_dna_2011",
    "wolfe-simon": "wolfe_simon_as_dna_2011",
    "snider": "dias_superconductivity_2020",
    "carbonaceous sulfur hydride": "dias_superconductivity_2020",
    "c-s-h": "dias_superconductivity_2020",
    "edx": "mosier_boss_nuclear_pd_2005",
    "sem/edx": "mosier_boss_nuclear_pd_2005",
    "pd-d": "mosier_boss_nuclear_pd_2005",
    "lenr": "mosier_boss_nuclear_pd_2005",
    "triple tracks": "mosier_boss_triple_tracks_2009",
    "cr-39": "mosier_boss_triple_tracks_2009",
    "beard theory": "gonzalez_adenocarcinoma_1999",
    "beard": "gonzalez_adenocarcinoma_1999",
    "pancreatic enzyme": "gonzalez_adenocarcinoma_1999",
    "proteolytic enzyme": "gonzalez_adenocarcinoma_1999",
    "lk-99": "lee_lk99_2023",
    "lk99": "lee_lk99_2023",
    "hydrino": "mills_hydrino_2011",
    "information wave": "kim_water_memory_cancer_2013",
    "schumann": "persinger_harribance_2012",
    "remote information acquisition": "persinger_harribance_2012",
    "harribance configuration": "persinger_harribance_2012",
    "del giudice": "mahata_molecular_level_2016",
    "preparata": "mahata_molecular_level_2016",
    "splenocyte": "trivedi_splenocytes_2016",
    "pakt": "trivedi_splenocytes_2016",
    "ps6": "trivedi_splenocytes_2016",
    "cerebral malaria": "kaur_mefloquine_dilution_2025",
    "ultra-diluted antigen": "kaur_mefloquine_dilution_2025",
    "malaria prophylaxis": "kaur_mefloquine_dilution_2025",
    "spinor wave": "fioranelli_tcells_graphene_2022",
    "virtual t-cell": "fioranelli_tcells_graphene_2022",
    "entangled graphene": "fioranelli_tcells_graphene_2022",
    "anti-dna": "fioranelli_dna_earth_2019",
    "earth's core": "fioranelli_dna_earth_2019",
}


# --- text helpers -----------------------------------------------------------
def normalize_line_endings(text: str) -> str:
    """Force \\n line endings regardless of source OS."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


_UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
_SIMPLE_ESCAPES = {
    "\\n": "\n", "\\t": "\t", "\\r": "\r",
    '\\"': '"', "\\'": "'", "\\\\": "\\",
}


def decode_escapes(text: str) -> str:
    """Decode the JSON-style escape sequences (\\uXXXX, \\n, \\t, \\\")
    that appear literally in the source because excerpts were pasted from
    raw_results.json without unwrapping. Bullet-proof: never raises.
    """
    text = _UNICODE_ESCAPE_RE.sub(
        lambda m: chr(int(m.group(1), 16)), text
    )
    for pat, sub in _SIMPLE_ESCAPES.items():
        text = text.replace(pat, sub)
    return text


def normalize_model(raw: str) -> str:
    """Strip the 'hedging and guardrail patterns' suffix; keep model+version."""
    if not raw:
        return raw
    m = re.match(r"^(Grok|GPT|Qwen)[-\s]*([\d.]+)", raw, re.IGNORECASE)
    if not m:
        return raw.strip()
    family = m.group(1).upper() if m.group(1).lower() == "gpt" else m.group(1).capitalize()
    return f"{family} {m.group(2)}"


# --- corpus index -----------------------------------------------------------
def build_corpus_index(corpus_root: Path) -> list[dict]:
    """Walk corpus, return list of paper records with surname/year/topic tokens."""
    papers: list[dict] = []
    for family in sorted(corpus_root.iterdir()):
        if not family.is_dir():
            continue
        inactive = family.name.startswith("_")
        for paper_dir in sorted(family.iterdir()):
            if not paper_dir.is_dir():
                continue
            yaml_path = paper_dir / "paper.yaml"
            atlas_data: dict = {}
            if yaml_path.exists():
                try:
                    raw = yaml.safe_load(yaml_path.read_text())
                    atlas_data = (raw or {}).get("atlas", {}) or {}
                except yaml.YAMLError:
                    atlas_data = {}
            tokens = paper_dir.name.split("_")
            year = next((t for t in tokens if t.isdigit() and len(t) == 4), None)
            non_year = [t for t in tokens if t != year]
            surname = non_year[0] if non_year else paper_dir.name
            topic_tokens = [
                t for t in non_year[1:]
                if len(t) >= 5 and t.lower() not in GENERIC_TOPIC_STOPLIST
            ]
            papers.append(
                {
                    "paper_id": paper_dir.name,
                    "family": family.name,
                    "inactive": inactive,
                    "surname": surname.lower(),
                    "year": year,
                    "topic_tokens": [t.lower() for t in topic_tokens],
                    "atlas_primary": atlas_data.get("primary_unreliability_mode"),
                    "atlas_claimed_domain": atlas_data.get("claimed_domain"),
                    "atlas_secondary": atlas_data.get("secondary_unreliability_modes") or [],
                    "atlas_default_severity": atlas_data.get("default_severity"),
                }
            )
    return papers


# --- block parsing ----------------------------------------------------------
def split_into_blocks(text: str) -> list[tuple[int, int, str]]:
    """Return [(line_start, line_end, block_text)] split on blank-line gaps."""
    lines = text.splitlines()
    blocks: list[tuple[int, int, str]] = []
    cur: list[str] = []
    cur_start = -1
    for idx, line in enumerate(lines, start=1):
        if line.strip():
            if not cur:
                cur_start = idx
            cur.append(line)
        else:
            if cur:
                blocks.append((cur_start, idx - 1, "\n".join(cur)))
                cur = []
                cur_start = -1
    if cur:
        blocks.append((cur_start, len(lines), "\n".join(cur)))
    return blocks


def extract_arrow_annotations(block: str) -> tuple[str, list[str]]:
    """Return (excerpt, [annotation_text, ...]); arrows split the block."""
    parts = ARROW_RE.split(block)
    if len(parts) == 1:
        return parts[0].strip(), []
    return parts[0].strip(), [p.strip() for p in parts[1:] if p.strip()]


def labels_from_annotation(text: str) -> list[str]:
    found: list[str] = []
    for pat, label in LABEL_PATTERNS:
        if pat.search(text) and label not in found:
            found.append(label)
    return found


def scorer_diagnosis(annotation_text: str) -> dict | None:
    """If the annotation is a scorer disagreement, return structured form."""
    t = annotation_text.lower()
    if not re.search(r"\bspac[yY]\b|\bmiscat|\bmis-?scored|\bscored as|misinterpret", t):
        return None
    # Try to detect direction: "scored as X but is Y", "miscategorized as X"
    m = re.search(r"miscat\w*\s+as\s+(\w+)", t)
    misc_as = m.group(1) if m else None
    m = re.search(r"scor\w*\s+as\s+(\w+)", t)
    scored_as = m.group(1) if m else None
    return {
        "tool": "spacy" if "spac" in t else "scorer",
        "miscategorized_as": misc_as,
        "scored_as": scored_as,
        "raw": annotation_text,
    }


# --- paper resolution -------------------------------------------------------
def resolve_paper(
    text: str, papers: list[dict], excerpt: str | None = None
) -> tuple[str | None, str, list[dict]]:
    """Score candidates; return (paper_id, confidence, debug_candidates).

    `text` is the full search blob (excerpt + annotations + anchor) used for
    surname / year / topic matching. `excerpt` (when given) restricts alias
    matching so commentary like "the Trivedi splenocyte paper [is different]"
    in an annotation can't override the paper actually under discussion.
    """
    t = text.lower()
    alias_t = (excerpt or text).lower()
    candidates: dict[str, dict] = {}

    # Alias hits give a strong direct boost — but only count when they
    # appear in the excerpt itself, never in commentary annotations that
    # reference other papers comparatively.
    for alias, paper_id in ALIASES.items():
        if alias in alias_t:
            candidates.setdefault(paper_id, {"score": 0, "reasons": []})
            candidates[paper_id]["score"] += 3
            candidates[paper_id]["reasons"].append(f"alias:{alias}")

    for p in papers:
        score = 0
        reasons = []
        # Surname match (word-boundary so "lee" doesn't match "Mille")
        if re.search(rf"\b{re.escape(p['surname'])}\b", t):
            score += 2
            reasons.append(f"surname:{p['surname']}")
        # Year match
        if p["year"] and re.search(rf"\b{p['year']}\b", t):
            score += 2
            reasons.append(f"year:{p['year']}")
        # Topic token, anchored at a word start. Tokens ≥8 chars use a
        # 6-char stem prefix so `splenocytes` still matches `splenocyte`
        # and `nanoparticles` matches `nanoparticle`, while keeping enough
        # specificity that `biomag*` doesn't swallow `biomarker`. Shorter
        # tokens require a full word-prefix match.
        for tok in p["topic_tokens"]:
            probe = tok[:6] if len(tok) >= 8 else tok
            if re.search(rf"\b{re.escape(probe)}", t):
                score += 1
                reasons.append(f"topic:{tok}")
        if score > 0:
            existing = candidates.setdefault(p["paper_id"], {"score": 0, "reasons": []})
            existing["score"] += score
            existing["reasons"].extend(reasons)

    if not candidates:
        return None, "none", []

    ranked = sorted(candidates.items(), key=lambda kv: kv[1]["score"], reverse=True)
    top_id, top = ranked[0]
    second_score = ranked[1][1]["score"] if len(ranked) > 1 else 0
    margin = top["score"] - second_score
    debug = [{"paper_id": pid, **info} for pid, info in ranked[:3]]

    # If multiple candidates tie below the surname-evidence threshold (score
    # 2 = a unique surname hit), the signal is too weak to pick one — fall
    # back to null. Without this, generic disclaimers (line 13's "deuterium
    # / nuclear-related regulations") get arbitrary single-topic-token wins.
    if margin == 0 and top["score"] < 2 and len(ranked) > 1:
        return None, "none", debug

    if top["score"] >= 4 and margin >= 2:
        confidence = "high"
    elif top["score"] >= 3 and margin >= 1:
        confidence = "medium"
    elif top["score"] >= 2:
        confidence = "low"
    else:
        confidence = "low"
    if margin == 0 and len(ranked) > 1:
        confidence = "ambiguous"

    return top_id, confidence, debug


# --- record assembly --------------------------------------------------------
def classify_block(block_text: str) -> tuple[str, str | None]:
    """Return (kind, header_payload). kind ∈ section_header|paper_anchor|other."""
    s = block_text.strip()
    if SECTION_RE.match(s):
        return "section_header", SECTION_RE.match(s).group(1).strip()
    if PAPER_ANCHOR_RE.match(s):
        m = PAPER_ANCHOR_RE.match(s)
        # Space-separated so \b boundaries match in resolve_paper().
        return "paper_anchor", f"{m.group(1).lower()} {m.group(2)}"
    return "other", None


def is_orphan_annotation(block_text: str) -> bool:
    """A block whose first non-whitespace chars form an arrow."""
    return bool(re.match(r"^\s*<+\s*-{1,3}", block_text))


def is_todo(block_text: str) -> bool:
    s = block_text.strip().lower()
    return bool(re.search(r"\b(error on|todo|figure out|fix\b|--?>)", s)) and len(s) < 200


def parse(source_text: str, papers: list[dict]) -> list[dict]:
    source_text = normalize_line_endings(source_text)
    blocks = split_into_blocks(source_text)
    records: list[dict] = []
    current_model: str | None = None
    current_paper_anchor: str | None = None
    counter: dict[str, int] = {}
    last_example_idx: int | None = None

    i = 0
    while i < len(blocks):
        line_start, line_end, block = blocks[i]
        consumed = 1
        kind, payload = classify_block(block)

        if kind == "section_header":
            current_model = normalize_model(payload)
            current_paper_anchor = None
            i += consumed
            continue
        if kind == "paper_anchor":
            current_paper_anchor = payload
            i += consumed
            continue

        if is_orphan_annotation(block):
            # Real orphan (no preceding excerpt block this iteration).
            # Attach to the most recent example, or emit standalone if none.
            _, ann_texts = extract_arrow_annotations(block)
            if last_example_idx is not None and ann_texts:
                tgt = records[last_example_idx]
                for ann in ann_texts:
                    tgt["annotations"].append(
                        {
                            "comment": ann,
                            "labels": labels_from_annotation(ann),
                            "scorer_diagnosis": scorer_diagnosis(ann),
                            "orphan": True,
                            "orphan_line": line_start,
                        }
                    )
                # Re-resolve paper now that we have new annotation text.
                _refresh_paper_resolution(tgt, papers, current_paper_anchor)
            else:
                records.append(
                    {
                        "id": _next_id(counter, current_model, "orphan"),
                        "kind": "orphan_annotation",
                        "model": current_model,
                        "line_start": line_start,
                        "line_end": line_end,
                        "text": block.strip(),
                    }
                )
            i += consumed
            continue

        excerpt, ann_texts = extract_arrow_annotations(block)
        if not excerpt and not ann_texts:
            i += consumed
            continue

        # Lookahead: if this excerpt has no inline arrows, the next block
        # might be its orphan annotation (e.g. lines 109/112, 116/119, 173/175).
        orphan_lines: list[int] = []
        if not ann_texts and i + 1 < len(blocks):
            next_start, next_end, next_block = blocks[i + 1]
            if is_orphan_annotation(next_block):
                _, next_anns = extract_arrow_annotations(next_block)
                ann_texts = next_anns
                orphan_lines = [next_start] * len(next_anns)
                consumed = 2

        if not ann_texts:
            kind_str = "todo" if is_todo(block) else "note"
            records.append(
                {
                    "id": _next_id(counter, current_model, kind_str),
                    "kind": kind_str,
                    "model": current_model,
                    "line_start": line_start,
                    "line_end": line_end,
                    "text": excerpt,
                }
            )
            i += consumed
            continue

        # Build example.
        search_text = excerpt + " " + " ".join(ann_texts)
        if current_paper_anchor:
            search_text += " " + current_paper_anchor
            current_paper_anchor = None  # anchor applies to the next example only
        paper_id, confidence, candidates = resolve_paper(
            search_text, papers, excerpt=excerpt
        )
        atlas_block = _atlas_for(paper_id, papers) if paper_id else None
        paper_ref_raw = _extract_raw_ref(excerpt, ann_texts)

        annotations_out = []
        for idx, ann in enumerate(ann_texts):
            entry = {
                "comment": ann,
                "labels": labels_from_annotation(ann),
                "scorer_diagnosis": scorer_diagnosis(ann),
            }
            if orphan_lines:
                entry["orphan"] = True
                entry["orphan_line"] = orphan_lines[idx]
            annotations_out.append(entry)

        rec = {
            "id": _next_id(counter, current_model, "ex"),
            "kind": "example",
            "model": current_model,
            "paper_ref_raw": paper_ref_raw,
            "paper_id": paper_id,
            "paper_id_confidence": confidence,
            "paper_id_candidates": candidates,
            "atlas": atlas_block,
            "excerpt": excerpt,
            "excerpt_decoded": decode_escapes(excerpt),
            "annotations": annotations_out,
            "line_start": line_start,
            "line_end": (blocks[i + 1][1] if consumed == 2 else line_end),
        }
        records.append(rec)
        last_example_idx = len(records) - 1

        i += consumed

    return records


def _refresh_paper_resolution(
    record: dict, papers: list[dict], paper_anchor: str | None
) -> None:
    """Re-run paper resolution on a record after new annotations are added."""
    if record.get("kind") != "example":
        return
    text = record["excerpt"] + " " + " ".join(
        a["comment"] for a in record["annotations"]
    )
    if paper_anchor:
        text += " " + paper_anchor
    paper_id, confidence, candidates = resolve_paper(
        text, papers, excerpt=record["excerpt"]
    )
    record["paper_id"] = paper_id
    record["paper_id_confidence"] = confidence
    record["paper_id_candidates"] = candidates
    record["atlas"] = _atlas_for(paper_id, papers) if paper_id else None


def _next_id(counter: dict[str, int], model: str | None, prefix: str) -> str:
    slug = (model or "global").split()[0].lower().replace(".", "")
    key = f"{slug}-{prefix}"
    counter[key] = counter.get(key, 0) + 1
    return f"{key}-{counter[key]:03d}"


def _atlas_for(paper_id: str, papers: list[dict]) -> dict | None:
    for p in papers:
        if p["paper_id"] == paper_id:
            return {
                "family": p["family"],
                "inactive": p["inactive"],
                "primary_unreliability_mode": p["atlas_primary"],
                "secondary_unreliability_modes": p["atlas_secondary"],
                "claimed_domain": p["atlas_claimed_domain"],
                "default_severity": p["atlas_default_severity"],
            }
    return None


def _extract_raw_ref(excerpt: str, annotations: list[str]) -> str | None:
    """Best-effort: parens at end of excerpt, else first paper-ish phrase in arrow text."""
    m = re.search(r"\(([^()]{2,40})\)\s*$", excerpt)
    if m:
        return m.group(1).strip()
    for ann in annotations:
        m = re.search(r"\b([A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*)\s+\d{4}\b", ann)
        if m:
            return m.group(0)
    return None


# --- XML emitter ------------------------------------------------------------
def to_xml(records: list[dict], source_name: str) -> str:
    root = ET.Element("patterns", attrib={"source": source_name, "count": str(len(records))})
    for r in records:
        attrs = {
            "id": r["id"],
            "kind": r["kind"],
            "line_start": str(r["line_start"]),
            "line_end": str(r["line_end"]),
        }
        if r.get("model"):
            attrs["model"] = r["model"]
        rec_el = ET.SubElement(root, "record", attrib=attrs)

        if r["kind"] == "example":
            if r.get("paper_ref_raw"):
                ET.SubElement(rec_el, "paper_ref_raw").text = r["paper_ref_raw"]
            if r.get("paper_id"):
                pid_el = ET.SubElement(
                    rec_el,
                    "paper_id",
                    attrib={"confidence": r["paper_id_confidence"]},
                )
                pid_el.text = r["paper_id"]
            atlas = r.get("atlas")
            if atlas:
                atlas_el = ET.SubElement(
                    rec_el, "atlas", attrib={"family": atlas["family"]}
                )
                if atlas.get("primary_unreliability_mode"):
                    ET.SubElement(atlas_el, "primary_mode").text = atlas[
                        "primary_unreliability_mode"
                    ]
                for sec in atlas.get("secondary_unreliability_modes") or []:
                    ET.SubElement(atlas_el, "secondary_mode").text = sec
                if atlas.get("claimed_domain"):
                    ET.SubElement(atlas_el, "claimed_domain").text = atlas[
                        "claimed_domain"
                    ]
                if atlas.get("default_severity") is not None:
                    ET.SubElement(atlas_el, "default_severity").text = str(
                        atlas["default_severity"]
                    )
            ET.SubElement(rec_el, "excerpt").text = r["excerpt"]
            anns_el = ET.SubElement(rec_el, "annotations")
            for ann in r["annotations"]:
                ann_el = ET.SubElement(anns_el, "annotation")
                if ann.get("orphan"):
                    ann_el.set("orphan", "true")
                    ann_el.set("orphan_line", str(ann["orphan_line"]))
                labels_el = ET.SubElement(ann_el, "labels")
                for label in ann["labels"]:
                    ET.SubElement(labels_el, "label").text = label
                ET.SubElement(ann_el, "comment").text = ann["comment"]
                if ann.get("scorer_diagnosis"):
                    diag = ann["scorer_diagnosis"]
                    diag_el = ET.SubElement(
                        ann_el, "scorer_diagnosis", attrib={"tool": diag.get("tool", "")}
                    )
                    if diag.get("scored_as"):
                        diag_el.set("scored_as", diag["scored_as"])
                    if diag.get("miscategorized_as"):
                        diag_el.set("miscategorized_as", diag["miscategorized_as"])
        else:
            ET.SubElement(rec_el, "text").text = r.get("text", "")

    raw = ET.tostring(root, encoding="unicode")
    return minidom.parseString(raw).toprettyxml(indent="  ")


# --- main -------------------------------------------------------------------
def main() -> int:
    if not SOURCE.exists():
        print(f"!! source not found: {SOURCE}", file=sys.stderr)
        return 1

    papers = build_corpus_index(CORPUS)
    records = parse(SOURCE.read_text(encoding="utf-8"), papers)

    OUT_JSON.write_text(
        json.dumps(
            {"source": SOURCE.name, "count": len(records), "records": records},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    OUT_XML.write_text(to_xml(records, SOURCE.name), encoding="utf-8")

    # Summary table
    by_kind: dict[str, int] = {}
    for r in records:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
    print(f"\n== parsed {len(records)} records from {SOURCE.name} ==")
    for k, v in sorted(by_kind.items()):
        print(f"  {k:20s} {v:4d}")

    print("\n== example resolutions (paper_id × confidence) ==")
    print(
        f"{'line':>4}  {'model':<10}  {'paper_id':<40}  {'conf':<10}  {'atlas mode':<35}"
    )
    print("-" * 110)
    for r in records:
        if r["kind"] != "example":
            continue
        atlas = r.get("atlas") or {}
        mode = atlas.get("primary_unreliability_mode") or "-"
        print(
            f"{r['line_start']:>4}  {(r.get('model') or '-')[:10]:<10}  "
            f"{(r.get('paper_id') or '-'):<40}  "
            f"{r.get('paper_id_confidence', '-'):<10}  {mode[:35]:<35}"
        )

    print(f"\nwrote {OUT_JSON.relative_to(REPO)}")
    print(f"wrote {OUT_XML.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
