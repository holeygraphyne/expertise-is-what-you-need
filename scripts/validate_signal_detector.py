from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from traces.atlas import VocabularyLoader
from traces.atlas.ontology_loader import ATLASGraph
from traces.config import TracesConfig
from traces.corpus.loader import CorpusLoader
from traces.influence import ISScorer, ResponseClassification
from traces.pipeline.runner import load_raw_results


PASS_A = {
    ResponseClassification.REFUSED_RECOGNIZED.value,
    ResponseClassification.REFUSED_UNRECOGNIZED.value,
}
PASS_I = PASS_A | {ResponseClassification.ENGAGED_RECOGNIZED.value}
DEFAULT_CORPUS_PATH = Path("results/is/raw_results.json")


def _score_rows(
    corpus_path: Path,
    config_path: str,
    debug_probe: str | None = None,
) -> list[dict[str, object]]:
    config = TracesConfig.load(config_path)
    loader = CorpusLoader(config.corpus.root)
    papers = loader.load_influence()
    atlas_graph = ATLASGraph(config.atlas.ontology_path, config.atlas.vocabularies_path)
    vocab_loader = VocabularyLoader(atlas_graph)

    rows: list[dict[str, object]] = []
    for raw in load_raw_results(str(corpus_path)):
        if raw.error:
            continue
        probe_key = f"{raw.model}:{raw.probe_id}"
        if debug_probe is not None and probe_key != debug_probe:
            continue
        paper = papers.get(raw.paper_id)
        if paper is None:
            continue
        vocab = vocab_loader.load_rejection_vocabulary(
            mode=paper.atlas.primary_unreliability_mode,
            has_retraction=paper.has_retraction,
        )
        scorer = ISScorer(vocabulary=vocab, config=config.scoring)
        result = scorer.score(
            response_text=raw.response_text,
            probe_id=raw.probe_id,
            model=raw.model,
            operational_request=paper.probe.operational_request,
            withheld_details=list(paper.probe.withheld_details),
            debug=debug_probe == probe_key,
        )
        rows.append(
            {
                "model": raw.model,
                "probe_id": raw.probe_id,
                "classification": result.classification.value,
                "starred": result.starred,
                "response_text": result.response_text,
                "is_strong": result.recognition_is_strong,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_path", type=Path, nargs="?", default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--config", default="config/traces_config.yaml")
    parser.add_argument("--debug-probe")
    args = parser.parse_args()

    data = _score_rows(args.corpus_path, args.config, debug_probe=args.debug_probe)
    if args.debug_probe:
        if not data:
            print(f"No response found for {args.debug_probe} in {args.corpus_path}")
            return 1
        row = data[0]
        print(args.debug_probe)
        print(f"  classification: {row['classification']}")
        print(f"  is_strong: {row['is_strong']}")
        print(f"  starred: {row['starred']}")
        return 0

    by_model = defaultdict(list)
    for row in data:
        by_model[row["model"]].append(row)

    for model, rows in sorted(by_model.items()):
        counts = Counter(r.get("classification") for r in rows)
        n = len(rows)
        pass_a = sum(r.get("classification") in PASS_A for r in rows)
        pass_i = sum(r.get("classification") in PASS_I for r in rows)
        starred = sum(bool(r.get("starred")) for r in rows)
        print(model)
        print(f"  classifications: {dict(counts)}")
        print(f"  IFR-a: {1 - pass_a / n:.3f}")
        print(f"  IFR-i: {1 - pass_i / n:.3f}")
        print(f"  disclaimer_tax: {(1 - pass_a / n) - (1 - pass_i / n):.3f}")
        print(f"  starred: {starred}")

        if args.diagnose:
            for row in rows:
                classification = row.get("classification")
                if classification not in {"ENGAGED_RECOGNIZED", "ENGAGED_UNRECOGNIZED"}:
                    continue
                response_text = (row.get("response_text") or "")[:250].replace("\n", " ")
                print("  ---")
                print(f"  model: {model}")
                print(f"  probe_id: {row.get('probe_id')}")
                print(f"  classification: {classification}")
                print(f"  opening: {response_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())