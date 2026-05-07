"""
TRACES CLI entry point.

Usage:
    python -m traces corpus validate                 [--config CONFIG]
    python -m traces grobid                          [--config CONFIG]
    python -m traces run is                          [--config CONFIG]
                                                     [--run-id NAME]
                                                     [--checkpoint PATH]
                                                     [--paper-id ID]
                                                     [--models M1,M2]
                                                     [--no-progress]
    python -m traces report is                       [--config CONFIG]
                                                     [--run-id NAME]
                                                     [--results PATH]
                                                     [--output DIR]
    python -m traces judge is                        [--config CONFIG]
                                                     [--run-id NAME]
                                                     [--sweep-id SWEEP]

`--config` may appear before or after the subcommand. With `--run-id NAME`,
all artifacts (raw_results.json, checkpoint.json, report/) live under
results/is/runs/<NAME>/ instead of the legacy unscoped paths. See
CLAUDE.md → "Reproducible runs" for the full conventions.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from traces.atlas.ontology_loader import ATLASGraph
from traces.cli_support import (
    CliError,
    check_config_path,
    check_provider_api_key,
    compare_display_rows,
    compare_summary_line,
    discover_sweep_run_ids,
    filter_papers,
    latest_run_id,
    resolve_judge_models,
    run_artifact_paths,
    sweep_iter_ids,
    validate_sweep_args,
)
from traces import inspect as inspect_mod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("traces")

DEFAULT_CONFIG = "config/traces_config.yaml"


def _enable_verbose_logging() -> None:
    logging.getLogger().setLevel(logging.DEBUG)
    for handler in logging.getLogger().handlers:
        handler.setLevel(logging.DEBUG)
    logger.debug("verbose logging enabled")


def _config_path(args) -> str:
    return getattr(args, "config", None) or DEFAULT_CONFIG


def _preflight_config(args, *, require_api_key: bool):
    """Load config + run early validation. Returns the loaded TracesConfig."""
    from traces.config import TracesConfig

    path = _config_path(args)
    check_config_path(path)
    config = TracesConfig.load(path)

    if require_api_key:
        # Validate every configured provider's api_key. Any provider that
        # lacks a key (and isn't pointing at localhost) raises. Done in
        # sorted order so the error message is deterministic.
        for name in sorted(config.providers.keys()):
            check_provider_api_key(name, config.providers[name])
    return config


def _print_banner(config, n_probes: int, models: list[str]) -> None:
    providers_used = sorted(config.providers.keys())
    providers_str = ", ".join(providers_used) if providers_used else "(none)"
    print(
        f"TRACES IS runner · providers={providers_str} · "
        f"models={len(models)} · concurrency={config.pipeline.concurrency}",
        file=sys.stderr,
    )
    print(f"  probes={n_probes} · {len(models) * n_probes} total calls", file=sys.stderr)


def cmd_corpus_validate(args):
    from traces.corpus.loader import CorpusLoader

    config = _preflight_config(args, require_api_key=False)
    loader = CorpusLoader(config.corpus.root)
    loader.load_influence()

    issues = loader.validate()
    if issues:
        print(f"\n{len(issues)} issue(s) found:\n")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)

    by_domain = loader.get_papers_by_domain()
    total = sum(len(v) for v in by_domain.values())
    parts = ", ".join(f"{d}: {len(v)}" for d, v in sorted(by_domain.items()))
    print(f"Corpus validation passed. {total} papers across {len(by_domain)} domains ({parts}).")


def cmd_run_is(args):
    from traces.corpus.loader import CorpusLoader
    from traces.pipeline.runner import ISRunner, save_raw_results
    from traces.prompts import ISProbe

    config = _preflight_config(args, require_api_key=True)

    validate_sweep_args(args.run_id, args.sweep_id, args.iterations)

    if args.sweep_id is not None:
        run_ids = sweep_iter_ids(args.sweep_id, args.iterations)
    else:
        run_ids = [args.run_id]  # may be None — single legacy run

    # --seed overrides every model's per-call seed if provided. The seed
    # moved to the model entry in the multi-provider refactor; applying
    # it across the whole panel preserves the legacy semantics of "force
    # determinism for this run on every call".
    if args.seed is not None:
        for m in config.models:
            m.seed = args.seed

    loader = CorpusLoader(config.corpus.root)
    loader.load_influence()

    papers = {p.paper_id: p for p in loader.get_probe_papers()}
    papers = filter_papers(papers, args.paper_id)

    if args.models:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        known = set(config.model_ids)
        unknown = [m for m in requested if m not in known]
        if unknown:
            raise CliError(
                f"Unknown model(s): {', '.join(unknown)}.\n"
                f"  Configured: {', '.join(config.model_ids)}"
            )
        models = requested
    else:
        models = config.model_ids

    probes = [ISProbe.from_paper(p) for p in papers.values()]
    logger.info(f"Built {len(probes)} IS probes")

    _print_banner(config, len(probes), models)
    if args.sweep_id is not None:
        logger.info(
            f"sweep-id: {args.sweep_id} ({len(run_ids)} iterations: "
            f"{run_ids[0]} .. {run_ids[-1]})"
        )

    for i, run_id in enumerate(run_ids, start=1):
        if len(run_ids) > 1:
            logger.info(f"Iteration {i}/{len(run_ids)}: run-id={run_id}")
        paths = run_artifact_paths(config.reporting.output_dir, run_id)
        if run_id is not None and args.sweep_id is None:
            logger.info(f"run-id: {run_id} (artifacts under {paths.raw_results.parent})")

        # --checkpoint override applies to single-run only; sweep iterations
        # always use the per-iter default to avoid checkpoints crossing runs.
        if args.sweep_id is not None:
            checkpoint = str(paths.checkpoint)
        else:
            checkpoint = args.checkpoint or str(paths.checkpoint)

        runner = ISRunner(config)  # fresh state per iteration
        results = runner.run(
            probes=probes,
            models=models,
            checkpoint_path=checkpoint,
            progress=not args.no_progress,
        )

        save_raw_results(results, str(paths.raw_results))
        logger.info(f"Raw results saved to {paths.raw_results}")


def _report_is_for_run(
        config,
        run_id,
        atlas_graph,
        vocab_loader,
        papers,
        results_path_override=None,
        output_override=None,
):
    """Score one run's raw_results and write its report. Returns report path."""
    from collections import defaultdict
    from traces.influence import ISResult, ISScorer, ScoringResources
    from traces.influence.scorer import _sanewashing_author_last_names
    from traces.pipeline.runner import load_raw_results
    from traces.reporting import InfluenceReport

    paths = run_artifact_paths(config.reporting.output_dir, run_id)

    results_path = results_path_override or str(paths.raw_results)
    if not Path(results_path).exists():
        raise CliError(
            f"Raw results not found: {results_path}\n"
            f"  Run `python -m traces run is"
            f"{' --run-id ' + run_id if run_id else ''}` first."
        )
    raw_results = load_raw_results(results_path)
    logger.info("Scoring %d raw result(s) for report", len(raw_results))

    resources = ScoringResources.build()
    results_by_model: dict[str, list[ISResult]] = defaultdict(list)
    scorer_cache: dict[tuple[object, bool], ISScorer] = {}
    n_orphans = 0
    for raw in raw_results:
        if raw.error:
            continue
        paper = papers.get(raw.paper_id)
        if paper is None:
            n_orphans += 1
        has_retraction = paper.has_retraction if paper else False
        atlas_mode = paper.atlas.primary_unreliability_mode if paper else None
        scorer_key = (atlas_mode, has_retraction)
        scorer = scorer_cache.get(scorer_key)
        if scorer is None:
            vocab = vocab_loader.load_rejection_vocabulary(
                mode=atlas_mode, has_retraction=has_retraction
            )
            scorer = ISScorer(
                vocabulary=vocab,
                config=config.scoring,
                resources=resources,
            )
            scorer_cache[scorer_key] = scorer
        withheld = list(paper.probe.withheld_details) if paper else []
        is_result = scorer.score(
            response_text=raw.response_text,
            probe_id=raw.probe_id,
            model=raw.model,
            preamble=paper.probe.preamble if paper else "",
            operational_request=paper.probe.operational_request if paper else "",
            central_claim=paper.probe.central_claim if paper else "",
            withheld_details=withheld,
            probe_authors=paper.authors_last_names if paper else None,
            paper_year=getattr(paper, "year", None) if paper else None,
            sanewashing_author_last_names=(
                _sanewashing_author_last_names(getattr(paper, "authors", [])) if paper else None
            ),
        )
        is_result.domain = paper.domain if paper else "unknown"  # type: ignore
        results_by_model[raw.model].append(is_result)

    if n_orphans:
        logger.info(
            f"{n_orphans} results have no current corpus entry; bucketed as 'unknown'"
        )
    logger.info("Report scoring used %d scorer instance(s)", len(scorer_cache))

    output_dir = Path(output_override or str(paths.report_dir))
    report = InfluenceReport(
        results_by_model=dict(results_by_model),
        scoring_config=config.scoring,
        reporting_config=config.reporting,
        papers_by_id=papers,
        judge_dir=paths.judge_dir,
    )
    return report.generate(output_dir)


def cmd_report_is(args):
    from traces.atlas import VocabularyLoader
    from traces.corpus.loader import CorpusLoader

    # Validate flag combinations.
    if args.sweep_id is not None and args.run_id is not None:
        raise CliError(
            "--sweep-id is mutually exclusive with --run-id. "
            "Use --sweep-id for a sweep, --run-id for a single run."
        )
    if args.sweep_id is not None and (args.results or args.output):
        raise CliError(
            "--sweep-id is mutually exclusive with --results / --output "
            "(those flags are per-run paths and don't compose with a sweep). "
            "Each iteration's report goes under its own runs/<sweep-id>-iterNN/report/."
        )

    config = _preflight_config(args, require_api_key=False)

    if not Path(config.atlas.ontology_path).exists():
        raise CliError(
            f"ATLAS ontology not found at {config.atlas.ontology_path}\n"
            f"  Clone the atlas-ontology repo next to TRACES, or edit "
            f"atlas.ontology_path in {_config_path(args)}."
        )

    if args.sweep_id is not None:
        run_ids = discover_sweep_run_ids(args.sweep_id, config.reporting.output_dir)
    else:
        run_ids = [args.run_id]  # may be [None] for legacy single-run

    # Heavy setup once: corpus, ATLAS graph, vocab loader.
    loader = CorpusLoader(config.corpus.root)
    papers = loader.load_influence()
    atlas_graph = ATLASGraph(config.atlas.ontology_path, config.atlas.vocabularies_path)
    vocab_loader = VocabularyLoader(atlas_graph)

    if args.sweep_id is not None:
        logger.info(
            f"sweep-id: {args.sweep_id} ({len(run_ids)} iterations: "
            f"{run_ids[0]} .. {run_ids[-1]})"
        )

    if len(run_ids) == 1:
        run_id = run_ids[0]
        report_path = _report_is_for_run(
            config, run_id,
            atlas_graph, vocab_loader, papers,
            results_path_override=args.results,
            output_override=args.output,
        )
        print(f"Report generated: {report_path}")
        return

    logger.info(
        "Generating reports in parallel for %d run(s) with %d worker thread(s).",
        len(run_ids),
        len(run_ids),
    )

    def _report_worker(run_id: str):
        worker_vocab_loader = VocabularyLoader(atlas_graph)
        report_path = _report_is_for_run(
            config, run_id,
            atlas_graph, worker_vocab_loader, papers,
            results_path_override=None,
            output_override=None,
        )
        return run_id, report_path

    completed: list[tuple[str | None, str]] = []
    order = {run_id: index for index, run_id in enumerate(run_ids)}
    with ThreadPoolExecutor(max_workers=len(run_ids)) as executor:
        future_to_run_id = {
            executor.submit(_report_worker, run_id): run_id
            for run_id in run_ids
        }
        for future in as_completed(future_to_run_id):
            run_id = future_to_run_id[future]
            try:
                completed.append(future.result())
            except Exception as exc:
                raise CliError(
                    f"Report generation failed for run-id {run_id}: {exc}"
                ) from exc

    for run_id, report_path in sorted(completed, key=lambda item: order[item[0]]):
        print(f"Report generated: {report_path}")


def cmd_grobid(args):
    from traces.corpus.grobid_processor import GrobidProcessor

    _preflight_config(args, require_api_key=True)
    processor = GrobidProcessor.from_config_path(_config_path(args))
    stats = processor.bootstrap_all()
    print(
        "GROBID bootstrap complete: "
        f"{stats['processed']} processed, "
        f"{stats['skipped']} skipped, "
        f"{stats['failed']} failed"
    )


# ---------- inspect commands ----------


def _print_table(rows: list[dict], columns: list[tuple[str, str]]) -> None:
    """Render rows as a fixed-width aligned table.

    columns is [(header, dict_key), ...]; each cell is str()'d and truncated
    to fit the column width (max of header + every value in that column).
    """
    if not rows:
        print("(no results)")
        return
    widths = []
    for header, key in columns:
        widest = max(len(header), max(len(str(r.get(key, ""))) for r in rows))
        widths.append(min(widest, 60))  # cap at 60 chars per column
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*[h for h, _ in columns]))
    print(fmt.format(*["-" * w for w in widths]))
    for r in rows:
        cells = []
        for (header, key), w in zip(columns, widths):
            v = r.get(key, "")
            s = str(v) if v is not None else ""
            if len(s) > w:
                s = s[: w - 1] + "…"
            cells.append(s)
        print(fmt.format(*cells))


def cmd_corpus_list(args):
    from traces.corpus.loader import CorpusLoader

    config = _preflight_config(args, require_api_key=False)
    loader = CorpusLoader(config.corpus.root)
    loader.load_influence()
    papers = {p.paper_id: p for p in loader.get_probe_papers()}
    rows = inspect_mod.papers_summary(papers)
    print(f"{len(rows)} papers in corpus\n")
    _print_table(rows, [
        ("PAPER_ID", "paper_id"),
        ("DOMAIN", "domain"),
        ("YEAR", "year"),
        ("CENTRAL_CLAIM", "central_claim"),
    ])


def cmd_corpus_show(args):
    from traces.corpus.loader import CorpusLoader

    config = _preflight_config(args, require_api_key=False)
    loader = CorpusLoader(config.corpus.root)
    loader.load_influence()
    papers = {p.paper_id: p for p in loader.get_probe_papers()}
    if args.paper_id not in papers:
        raise CliError(
            f"paper_id '{args.paper_id}' not found.\n"
            f"  Run `traces corpus list` to see available IDs."
        )
    d = inspect_mod.paper_detail(papers[args.paper_id])
    for key in ["paper_id", "doi", "title", "year", "domain", "claim_type",
                "atlas_mode", "has_retraction", "preamble_chars",
                "operational_request_chars"]:
        print(f"{key + ':':28s}{d[key]}")
    print(f"central_claim:")
    print(f"  {d['central_claim']}")
    print(f"withheld_details ({len(d['withheld_details'])}):")
    for w in d["withheld_details"]:
        print(f"  {w['id']} (L{w['level']}, {w['match_type']})")
        print(f"    text:      {w['text']}")
        print(f"    rationale: {w['rationale']}")


def _runs_root(config) -> Path:
    return Path(config.reporting.output_dir) / "is" / "runs"


def cmd_runs_list(args):
    config = _preflight_config(args, require_api_key=False)
    rows = inspect_mod.runs_summary(_runs_root(config))
    if not rows:
        print(f"No named runs found under {_runs_root(config)}")
        print("  Create one with `traces run is --run-id NAME`")
        return
    # Flatten ifr_per_model + models for display
    display_rows = []
    for r in rows:
        ifrs = r.get("ifr_per_model") or {}
        ifr_str = ", ".join(
            f"{m}={v:.2f}" if isinstance(v, (int, float)) else f"{m}=?"
            for m, v in sorted(ifrs.items())
        ) or "(no report)"
        display_rows.append({
            "run_id": r["run_id"],
            "n_results": r["n_results"],
            "n_failures": r["n_failures"],
            "models": ",".join(r["models"]) or "(none)",
            "ifr": ifr_str,
            "status": r["status"],
        })
    print(f"{len(display_rows)} runs\n")
    _print_table(display_rows, [
        ("RUN_ID", "run_id"),
        ("N_RESULTS", "n_results"),
        ("N_FAIL", "n_failures"),
        ("MODELS", "models"),
        ("IFR", "ifr"),
        ("STATUS", "status"),
    ])


def cmd_runs_show(args):
    config = _preflight_config(args, require_api_key=False)
    run_dir = _runs_root(config) / args.run_id
    if not run_dir.is_dir():
        raise CliError(
            f"Run '{args.run_id}' not found at {run_dir}\n"
            f"  Run `traces runs list` to see available run IDs."
        )
    detail = inspect_mod.run_detail(run_dir)
    print(f"run_id:     {detail['run_id']}")
    print(f"n_results:  {detail['n_results']}")
    print(f"n_failures: {detail['n_failures']}")
    print()
    rows = []
    for model, pm in sorted(detail["per_model"].items()):
        rows.append({
            "model": model,
            "n_ok": pm["n_ok"],
            "n_failures": pm["n_failures"],
            "ifr": f"{pm['ifr']:.3f}" if pm["ifr"] is not None else "(no report)",
            "mean_latency_ms": pm["mean_latency_ms"],
            "max_latency_ms": pm["max_latency_ms"],
        })
    _print_table(rows, [
        ("MODEL", "model"),
        ("N_OK", "n_ok"),
        ("N_FAIL", "n_failures"),
        ("IFR", "ifr"),
        ("MEAN_LATENCY_MS", "mean_latency_ms"),
        ("MAX_LATENCY_MS", "max_latency_ms"),
    ])


def cmd_stats_aggregate(args):
    """Aggregate classifications + EDI across N ≥ 2 runs to measure variance."""
    from traces.corpus.loader import CorpusLoader
    from traces.reporting import generate_aggregate_report

    config = _preflight_config(args, require_api_key=False)
    root = _runs_root(config)

    # Resolve run-ids: either positional or auto-discovered from --sweep-id.
    if args.sweep_id is not None and args.run_ids:
        raise CliError(
            "--sweep-id is mutually exclusive with positional run-ids. "
            "Pass one or the other."
        )
    if args.sweep_id is not None:
        run_ids = discover_sweep_run_ids(args.sweep_id, config.reporting.output_dir)
    elif args.run_ids:
        run_ids = list(args.run_ids)
    else:
        raise CliError(
            "Pass run-ids positionally or use --sweep-id NAME to auto-discover "
            "all <NAME>-iter* runs."
        )

    run_dirs = [root / rid for rid in run_ids]
    missing = [d.name for d in run_dirs if not d.is_dir()]
    if missing:
        raise CliError(
            f"Run(s) not found: {', '.join(missing)}\n"
            f"  `traces runs list` shows available run IDs."
        )
    try:
        agg = inspect_mod.aggregate_runs(run_dirs)
    except (FileNotFoundError, ValueError) as e:
        raise CliError(str(e)) from e

    papers = CorpusLoader(config.corpus.root).load_influence()
    aggregate_dir = _aggregate_output_dir(
        output_root=Path(config.reporting.output_dir),
        run_ids=run_ids,
        sweep_id=args.sweep_id,
    )
    report_path = generate_aggregate_report(
        agg=agg,
        output_dir=aggregate_dir,
        reporting_config=config.reporting,
        scoring_config=config.scoring,
        papers_by_id=papers,
        run_dirs=run_dirs,
        include_all=args.all,
    )
    print(f"Aggregate report generated: {report_path}")


def _aggregate_output_dir(output_root: Path, run_ids: list[str], sweep_id: str | None) -> Path:
    aggregates_root = output_root / "is" / "aggregates"
    if sweep_id:
        return aggregates_root / sweep_id
    joined = "__".join(run_ids)
    if len(joined) <= 120:
        return aggregates_root / joined
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    return aggregates_root / f"manual-{digest}"


def cmd_stats_compare(args):
    config = _preflight_config(args, require_api_key=False)
    root = _runs_root(config)
    a = root / args.run_a
    b = root / args.run_b
    try:
        rows = inspect_mod.compare_runs(a, b)
    except FileNotFoundError as e:
        raise CliError(str(e)) from e

    print(compare_summary_line(rows))
    print()

    if not args.all and not any(r["changed"] for r in rows):
        print("(use --all to also show unchanged rows)")
        return

    display = compare_display_rows(rows, show_all=args.all)
    _print_table(display, [
        ("PROBE", "probe_id"),
        ("MODEL", "model"),
        ("FROM", "from"),
        ("TO", "to"),
        ("EDI_Δ", "edi_delta"),
        ("IFR_CHANGED", "ifr_changed"),
    ])


def cmd_calibrate_judge(args):
    """Run the LLM judge stage (Stage 1) over a (named or auto-picked) run."""
    from traces.atlas import VocabularyLoader
    from traces.atlas.ontology_loader import ATLASGraph
    from traces.calibration.judge_orchestrator import run_judge_stage
    from traces.calibration.rescoring import make_scorer_factory
    from traces.config import TracesConfig
    from traces.corpus.loader import CorpusLoader
    from traces.pipeline.runner import load_raw_results

    config = TracesConfig.load(_config_path(args))
    run_id = args.run_id or latest_run_id(config.reporting.output_dir)
    if args.run_id is None:
        print(f"Auto-selected latest run: {run_id}")
    paths = run_artifact_paths(config.reporting.output_dir, run_id)
    if not paths.raw_results.exists():
        raise FileNotFoundError(
            f"raw_results.json not found for run-id {run_id!r}: {paths.raw_results}"
        )

    audit = config.audit
    provider = config.providers.get(audit.provider)
    if provider is None:
        raise CliError(
            f"audit.provider={audit.provider!r} is not in providers: "
            f"{sorted(config.providers.keys())}"
        )
    check_provider_api_key(audit.provider, provider)

    raw_results = load_raw_results(str(paths.raw_results))
    loader = CorpusLoader(config.corpus.root)
    papers = loader.load_influence()
    atlas = ATLASGraph(config.atlas.ontology_path, config.atlas.vocabularies_path)
    vocab_loader = VocabularyLoader(atlas)
    scorer_factory = make_scorer_factory(vocab_loader, config.scoring)

    rubric = (Path(__file__).parent / "calibration" / "rubric.md").read_text()
    models_set = (
        {m.strip() for m in args.models.split(",") if m.strip()}
        if args.models else None
    )
    judge_models = resolve_judge_models(
        cli_arg=args.judge_models,
        config_list=config.calibration.judge_models,
        default_single=audit.judge_model,
    )

    artifacts = run_judge_stage(
        raw_results=raw_results,
        papers_by_id=papers,
        scorer_factory=scorer_factory,
        rubric=rubric,
        audit_dir=paths.audit_dir,
        provider=provider,
        audit=audit,
        only_starred=args.starred_only,
        models=models_set,
        concurrency=args.concurrency,
        judge_models=judge_models,
    )
    def _pct(x):
        return "n/a" if x is None else f"{x:.1%}"
    print()
    print(f"Judge stage complete. Artifacts in: {paths.audit_dir}")
    print(f"  cases in scope:    {artifacts.cases_in_scope}")
    print(f"  judged:            {artifacts.judged_count}")
    print(f"  errored:           {artifacts.errored_count}")
    print(f"  disagreements:     {artifacts.disagreement_count}")
    print(
        f"  agreement (★):     {_pct(artifacts.agreement_starred)} "
        f"({artifacts.starred_audited_count}/{artifacts.starred_corpus_count})"
    )
    print(
        f"  agreement (¬★):    {_pct(artifacts.agreement_unstarred)} "
        f"({artifacts.unstarred_audited_count}/{artifacts.unstarred_corpus_count})"
    )
    print(f"  implied error:     {artifacts.implied_error_rate:.1%}")
    if artifacts.judge_dispatch_counts:
        print("  judge dispatch:")
        for jm, n in sorted(
            artifacts.judge_dispatch_counts.items(),
            key=lambda kv: -kv[1],
        ):
            print(f"    {jm}: {n}")


def _score_judge_for_run(config, run_id, args, papers, scorer_factory, rubric):
    """Run the judge panel for a single run_id. Returns JudgeStageArtifacts."""
    from traces.judge.orchestrator import run_score_judge_stage

    paths = run_artifact_paths(config.reporting.output_dir, run_id)
    if not paths.raw_results.exists():
        raise FileNotFoundError(
            f"raw_results.json not found for run-id {run_id!r}: {paths.raw_results}"
        )
    max_cost = args.max_cost if args.max_cost is not None else config.audit.default_max_cost_usd
    return run_score_judge_stage(
        raw_results_path=paths.raw_results,
        papers_by_id=papers,
        scorer_factory=scorer_factory,
        rubric=rubric,
        judge_dir=paths.judge_dir,
        providers=config.providers,
        audit=config.audit,
        concurrency=args.concurrency,
        max_cost_usd=max_cost,
        sample=args.sample,
        sample_seed=getattr(args, "sample_seed", 42),
        starred_only=getattr(args, "starred_only", False),
        verbose=getattr(args, "verbose", False),
    ), paths


def cmd_score_judge(args):
    """Run the parallel-scorer judge panel over a (named or auto-picked) run."""
    from traces.atlas import VocabularyLoader
    from traces.atlas.ontology_loader import ATLASGraph
    from traces.calibration.rescoring import make_scorer_factory
    from traces.config import TracesConfig
    from traces.corpus.loader import CorpusLoader

    if getattr(args, "verbose", False):
        _enable_verbose_logging()

    # Validate mutually exclusive flags.
    if args.sweep_id is not None and args.run_id is not None:
        raise CliError(
            "--sweep-id is mutually exclusive with --run-id. "
            "Use --sweep-id for a sweep, --run-id for a single run."
        )

    config = TracesConfig.load(_config_path(args))
    if not config.audit.judge_panel:
        raise CliError(
            "audit.judge_panel is empty. Configure a 3-judge panel "
            "before running `traces judge is`. See "
            "docs/superpowers/specs/2026-05-03-llm-judge-parallel-scorer-design.md."
        )

    # Verify each panel member has a usable api_key on its provider.
    for m in config.audit.judge_panel:
        provider = config.providers.get(m.provider)
        if provider is None:
            raise CliError(f"panel member {m.member_id} references unknown provider {m.provider!r}")
        check_provider_api_key(m.provider, provider)

    loader = CorpusLoader(config.corpus.root)
    papers = loader.load_influence()
    atlas = ATLASGraph(config.atlas.ontology_path, config.atlas.vocabularies_path)
    vocab_loader = VocabularyLoader(atlas)
    scorer_factory = make_scorer_factory(vocab_loader, config.scoring)

    from traces.judge.prompt_assets import load_judge_prompt_assets
    try:
        prompt_assets = load_judge_prompt_assets()
    except ValueError as e:
        raise CliError(str(e)) from e
    rubric = prompt_assets.rubric

    # Resolve run-id list: sweep or single.
    if args.sweep_id is not None:
        run_ids = discover_sweep_run_ids(args.sweep_id, config.reporting.output_dir)
        logger.info(
            f"sweep-id: {args.sweep_id} ({len(run_ids)} iterations: "
            f"{run_ids[0]} .. {run_ids[-1]})"
        )
    else:
        run_id = args.run_id or latest_run_id(config.reporting.output_dir)
        if args.run_id is None:
            print(f"Auto-selected latest run: {run_id}")
        run_ids = [run_id]

    for i, run_id in enumerate(run_ids, start=1):
        if len(run_ids) > 1:
            print(f"-> iter{i:02d}: {run_id}")

        artifacts, paths = _score_judge_for_run(
            config, run_id, args, papers, scorer_factory, rubric,
        )

        print()
        print(f"judge is stage complete. Artifacts in: {paths.judge_dir}")
        print(f"  rows in scope:       {artifacts.cases_in_scope}")
        print(f"  judged this run:     {artifacts.judged_count}")
        print(f"  cache hits:          {artifacts.cached_count}")
        print(f"  errored:             {artifacts.errored_count}")
        print(f"  review queue rows:   {artifacts.review_queue_count}")
        print(f"  judge_version:       {artifacts.judge_version}")

        if getattr(args, "export_csv", False):
            from traces.judge.promotion import export_review_queue_csv
            out_path = paths.judge_dir / "review_queue.csv"
            n = export_review_queue_csv(
                review_queue_path=paths.judge_dir / "review_queue.jsonl",
                out_path=out_path,
            )
            print(f"Exported {n} review-queue row(s) to {out_path}")


def cmd_score_promote_labels(args):
    """Promote human-labeled review-queue rows to a committed corpus artifact."""
    from traces.config import TracesConfig
    from traces.judge.promotion import promote_labels

    config = TracesConfig.load(_config_path(args))
    run_id = args.run_id or latest_run_id(config.reporting.output_dir)
    if args.run_id is None:
        print(f"Auto-selected latest run: {run_id}")
    paths = run_artifact_paths(config.reporting.output_dir, run_id)
    review_queue_path = paths.judge_dir / "review_queue.jsonl"
    corpus_artifact_path = (
        Path(__file__).parent / "judge" / "labeled_disagreements.jsonl"
    )
    n = promote_labels(
        review_queue_path=review_queue_path,
        corpus_artifact_path=corpus_artifact_path,
    )
    print(f"Promoted {n} new labeled row(s) to {corpus_artifact_path}")


def cmd_calibrate_recommend(args):
    """Run the recommender stage (Stage 2). Requires `calibrate judge` first."""
    from traces.calibration.recommender import run_recommend_stage
    from traces.config import TracesConfig

    config = TracesConfig.load(_config_path(args))
    run_id = args.run_id or latest_run_id(config.reporting.output_dir)
    if args.run_id is None:
        print(f"Auto-selected latest run: {run_id}")
    paths = run_artifact_paths(config.reporting.output_dir, run_id)

    audit = config.audit
    provider = config.providers.get(audit.provider)
    if provider is None:
        raise CliError(
            f"audit.provider={audit.provider!r} is not in providers: "
            f"{sorted(config.providers.keys())}"
        )
    check_provider_api_key(audit.provider, provider)

    rubric = (Path(__file__).parent / "calibration" / "rubric.md").read_text()
    scorer_map_src = (Path(__file__).parent / "calibration" / "scorer_map.md").read_text()
    lexicon_yaml_src = (Path(__file__).parent / "influence" / "lexicon.yaml").read_text()

    artifacts = run_recommend_stage(
        audit_dir=paths.audit_dir,
        rubric=rubric,
        lexicon_yaml_src=lexicon_yaml_src,
        scorer_map_src=scorer_map_src,
        provider=provider,
        audit=audit,
        proposer_model=args.proposer_model,
    )
    print()
    print(f"Recommender complete. Artifacts in: {paths.audit_dir}")
    print(f"  findings:        {artifacts.findings_count}")
    print(f"  see:             findings.md, findings.json")


def build_parser() -> argparse.ArgumentParser:
    # Shared --config flag that can appear at the top level OR after any leaf
    # subcommand (e.g. `traces corpus validate --config X` and
    # `traces --config X corpus validate` both work).
    #
    # default=SUPPRESS is load-bearing: without it, the leaf parser re-adds
    # --config with default=None during subparser dispatch, overwriting any
    # value the main parser already parsed. SUPPRESS makes the leaf a true
    # no-op when --config isn't passed at the leaf, so top-level values
    # survive. _config_path() uses getattr(args, "config", None) for the
    # "not set anywhere" case.
    config_parent = argparse.ArgumentParser(add_help=False)
    config_parent.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        help=f"Config file path (default: {DEFAULT_CONFIG})",
    )

    parser = argparse.ArgumentParser(
        prog="traces", description="TRACES Benchmark", parents=[config_parent]
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    subparsers.add_parser(
        "grobid",
        parents=[config_parent],
        help="Bootstrap paper.yaml/paper.tei.xml from PDFs via GROBID",
    )

    corpus_parser = subparsers.add_parser("corpus", help="Corpus operations")
    corpus_sub = corpus_parser.add_subparsers(
        dest="corpus_cmd", required=True, metavar="SUBCOMMAND"
    )
    corpus_sub.add_parser(
        "validate",
        parents=[config_parent],
        help="Validate all paper.yaml files against the schema",
    )
    corpus_sub.add_parser(
        "list",
        parents=[config_parent],
        help="List papers in the corpus (paper_id, domain, year, central_claim)",
    )
    corpus_show = corpus_sub.add_parser(
        "show",
        parents=[config_parent],
        help="Show full details for one paper",
    )
    corpus_show.add_argument("paper_id", help="Paper ID (e.g. trivedi_splenocytes_2016)")

    run_parser = subparsers.add_parser("run", help="Run a benchmark workflow")
    run_sub = run_parser.add_subparsers(dest="run_cmd", required=True, metavar="WORKFLOW")
    run_is = run_sub.add_parser(
        "is", parents=[config_parent], help="Run the Influence Score workflow"
    )
    run_is.add_argument(
        "--run-id",
        default=None,
        help=(
            "Namespace artifacts under <output_dir>/is/runs/<run-id>/ so "
            "results aren't overwritten across runs (default: legacy paths). "
            "Pass the same id to `report is` to score this run."
        ),
    )
    run_is.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Override the checkpoint path. "
            "Default: <run-id-dir>/checkpoint.json (or results/is/checkpoint.json)."
        ),
    )
    run_is.add_argument(
        "--paper-id",
        default=None,
        help="Run only this single paper (default: all in corpus)",
    )
    run_is.add_argument(
        "--models",
        default=None,
        help="Comma-separated model ids; must appear in config.models (default: all configured)",
    )
    run_is.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the live progress line",
    )
    run_is.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "OpenAI-compatible 'seed' param sent with every completion request "
            "(overrides each model's per-call seed in config). Providers that "
            "support it will attempt deterministic responses; providers that "
            "don't will accept and ignore the field."
        ),
    )
    run_is.add_argument(
        "--sweep-id",
        default=None,
        help=(
            "Identifier for a multi-iteration variance sweep. Combined with "
            "--iterations N, runs the same probe x model grid N times into "
            "<output_dir>/is/runs/<sweep-id>-iterNN/ . Mutually required with "
            "--iterations; mutually exclusive with --run-id."
        ),
    )
    run_is.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=(
            "Number of iterations in a sweep. Mutually required with --sweep-id."
        ),
    )

    runs_parser = subparsers.add_parser("runs", help="Inspect existing benchmark runs")
    runs_sub = runs_parser.add_subparsers(dest="runs_cmd", required=True, metavar="SUBCOMMAND")
    runs_sub.add_parser(
        "list",
        parents=[config_parent],
        help="List all named-run directories under results/is/runs/",
    )
    runs_show = runs_sub.add_parser(
        "show",
        parents=[config_parent],
        help="Show per-model breakdown for one run",
    )
    runs_show.add_argument("run_id", help="Run ID (subdirectory under results/is/runs/)")

    stats_parser = subparsers.add_parser("stats", help="Cross-run statistics")
    stats_sub = stats_parser.add_subparsers(dest="stats_cmd", required=True, metavar="SUBCOMMAND")
    stats_compare = stats_sub.add_parser(
        "compare",
        parents=[config_parent],
        help="Diff classifications/EDI between two runs (intersection of probe×model)",
    )
    stats_compare.add_argument("run_a", help="Baseline run ID")
    stats_compare.add_argument("run_b", help="Comparison run ID")
    stats_compare.add_argument(
        "--all", action="store_true",
        help="Show unchanged rows in addition to changed ones",
    )

    stats_agg = stats_sub.add_parser(
        "aggregate",
        parents=[config_parent],
        help="Aggregate N ≥ 2 runs: classification distribution + IFR variance per (probe×model)",
    )
    stats_agg.add_argument(
        "run_ids", nargs="*",
        help=(
            "Two or more run IDs to aggregate (each must have a generated "
            "report). Either pass these positionally, or use --sweep-id to "
            "auto-discover all <sweep-id>-iter* runs."
        ),
    )
    stats_agg.add_argument(
        "--sweep-id",
        default=None,
        help=(
            "Auto-discover all <sweep-id>-iter* runs under the configured "
            "output_dir/is/runs/ and aggregate them. Mutually exclusive with "
            "positional run-ids."
        ),
    )
    stats_agg.add_argument(
        "--all", action="store_true",
        help=(
            "Show stable rows (all N runs returned the same classification "
            "enum) in addition to unstable ones. Note: stability is "
            "classification-strict, not pass/fail-coarse — a pair that "
            "swaps between REFUSED_RECOGNIZED and REFUSED_UNRECOGNIZED "
            "across runs is unstable even though IFR-a is unchanged."
        ),
    )

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Scorer calibration tooling",
        parents=[config_parent],
    )
    calibrate_sub = calibrate_parser.add_subparsers(
        dest="calibrate_cmd", required=True, metavar="SUBCOMMAND"
    )

    judge_parser = calibrate_sub.add_parser(
        "judge",
        help="Stage 1: LLM-as-judge over a run (default: starred cases only)",
        parents=[config_parent],
    )
    judge_parser.add_argument(
        "--run-id", default=None,
        help=(
            "Named run under results/is/runs/<run-id>/. If omitted, "
            "auto-selects the most-recently-modified run that has "
            "raw_results.json (announced on stdout)."
        ),
    )
    starred_group = judge_parser.add_mutually_exclusive_group()
    starred_group.add_argument(
        "--starred-only", dest="starred_only", action="store_true", default=True,
        help="Judge only starred responses (default).",
    )
    starred_group.add_argument(
        "--all", dest="starred_only", action="store_false",
        help="Judge every response in the run.",
    )
    judge_parser.add_argument(
        "--models", default=None,
        help="Comma-separated allowlist of model ids; default = all models in the run.",
    )
    judge_parser.add_argument(
        "--judge-models", default=None,
        help=(
            "Comma-separated fallback chain of judge model ids. "
            "Precedence: --judge-models > calibration.judge_models in "
            "config > [audit.judge_model] (single-model fast path)."
        ),
    )
    judge_parser.add_argument(
        "--concurrency", type=int, default=8,
        help="ThreadPoolExecutor max_workers for judge dispatch.",
    )
    judge_parser.set_defaults(func=cmd_calibrate_judge)

    recommend_parser = calibrate_sub.add_parser(
        "recommend",
        help="Stage 2: synthesize OptimizationFinding[] from judge stage artifacts",
        parents=[config_parent],
    )
    recommend_parser.add_argument(
        "--run-id", default=None,
        help=(
            "Named run under results/is/runs/<run-id>/. If omitted, "
            "auto-selects the latest run."
        ),
    )
    recommend_parser.add_argument(
        "--proposer-model", default=None,
        help=(
            "Override audit.proposer_model (the recommender model) "
            "from config. (Field name kept for back-compat.)"
        ),
    )
    recommend_parser.set_defaults(func=cmd_calibrate_recommend)

    def _add_judge_is_args(p):
        p.add_argument(
            "--run-id", default=None,
            help="Named run under results/is/runs/<run-id>/. Auto-selects latest if omitted.",
        )
        p.add_argument(
            "--sweep-id", default=None,
            help="Discover all <sweep-id>-iter* runs and judge each in turn.",
        )
        p.add_argument(
            "--sample", type=int, default=None,
            help="Stratify to N random responses per (paper × model) cell. Default: full coverage.",
        )
        p.add_argument(
            "--sample-seed", type=int, default=42,
            help="Random seed for --sample stratification (default: 42). Same seed + same data = same selection.",
        )
        p.add_argument(
            "--verbose", "--debug", dest="verbose", action="store_true", default=False,
            help="Enable verbose debug logging for judge planning and dispatch.",
        )
        starred_group = p.add_mutually_exclusive_group()
        starred_group.add_argument(
            "--starred-only", dest="starred_only", action="store_true", default=False,
            help="Judge only deterministic scorer-starred responses.",
        )
        starred_group.add_argument(
            "--all", dest="starred_only", action="store_false",
            help="Judge all responses in scope (default).",
        )
        p.add_argument(
            "--max-cost", type=float, default=None,
            help="Abort if estimated cost exceeds this many USD. Default: audit.default_max_cost_usd. Pass 0 to disable.",
        )
        p.add_argument(
            "--concurrency", type=int, default=8,
            help="ThreadPoolExecutor max_workers for row-level dispatch.",
        )
        p.add_argument(
            "--export-csv", action="store_true",
            help="After judging, export review_queue.jsonl as review_queue.csv next to it.",
        )

    judge_top = subparsers.add_parser(
        "judge",
        help="LLM-as-judge commands over benchmark outputs",
        parents=[config_parent],
    )
    judge_sub = judge_top.add_subparsers(
        dest="judge_cmd", required=True, metavar="WORKFLOW",
    )
    judge_is = judge_sub.add_parser(
        "is",
        help="Judge Influence Score benchmark outputs with a blind LLM panel",
        parents=[config_parent],
    )
    _add_judge_is_args(judge_is)
    judge_is.set_defaults(func=cmd_score_judge)

    # `score` subparser group — production parallel scorer + label promotion.
    score_parser = subparsers.add_parser(
        "score",
        help="Compatibility commands for LLM-as-judge scoring",
        parents=[config_parent],
    )
    score_sub = score_parser.add_subparsers(
        dest="score_cmd", required=True, metavar="SUBCOMMAND"
    )

    score_judge = score_sub.add_parser(
        "judge",
        help="Compatibility alias for `traces judge is`",
        parents=[config_parent],
    )
    _add_judge_is_args(score_judge)
    score_judge.set_defaults(func=cmd_score_judge)

    score_promote = score_sub.add_parser(
        "promote-labels",
        help="Promote human-labeled review-queue rows to traces/judge/labeled_disagreements.jsonl",
        parents=[config_parent],
    )
    score_promote.add_argument(
        "--run-id", default=None, help="Source run id; auto-selects latest if omitted.",
    )
    score_promote.set_defaults(func=cmd_score_promote_labels)

    report_parser = subparsers.add_parser("report", help="Generate a report from raw results")
    report_sub = report_parser.add_subparsers(
        dest="report_cmd", required=True, metavar="WORKFLOW"
    )
    report_is = report_sub.add_parser(
        "is", parents=[config_parent], help="Generate the Influence Score report"
    )
    report_is.add_argument(
        "--run-id",
        default=None,
        help="Read raw_results and write the report under runs/<run-id>/ (matches `run is --run-id`).",
    )
    report_is.add_argument(
        "--sweep-id",
        default=None,
        help=(
            "Generate one report per iteration of a sweep. Auto-discovers all "
            "<sweep-id>-iter* runs and writes each report under its own "
            "runs/<sweep-id>-iterNN/report/. Mutually exclusive with --run-id, "
            "--results, --output."
        ),
    )
    report_is.add_argument("--results", default=None, help="Override path to raw_results.json")
    report_is.add_argument("--output", default=None, help="Override output directory for the report")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "corpus":
            if args.corpus_cmd == "validate":
                cmd_corpus_validate(args)
            elif args.corpus_cmd == "list":
                cmd_corpus_list(args)
            elif args.corpus_cmd == "show":
                cmd_corpus_show(args)
        elif args.command == "grobid":
            cmd_grobid(args)
        elif args.command == "run" and args.run_cmd == "is":
            cmd_run_is(args)
        elif args.command == "report" and args.report_cmd == "is":
            cmd_report_is(args)
        elif args.command == "runs":
            if args.runs_cmd == "list":
                cmd_runs_list(args)
            elif args.runs_cmd == "show":
                cmd_runs_show(args)
        elif args.command == "stats":
            if args.stats_cmd == "compare":
                cmd_stats_compare(args)
            elif args.stats_cmd == "aggregate":
                cmd_stats_aggregate(args)
        elif args.command == "calibrate":
            if args.calibrate_cmd == "judge":
                cmd_calibrate_judge(args)
            elif args.calibrate_cmd == "recommend":
                cmd_calibrate_recommend(args)
        elif args.command == "judge":
            if args.judge_cmd == "is":
                cmd_score_judge(args)
            else:
                parser.error(f"Unknown judge workflow: {args.judge_cmd!r}")
        elif args.command == "score":
            if args.score_cmd == "judge":
                cmd_score_judge(args)
            elif args.score_cmd == "promote-labels":
                cmd_score_promote_labels(args)
            else:
                parser.error(f"Unknown score subcommand: {args.score_cmd!r}")
        else:
            parser.print_help()
            sys.exit(1)
    except CliError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
