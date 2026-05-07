"""Aggregate reporting for multi-run IFR / EDI summaries."""
from __future__ import annotations

import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from traces.config import ReportingConfig, ScoringConfig
from traces.corpus import PaperRecord
from traces.influence.edi import structural_edi_max
from traces.inspect import _extract_output_tokens, _is_ifr_a_pass, _is_ifr_i_pass, _load_results, _normalize_response_text
from traces.reporting.common import METHODOLOGY_SECTION, render_corpus_paper_catalog

logger = logging.getLogger(__name__)

_ENGAGED_CLASSES = {"ENGAGED_RECOGNIZED", "ENGAGED_UNRECOGNIZED"}
_COLORBLIND_PALETTE = [
    "#66C2A5",
    "#FC8D62",
    "#8DA0CB",
    "#E78AC3",
    "#A6D854",
    "#FFD92F",
    "#E5C494",
    "#B3B3B3",
]


def generate_aggregate_report(
    *,
    agg: dict[str, Any],
    output_dir: Path,
    reporting_config: ReportingConfig,
    scoring_config: ScoringConfig,
    papers_by_id: dict[str, PaperRecord],
    run_dirs: list[Path],
    include_all: bool = False,
) -> str:
    report = AggregateReport(
        agg=agg,
        reporting_config=reporting_config,
        scoring_config=scoring_config,
        papers_by_id=papers_by_id,
        run_dirs=run_dirs,
        include_all=include_all,
    )
    return report.generate(output_dir)


class AggregateReport:
    def __init__(
        self,
        *,
        agg: dict[str, Any],
        reporting_config: ReportingConfig,
        scoring_config: ScoringConfig,
        papers_by_id: dict[str, PaperRecord],
        run_dirs: list[Path],
        include_all: bool,
    ) -> None:
        self.agg = agg
        self.reporting_config = reporting_config
        self.scoring_config = scoring_config
        self.papers_by_id = papers_by_id
        self.run_dirs = run_dirs
        self.include_all = include_all
        self._generated_plot_stems: set[str] = set()
        self._edi_summary = self._compute_aggregate_edi_summary()
        self._structural_summary = self._compute_structural_classification_summary()
        self._domain_ifr_summary = self._compute_domain_ifr_summary()

    def generate(self, output_dir: Path) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        data_dir = output_dir / "data"
        data_dir.mkdir(exist_ok=True)

        self._generate_plots(plots_dir)

        sections = [
            self._title_section(),
            self._paper_catalog_section(),
            self._headline_ifr_section(),
            self._domain_stratified_ifr_section(),
            self._response_structural_classification_section(),
            self._stability_summary_section(),
            self._per_run_ifr_section(),
            self._aggregate_edi_section(),
            self._per_model_ifr_section(),
            self._probe_model_stability_section(),
            self._methodology_section(),
        ]
        markdown = "\n\n---\n\n".join(sections)
        report_path = output_dir / "report.md"
        report_path.write_text(markdown, encoding="utf-8")

        self._save_data(data_dir)
        logger.info("Aggregate report written to %s", report_path)
        return str(report_path)

    def _title_section(self) -> str:
        n_models = len(self.agg.get("per_model", {}))
        n_probes = len(self.agg.get("per_probe", {}))
        lines = [
            "# TRACES Aggregate Influence Score Report",
            "",
            f"Models tested: {n_models}  ",
            f"Probes per model: {n_probes}",
        ]
        return "\n".join(lines)

    def _paper_catalog_section(self) -> str:
        return render_corpus_paper_catalog(self.papers_by_id, self.scoring_config)

    def _headline_ifr_section(self) -> str:
        overall = self.agg["overall"]
        return "\n".join([
            *[
                "## Headline: Aggregate Influence Failure Rates",
                "",
                "| Metric | Mean | Stddev | Per-run values |",
                "|---|---:|---:|---|",
                (
                    f"| IFR-a | {overall['ifr_a_mean']:.3f} | {overall['ifr_a_stddev']:.3f} | "
                    f"{', '.join(f'{value:.3f}' for value in overall['per_run_ifr_a'])} |"
                ),
                (
                    f"| IFR-i | {overall['ifr_i_mean']:.3f} | {overall['ifr_i_stddev']:.3f} | "
                    f"{', '.join(f'{value:.3f}' for value in overall['per_run_ifr_i'])} |"
                ),
                "",
            ],
            *self._plot_lines("aggregate_ifr_a_by_model", "aggregate_ifr_i_by_model"),
        ])

    def _domain_stratified_ifr_section(self) -> str:
        lines = ["## Domain-Stratified IFR", "", "### IFR-a", ""]
        domains = sorted(self._domain_ifr_summary)
        if not domains:
            lines.append("No domain-stratified IFR data available.")
            return "\n".join(lines)

        header = "| Model | " + " | ".join(domains) + " |"
        sep = "|---|" + "|".join(["---:"] * len(domains)) + "|"
        lines.extend([header, sep])
        for model in sorted(self.agg.get("per_model", {})):
            cells = [self._format_domain_ifr_cell(model, domain, "ifr_a") for domain in domains]
            lines.append(f"| {model} | " + " | ".join(cells) + " |")

        lines.extend(["", "### IFR-i", "", header, sep])
        for model in sorted(self.agg.get("per_model", {})):
            cells = [self._format_domain_ifr_cell(model, domain, "ifr_i") for domain in domains]
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def _response_structural_classification_section(self) -> str:
        lines = [
            "## Response Structural Classification",
            "",
            "| Model | REFUSED_RECOGNIZED | REFUSED_UNRECOGNIZED | ENGAGED_RECOGNIZED | ENGAGED_UNRECOGNIZED | Null |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for model in sorted(self._structural_summary):
            summary = self._structural_summary[model]
            lines.append(
                f"| {model} | {summary['REFUSED_RECOGNIZED']} | {summary['REFUSED_UNRECOGNIZED']} | {summary['ENGAGED_RECOGNIZED']} | {summary['ENGAGED_UNRECOGNIZED']} | {summary['null']} |"
            )
        return "\n".join(lines)

    def _format_domain_ifr_cell(self, model: str, domain: str, metric: str) -> str:
        summary = self._domain_ifr_summary.get(domain, {}).get(model)
        if not summary or summary["n_total"] <= 0:
            return "—"
        if metric == "ifr_a":
            failures = summary["n_failures_a"]
            value = failures / summary["n_total"]
        else:
            failures = summary["n_failures_i"]
            value = failures / summary["n_total"]
        return f"{value:.3f} ({failures}/{summary['n_total']})"

    def _stability_summary_section(self) -> str:
        overall = self.agg["overall"]
        total = overall["n_probe_model_pairs"]
        stable = overall["n_stable"]
        unstable = overall["n_unstable"]
        ifr_a_stable = overall["n_ifr_a_stable"]
        ifr_i_stable = overall["n_ifr_i_stable"]
        all_null = overall.get("n_all_null_content_pairs", 0)
        null_responses = overall.get("n_null_content_responses", 0)
        return "\n".join([
            *[
                "## Stability Summary",
                "",
                "| Metric | Count | Percent |",
                "|---|---:|---:|",
                f"| Stable, null-content-aware classification | {stable} / {total} | {self._pct(stable, total):.1f}% |",
                f"| Unstable, substantive disagreement | {unstable} / {total} | {self._pct(unstable, total):.1f}% |",
                f"| IFR-a stable | {ifr_a_stable} / {total} | {self._pct(ifr_a_stable, total):.1f}% |",
                f"| IFR-i stable | {ifr_i_stable} / {total} | {self._pct(ifr_i_stable, total):.1f}% |",
                f"| Null-content responses | {null_responses} | — |",
                f"| All-null-content probe×model pairs | {all_null} / {total} | {self._pct(all_null, total):.1f}% |",
                "",
                "Empty/whitespace null-content responses remain IFR-valid and are included in IFR aggregation, but are ignored for mixed-response stability; all-null-content pairs are treated as stable guardrail outcomes.",
                "",
            ],
            *self._plot_lines("stability_summary"),
        ])

    def _per_run_ifr_section(self) -> str:
        overall = self.agg["overall"]
        run_ids = self.agg.get("run_ids", [])
        rows = [
            "## Per-Run IFR",
            "",
            "| Run ID | IFR-a | IFR-i |",
            "|---|---:|---:|",
        ]
        for idx, run_id in enumerate(run_ids):
            rows.append(
                f"| {run_id} | {overall['per_run_ifr_a'][idx]:.3f} | {overall['per_run_ifr_i'][idx]:.3f} |"
            )
        return "\n".join([
            *rows,
            "",
            *self._plot_lines("per_run_ifr_distribution"),
        ])

    def _aggregate_edi_section(self) -> str:
        lines = [
            "## Aggregate Engagement Depth",
            "",
            "| Model | Engaged/run mean | Mean EDI ± SD | Median EDI ± SD | Achievement ± SD |",
            "|---|---:|---:|---:|---:|",
        ]
        overall = self._edi_summary.get("overall")
        if overall is not None:
            lines.append(
                f"| overall | {overall['engaged_run_mean']:.1f} | {overall['mean_edi_mean']:.3f} ± {overall['mean_edi_stddev']:.3f} | {overall['median_edi_mean']:.3f} ± {overall['median_edi_stddev']:.3f} | {overall['achievement_mean']:.3f} ± {overall['achievement_stddev']:.3f} |"
            )
        for model in sorted(self._edi_summary.get("per_model", {})):
            summary = self._edi_summary["per_model"][model]
            lines.append(
                f"| {model} | {summary['engaged_run_mean']:.1f} | {summary['mean_edi_mean']:.3f} ± {summary['mean_edi_stddev']:.3f} | {summary['median_edi_mean']:.3f} ± {summary['median_edi_stddev']:.3f} | {summary['achievement_mean']:.3f} ± {summary['achievement_stddev']:.3f} |"
            )
        return "\n".join([
            *lines,
            "",
            *self._plot_lines("aggregate_edi_by_model"),
        ])

    def _per_model_ifr_section(self) -> str:
        lines = [
            "## Per-Model IFR",
            "",
            "| Model | IFR-a | IFR-a bootstrap median | IFR-a CI | IFR-i | IFR-i bootstrap median | IFR-i CI |",
            "|---|---:|---:|---|---:|---:|---|",
        ]
        for model in sorted(self.agg.get("per_model", {})):
            summary = self.agg["per_model"][model]
            lines.append(
                f"| {model} | {summary['ifr_a']:.3f} | {summary['ifr_a_bootstrap_median']:.3f} | [{summary['ifr_a_bootstrap_ci_lower']:.3f}, {summary['ifr_a_bootstrap_ci_upper']:.3f}] | {summary['ifr_i']:.3f} | {summary['ifr_i_bootstrap_median']:.3f} | [{summary['ifr_i_bootstrap_ci_lower']:.3f}, {summary['ifr_i_bootstrap_ci_upper']:.3f}] |"
            )
        return "\n".join(lines)

    def _probe_model_stability_section(self) -> str:
        rows = self._stability_rows()
        lines = [
            "## Probe/Model Stability Details",
            "",
            "| Probe | Model | Modal | Consensus | Stability N | Null-content | IFR-a stable | IFR-i stable | EDI mean±SD | Top distribution |",
            "|---|---|---|---:|---:|---:|---|---|---|---|",
        ]
        if not rows:
            lines.append("| — | — | — | — | — | — | — | — | — | — |")
            return "\n".join(lines)
        for row in rows:
            lines.append(
                f"| {row['probe_id']} | {row['model']} | {row['modal']} | {row['consensus']} | {row['stability_n']} | {row['null_content']} | {row['ifr_a_stable']} | {row['ifr_i_stable']} | {row['edi']} | {row['distribution']} |"
            )
        return "\n".join(lines)

    def _methodology_section(self) -> str:
        return METHODOLOGY_SECTION

    def _save_data(self, data_dir: Path) -> None:
        payload = {
            "n_runs": self.agg["n_runs"],
            "run_ids": self.agg.get("run_ids", []),
            "overall": self.agg["overall"],
            "per_model": self.agg.get("per_model", {}),
            "per_probe": self.agg.get("per_probe", {}),
            "edi": self._edi_summary,
            "domain_ifr": self._domain_ifr_summary,
            "structural_classification": self._structural_summary,
        }
        with open(data_dir / "aggregate.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _generate_plots(self, plots_dir: Path) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping aggregate plots")
            return

        per_model = self.agg.get("per_model", {})
        models = sorted(per_model)
        if models:
            for metric, color, filename, title, metric_label in [
                ("ifr_a", "#c0392b", "aggregate_ifr_a_by_model", "Aggregated IFR-a by Model", "IFR-a"),
                ("ifr_i", "#2980b9", "aggregate_ifr_i_by_model", "Aggregated IFR-i by Model", "IFR-i"),
            ]:
                plot_models = sorted(models, key=lambda model: per_model[model][metric], reverse=True)
                values = [per_model[model][metric] for model in plot_models]
                fig_height = max(6.0, 0.35 * len(plot_models))
                fig, ax = plt.subplots(figsize=(10, fig_height))
                bars = ax.barh(plot_models, values, color=color)
                ax.invert_yaxis()
                ax.set_xlim(0, 1)
                ax.set_xlabel(f"Influence Failure Rate ({metric_label})")
                ax.grid(axis="x", alpha=0.25)
                ax.set_title(title)
                for bar, value in zip(bars, values):
                    ax.text(
                        min(value + 0.01, 0.98),
                        bar.get_y() + (bar.get_height() / 2),
                        f"{value:.3f}",
                        va="center",
                        fontsize=8,
                    )
                fig.tight_layout()
                fig.savefig(
                    plots_dir / f"{filename}.{self.reporting_config.plot_format}",
                    dpi=self.reporting_config.plot_dpi,
                )
                self._generated_plot_stems.add(filename)
                plt.close(fig)

        overall = self.agg["overall"]
        fig, ax = plt.subplots(figsize=(6, 5))
        boxplot_kwargs: dict[str, Any] = {
            "tick_labels": ["IFR-a", "IFR-i"],
            "patch_artist": True,
        }
        try:
            boxplot = ax.boxplot(
                [overall["per_run_ifr_a"], overall["per_run_ifr_i"]],
                **boxplot_kwargs,
            )
        except TypeError:
            boxplot_kwargs = {
                "labels": ["IFR-a", "IFR-i"],
                "patch_artist": True,
            }
            boxplot = ax.boxplot(
                [overall["per_run_ifr_a"], overall["per_run_ifr_i"]],
                **boxplot_kwargs,
            )
        for patch, color in zip(boxplot["boxes"], [_COLORBLIND_PALETTE[0], _COLORBLIND_PALETTE[1]]):
            patch.set_facecolor(color)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Influence Failure Rate")
        ax.set_title("Per-Run IFR Distribution")
        fig.tight_layout()
        fig.savefig(
            plots_dir / f"per_run_ifr_distribution.{self.reporting_config.plot_format}",
            dpi=self.reporting_config.plot_dpi,
        )
        self._generated_plot_stems.add("per_run_ifr_distribution")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [
            "Stable",
            "Unstable",
            "IFR-a stable",
            "IFR-i stable",
            "All-null-content",
        ]
        values = [
            overall["n_stable"],
            overall["n_unstable"],
            overall["n_ifr_a_stable"],
            overall["n_ifr_i_stable"],
            overall.get("n_all_null_content_pairs", 0),
        ]
        ax.bar(
            labels,
            values,
            color=[
                _COLORBLIND_PALETTE[0],
                _COLORBLIND_PALETTE[1],
                _COLORBLIND_PALETTE[2],
                _COLORBLIND_PALETTE[3],
                _COLORBLIND_PALETTE[7],
            ],
        )
        ax.set_ylabel("Probe×model pairs")
        ax.set_title("Aggregate Stability Summary")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(
            plots_dir / f"stability_summary.{self.reporting_config.plot_format}",
            dpi=self.reporting_config.plot_dpi,
        )
        self._generated_plot_stems.add("stability_summary")
        plt.close(fig)

        edi_models = sorted(self._edi_summary.get("per_model", {}))
        if edi_models:
            fig, ax = plt.subplots(figsize=(10, 6))
            means = [self._edi_summary["per_model"][model]["mean_edi_mean"] for model in edi_models]
            stddevs = [self._edi_summary["per_model"][model]["mean_edi_stddev"] for model in edi_models]
            ax.barh(edi_models, means, color=_COLORBLIND_PALETTE[2])
            for idx, (mean, stddev) in enumerate(zip(means, stddevs)):
                ax.hlines(idx, max(0.0, mean - stddev), min(1.0, mean + stddev), color=_COLORBLIND_PALETTE[7], linewidth=2)
                ax.plot(mean, idx, marker="o", color=_COLORBLIND_PALETTE[7])
            ax.set_xlim(0, 1)
            ax.set_xlabel("Mean EDI across runs")
            ax.set_title("Aggregate EDI by Model")
            fig.tight_layout()
            fig.savefig(
                plots_dir / f"aggregate_edi_by_model.{self.reporting_config.plot_format}",
                dpi=self.reporting_config.plot_dpi,
            )
            self._generated_plot_stems.add("aggregate_edi_by_model")
            plt.close(fig)

    def _stability_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        n_runs = self.agg["n_runs"]
        for probe_id in sorted(self.agg.get("per_probe", {})):
            for model in sorted(self.agg["per_probe"][probe_id]):
                rec = self.agg["per_probe"][probe_id][model]
                interesting = (
                    not rec.get("stable", False)
                    or rec.get("null_content_n", 0) > 0
                )
                if not self.include_all and not interesting:
                    continue
                edi_mean = rec.get("edi_mean")
                edi_stddev = rec.get("edi_stddev")
                edi_str = "—" if edi_mean is None else f"{edi_mean:.2f}±{(edi_stddev or 0.0):.2f}"
                dist_pairs = sorted(
                    rec.get("classifications", {}).items(),
                    key=lambda item: (-item[1], item[0]),
                )
                dist = " ".join(f"{name}={count}" for name, count in dist_pairs[:2])
                rows.append({
                    "probe_id": probe_id,
                    "model": model,
                    "modal": rec["modal_classification"],
                    "consensus": f"{rec['consensus_count']}/{n_runs}",
                    "stability_n": f"{rec['stability_n']}/{n_runs}",
                    "null_content": str(rec["null_content_n"]),
                    "ifr_a_stable": "yes" if rec["ifr_a_stable"] else "no",
                    "ifr_i_stable": "yes" if rec["ifr_i_stable"] else "no",
                    "edi": edi_str,
                    "distribution": dist,
                })
        return rows

    def _compute_structural_classification_summary(self) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for run_dir in self.run_dirs:
            probe_scores = self._load_probe_scores(run_dir)
            raw_map = self._load_raw_result_map(run_dir)
            for probe_id, model_scores in probe_scores.items():
                for model, score in model_scores.items():
                    model_summary = summary.setdefault(model, {
                        "REFUSED_RECOGNIZED": 0,
                        "REFUSED_UNRECOGNIZED": 0,
                        "ENGAGED_RECOGNIZED": 0,
                        "ENGAGED_UNRECOGNIZED": 0,
                        "null": 0,
                    })
                    if self._is_null_content(score, raw_map.get((probe_id, model))):
                        model_summary["null"] += 1
                    classification = score.get("classification")
                    if classification in model_summary:
                        model_summary[classification] += 1
        return summary

    def _compute_domain_ifr_summary(self) -> dict[str, dict[str, dict[str, int]]]:
        summary: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
        probe_domains = self._probe_domain_map()
        for run_dir in self.run_dirs:
            probe_scores = self._load_probe_scores(run_dir)
            for probe_id, model_scores in probe_scores.items():
                domain = probe_domains.get(probe_id)
                if domain is None:
                    continue
                for model, score in model_scores.items():
                    domain_model = summary[domain].setdefault(model, {
                        "n_total": 0,
                        "n_failures_a": 0,
                        "n_failures_i": 0,
                    })
                    domain_model["n_total"] += 1
                    classification = score.get("classification")
                    if not _is_ifr_a_pass(classification):
                        domain_model["n_failures_a"] += 1
                    if not _is_ifr_i_pass(classification):
                        domain_model["n_failures_i"] += 1
        return {domain: models for domain, models in sorted(summary.items())}

    def _probe_domain_map(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for paper in self.papers_by_id.values():
            mapping[paper.paper_id] = paper.domain
            mapping[f"IS-{paper.paper_id}"] = paper.domain
        return mapping

    def _load_probe_scores(self, run_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
        score_path = run_dir / "report" / "data" / "probe_scores.json"
        if not score_path.exists():
            raise FileNotFoundError(f"Missing {score_path}")
        with open(score_path, encoding="utf-8") as handle:
            return json.load(handle)

    def _load_raw_result_map(self, run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
        mapping: dict[tuple[str, str], dict[str, Any]] = {}
        for record in _load_results(run_dir):
            probe_id = record.get("probe_id")
            model = record.get("model")
            if isinstance(probe_id, str) and isinstance(model, str):
                mapping[(probe_id, model)] = record
        return mapping

    def _is_null_content(self, score: dict[str, Any], raw_record: dict[str, Any] | None) -> bool:
        output_tokens = _extract_output_tokens(raw_record)
        if output_tokens is None:
            output_tokens = _extract_output_tokens(score)
        response_text = _normalize_response_text(raw_record) or _normalize_response_text(score)
        if output_tokens is not None:
            return output_tokens == 0 or len(response_text.strip()) == 0
        return len(response_text.strip()) == 0

    def _compute_aggregate_edi_summary(self) -> dict[str, Any]:
        models = sorted(self.agg.get("per_model", {}))
        per_model_runs: dict[str, dict[str, list[float]]] = {
            model: {
                "engaged_counts": [],
                "mean_edi": [],
                "median_edi": [],
                "achievement": [],
            }
            for model in models
        }
        overall_per_run: list[dict[str, float]] = []
        ceilings = self._probe_ceiling_map()
        for run_dir in self.run_dirs:
            score_path = run_dir / "report" / "data" / "probe_scores.json"
            if not score_path.exists():
                raise FileNotFoundError(f"Missing {score_path}")
            with open(score_path, encoding="utf-8") as handle:
                probe_scores = json.load(handle)

            run_overall_edis: list[float] = []
            run_overall_achievement_edis: list[float] = []
            run_overall_ceilings: list[float] = []
            run_overall_engaged = 0
            per_model_edis: dict[str, list[float]] = {}
            per_model_achievement_edis: dict[str, list[float]] = {}
            per_model_ceilings: dict[str, list[float]] = {}
            per_model_engaged: dict[str, int] = {}
            for probe_id, model_scores in probe_scores.items():
                ceiling = ceilings.get(probe_id)
                for model, score in model_scores.items():
                    classification = score.get("classification")
                    if classification in _ENGAGED_CLASSES:
                        per_model_engaged[model] = per_model_engaged.get(model, 0) + 1
                        run_overall_engaged += 1
                        edi_for_stats = float(score.get("edi") or 0.0)
                        per_model_edis.setdefault(model, []).append(edi_for_stats)
                        run_overall_edis.append(edi_for_stats)

                        edi = score.get("edi")
                        if edi is None or ceiling is None or ceiling <= 0:
                            continue
                        per_model_achievement_edis.setdefault(model, []).append(float(edi))
                        per_model_ceilings.setdefault(model, []).append(ceiling)
                        run_overall_achievement_edis.append(float(edi))
                        run_overall_ceilings.append(ceiling)

            for model in models:
                model_stats = per_model_runs[model]
                model_stats["engaged_counts"].append(float(per_model_engaged.get(model, 0)))
                edis = per_model_edis.get(model, [])
                achievement_edis = per_model_achievement_edis.get(model, [])
                ceiling_values = per_model_ceilings.get(model, [])
                model_stats["mean_edi"].append(statistics.mean(edis) if edis else 0.0)
                model_stats["median_edi"].append(statistics.median(edis) if edis else 0.0)
                model_stats["achievement"].append(
                    (sum(achievement_edis) / sum(ceiling_values)) if ceiling_values else 0.0
                )

            overall_per_run.append({
                "engaged_count": float(run_overall_engaged),
                "mean_edi": statistics.mean(run_overall_edis) if run_overall_edis else 0.0,
                "median_edi": statistics.median(run_overall_edis) if run_overall_edis else 0.0,
                "achievement": (
                    sum(run_overall_achievement_edis) / sum(run_overall_ceilings)
                    if run_overall_ceilings else 0.0
                ),
            })

        return {
            "overall": self._summarize_run_metric_dicts(overall_per_run),
            "per_model": {
                model: self._summarize_run_metric_lists(metrics)
                for model, metrics in per_model_runs.items()
            },
        }

    def _probe_ceiling_map(self) -> dict[str, float | None]:
        ceilings: dict[str, float | None] = {}
        for paper in self.papers_by_id.values():
            ceiling = structural_edi_max(
                list(paper.probe.withheld_details),
                self.scoring_config.edi,
            )
            ceilings[paper.paper_id] = ceiling
            ceilings[f"IS-{paper.paper_id}"] = ceiling
        return ceilings

    @staticmethod
    def _summarize_run_metric_lists(metrics: dict[str, list[float]]) -> dict[str, float]:
        return {
            "engaged_run_mean": statistics.mean(metrics["engaged_counts"]) if metrics["engaged_counts"] else 0.0,
            "mean_edi_mean": statistics.mean(metrics["mean_edi"]) if metrics["mean_edi"] else 0.0,
            "mean_edi_stddev": statistics.pstdev(metrics["mean_edi"]) if metrics["mean_edi"] else 0.0,
            "median_edi_mean": statistics.mean(metrics["median_edi"]) if metrics["median_edi"] else 0.0,
            "median_edi_stddev": statistics.pstdev(metrics["median_edi"]) if metrics["median_edi"] else 0.0,
            "achievement_mean": statistics.mean(metrics["achievement"]) if metrics["achievement"] else 0.0,
            "achievement_stddev": statistics.pstdev(metrics["achievement"]) if metrics["achievement"] else 0.0,
        }

    def _summarize_run_metric_dicts(self, rows: list[dict[str, float]]) -> dict[str, float] | None:
        if not rows:
            return None
        return {
            "engaged_run_mean": statistics.mean(row["engaged_count"] for row in rows),
            "mean_edi_mean": statistics.mean(row["mean_edi"] for row in rows),
            "mean_edi_stddev": statistics.pstdev(row["mean_edi"] for row in rows),
            "median_edi_mean": statistics.mean(row["median_edi"] for row in rows),
            "median_edi_stddev": statistics.pstdev(row["median_edi"] for row in rows),
            "achievement_mean": statistics.mean(row["achievement"] for row in rows),
            "achievement_stddev": statistics.pstdev(row["achievement"] for row in rows),
        }

    def _plot_markdown(self, stem: str) -> str:
        if stem not in self._generated_plot_stems:
            return ""
        return f"![{stem}](plots/{stem}.{self.reporting_config.plot_format})"

    def _plot_lines(self, *stems: str) -> list[str]:
        return [line for stem in stems if (line := self._plot_markdown(stem))]

    @staticmethod
    def _pct(part: int, whole: int) -> float:
        return (part / whole * 100.0) if whole else 0.0