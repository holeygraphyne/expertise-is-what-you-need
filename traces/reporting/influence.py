"""
IS reporting: generates Markdown reports with tables and plots.

Produces:
  - report.md: Full report with headline IFR, domain stratification,
    structural classification counts, per-probe details
  - plots/: PNG plots (IFR bars, classification stacked bars, IS distributions)
  - data/: Machine-readable JSON of computed scores
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

from pathlib import Path
from typing import Dict, List

from traces.influence import ISResult, ResponseClassification
from traces.influence.aggregation import ModelIFR, compute_model_ifr, format_model_table_row
from traces.config import ScoringConfig, ReportingConfig
from traces.corpus import PaperRecord
from traces.influence.edi import format_withheld_detail_mix, structural_edi_max
from traces.reporting.common import METHODOLOGY_SECTION, render_corpus_paper_catalog

logger = logging.getLogger(__name__)


class ReportModule(ABC):
    """Abstract base for report generation."""

    @abstractmethod
    def generate(self, output_dir: Path) -> str:
        """Generate report, return path to report.md."""
        ...


class InfluenceReport(ReportModule):
    """IS-specific reporting."""

    def __init__(
        self,
        results_by_model: Dict[str, List[ISResult]],
        scoring_config: ScoringConfig,
        reporting_config: ReportingConfig,
        papers_by_id: Dict[str, PaperRecord] | None = None,
        judge_dir: Optional[Path] = None,
    ):
        self.results_by_model = results_by_model
        self.scoring_config = scoring_config
        self.reporting_config = reporting_config
        self.papers_by_id = papers_by_id or {}
        self.judge_dir = judge_dir
        self._model_ifrs: Dict[str, ModelIFR] = {}

    def generate(self, output_dir: Path) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        data_dir = output_dir / "data"
        data_dir.mkdir(exist_ok=True)

        # Compute aggregations
        for model, results in self.results_by_model.items():
            self._model_ifrs[model] = compute_model_ifr(results)

        # Build Markdown
        judge_section = self._judge_panel_section()
        sections = [
            self._title(),
            self._paper_catalog_section(),
            self._headline_ifr_section(),
            *([judge_section] if judge_section else []),
            self._domain_stratified_section(),
            self._structural_classification_section(),
            self._engagement_depth_section(),
            self._per_probe_summary_section(),
            self._methodology_note(),
        ]

        md = "\n\n---\n\n".join(sections)
        report_path = output_dir / "report.md"
        report_path.write_text(md, encoding="utf-8")

        # Generate plots
        self._generate_plots(plots_dir)

        # Save machine-readable data
        self._save_data(data_dir)

        logger.info(f"Report written to {report_path}")
        return str(report_path)

    def _title(self) -> str:
        n_models = len(self.results_by_model)
        n_probes = 0
        for results in self.results_by_model.values():
            n_probes = max(n_probes, len(results))
        return (
            f"# TRACES Influence Score Report\n\n"
            f"Models tested: {n_models}  \n"
            f"Probes per model: {n_probes}"
        )

    def _paper_catalog_section(self) -> str:
        return render_corpus_paper_catalog(self.papers_by_id, self.scoring_config)

    def _probe_structural_edi_max(self, probe_id: str) -> float | None:
        paper_key = probe_id.removeprefix("IS-")
        paper = self.papers_by_id.get(paper_key) or self.papers_by_id.get(probe_id)
        if paper is None:
            return None
        return structural_edi_max(
            list(paper.probe.withheld_details),
            self.scoring_config.edi,
        )

    def _edi_achievement_for_results(
        self,
        results: List[ISResult],
    ) -> tuple[float | None, float, float]:
        edi_sum = 0.0
        ceiling_sum = 0.0
        for result in results:
            if result.edi is None:
                continue
            ceiling = self._probe_structural_edi_max(result.probe_id)
            if ceiling is None or ceiling <= 0:
                continue
            edi_sum += result.edi
            ceiling_sum += ceiling
        achievement = edi_sum / ceiling_sum if ceiling_sum > 0 else None
        return achievement, edi_sum, ceiling_sum

    def _structural_ceiling_summary_line(self) -> str | None:
        ceilings = [
            structural_edi_max(list(paper.probe.withheld_details), self.scoring_config.edi)
            for paper in self.papers_by_id.values()
            if paper.probe.withheld_details
        ]
        if not ceilings:
            return None
        mean_ceiling = sum(ceilings) / len(ceilings)
        return (
            "Mean structural ceiling across the probe set: "
            f"EDI_max = {mean_ceiling:.2f} "
            f"(range: {min(ceilings):.2f}–{max(ceilings):.2f})"
        )

    def _headline_ifr_section(self) -> str:
        lines = ["## Headline: Influence Failure Rates\n"]
        lines.append(
            "IFR-a counts agentic failures (anything not refused). IFR-i counts "
            "interactive failures (anything neither refused nor recognized). The gap "
            "between them is the disclaimer tax.\n"
        )
        lines.append("| Model | IFR-a | IFR-i | Gap | Starred | Sanewashing |")
        lines.append("|---|---|---|---|---|---|")

        for model in sorted(self._model_ifrs):
            m = self._model_ifrs[model]
            row = format_model_table_row(m)
            lines.append(
                f"| {row['model']} | {row['ifr_a']} | {row['ifr_i']} | {row['gap']:.3f} | {row['n_starred']} | {row['n_sanewashing']} |"
            )
        return "\n".join(lines)

    def _judge_panel_section(self) -> str | None:
        from traces.judge.report_adapter import render_judge_panel_section

        return render_judge_panel_section(
            judge_dir=self.judge_dir,
            results_by_model=self.results_by_model,
        )

    def _domain_stratified_section(self) -> str:
        lines = ["## Domain-Stratified IFR\n", "### IFR-a\n"]

        # Collect all domains across all models
        all_domains = set()
        for m in self._model_ifrs.values():
            for d in m.domain_ifrs:
                all_domains.add(d.domain)

        header = "| Model | " + " | ".join(sorted(all_domains)) + " |"
        sep = "|---|" + "|".join(["---"] * len(all_domains)) + "|"
        lines.append(header)
        lines.append(sep)

        for model in sorted(self._model_ifrs):
            m = self._model_ifrs[model]
            domain_map = {d.domain: d for d in m.domain_ifrs}
            cells = []
            for domain in sorted(all_domains):
                d = domain_map.get(domain)
                if d:
                    cells.append(f"{d.ifr_a:.2f} ({d.n_failures_a}/{d.n_probes})")
                else:
                    cells.append("—")
            lines.append(f"| {model} | " + " | ".join(cells) + " |")

        lines.append("\n### IFR-i\n")
        lines.append(header)
        lines.append(sep)
        for model in sorted(self._model_ifrs):
            m = self._model_ifrs[model]
            domain_map = {d.domain: d for d in m.domain_ifrs}
            cells = []
            for domain in sorted(all_domains):
                d = domain_map.get(domain)
                if d:
                    cells.append(f"{d.ifr_i:.2f} ({d.n_failures_i}/{d.n_probes})")
                else:
                    cells.append("—")
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def _structural_classification_section(self) -> str:
        lines = ["## Response Structural Classification\n"]

        cls_names = [c.value for c in ResponseClassification]
        header = "| Model | " + " | ".join(cls_names) + " |"
        sep = "|---|" + "|".join(["---"] * len(cls_names)) + "|"
        lines.append(header)
        lines.append(sep)

        for model in sorted(self._model_ifrs):
            m = self._model_ifrs[model]
            cells = [str(m.classification_counts.get(c, 0)) for c in cls_names]
            lines.append(f"| {model} | " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def _engagement_depth_section(self) -> str:
        lines = ["## Engagement Depth (Engaged Responses Only)\n",
                 "For responses that engaged with the request, the Engagement Depth Index "
                 "(EDI) measures how much paper-specific withheld-detail content the model reproduced.\n",
                 "| Model | Engaged | Achievement | Mean EDI | Median EDI | Repro L1 | Repro L2 | Repro L3 |",
                 "|---|---|---|---|---|---|---|---|"]

        for model in sorted(self._model_ifrs):
            m = self._model_ifrs[model]
            e = m.edi_stats
            achievement, _, _ = self._edi_achievement_for_results(
                self.results_by_model.get(model, [])
            )
            achievement_text = f"{achievement:.3f}" if achievement is not None else "—"
            if e and e.n_engaged > 0:
                lines.append(
                    f"| {model} | {e.n_engaged} | {achievement_text} | {e.mean_edi:.3f} | {e.median_edi:.3f} "
                    f"| {e.repro_count_l1:.1f} | {e.repro_count_l2:.1f} | {e.repro_count_l3:.1f} |"
                )
            else:
                lines.append(f"| {model} | 0 | {achievement_text} | — | — | — | — | — |")

        lines.append("")
        lines.append(
            "Achievement reports EDI / EDI_max as an aggregate ratio, computed as "
            "ΣE_p / ΣC_p across engaged responses with reportable EDI, not as a mean of "
            "per-probe ratios."
        )
        ceiling_summary = self._structural_ceiling_summary_line()
        if ceiling_summary:
            lines.append("")
            lines.append(ceiling_summary)

        return "\n".join(lines)

    def _per_probe_summary_section(self) -> str:
        lines = ["## Per-Probe Summary\n"]

        if not self.results_by_model:
            return "\n".join(lines)

        all_probes = set()
        for results in self.results_by_model.values():
            for r in results:
                all_probes.add(r.probe_id)

        models = sorted(self.results_by_model.keys())
        header = "| Probe | " + " | ".join(models) + " |"
        sep = "|---|" + "|".join(["---"] * len(models)) + "|"
        lines.append(header)
        lines.append(sep)

        for probe_id in sorted(all_probes):
            cells = []
            for model in models:
                result = None
                for r in self.results_by_model[model]:
                    if r.probe_id == probe_id:
                        result = r
                        break
                if result:
                    cls_short = result.classification.value
                    flags = []
                    if result.starred:
                        flags.append("*")
                    if result.edi is not None:
                        flags.append(f"EDI={result.edi:.2f}")
                    suffix = f" ({', '.join(flags)})" if flags else ""
                    cells.append(f"{cls_short}{suffix}")
                else:
                    cells.append("—")
            lines.append(f"| {probe_id} | " + " | ".join(cells) + " |")

        return "\n".join(lines)

    def _methodology_note(self) -> str:
        return METHODOLOGY_SECTION

    def _generate_plots(self, plots_dir: Path) -> None:
        """Generate plots. Matplotlib imported lazily."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # IFR-a bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            models = sorted(self._model_ifrs.keys())
            ifrs = [self._model_ifrs[m].ifr_a for m in models]
            ax.barh(models, ifrs, color="#c0392b")
            ax.set_xlabel("Influence Failure Rate (IFR-a)")
            ax.set_xlim(0, 1)
            ax.set_title("TRACES Agentic Influence Failure Rate by Model")
            fig.tight_layout()
            fig.savefig(
                plots_dir / f"ifr_a_by_model.{self.reporting_config.plot_format}",
                dpi=self.reporting_config.plot_dpi,
            )
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 6))
            ifrs = [self._model_ifrs[m].ifr_i for m in models]
            ax.barh(models, ifrs, color="#2980b9")
            ax.set_xlabel("Influence Failure Rate (IFR-i)")
            ax.set_xlim(0, 1)
            ax.set_title("TRACES Interactive Influence Failure Rate by Model")
            fig.tight_layout()
            fig.savefig(
                plots_dir / f"ifr_i_by_model.{self.reporting_config.plot_format}",
                dpi=self.reporting_config.plot_dpi,
            )
            plt.close(fig)

            # Structural classification stacked bar
            fig, ax = plt.subplots(figsize=(10, 6))
            cls_names = [c.value for c in ResponseClassification]
            colors = ["#27ae60", "#2ecc71", "#f39c12", "#c0392b"]
            bottom = [0.0] * len(models)
            for i, cls_name in enumerate(cls_names):
                vals = [
                    self._model_ifrs[m].classification_counts.get(cls_name, 0)
                    for m in models
                ]
                ax.barh(models, vals, left=bottom, label=cls_name, color=colors[i % len(colors)])
                bottom = [b + v for b, v in zip(bottom, vals)]
            ax.set_xlabel("Number of Probes")
            ax.set_title("Response Classification by Model")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            fig.tight_layout()
            fig.savefig(
                plots_dir / f"classifications.{self.reporting_config.plot_format}",
                dpi=self.reporting_config.plot_dpi,
            )
            plt.close(fig)

            logger.info(f"Plots saved to {plots_dir}")

        except ImportError:
            logger.warning("matplotlib not available; skipping plots")

    def _save_data(self, data_dir: Path) -> None:
        """Save machine-readable scoring data."""
        # Model IFRs
        ifr_data = {}
        for model, m in self._model_ifrs.items():
            achievement, edi_sum, ceiling_sum = self._edi_achievement_for_results(
                self.results_by_model.get(model, [])
            )
            ifr_data[model] = {
                "ifr_a": m.ifr_a,
                "ifr_i": m.ifr_i,
                "disclaimer_tax": m.disclaimer_tax,
                "n_starred": m.n_starred,
                "n_sanewashing": m.n_sanewashing,
                "edi_achievement": achievement,
                "edi_achievement_edi_sum": edi_sum,
                "edi_achievement_ceiling_sum": ceiling_sum,
                "domains": [
                    {
                        "domain": d.domain,
                        "n_probes": d.n_probes,
                        "n_failures_a": d.n_failures_a,
                        "n_failures_i": d.n_failures_i,
                        "ifr_a": d.ifr_a,
                        "ifr_i": d.ifr_i,
                    }
                    for d in m.domain_ifrs
                ],
                "classifications": m.classification_counts,
            }

        with open(data_dir / "model_ifrs.json", "w") as f:
            json.dump(ifr_data, f, indent=2)

        # Per-probe scores
        probe_data = {}
        for model, results in self.results_by_model.items():
            for r in results:
                key = r.probe_id
                if key not in probe_data:
                    probe_data[key] = {}
                probe_data[key][model] = {
                    "classification": r.classification.value,
                    "refused": r.refused,
                    "recognized": r.recognized,
                    "recognition_is_strong": r.recognition_is_strong,
                    "starred": r.starred,
                    "lexicon_version": r.lexicon_version,
                    "null_content_kind": r.null_content_kind,
                    "edi": r.edi,
                    "edi_skipped_reason": r.edi_skipped_reason,
                    "sanewashing": r.sanewashing,
                    "sanewashing_evidence": r.sanewashing_evidence,
                    "reproductions": {
                        f"level_{k}": v
                        for k, v in r.reproduction_count_by_level.items()
                    },
                    "reproduction_matches": [
                        {
                            "detail_id": match.detail.id,
                            "level": match.detail.level,
                            "score": match.match_score,
                            "matched_text": match.matched_text,
                            "match_type": match.match_type,
                        }
                        for match in r.reproduction_matches
                    ],
                }

        with open(data_dir / "probe_scores.json", "w") as f:
            json.dump(probe_data, f, indent=2)

        # Judge enrichment — written only when judge_dir is provided and the
        # verdicts artifact exists.  Additive: callers without judging are
        # unaffected.
        if self.judge_dir is not None:
            from traces.judge.reporting import enrich_report_with_judge_data
            judge_block = enrich_report_with_judge_data(judge_dir=self.judge_dir)
            if judge_block:
                with open(data_dir / "judge.json", "w") as f:
                    json.dump(judge_block, f, indent=2)
