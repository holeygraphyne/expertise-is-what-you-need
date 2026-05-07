"""
IS Runner: executes influence probes across models.

Reads ISProbe bundles, calls OpenRouter, saves raw responses,
invokes the scorer, and checkpoints progress.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from contextlib import contextmanager
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from traces.config import TracesConfig, ModelConfig
from traces.pipeline import ProviderClient, ThreadSafeRpmLimiter
from traces.prompts import ISProbe

logger = logging.getLogger(__name__)


@contextmanager
def _progress_logging(console: Console, enabled: bool):
    """Route logs through Rich while the live progress UI is active."""
    if not enabled:
        yield
        return

    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level

    handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=False,
        rich_tracebacks=False,
    )
    handler.setLevel(logging.INFO)

    root.handlers = [handler]
    root.setLevel(logging.INFO if old_level <= 0 else min(old_level, logging.INFO))

    try:
        yield
    finally:
        root.handlers = old_handlers
        root.setLevel(old_level)


def _build_work_items(
    probes: List[ISProbe],
    model_ids: List[str],
    completed: set[tuple[str, str]],
) -> list[tuple[ISProbe, str]]:
    """Build pending work in probe-major order, skipping completed pairs."""
    work: list[tuple[ISProbe, str]] = []
    for probe in probes:
        for model_id in model_ids:
            key = (probe.probe_id, model_id)
            if key not in completed:
                work.append((probe, model_id))
    return work


def _pop_next_runnable_work(
    pending: deque[tuple[ISProbe, str]],
    inflight_by_model: dict[str, int],
    max_inflight_by_model: dict[str, int],
) -> tuple[ISProbe, str] | None:
    """Pop the next pending item whose model still has available capacity."""
    if not pending:
        return None

    for _ in range(len(pending)):
        probe, model_id = pending.popleft()
        max_inflight = max_inflight_by_model.get(model_id, 2)
        if inflight_by_model.get(model_id, 0) < max_inflight:
            return probe, model_id
        pending.append((probe, model_id))

    return None


@dataclass
class RawProbeResult:
    """Raw result from executing a probe, before scoring."""
    probe_id: str
    paper_id: str
    model: str
    response_text: str
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: Optional[str] = None
    timestamp: str = ""
    error: Optional[str] = None
    # Deprecated: domain now derived from corpus folder at report time.
    # Kept Optional for backwards-compat read of legacy raw_results.json.
    domain: Optional[str] = None


class ProgressConsole:
    """Thread-safe live stderr renderer for runner progress and warnings."""

    @property
    def console(self) -> Console:
        return self._console

    def __init__(self, total: int, enabled: bool, stream=None):
        self.total = total
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self._lock = threading.Lock()
        self._active: dict[tuple[str, str], float] = {}
        self._warning_message = "—"
        self._done_count = 0
        self._fail_count = 0
        self._elapsed = 0.0
        self._session_done = 0
        self._console = Console(file=self.stream, force_terminal=self._should_force_terminal(), color_system=None)
        self._progress = Progress(
            TextColumn("progress", justify="right"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TextColumn("failures={task.fields[fail_count]}"),
            TimeElapsedColumn(),
            TextColumn("eta"),
            TimeRemainingColumn(),
            console=self._console,
            transient=True,
            expand=True,
        )
        self._task_id = self._progress.add_task("run", total=max(1, total), completed=0, fail_count=0)
        self._live: Live | None = None
        if self.enabled:
            self._live = Live(
                self._build_renderable(),
                console=self._console,
                auto_refresh=False,
                refresh_per_second=10,
                transient=True,
            )
            self._live.start()

    def started(self, probe_id: str, model: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._active[(probe_id, model)] = time.monotonic()
            self._refresh_locked()

    def finished(self, probe_id: str, model: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._active.pop((probe_id, model), None)
            self._refresh_locked()

    def warn(self, message: str) -> None:
        if not self.enabled:
            logger.warning(message)
            return
        with self._lock:
            self._warning_message = message
            if self._live is not None:
                self._live.console.print(message)
            self._refresh_locked()

    def update(
        self,
        *,
        done_count: int,
        fail_count: int,
        elapsed: float,
        session_done: int,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._done_count = done_count
            self._fail_count = fail_count
            self._elapsed = elapsed
            self._session_done = session_done
            self._progress.update(
                self._task_id,
                total=max(1, self.total),
                completed=done_count,
                fail_count=fail_count,
            )
            self._refresh_locked()

    def close(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._live is not None:
                self._refresh_locked()
                self._live.stop()
            self._live = None

    @staticmethod
    def _format_eta(eta: Optional[float]) -> str:
        if eta is None:
            return "--"
        return f"{eta:.0f}s"

    @staticmethod
    def _format_seconds(value: float) -> str:
        seconds = max(0, int(value))
        minutes, sec = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}h{minutes:02d}m{sec:02d}s"
        if minutes:
            return f"{minutes:d}m{sec:02d}s"
        return f"{sec:d}s"

    def _should_force_terminal(self) -> bool:
        isatty = getattr(self.stream, "isatty", None)
        if callable(isatty):
            try:
                return bool(isatty())
            except Exception:
                return False
        return False

    def _build_renderable(self):
        return Group(
            Text(
                f"{self._format_status()} | latest event: {self._warning_message}",
                overflow="ellipsis",
                no_wrap=True,
            ),
            self._progress,
        )

    def _refresh_locked(self) -> None:
        if self._live is None:
            return
        self._live.update(self._build_renderable(), refresh=True)

    def _format_status(self) -> str:
        rate = self._session_done / self._elapsed if self._elapsed > 0 else 0.0
        remaining = max(0, self.total - self._done_count)
        eta = remaining / rate if rate > 0 else None
        prefix = (
            f"done {self._done_count}/{self.total} | failures {self._fail_count} | "
            f"elapsed {self._format_seconds(self._elapsed)} | eta {self._format_eta(eta)}"
        )
        if not self._active:
            return f"{prefix} | idle"
        now = time.monotonic()
        active_items = sorted(self._active.items(), key=lambda item: item[1])
        probe_id, model = active_items[0][0]
        started = active_items[0][1]
        active_summary = f"current {probe_id} @ {model} ({self._format_seconds(now - started)})"
        if len(active_items) == 1:
            return f"{prefix} | {active_summary}"
        return f"{prefix} | {active_summary}; +{len(active_items) - 1} more"


class ISRunner:
    """Executes IS probes and scores results."""

    def __init__(self, config: TracesConfig):
        self.config = config
        self._model_cfg: Dict[str, ModelConfig] = {m.id: m for m in config.models}
        # One rate limiter + one client per provider used by the panel.
        # ModelConfig.provider is required, so every model has a provider
        # entry; the validator on TracesConfig guarantees the lookup hits.
        self._provider_clients: Dict[str, ProviderClient] = {}
        for m in config.models:
            if m.provider in self._provider_clients:
                continue
            provider_cfg = config.providers.get(m.provider)
            if provider_cfg is None:
                # Validator should have caught this, but defend.
                raise RuntimeError(
                    f"models[{m.id!r}].provider={m.provider!r} not in providers"
                )
            limiter = ThreadSafeRpmLimiter(provider_cfg.rpm_limit)
            self._provider_clients[m.provider] = ProviderClient(
                provider_cfg, rpm_limiter=limiter,
            )

        # Build dispatcher.
        from traces.pipeline.dispatcher import (
            ModelDispatcher,
        )

        model_configs: dict[str, ModelConfig] = {
            m.id: m for m in config.models
        }

        self._dispatcher = ModelDispatcher(
            model_configs=model_configs,
            provider_clients=self._provider_clients,
            trip_thresholds=config.pipeline.trip_thresholds.to_thresholds(),
        )

        # Work-gate: (probe_id, model_id) is "done" when it completed successfully.
        # Non-successful results are retried on resume.
        self._completed: set = set()
        self._raw_results: List[RawProbeResult] = []

    def run(
        self,
        probes: List[ISProbe],
        models: Optional[List[str]] = None,
        checkpoint_path: Optional[str] = None,
        progress: bool = False,
    ) -> List[RawProbeResult]:
        """Execute probes across models, return raw results.

        If progress=True, writes a live single-line status to stderr
        after each completion (suppresses the periodic log line).
        """
        model_ids = models or self.config.model_ids
        concurrency = self.config.pipeline.concurrency
        unknown_model_ids = [model_id for model_id in model_ids if model_id not in self._model_cfg]
        if unknown_model_ids:
            raise ValueError(f"Unknown model id(s): {', '.join(unknown_model_ids)}")
        max_inflight_by_model = {
            model_id: self._model_cfg[model_id].max_inflight
            for model_id in model_ids
        }
        for model_id, max_inflight in max_inflight_by_model.items():
            if max_inflight < 1:
                raise ValueError(
                    f"models[{model_id!r}].max_inflight must be >= 1, got {max_inflight}"
                )

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)

        work = _build_work_items(probes, model_ids, self._completed)

        total = len(work) + len(self._completed)
        logger.info(
            f"IS runner: {len(work)} API calls remaining "
            f"({len(self._completed)} done), concurrency={concurrency}"
        )
        active_models_with_work = {model_id for _, model_id in work}
        model_slot_capacity = sum(
            max_inflight_by_model[model_id]
            for model_id in active_models_with_work
        )
        effective_capacity = min(concurrency, model_slot_capacity) if active_models_with_work else 0
        logger.info(
            "IS runner effective capacity: %d active slot(s) across %d model(s)",
            effective_capacity,
            len(active_models_with_work),
        )

        done_count = len(self._completed)
        fail_count = 0
        t_start = time.monotonic()
        session_start_done = done_count
        console = ProgressConsole(total=total, enabled=progress)

        def process_result(probe, model_id, result):
            nonlocal done_count, fail_count
            self._raw_results.append(result)
            if not result.error:
                self._completed.add((probe.probe_id, model_id))
            done_count += 1
            if result.error:
                fail_count += 1
                console.warn(
                    f"warning: {probe.probe_id} @ {model_id} failed: {result.error}"
                )
            if checkpoint_path and done_count % self.config.pipeline.checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_path)
            if progress:
                elapsed = time.monotonic() - t_start
                session_done = done_count - session_start_done
                console.update(
                    done_count=done_count,
                    fail_count=fail_count,
                    elapsed=elapsed,
                    session_done=session_done,
                )
            elif done_count % 10 == 0:
                logger.info(f"  Progress: {done_count}/{total} ({fail_count} failed)")

        try:
            with _progress_logging(console.console, enabled=progress):
                if concurrency <= 1:
                    for probe, model_id in work:
                        console.started(probe.probe_id, model_id)
                        try:
                            result = self._dispatcher.execute(probe, model_id=model_id)
                        except Exception as e:
                            # Defense-in-depth: dispatcher.execute() shouldn't raise, but we
                            # don't want one bogus model_id to abort the whole run.
                            result = RawProbeResult(
                                probe_id=probe.probe_id, paper_id=probe.paper_id,
                                model=model_id, response_text="", latency_ms=0,
                                error=str(e),
                                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                            )
                        console.finished(probe.probe_id, model_id)
                        process_result(probe, model_id, result)
                else:
                    pending = deque(work)
                    inflight_by_model: dict[str, int] = {}

                    with ThreadPoolExecutor(max_workers=concurrency) as ex:
                        fut_map = {}

                        def submit_available() -> None:
                            while len(fut_map) < concurrency:
                                item = _pop_next_runnable_work(
                                    pending,
                                    inflight_by_model,
                                    max_inflight_by_model,
                                )
                                if item is None:
                                    break

                                probe, model_id = item
                                inflight_by_model[model_id] = inflight_by_model.get(model_id, 0) + 1
                                fut = ex.submit(self._execute_via_dispatcher, console, probe, model_id)
                                fut_map[fut] = (probe, model_id)

                        submit_available()

                        while fut_map:
                            done, _ = wait(fut_map.keys(), return_when=FIRST_COMPLETED)

                            for fut in done:
                                probe, model_id = fut_map.pop(fut)
                                current = inflight_by_model.get(model_id, 0) - 1
                                if current > 0:
                                    inflight_by_model[model_id] = current
                                else:
                                    inflight_by_model.pop(model_id, None)

                                try:
                                    result = fut.result()
                                except Exception as e:
                                    # Defense-in-depth: dispatcher.execute() shouldn't raise, but we
                                    # don't want one bogus model_id to abort the whole run.
                                    result = RawProbeResult(
                                        probe_id=probe.probe_id, paper_id=probe.paper_id,
                                        model=model_id, response_text="", latency_ms=0,
                                        error=str(e),
                                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                                    )
                                process_result(probe, model_id, result)

                            submit_available()
        finally:
            if progress:
                console.close()

        if checkpoint_path:
            self._save_checkpoint(checkpoint_path)

        logger.info(f"IS run complete: {done_count} calls, {fail_count} failures")

        # Model availability summary (observability). Limit to models
        # that were *exercised* this run — when the user uses
        # --paper-id or --models to subset the work, untouched models
        # have attempts=0 and would otherwise inflate the "fully
        # available" count.
        snap = self._dispatcher.health_snapshot()
        exercised = {m: info for m, info in snap.items() if info["attempts"] > 0}
        tripped_models = [m for m, info in exercised.items() if info["tripped"]]
        skipped_count = len(snap) - len(exercised)
        msg = (
            "Model availability: %d/%d models fully available, "
            "%d had at least one trip"
        )
        args: tuple = (
            len(exercised) - len(tripped_models),
            len(exercised),
            len(tripped_models),
        )
        if skipped_count:
            msg += " (%d configured models not exercised this run)"
            args = args + (skipped_count,)
        logger.info(msg, *args)

        return self._raw_results

    def _execute_via_dispatcher(
        self,
        console: ProgressConsole,
        probe: ISProbe,
        model_id: str,
    ) -> RawProbeResult:
        console.started(probe.probe_id, model_id)
        try:
            return self._dispatcher.execute(probe, model_id=model_id)
        finally:
            console.finished(probe.probe_id, model_id)

    def _save_checkpoint(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "completed": [list(k) for k in self._completed],
            "results": [asdict(r) for r in self._raw_results],
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def _load_checkpoint(self, path: str):
        p = Path(path)
        if not p.exists():
            return
        with open(p) as f:
            data = json.load(f)

        all_results = []
        for r in data.get("results", []):
            # Strip legacy fields not present in the current schema.
            r.pop("slot_id", None)
            r.pop("is_terminal_failure", None)
            all_results.append(RawProbeResult(**r))

        successful_results = [r for r in all_results if not r.error]

        # Work gate: only successes skip re-execution. Errors retry on resume.
        self._completed = {
            (r.probe_id, r.model) for r in successful_results
        }

        # Preserve only successful results for the next checkpoint write.
        # Errors are excluded so they are re-run fresh.
        self._raw_results = successful_results

        skipped = len(all_results) - len(successful_results)
        msg = f"Resumed from checkpoint: {len(self._completed)} completed"
        if skipped:
            msg += f" ({skipped} prior error(s) will retry)"
        logger.info(msg)


def save_raw_results(results: List[RawProbeResult], path: str):
    """Save raw results to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def load_raw_results(path: str) -> List[RawProbeResult]:
    """Load raw results from JSON.

    Strips legacy fields (`slot_id`, `is_terminal_failure`) that may be
    present on records produced before the slot/fallback machinery was
    removed. Same backward-compat handling as `_load_checkpoint`.
    """
    with open(path) as f:
        data = json.load(f)
    out: List[RawProbeResult] = []
    for r in data:
        r.pop("slot_id", None)
        r.pop("is_terminal_failure", None)
        out.append(RawProbeResult(**r))
    return out
