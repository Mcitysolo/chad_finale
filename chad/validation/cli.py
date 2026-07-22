"""chad/validation/cli.py — Phase 5 assembly + CLI (SSOT Part 6, Phase 5).

The capstone: it wires Phases 0-4 (bar audit → feature parity → backtest → cost/ splits/
significance) to the Phase-5 lockbox / config-freeze / verdict / report layer into one
runnable machine that emits a signed verdict artifact (SSOT
``docs/CHAD_EDGE_VALIDATION_HARNESS_DESIGN_v1.1.md`` Part 6, Phase 5). It adds ONLY
assembly — every metric, cost, split, significance figure, and parity classification is
computed by the phase that owns it (DRY; no re-implementation).

    python -m chad.validation.cli --stage historical [--final-run]

Default (no ``--final-run``): runs against train/validation + a **synthetic decoy OOS**
only — the real out-of-sample partition is hash-sealed but NEVER opened, so the run ends
with OOS access count ``0`` (SSOT §3.1). ``--final-run`` opens the sealed OOS through the
lockbox (one logged access) and scores it. Either way the machine prints the verdict
summary and writes ``edge_report_<ts>.json`` (+ ``.md``) — and NEVER writes
``ready_for_live`` or any ``runtime/`` state (SSOT §1.2 / Part 0).

The replay seam (SSOT §1.2 / §V1). The isolation wall forbids importing a live strategy
module, so a REPLAYABLE head is replayed by a harness-side decision function defined here
that reads ONLY reconstructable daily-bar inputs (EMA / momentum / ATR — exactly the
input families the Phase-4 parity audit detects for ``alpha_forex``). This is a
reconstruction, honestly flagged in the report, NOT an import of the live head (which
would fail isolation) and NOT an assertion of behavioural fidelity.

Isolation (SSOT §1.2 / §2): standard-library + sibling ``chad.validation`` modules only.
It reads strategy source as TEXT (never imports it) via
:func:`chad.validation.feature_parity.classify_head_file`, and its only writes are the
report artifact + the lockbox/freeze files under the caller-supplied output directory.
``subprocess`` (for the git commit stamp) and ``datetime`` are used at RUN time only, so
importing this module has no side effects and the transitive-import isolation test stays
green.
"""

from __future__ import annotations

import argparse
import datetime
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from chad.validation.backtest_engine import (
    BacktestResult,
    BarWindow,
    ExecutionSpec,
    Signal,
    load_bars_file,
    run_backtest,
)
from chad.validation.bar_audit import Status, SymbolAudit, audit_bar_file
from chad.validation.config_freeze import FreezeLedger, FreezeState, deflation_trials
from chad.validation.cost_model import (
    DEFAULT_COST_CONFIG,
    CostConfig,
    InstrumentClass,
    LiquidityTier,
    Trade,
    apply_costs,
)
from chad.validation.feature_parity import (
    ParityStatus,
    classify_head_file,
    render_parity_summary,
)
from chad.validation.oos_lockbox import OOSLockbox
from chad.validation.regime_labeler import DEFAULT_REGIME_CONFIG, Regime, label_series
from chad.validation.report_writer import (
    build_report,
    report_basename,
    sign_report,
    verify_signature,
    write_report,
)
from chad.validation.scoring_spine import score_returns
from chad.validation.splits import generate_walk_forward
from chad.validation.trade_log_adapter import (
    EXIT_OVERLAY_BOUNDARY,
    AdmittedTrade,
    LedgerChainError,
    ScrCrosscheckError,
    era_of,
    run_adapter,
)
from chad.validation.significance import (
    DEFAULT_TRIAL_MULTIPLE,
    RuinConfig,
    punitive_trial_count,
    seed_sweep,
)
from chad.validation.verdict import (
    DEFAULT_THRESHOLDS,
    HeadMetrics,
    PortfolioMetrics,
    Verdict,
    VerdictResult,
    VerdictThresholds,
    decide_portfolio_verdict,
    decide_verdict,
)

__all__ = [
    "HeadSpec",
    "DEFAULT_CATALOG",
    "run_stage_historical",
    "run_stage2_trade_log",
    "current_code_commit",
    "main",
]

# The seed-sweep ruin bootstrap is trimmed for the CLI (the OOS series is tiny); the
# full 2000-path default lives in significance.DEFAULT_RUIN_CONFIG for the sealed run.
_CLI_RUIN_CONFIG: RuinConfig = RuinConfig(bootstrap_paths=500)

# Worst-quantile tail the verdict thresholds against (SSOT §3.6: worst quantile, 5%).
_WORST_TAIL: float = 0.05


# --------------------------------------------------------------------------- #
# The harness-side REPLAYABLE decision function (reconstruction, not an import).
# --------------------------------------------------------------------------- #
def _ema(values: Sequence[float], span: int) -> float:
    """Exponential moving average of ``values`` (causal; uses only the passed series)."""
    k = 2.0 / (span + 1.0)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1.0 - k)
    return e


def _atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int) -> float:
    """Average true range over the last ``period`` bars (causal)."""
    trs: list[float] = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if not trs:
        return 0.0
    window = trs[-period:]
    return sum(window) / len(window)


def _alpha_forex_decide(w: BarWindow) -> Optional[Signal]:
    """A REPLAYABLE EMA/momentum/ATR breakout over daily bars (SSOT §V1 reconstruction).

    Reads ONLY the :class:`BarWindow` (bars ``<= t``) — no future data, no runtime state,
    no live feed — so it is structurally no-lookahead and matches the reconstructable
    input families the Phase-4 parity audit classifies REPLAYABLE for ``alpha_forex``
    (daily bars, realized vol, EMA/ATR/momentum). Goes long on an up-trend + positive
    momentum, short on the mirror, with an ATR-scaled stop/target band; stays flat
    otherwise. Deterministic.
    """
    closes = w.closes()
    n = len(closes)
    if n < 26:  # need enough history for the slow EMA + ATR
        return None
    highs = w.highs()
    lows = w.lows()
    fast = _ema(closes[-10:], 5)
    slow = _ema(closes[-25:], 20)
    momentum = closes[-1] - closes[-11]  # 10-bar momentum
    atr = _atr(highs, lows, closes, 14)
    price = closes[-1]
    if atr <= 0.0 or price <= 0.0:
        return None
    if fast > slow and momentum > 0.0:
        stop = price - 2.0 * atr
        target = price + 3.0 * atr
        if stop <= 0.0:
            return None
        return Signal(direction="long", stop=stop, target=target, label="alpha_forex")
    if fast < slow and momentum < 0.0:
        stop = price + 2.0 * atr
        target = price - 3.0 * atr
        if target <= 0.0:
            return None
        return Signal(direction="short", stop=stop, target=target, label="alpha_forex")
    return None


# --------------------------------------------------------------------------- #
# Head catalog.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HeadSpec:
    """One head's wiring: source (for parity), symbol (for bars), decide fn, sizing."""

    head: str
    source_rel: str            # strategy source path (read as text; never imported)
    symbol: str                # bar-corpus symbol backing the replay
    decide: Callable[[BarWindow], Optional[Signal]]
    label_horizon: int
    execution_spec: ExecutionSpec


# ``alpha_forex`` is the one head the Phase-4 parity audit classifies REPLAYABLE over the
# daily-bar corpus (verified 2026-07-04). M6E (Micro EUR/USD future) is its backing bar
# series. Other heads depend on category-(c) inputs (news/options/intraday/runtime) and
# would classify NOT_REPLAYABLE — they are not in the default backtest catalog.
DEFAULT_CATALOG: tuple[HeadSpec, ...] = (
    HeadSpec(
        head="alpha_forex",
        source_rel="chad/strategies/alpha_forex.py",
        symbol="M6E",
        decide=_alpha_forex_decide,
        label_horizon=5,
        execution_spec=ExecutionSpec(
            instrument_class=InstrumentClass.FUT,
            quantity=1.0,
            multiplier=12500.0,   # M6E contract multiplier (Micro EUR/USD)
            liquidity_tier=LiquidityTier.MID,
        ),
    ),
)


# --------------------------------------------------------------------------- #
# Small helpers.
# --------------------------------------------------------------------------- #
def current_code_commit(repo_root: Path) -> str:
    """The current git commit (``git rev-parse HEAD``); ``"unknown"`` if unavailable.

    Run-time only (never at import). A read-only git query — no network, no broker.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = out.stdout.strip()
    return commit if (out.returncode == 0 and commit) else "unknown"


def _now_iso() -> str:
    """Current UTC timestamp (run-time only; report determinism is per-input, not clock)."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _oos_returns_from_result(result: BacktestResult) -> list[float]:
    """Extract the per-trade NET returns of trades whose entry AND exit fall in OOS.

    Uses the engine's own partition (``result.partition['oos_indices']``) and net
    ``ret`` — the same attribution rule the engine uses, no re-derivation. Sealing
    fingerprints THIS series; the verdict consumes it only via the lockbox gate.
    """
    if result.partition is None:
        return []
    oos_idx = set(result.partition.get("oos_indices", ()))
    return [
        t.ret
        for t in result.trades
        if t.entry_index in oos_idx and t.exit_index in oos_idx
    ]


def _oos_regime_stats(oos_returns: Sequence[float]) -> tuple[int, int]:
    """(distinct OOS regimes, regimes with a positive net edge) via the independent labeler.

    Builds a normalized price path from the OOS returns and labels it with the Phase-2
    independent regime labeler (SSOT §3.4). ``regimes_with_edge`` counts distinct regimes
    whose summed OOS return is positive — a simple, deterministic breadth proxy the
    verdict's §4.1 regime check reads. With only a handful of OOS returns every bar is
    UNKNOWN (insufficient trailing history) → (0, 0), the honest below-R_min default.
    """
    rets = [float(r) for r in oos_returns]
    if len(rets) < 2:
        return (0, 0)
    prices = [1.0]
    for r in rets:
        prices.append(prices[-1] * (1.0 + r))
    series = label_series(prices, config=DEFAULT_REGIME_CONFIG)
    per_regime: dict[str, float] = {}
    for k, r in enumerate(rets):
        # returns[k] drives the transition into price index k+1 → label at index k+1.
        label = series.labels[k + 1].regime
        if label is Regime.UNKNOWN:
            continue
        per_regime[label.value] = per_regime.get(label.value, 0.0) + r
    distinct = len(per_regime)
    with_edge = sum(1 for v in per_regime.values() if v > 0.0)
    return (distinct, with_edge)


def _oos_metrics(
    oos_returns: Sequence[float],
    *,
    periods_per_year: float,
    n_effective_trials: int,
) -> dict[str, Any]:
    """Compute the OOS-derived verdict inputs from a chosen OOS return series (DRY-reuse).

    Runs the Phase-1 spine (cost-adjusted CAGR) and the Phase-3 seed sweep
    (worst-quantile Deflated Sharpe + worst-quantile ruin) over the ALREADY-chosen OOS
    series (real sealed series in a final run, decoy otherwise). Everything numeric comes
    from the owning phase; this only selects the worst-quantile figures the verdict reads.
    """
    n = len(oos_returns)
    cagr: Optional[float] = None
    if n >= 1:
        cagr = score_returns(oos_returns, periods_per_year=periods_per_year).cagr

    dsr_worst: Optional[float] = None
    ruin_worst: Optional[float] = None
    sweep_dict: dict[str, Any] = {}
    if n >= 2:
        sweep = seed_sweep(
            oos_returns,
            n_effective_trials=n_effective_trials,
            config=_CLI_RUIN_CONFIG,
        )
        dsr_worst = sweep.dsr_worst_quantile(_WORST_TAIL)
        ruin_worst = sweep.ruin_worst_quantile(_WORST_TAIL)
        sweep_dict = {
            "n_effective_trials": sweep.n_effective_trials,
            "dsr_worst_quantile": dsr_worst,
            "ruin_worst_quantile": ruin_worst,
            "mean_dsr": sweep.mean_dsr(),
            "mean_ruin_worse": sweep.mean_ruin_worse(),
            "warnings": list(sweep.warnings),
        }
    n_regimes, regimes_with_edge = _oos_regime_stats(oos_returns)
    return {
        "n_oos_trades": n,
        "cost_adj_cagr": cagr,
        "deflated_sharpe_worst": dsr_worst,
        "worst_quantile_ruin": ruin_worst,
        "n_regimes_in_oos": n_regimes,
        "regimes_with_edge": regimes_with_edge,
        "seed_sweep": sweep_dict,
    }


# --------------------------------------------------------------------------- #
# The assembly.
# --------------------------------------------------------------------------- #
@dataclass
class _HeadRun:
    """Internal per-head bundle carried from replay to the report."""

    spec: HeadSpec
    parity_status: str
    replayable: bool
    symbol_audit: Optional[SymbolAudit]
    backtest_summary: dict[str, Any]
    metrics: HeadMetrics
    verdict: VerdictResult
    oos_source: str
    oos_access_count: int
    oos_returns: list[float]
    periods_per_year: float


def run_stage_historical(
    *,
    repo_root: Path,
    out_dir: Path,
    final_run: bool,
    catalog: Sequence[HeadSpec] = DEFAULT_CATALOG,
    thresholds: VerdictThresholds = DEFAULT_THRESHOLDS,
    cost_config: CostConfig = DEFAULT_COST_CONFIG,
    bars_dir: Optional[Path] = None,
    generated_at: Optional[str] = None,
    code_commit: Optional[str] = None,
    trial_multiple: float = DEFAULT_TRIAL_MULTIPLE,
) -> dict[str, Any]:
    """Run the Stage-1 historical harness end-to-end and return the SIGNED report dict.

    Pipeline per head: feature parity (Phase 4) → REPLAYABLE ? backtest (Phase 4, costs
    Phase 2) : skip → seal the OOS partition (Phase 5 lockbox) → open real OOS only on
    ``final_run`` else use the decoy → OOS metrics (Phases 1/3) → verdict (Phase 5). Then
    a portfolio verdict (§4.2), a signed report (Phase 5), and the config-freeze ledger
    is updated (Phase 5 §3.2). Writes ONLY the report + lockbox/freeze files under
    ``out_dir``; never ``runtime/``.
    """
    if bars_dir is None:
        bars_dir = repo_root / "data" / "bars" / "1d"
    if generated_at is None:
        generated_at = _now_iso()
    if code_commit is None:
        code_commit = current_code_commit(repo_root)

    lockbox_root = out_dir / "lockbox"
    freeze_dir = out_dir / "freeze"

    # --- Config freeze (SSOT §3.2): freeze once, then reuse/amend on re-runs. ---- #
    thresholds_dict = thresholds.to_dict()
    cost_dict = cost_config.to_dict()
    ledger = FreezeLedger(freeze_dir / "config_freeze_ledger.json")
    state: FreezeState
    loaded = ledger.load()
    if loaded is None:
        state = ledger.freeze(thresholds_dict, cost_dict, timestamp=generated_at)
    elif loaded.frozen.verify(thresholds_dict, cost_dict):
        state = loaded
    else:
        # Config changed vs the frozen one → amend (penalised only after a prior FAIL).
        state = ledger.amend(thresholds_dict, cost_dict, timestamp=generated_at)

    # Punitive N base = trial-multiple × visible survivors (the REPLAYABLE heads); the
    # ledger's trial_count (post-FAIL goalpost moves) is added on top (SSOT §3.3/§3.2).
    survivors = sum(
        1
        for s in catalog
        if classify_head_file(s.head, repo_root / s.source_rel).status is ParityStatus.REPLAYABLE
    )
    base_n = punitive_trial_count(max(1, survivors), multiple=trial_multiple)
    n_effective_trials = deflation_trials(base_n, state)

    head_runs: list[_HeadRun] = []
    involved_audits: list[SymbolAudit] = []

    for spec in catalog:
        parity = classify_head_file(spec.head, repo_root / spec.source_rel)
        if parity.status is not ParityStatus.REPLAYABLE:
            metrics = HeadMetrics(
                head=spec.head,
                parity_status=parity.status.value,
                replayable=False,
                data_quality_status="UNKNOWN",
                oos_access_count=0,
                n_oos_trades=0,
                n_walk_forward_windows=0,
                n_regimes_in_oos=0,
                deflated_sharpe_worst=None,
                cost_adj_cagr=None,
                worst_quantile_ruin=None,
                regimes_with_edge=0,
                regime_scoped_sizing=False,
                final_run=final_run,
                oos_source="none",
            )
            head_runs.append(
                _HeadRun(
                    spec=spec,
                    parity_status=parity.status.value,
                    replayable=False,
                    symbol_audit=None,
                    backtest_summary={"backtested": False, "skip_reason": parity.status.value},
                    metrics=metrics,
                    verdict=decide_verdict(metrics, thresholds),
                    oos_source="none",
                    oos_access_count=0,
                    oos_returns=[],
                    periods_per_year=252.0,
                )
            )
            continue

        # --- REPLAYABLE: audit + load bars, then replay (Phase 4). --------------- #
        bar_path = bars_dir / f"{spec.symbol}.json"
        symbol_audit = audit_bar_file(bar_path)
        involved_audits.append(symbol_audit)
        bars = load_bars_file(str(bar_path))  # Phase-0 audit gate raises on FAIL
        result = run_backtest(
            spec.head,
            spec.decide,
            bars,
            parity=parity,
            label_horizon=spec.label_horizon,
            symbol=spec.symbol,
            execution_spec=spec.execution_spec,
            cost_config=cost_config,
        )
        ppy = result.tracks[0].periods_per_year if result.tracks else 252.0

        # --- Seal the OOS partition at split time (Phase 5 §3.1). ---------------- #
        oos_real = _oos_returns_from_result(result)
        lockbox = OOSLockbox(lockbox_root / spec.head)
        lockbox.seal(
            oos_real,
            config_hash=state.frozen.config_hash,
            code_commit=code_commit,
            timestamp=generated_at,
        )

        # --- Route the OOS series: sealed (final) vs decoy (dev). ---------------- #
        if final_run:
            chosen_oos = lockbox.open_oos(
                oos_real,
                final_run=True,
                config_hash=state.frozen.config_hash,
                code_commit=code_commit,
                timestamp=generated_at,
            )
            oos_source = "sealed_oos"
        else:
            chosen_oos = lockbox.decoy_oos(len(oos_real))
            oos_source = "decoy"
        access_count = lockbox.access_count()

        om = _oos_metrics(
            chosen_oos, periods_per_year=ppy, n_effective_trials=n_effective_trials
        )
        metrics = HeadMetrics(
            head=spec.head,
            parity_status=parity.status.value,
            replayable=True,
            data_quality_status=symbol_audit.status.value,
            oos_access_count=access_count,
            n_oos_trades=int(om["n_oos_trades"]),
            n_walk_forward_windows=result.n_walk_forward_windows,
            n_regimes_in_oos=int(om["n_regimes_in_oos"]),
            deflated_sharpe_worst=om["deflated_sharpe_worst"],
            cost_adj_cagr=om["cost_adj_cagr"],
            worst_quantile_ruin=om["worst_quantile_ruin"],
            regimes_with_edge=int(om["regimes_with_edge"]),
            regime_scoped_sizing=False,
            final_run=final_run,
            oos_source=oos_source,
        )
        # Report-embedded backtest summary: in-sample tracks + counts ONLY. The isolated
        # OOS track and raw trade prices are NEVER embedded. On a NON-final (decoy) run
        # the "overall" track is ALSO withheld, because it is scored over the full trade
        # set (which includes the real OOS trades) and would otherwise leak a blended OOS
        # figure into a dry-run artifact; only the pure in-sample train/val tracks remain.
        # On a --final-run the OOS is legitimately opened, so "overall" is included.
        visible_names = ("overall", "train", "val") if final_run else ("train", "val")
        visible_tracks = [t.to_dict() for t in result.tracks if t.name in visible_names]
        backtest_summary = {
            "backtested": result.backtested,
            "symbol": result.symbol,
            "n_bars": result.n_bars,
            "label_horizon": result.label_horizon,
            "n_decisions": result.n_decisions,
            "n_signals": result.n_signals,
            "ambiguous_exit_count": result.ambiguous_exit_count,
            "excluded_trade_count": result.excluded_trade_count,
            "n_walk_forward_windows": result.n_walk_forward_windows,
            "span_years": result.span_years,
            "partition": result.partition,
            "execution_spec": result.execution_spec,
            "cost_config_echo": result.cost_config_echo,
            "visible_tracks": visible_tracks,
            "oos_metrics": {"source": oos_source, **om},
            "warnings": list(result.warnings),
        }
        head_runs.append(
            _HeadRun(
                spec=spec,
                parity_status=parity.status.value,
                replayable=True,
                symbol_audit=symbol_audit,
                backtest_summary=backtest_summary,
                metrics=metrics,
                verdict=decide_verdict(metrics, thresholds),
                oos_source=oos_source,
                oos_access_count=access_count,
                oos_returns=list(chosen_oos),
                periods_per_year=ppy,
            )
        )

    # --- Portfolio track (SSOT §4.2, S5): equal-weight harness-side allocator. ---- #
    portfolio_verdict, portfolio_block = _portfolio(
        head_runs, thresholds, final_run=final_run, n_effective_trials=n_effective_trials
    )

    # --- Assemble sections + sign (Phase 5 report). ------------------------------ #
    parity_map = [
        classify_head_file(s.head, repo_root / s.source_rel).to_dict() for s in catalog
    ]
    parity_table = render_parity_summary(
        classify_head_file(s.head, repo_root / s.source_rel) for s in catalog
    )
    dq_worst = _worst_status(involved_audits)
    heads_section = [
        {
            "head": hr.spec.head,
            "symbol": hr.spec.symbol,
            "metrics": hr.metrics.to_dict(),
            "verdict": hr.verdict.to_dict(),
            "backtest_summary": hr.backtest_summary,
        }
        for hr in head_runs
    ]
    counts: dict[str, int] = {v.value: 0 for v in Verdict}
    for hr in head_runs:
        counts[hr.verdict.verdict.value] += 1

    oos_first = next((hr for hr in head_runs if hr.replayable), None)
    oos_section = _oos_section(head_runs, lockbox_root, final_run=final_run)

    report = build_report(
        generated_at=generated_at,
        stage="historical",
        final_run=final_run,
        code_commit=code_commit,
        data_quality={
            "worst_status": dq_worst,
            "symbols": [a.to_dict() for a in involved_audits],
            "involved_symbols": [a.symbol for a in involved_audits],
        },
        parity_map=parity_map,
        parity_table=parity_table,
        heads=heads_section,
        portfolio=portfolio_block,
        frozen_config=state.to_dict(),
        oos=oos_section,
        thresholds=thresholds.to_dict(),
        verdict_summary={
            "counts": counts,
            "portfolio_verdict": portfolio_verdict.label,
        },
        extra_notes=[
            "Stage 1 (historical backtest). Stage 2 (real trade-log) plugs into the same "
            "scoring spine later (SSOT §1.3).",
        ],
    )
    signed = sign_report(report)

    # --- Record this run's headline verdict into the freeze ledger (§3.2). ------- #
    ledger.record_verdict(portfolio_verdict.verdict.value)

    _ = oos_first  # (kept for clarity; oos_section already summarises the first head)
    return signed


def _portfolio(
    head_runs: Sequence[_HeadRun],
    thresholds: VerdictThresholds,
    *,
    final_run: bool,
    n_effective_trials: int,
) -> tuple[VerdictResult, dict[str, Any]]:
    """Build the portfolio verdict (§4.2) from an equal-weight replay of REPLAYABLE heads.

    The 50/30/20 sleeve allocator cannot be imported (isolation), so the portfolio track
    is a harness-side equal-weight pool of the REPLAYABLE heads' OOS return streams —
    honestly flagged as a reconstruction. Portfolio PASS additionally requires a stated
    fraction of capital in individually-surviving heads (S5), enforced in
    :func:`chad.validation.verdict.decide_portfolio_verdict`.
    """
    replayable = [hr for hr in head_runs if hr.replayable]
    total_heads = len(replayable)
    surviving = sum(1 for hr in replayable if hr.verdict.verdict is Verdict.PASS)
    capital_fraction = (surviving / total_heads) if total_heads else 0.0

    pooled: list[float] = []
    for hr in replayable:
        pooled.extend(hr.oos_returns)
    ppy = replayable[0].periods_per_year if replayable else 252.0
    access_count = max((hr.oos_access_count for hr in replayable), default=0)
    dq_worst = _worst_status([hr.symbol_audit for hr in replayable if hr.symbol_audit])
    n_wf = min((hr.metrics.n_walk_forward_windows for hr in replayable), default=0)
    oos_source = replayable[0].oos_source if replayable else "none"

    om = _oos_metrics(pooled, periods_per_year=ppy, n_effective_trials=n_effective_trials)
    portfolio_metrics = HeadMetrics(
        head="portfolio",
        parity_status="PORTFOLIO",
        replayable=bool(replayable),
        data_quality_status=dq_worst,
        oos_access_count=access_count,
        n_oos_trades=int(om["n_oos_trades"]),
        n_walk_forward_windows=n_wf,
        n_regimes_in_oos=int(om["n_regimes_in_oos"]),
        deflated_sharpe_worst=om["deflated_sharpe_worst"],
        cost_adj_cagr=om["cost_adj_cagr"],
        worst_quantile_ruin=om["worst_quantile_ruin"],
        regimes_with_edge=int(om["regimes_with_edge"]),
        regime_scoped_sizing=False,
        final_run=final_run,
        oos_source=oos_source,
    )
    pm = PortfolioMetrics(
        portfolio=portfolio_metrics,
        surviving_heads=surviving,
        total_heads=total_heads,
        capital_fraction_in_surviving_heads=capital_fraction,
    )
    pv = decide_portfolio_verdict(pm, thresholds)
    block = {
        "verdict": pv.to_dict(),
        "metrics": portfolio_metrics.to_dict(),
        "surviving_heads": surviving,
        "total_heads": total_heads,
        "capital_fraction_in_surviving_heads": capital_fraction,
        "allocator_note": (
            "harness-side equal-weight reconstruction of the 50/30/20 sleeve allocator "
            "(the live allocator cannot be imported under §1.2 isolation)"
        ),
    }
    return pv, block


def _oos_section(
    head_runs: Sequence[_HeadRun], lockbox_root: Path, *, final_run: bool
) -> dict[str, Any]:
    """The report's OOS-discipline section (access count, seal, integrity) for the run.

    Each head has its OWN sealed OOS partition/lockbox, so contamination is per-box: a
    legitimate ``--final-run`` opens each box exactly once. ``access_count`` is the MAX
    per-head count (the number that matters for the > 1 ⇒ CONTAMINATED rule); the sum of
    opens across boxes is reported separately as ``total_access_events`` for transparency
    (summing independent boxes would falsely read as contamination on a clean multi-head
    run).
    """
    replayable = [hr for hr in head_runs if hr.replayable]
    per_head_counts = {hr.spec.head: hr.oos_access_count for hr in replayable}
    counts = list(per_head_counts.values())
    max_access = max(counts, default=0)
    total_events = sum(counts)
    source = replayable[0].oos_source if replayable else "none"
    seal_dict: Optional[dict[str, Any]] = None
    integrity_ok = True
    sealed_any = False
    for hr in replayable:
        box = OOSLockbox(lockbox_root / hr.spec.head)
        if box.is_sealed():
            sealed_any = True
            if seal_dict is None:
                seal_dict = box.load_seal().to_dict()
        if not box.verify_log_integrity():
            integrity_ok = False
    return {
        "access_count": max_access,          # per-box max; > 1 ⇒ a box opened > once
        "total_access_events": total_events,  # sum across all heads' boxes (transparency)
        "per_head_access_counts": per_head_counts,
        "source": source,
        "final_run": final_run,
        "sealed": sealed_any,
        "seal": seal_dict,
        "log_integrity_ok": integrity_ok,
        "contaminated": any(c > 1 for c in counts),
    }


def _worst_status(audits: Sequence[Optional[SymbolAudit]]) -> str:
    """Worst Phase-0 status across involved symbols (CLEAN < WARN < FAIL)."""
    rank = {"CLEAN": 0, "WARN": 1, "FAIL": 2}
    worst = "CLEAN"
    for a in audits:
        if a is None:
            continue
        if rank.get(a.status.value, 0) > rank.get(worst, 0):
            worst = a.status.value
    return worst


# --------------------------------------------------------------------------- #
# Stage 2 — real trade-log validation (SSOT §1.3 / Part 6, Phase 6).
# --------------------------------------------------------------------------- #
def _finite_or_none(value: Any) -> Any:
    """A real, finite number passes through; NaN/±inf/bool/None collapse to ``None``.

    Keeps every metric embedded in the (allow_nan=False) signed report finitely
    representable — an undefined metric is honestly ``None``, which the strict verdict
    treats as *not confirmed* (fails that check), never a fabricated pass.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _finitize(obj: Any) -> Any:
    """Recursively replace every non-finite float (NaN/±inf) in a JSON-ish structure with
    ``None`` so the whole embedded metrics block is safe for ``allow_nan=False`` signing.

    Covers NESTED dicts (e.g. the ``seed_sweep`` sub-dict), lists, and scalars — not just the
    top-level metric keys — so a future upstream NaN cannot slip into the signed report via a
    nested value. Bools and ints pass through unchanged (only float NaN/inf collapse)."""
    if isinstance(obj, dict):
        return {k: _finitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# --------------------------------------------------------------------------- #
# W3A-4 (D5) — day-bucketed walk-forward over the real trade-log time axis.
# --------------------------------------------------------------------------- #
# Per-strategy label horizon in TRADING DAYS, CITED from real observed holds (not guessed,
# per D5): gamma median hold = 70.0h ≈ 2.9 trading days (measured over n=74 closed_trade.v1
# laps, 2026-07-22) → 3. Strategies with no closed-lap history yet fall back to a conservative
# default of 2. Static now; a horizon derived from live holds is the "derived later" refinement.
_STRATEGY_LABEL_HORIZON_DAYS: dict[str, int] = {"gamma": 3}
_DEFAULT_LABEL_HORIZON_DAYS: int = 2

# Walk-forward sizing over the exit-DAY axis (units = distinct trading days): a 5-day train
# block and a 1-day forward test block. Reaching W_min=6 windows therefore needs many
# separable trading days — a corpus concentrated in a few bursts honestly yields 0 windows.
_WF_TRAIN_DAYS: int = 5
_WF_TEST_DAYS: int = 1


def _exit_date(t: AdmittedTrade) -> Optional[str]:
    """The YYYY-MM-DD realization date of an admitted lap (exit, fallback entry)."""
    for src in (t.provenance.get("exit_time_utc"), t.provenance.get("entry_time_utc")):
        if isinstance(src, str) and len(src) >= 10 and src[4] == "-" and src[7] == "-":
            return src[:10]
    return None


def _stage2_walk_forward_windows(admitted: Sequence[AdmittedTrade], strategy: str) -> int:
    """Count purged/embargoed walk-forward windows over the DISTINCT exit-day axis (W3A-4).

    A trade log's laps cluster into bursts (gamma's 74 laps exit on only 2 distinct days), so
    windowing over trade-INDICES would let a handful of time-points masquerade as many
    "independent" windows — a leakage cheat. Windowing over DISTINCT EXIT DAYS makes
    ">= W_min windows" mean ">= W_min separable forward time-periods", which clustered trades
    cannot fabricate. The per-strategy ``label_horizon`` (days, D5) purges correlated adjacent
    days. NOTE: this is the temporal-SEPARABILITY gate feeding the §4.3 W_min minimum; genuine
    per-window cross-consistency re-scoring is a documented future refinement, so a window
    count >= W_min is necessary-but-not-sufficient and never on its own a pass.
    """
    distinct_days = sorted({d for d in (_exit_date(t) for t in admitted) if d})
    n_days = len(distinct_days)
    if n_days < 1:
        return 0
    h = _STRATEGY_LABEL_HORIZON_DAYS.get(strategy, _DEFAULT_LABEL_HORIZON_DAYS)
    windows = generate_walk_forward(
        n_days, train_size=_WF_TRAIN_DAYS, test_size=_WF_TEST_DAYS, label_horizon=h
    )
    return len(windows)


def _stage2_net_returns(
    admitted: Sequence[AdmittedTrade], cost_config: CostConfig
) -> tuple[list[float], dict[str, Any]]:
    """Net per-trade returns for admitted real fills, via the IDENTICAL cost path (S4).

    Each admitted trade is mapped onto :meth:`Trade.from_fill` and charged by
    :func:`apply_costs` — the same commission + half-spread + slippage haircut synthetic
    Stage-1 trades get — then ``return_i = (gross_pnl_i - total_cost_i) / notional_i``. No
    scoring logic is duplicated here (the spine consumes the returns). A trade whose notional
    is non-positive or whose net return is non-finite is skipped and counted (never crashes).
    """
    returns: list[float] = []
    gross_sum = net_sum = cost_sum = 0.0
    skipped = 0
    for t in admitted:
        try:
            breakdown = apply_costs(Trade.from_fill(t.to_fill_mapping()), cost_config)
        except (ValueError, KeyError):
            skipped += 1
            continue
        # apply_costs always sets net_pnl when a gross_pnl is supplied (AdmittedTrade always
        # carries one); the fallback is belt-and-suspenders for a None net.
        net = breakdown.net_pnl if breakdown.net_pnl is not None else (t.gross_pnl - breakdown.total_cost)
        if not (t.notional > 0.0) or not math.isfinite(net) or not math.isfinite(net / t.notional):
            skipped += 1
            continue
        returns.append(net / t.notional)
        gross_sum += t.gross_pnl
        net_sum += net
        cost_sum += breakdown.total_cost
    summary = {
        "n_trades": len(returns),
        "n_skipped_uncostable": skipped,
        "gross_pnl": gross_sum,
        "net_pnl": net_sum,
        "total_cost": cost_sum,
    }
    return returns, summary


def _stage2_head_metrics(
    head: str,
    returns: Sequence[float],
    *,
    n_effective_trials: int,
    final_run: bool,
    n_walk_forward_windows: int = 0,
) -> tuple[HeadMetrics, dict[str, Any]]:
    """Build a head's :class:`HeadMetrics` from a real-fill net-return series (DRY-reuse).

    Reuses the Phase-1/3 :func:`_oos_metrics` spine so a live-trade-log head is scored by the
    IDENTICAL machinery as a Stage-1 head. ``n_walk_forward_windows`` (W3A-4) is now derived
    from the DISTINCT exit-day axis by :func:`_stage2_walk_forward_windows` — clustered laps
    (few separable days) honestly yield few/zero windows, which with ``W_min = 6`` keeps the
    verdict at ``INSUFFICIENT_DATA`` until the soak spans enough time (SSOT Part 0), never a
    fabricated pass. ``oos_source`` is ``"live_trade_log"``; there is no sealed OOS box for a
    live log, so ``oos_access_count = 0`` (never CONTAMINATED by construction).
    """
    # Finitize the WHOLE metrics block (incl. the nested seed_sweep sub-dict) so no non-finite
    # float can reach the allow_nan=False report signing via any nested value.
    om = _finitize(_oos_metrics(returns, periods_per_year=252.0, n_effective_trials=n_effective_trials))
    metrics = HeadMetrics(
        head=head,
        parity_status="LIVE_TRADE_LOG",
        replayable=True,          # these ARE real executed trades — scored, not skipped
        data_quality_status="CLEAN",  # Phase-0 audits BAR data, not live fills (noted in report)
        oos_access_count=0,
        n_oos_trades=int(om["n_oos_trades"]),
        n_walk_forward_windows=n_walk_forward_windows,
        n_regimes_in_oos=int(om["n_regimes_in_oos"]),
        deflated_sharpe_worst=om["deflated_sharpe_worst"],
        cost_adj_cagr=om["cost_adj_cagr"],
        worst_quantile_ruin=om["worst_quantile_ruin"],
        regimes_with_edge=int(om["regimes_with_edge"]),
        regime_scoped_sizing=False,
        final_run=final_run,
        oos_source="live_trade_log",
    )
    return metrics, om


def run_stage2_trade_log(
    *,
    repo_root: Path,
    out_dir: Path,
    since: Optional[str] = None,
    until: Optional[str] = None,
    trades_dir: Optional[Path] = None,
    final_run: bool = False,
    thresholds: VerdictThresholds = DEFAULT_THRESHOLDS,
    cost_config: CostConfig = DEFAULT_COST_CONFIG,
    generated_at: Optional[str] = None,
    code_commit: Optional[str] = None,
    trial_multiple: float = DEFAULT_TRIAL_MULTIPLE,
    include_futures: bool = False,
    verify_chain: bool = False,
    scr_crosscheck: bool = False,
    allow_era_pooling: bool = False,
    runtime_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run the Stage-2 real-trade-log validation end-to-end and return the SIGNED report.

    Pipeline: :func:`chad.validation.trade_log_adapter.run_adapter` (fail-closed trust gate) →
    per-strategy net returns via the IDENTICAL cost path (S4) → :func:`_oos_metrics` (Phases
    1/3 spine) → :func:`decide_verdict` per head → :func:`decide_portfolio_verdict` on the
    pooled series → signed report (Phase 5 report writer). It NEVER seals/opens an OOS box
    (there is no sealed OOS for a live log — the lockbox/config-freeze machinery of Phases 0-5
    is untouched), and NEVER writes ``runtime/`` or ``ready_for_live`` (SSOT §1.2 / Part 0).
    The adapter's ndjson + manifest are written under ``out_dir/stage2``.
    """
    if trades_dir is None:
        trades_dir = repo_root / "data" / "trades"
    if generated_at is None:
        generated_at = _now_iso()
    if code_commit is None:
        code_commit = current_code_commit(repo_root)
    if runtime_dir is None:
        runtime_dir = repo_root / "runtime"
    # D6/D2: the real Stage-2 run verifies the ledger chain and reconciles vs SCR (both
    # fail-loud). Library callers keep them off by default (synthetic fixtures); the CLI turns
    # them on. scr_crosscheck reads runtime/scr_state.json read-only (never written).
    scr_state_path = (runtime_dir / "scr_state.json") if scr_crosscheck else None

    result = run_adapter(
        trades_dir=trades_dir,
        since=since,
        until=until,
        out_dir=out_dir / "stage2",
        generated_at=generated_at,
        runtime_dir=runtime_dir,          # W3A-5: honour the operator quarantine manifest (was missing)
        include_futures=include_futures,  # D2: futures excluded by default (Bug-B)
        verify_chain=verify_chain,        # D6: fail-loud on a broken ledger chain
        scr_state_path=scr_state_path,    # D2: read-only SCR reconciliation cross-check
    )

    # D4: group admitted laps by (strategy, ERA) — a head NEVER pools the pre/post exit-overlay
    # eras (different data-generating processes). The era is stamped in every head name so a
    # cross-era pooled count can never be mistaken for one population.
    by_head: dict[str, list[AdmittedTrade]] = {}
    for t in result.admitted:
        era = era_of(_exit_date(t))
        by_head.setdefault(f"{t.strategy}|{era}", []).append(t)

    n_heads = max(1, len(by_head))
    n_effective_trials = punitive_trial_count(n_heads, multiple=trial_multiple)

    heads_section: list[dict[str, Any]] = []
    head_verdicts: list[VerdictResult] = []
    counts: dict[str, int] = {v.value: 0 for v in Verdict}
    pooled_by_era: dict[str, list[float]] = {}
    for head_key in sorted(by_head):
        admitted = by_head[head_key]
        strategy, era = head_key.rsplit("|", 1)
        returns, trade_summary = _stage2_net_returns(admitted, cost_config)
        pooled_by_era.setdefault(era, []).extend(returns)
        n_wf = _stage2_walk_forward_windows(admitted, strategy)  # W3A-4: exit-day windows
        metrics, om = _stage2_head_metrics(
            head_key, returns, n_effective_trials=n_effective_trials, final_run=final_run,
            n_walk_forward_windows=n_wf,
        )
        verdict = decide_verdict(metrics, thresholds)
        head_verdicts.append(verdict)
        counts[verdict.verdict.value] += 1
        heads_section.append(
            {
                "head": head_key,
                "strategy": strategy,
                "era": era,
                "symbol": "(live fills)",
                "metrics": metrics.to_dict(),
                "verdict": verdict.to_dict(),
                "trade_summary": {
                    **trade_summary,
                    "n_admitted": len(admitted),
                    "oos_metrics": {"source": "live_trade_log", **om},
                },
            }
        )

    # D4: the HEADLINE portfolio is the POST_OVERLAY era only (the genuine engine-driven
    # population). Pooling across eras is a heterogeneous mix and is offered ONLY as an
    # explicitly-labeled sensitivity view (allow_era_pooling), never the headline.
    if allow_era_pooling:
        pooled = [r for rs in pooled_by_era.values() for r in rs]
        portfolio_era = "POOLED_HETEROGENEOUS_POPULATION"
        portfolio_admitted = list(result.admitted)
    else:
        portfolio_era = "POST_OVERLAY"
        pooled = list(pooled_by_era.get("POST_OVERLAY", []))
        portfolio_admitted = [t for t in result.admitted if era_of(_exit_date(t)) == "POST_OVERLAY"]

    surviving = sum(1 for v in head_verdicts if v.verdict is Verdict.PASS)
    total_heads = len(head_verdicts)
    capital_fraction = (surviving / total_heads) if total_heads else 0.0
    portfolio_wf = _stage2_walk_forward_windows(portfolio_admitted, "portfolio")
    portfolio_metrics, _pom = _stage2_head_metrics(
        "portfolio", pooled, n_effective_trials=n_effective_trials, final_run=final_run,
        n_walk_forward_windows=portfolio_wf,
    )
    pm = PortfolioMetrics(
        portfolio=portfolio_metrics,
        surviving_heads=surviving,
        total_heads=total_heads,
        capital_fraction_in_surviving_heads=capital_fraction,
    )
    portfolio_verdict = decide_portfolio_verdict(pm, thresholds)
    portfolio_block = {
        "verdict": portfolio_verdict.to_dict(),
        "metrics": portfolio_metrics.to_dict(),
        "portfolio_era": portfolio_era,
        "surviving_heads": surviving,
        "total_heads": total_heads,
        "capital_fraction_in_surviving_heads": capital_fraction,
        "allocator_note": (
            "Stage-2 portfolio track is an equal-weight pool of the admitted heads' real "
            "net-return streams (no live allocator imported under §1.2 isolation). Headline "
            f"portfolio_era = {portfolio_era!r}: by default only POST_OVERLAY laps (the "
            "engine-driven population) form the headline; --allow-era-pooling mixes eras and "
            "is stamped POOLED_HETEROGENEOUS_POPULATION as a sensitivity view only (D4)."
        ),
    }

    oos_section = {
        "access_count": 0,
        "total_access_events": 0,
        "per_head_access_counts": {},
        "source": "live_trade_log",
        "final_run": final_run,
        "sealed": False,
        "seal": None,
        "log_integrity_ok": True,
        "contaminated": False,
        "note": (
            "No sealed OOS partition exists for a live trade log; the OOS lockbox is not used "
            "in Stage 2 (SSOT §3.1 applies to the historical backtest). Admission is governed "
            "by the fail-closed trust gate instead — see data_quality.stage2_adapter_manifest."
        ),
    }

    report = build_report(
        generated_at=generated_at,
        stage="stage2_trade_log",
        final_run=final_run,
        code_commit=code_commit,
        data_quality={
            "worst_status": "N/A (Stage-2 live fills; Phase-0 audits BAR data, not fills)",
            "symbols": [],
            "involved_symbols": [],
            "stage2_adapter_manifest": result.manifest.to_dict(),
        },
        parity_map=[],
        parity_table=(
            "n/a — Stage-2 heads are REAL executed trades (not replayed historical logic); "
            "feature parity is a Stage-1 concept."
        ),
        heads=heads_section,
        portfolio=portfolio_block,
        frozen_config={
            "note": (
                "Stage 2 echoes the frozen thresholds + cost config for transparency; it does "
                "NOT seal an OOS box or touch the Phase-5 config-freeze ledger."
            ),
            "thresholds": thresholds.to_dict(),
            "cost_config": cost_config.to_dict(),
        },
        oos=oos_section,
        thresholds=thresholds.to_dict(),
        verdict_summary={
            "counts": counts,
            "portfolio_verdict": portfolio_verdict.label,
        },
        extra_notes=[
            "EVIDENCE — NOT AN AUTHORIZATION TO TRADE LIVE. This report is evidence for a human "
            "decision only; the harness never flips ready_for_live and a pass is only ever "
            "'PASS (candidate)' (D7 / SSOT Part 0).",
            "Stage 2 (real trade-log validation, SSOT §1.3 / Part 6). Real paper fills pass "
            "the fail-closed trust gate, get the IDENTICAL S4 cost haircut, and are scored by "
            "the SAME spine + verdict as Stage 1 — only the input adapter differs.",
            "n_walk_forward_windows is derived (W3A-4) over the DISTINCT exit-day axis with a "
            "per-strategy label horizon cited from real holds (D5): clustered laps span few "
            "separable days → few/zero windows → INSUFFICIENT_DATA until the soak spans enough "
            "time (never a fabricated pass). The window count is the temporal-separability gate "
            "for W_min; genuine per-window cross-consistency re-scoring is a documented future "
            "refinement, so >= W_min windows is necessary-but-not-sufficient. The machine never "
            "flips ready_for_live (Part 0).",
            "D4: heads are split by (strategy, ERA) at the frozen exit-overlay boundary "
            f"({EXIT_OVERLAY_BOUNDARY}); pre/post are DIFFERENT populations and are NEVER pooled "
            "into a headline. The headline portfolio is POST_OVERLAY only; --allow-era-pooling "
            "yields a POOLED_HETEROGENEOUS_POPULATION sensitivity view, never the headline. "
            "Every report carries the adapter's era_partition + scr_reconciliation blocks.",
        ],
        # Stage 2 has neither a sealed OOS nor replayed heads — override the Stage-1 provenance
        # so the SIGNED artifact does not carry a hash-seal / replay claim that is false here.
        provenance_overrides={
            "oos_discipline": (
                "N/A for Stage 2: a live trade log has no sealed OOS partition (SSOT §3.1 "
                "applies to the Stage-1 backtest). Admission is governed by the fail-closed "
                "trust gate — see data_quality.stage2_adapter_manifest; oos.contaminated is "
                "false by construction (access_count = 0)."
            ),
            "replay_reconstruction": (
                "N/A for Stage 2: heads are REAL executed paper fills, not replayed historical "
                "logic; no strategy module is imported and no reconstruction is asserted."
            ),
        },
    )
    return sign_report(report)


# --------------------------------------------------------------------------- #
# CLI entry point + summary printing.
# --------------------------------------------------------------------------- #
def _print_summary(signed: dict[str, Any], json_path: Path) -> None:
    """Print a concise, human-readable verdict summary to stdout (deterministic in input)."""
    print("=" * 72)
    print("CHAD Edge-Validation Harness — verdict summary (SSOT Part 4)")
    print("=" * 72)
    # D7 (W3A-6): Stage-2 output LEADS with the evidence banner — never mistakable for a GO.
    if str(signed.get("stage") or "") == "stage2_trade_log":
        print("EVIDENCE — NOT AN AUTHORIZATION TO TRADE LIVE. The harness reports evidence; the")
        print("operator alone decides. The machine never flips ready_for_live (SSOT Part 0).")
        print("-" * 72)
    fr = bool(signed.get("final_run"))
    oos = signed.get("oos", {})
    fc = signed.get("config_frozen", {})
    frozen = fc.get("frozen", {}) if isinstance(fc, dict) else {}
    stage = str(signed.get("stage") or "")
    if stage == "stage2_trade_log":
        note = "(real trade-log; no sealed OOS — admission via fail-closed trust gate)"
    elif fr:
        note = ""
    else:
        note = "(decoy OOS — real OOS SEALED, not opened)"
    print(f"stage: {stage} | final_run: {fr} " + note)
    print(f"code_commit: {signed.get('code_commit')}")
    print(f"frozen config hash: {str(frozen.get('config_hash', '?'))[:16]}  "
          f"trial_count(deflation add-on): {fc.get('trial_count')}")
    print(
        f"OOS access count: {oos.get('access_count')}  source: {oos.get('source')}  "
        f"log_integrity_ok: {oos.get('log_integrity_ok')}  "
        f"contaminated: {oos.get('contaminated')}"
    )
    print("-" * 72)
    for h in signed.get("heads", []):
        m = h.get("metrics", {})
        v = h.get("verdict", {})
        print(f"HEAD {h.get('head')} [{m.get('parity_status')}] → {v.get('label')}")
        print(
            f"   OOS trades={m.get('n_oos_trades')}  WF windows={m.get('n_walk_forward_windows')}  "
            f"OOS regimes={m.get('n_regimes_in_oos')}  "
            f"DSR(worst)={m.get('deflated_sharpe_worst')}  "
            f"cost-adj CAGR={m.get('cost_adj_cagr')}  worst ruin={m.get('worst_quantile_ruin')}"
        )
        for r in v.get("reasons", []):
            print(f"     - {r}")
    print("-" * 72)
    vs = signed.get("verdict_summary", {})
    print(f"PORTFOLIO verdict: {vs.get('portfolio_verdict')}")
    counts = vs.get("counts", {})
    tally = "  ".join(f"{k}={counts[k]}" for k in sorted(counts) if counts[k])
    print(f"head verdict tally: {tally or '(none)'}")
    sig = signed.get("signature", {})
    print(f"signature: {sig.get('algo')}:{str(sig.get('content_sha256', ''))[:16]}… "
          f"(verifies: {verify_signature(signed)})")
    print(f"report artifact: {json_path}")
    print("=" * 72)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry: parse args, run the historical stage, print the verdict, write the report.

    Never writes ``runtime/`` or ``ready_for_live``. Exit code 0 on a completed run
    (regardless of the verdict — INSUFFICIENT_DATA is a successful, honest outcome), 2 on
    a usage/config error.
    """
    parser = argparse.ArgumentParser(
        prog="python -m chad.validation.cli",
        description="CHAD offline edge-validation harness (Stage 1: historical backtest).",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["historical", "stage2"],
        help="validation stage: 'historical' (Stage-1 backtest) or 'stage2' (real trade-log)",
    )
    parser.add_argument(
        "--final-run",
        action="store_true",
        help="OPEN and score the sealed OOS partition (one logged access). Default: "
             "run against train/val + a synthetic decoy OOS; the real OOS stays sealed. "
             "(Stage-1 only; Stage-2 has no sealed OOS.)",
    )
    parser.add_argument("--repo-root", default=None, help="repo root (default: inferred)")
    parser.add_argument("--bars-dir", default=None, help="daily-bar corpus dir (default: <repo>/data/bars/1d)")
    parser.add_argument("--out-dir", default=None, help="artifact output dir (default: <repo>/edge_reports)")
    parser.add_argument("--now", default=None, help="override the report timestamp (ISO 8601)")
    parser.add_argument("--code-commit", default=None, help="override the recorded code commit")
    # Stage-2 (real trade-log) inputs.
    parser.add_argument("--since", default=None, help="[stage2] earliest UTC date to admit (YYYY-MM-DD)")
    parser.add_argument("--until", default=None, help="[stage2] latest UTC date to admit (YYYY-MM-DD)")
    parser.add_argument("--trades-dir", default=None, help="[stage2] ledger dir (default: <repo>/data/trades)")
    parser.add_argument("--include-futures", action="store_true",
                        help="[stage2] admit futures rows (default excludes them as Bug-B-contaminated, D2)")
    parser.add_argument("--no-verify-chain", action="store_true",
                        help="[stage2] skip ledger hash-chain verification (default verifies + fails loud, D6)")
    parser.add_argument("--no-scr-crosscheck", action="store_true",
                        help="[stage2] skip the read-only SCR reconciliation cross-check (default on, D2)")
    parser.add_argument("--allow-era-pooling", action="store_true",
                        help="[stage2] pool pre/post exit-overlay eras as a labeled sensitivity view "
                             "(default: POST_OVERLAY headline only — eras are never pooled, D4)")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir).resolve() if args.out_dir else repo_root / "edge_reports"
    bars_dir = Path(args.bars_dir).resolve() if args.bars_dir else None

    for name in ("since", "until"):
        val = getattr(args, name)
        if val is not None and not re.match(r"^\d{4}-\d{2}-\d{2}$", val):
            print(f"error: --{name} must be YYYY-MM-DD, got {val!r}")
            return 2
    if args.since and args.until and args.since > args.until:
        print(f"error: --since ({args.since}) is after --until ({args.until})")
        return 2

    try:
        if args.stage == "stage2":
            trades_dir = Path(args.trades_dir).resolve() if args.trades_dir else None
            signed = run_stage2_trade_log(
                repo_root=repo_root,
                out_dir=out_dir,
                since=args.since,
                until=args.until,
                trades_dir=trades_dir,
                final_run=bool(args.final_run),
                generated_at=args.now,
                code_commit=args.code_commit,
                include_futures=bool(args.include_futures),
                verify_chain=not bool(args.no_verify_chain),      # D6: on by default for real runs
                scr_crosscheck=not bool(args.no_scr_crosscheck),  # D2: on by default for real runs
                allow_era_pooling=bool(args.allow_era_pooling),   # D4: off by default (never pool)
            )
        else:
            signed = run_stage_historical(
                repo_root=repo_root,
                out_dir=out_dir,
                final_run=bool(args.final_run),
                bars_dir=bars_dir,
                generated_at=args.now,
                code_commit=args.code_commit,
            )
    except (LedgerChainError, ScrCrosscheckError) as exc:
        # D6/D2 fail-loud: refuse to render a verdict from a tampered ledger or without SCR truth.
        print(f"error: {type(exc).__name__}: {exc}")
        return 2
    except (ValueError, OSError) as exc:
        print(f"error: {type(exc).__name__}: {exc}")
        return 2

    basename = report_basename(str(signed.get("generated_at")))
    json_path, _md_path = write_report(signed, out_dir, basename=basename)
    _print_summary(signed, json_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
