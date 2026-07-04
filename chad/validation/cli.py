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
# CLI entry point + summary printing.
# --------------------------------------------------------------------------- #
def _print_summary(signed: dict[str, Any], json_path: Path) -> None:
    """Print a concise, human-readable verdict summary to stdout (deterministic in input)."""
    print("=" * 72)
    print("CHAD Edge-Validation Harness — verdict summary (SSOT Part 4)")
    print("=" * 72)
    fr = bool(signed.get("final_run"))
    oos = signed.get("oos", {})
    fc = signed.get("config_frozen", {})
    frozen = fc.get("frozen", {}) if isinstance(fc, dict) else {}
    print(f"stage: {signed.get('stage')} | final_run: {fr} "
          + ("" if fr else "(decoy OOS — real OOS SEALED, not opened)"))
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
        choices=["historical"],
        help="validation stage to run (only 'historical' is implemented in Phase 5)",
    )
    parser.add_argument(
        "--final-run",
        action="store_true",
        help="OPEN and score the sealed OOS partition (one logged access). Default: "
             "run against train/val + a synthetic decoy OOS; the real OOS stays sealed.",
    )
    parser.add_argument("--repo-root", default=None, help="repo root (default: inferred)")
    parser.add_argument("--bars-dir", default=None, help="daily-bar corpus dir (default: <repo>/data/bars/1d)")
    parser.add_argument("--out-dir", default=None, help="artifact output dir (default: <repo>/edge_reports)")
    parser.add_argument("--now", default=None, help="override the report timestamp (ISO 8601)")
    parser.add_argument("--code-commit", default=None, help="override the recorded code commit")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir).resolve() if args.out_dir else repo_root / "edge_reports"
    bars_dir = Path(args.bars_dir).resolve() if args.bars_dir else None

    try:
        signed = run_stage_historical(
            repo_root=repo_root,
            out_dir=out_dir,
            final_run=bool(args.final_run),
            bars_dir=bars_dir,
            generated_at=args.now,
            code_commit=args.code_commit,
        )
    except (ValueError, OSError) as exc:
        print(f"error: {type(exc).__name__}: {exc}")
        return 2

    basename = report_basename(str(signed.get("generated_at")))
    json_path, _md_path = write_report(signed, out_dir, basename=basename)
    _print_summary(signed, json_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
