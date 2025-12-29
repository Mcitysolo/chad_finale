# PHASE 4 LOCKED — SCR / Trade Stats / Warmup

Date (UTC): 2025-12-28

Proof (authoritative):
- `python3 -m chad.analytics.trade_stats_engine` returns counts/metrics
- `python3 -m chad.analytics.shadow_confidence_router` prints SCR decision JSON

Current state at lock time:
- SCR state: WARMUP
- paper_only: true
- sizing_factor: 0.10
- effective_trades: 3 (warmup_min_trades=50)
- reasons:
  - Warmup: only 3 effective trades (< 50 required).
  - Current win_rate=0.000, sharpe_like=0.000, max_drawdown=0.00, total_pnl=0.00 (total_trades=65, effective_trades=3)

Guarantees:
- SCR decision is deterministic from trade ledgers + scr_config thresholds.
- Any future phase must not change these semantics without updating this lock + re-proofing.
