# BOX-034A — Canonical Equity Currency Unification (amends BOX-034)
Date: 2026-06-01 · Status: PENDING (spec; no implementation until operator GO)
Risk class: RISK MUTATION (equity feeds sizing, caps, VaR, drawdown, margin)

## 1. Root cause (confirmed live 2026-06-01)
Two services write the same key `ibkr_equity` to runtime/portfolio_snapshot.json in different currencies:
- chad-ibkr-collector (ibkr_portfolio_collector_v2, 120s): raw NetLiquidation, ignores currency tag -> CAD.
- chad-portfolio-snapshot (ops/portfolio_snapshot_publisher, 300s): applies Forex("USDCAD") -> USD.
Last writer wins; the file oscillates CAD<->USD. dynamic_caps.total_equity and pnl_state.account_equity inherit whichever currency is present (no FX in that leg). No system-wide base currency is declared.
Impact: portfolio_risk_cap = total_equity * daily_risk_fraction runs ~38% oversized whenever CAD is in play — a live risk-control defect and a contributor to the margin-maxed account (Bug B). Account DUK902770 base = CAD.

## 2. Decision
CHAD_BASE_CURRENCY = CAD (broker-native). All internal equity/cash/notional/sizing/caps/VaR/drawdown/margin math is CAD. USD appears ONLY in human-facing display fields suffixed `_usd_display`, never read by sizing or risk.

## 3. Single source of truth
- Exactly ONE writer owns portfolio_snapshot.ibkr_equity: the v2 collector (already emits CAD).
- v2 collector becomes currency-AWARE: read the NetLiquidation currency tag, assert == CAD, fail-closed (no bare ibkr_equity write) if it ever differs.
- ops/portfolio_snapshot_publisher MUST NOT write ibkr_equity. If USD display is wanted, it writes ibkr_equity_usd_display only (no risk path reads it).
- Every equity file (portfolio_snapshot, dynamic_caps, pnl_state) carries an explicit `currency` field; every reader asserts currency == CAD.

## 4. Consumers
dynamic_caps.total_equity, pnl_state.account_equity, portfolio_risk_cap, portfolio_var, drawdown_guard, profit_lock, position sizing — all derive from the single CAD source; none apply FX.

## 5. Reconciliation test (rewrite of BOX-034 §4a)
- Compare pnl_state.account_equity vs portfolio_snapshot total in CAD, within max(1.0 CAD, 0.05% of total).
- Assert both files carry currency == CAD.
- Bound timing skew: both samples within a defined freshness window (or derive pnl_state from the same snapshot atomically) so residual drift is sampling-lag only, never currency.

## 6. Acceptance criteria (PROVEN TO BIND, not "code exists")
- grep proof: exactly one service writes ibkr_equity.
- portfolio_snapshot / dynamic_caps / pnl_state all carry currency=CAD.
- Reconciliation test passes across a full 300s window (multiple live samples), not one lucky read.
- portfolio_risk_cap shows no CAD<->USD oscillation across a 10-min observation.
- Fail-closed test: NetLiquidation currency tag != CAD -> collector refuses the bare write.

## 7. Out of scope (tracked separately)
USD display reporting (cosmetic); the `manual`/`config_default` seeded positions (Bug B / position-truth); L1-L6 red-team items.

## 8. Implementation gate
Risk mutation: NO code until operator GO on this spec. Then Channel 2 implements with tests; verify §6 on the live box across a full timer window before commit.
