# BOX-034A — Canonical Equity Currency Unification (amends BOX-034)
Date: 2026-06-01 · Status: IN PROGRESS — Inc 2 + Inc 3 Step 1a/1b shipped & live-verified; Inc 3 Step 2 (warn-mode) shipped; enforce flip + Inc 4 remaining (see §9)
Related follow-ups: BOX-034B (kraken_equity single-writer), BOX-034C (snapshot oneshot watchdog).
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
- Currency-labeling convention (resolved 2026-06-01): every equity VALUE carries a sibling `<value>_currency` field — NOT a single file-level `currency`, which is ambiguous in multi-venue files (portfolio_snapshot holds ibkr/coinbase/kraken equities). Broker-sourced values also carry a `<value>_currency_ok` health flag. Canonical names: `ibkr_equity_currency` + `ibkr_equity_currency_ok` (portfolio_snapshot, shipped Inc 2), `total_equity_currency` (dynamic_caps), `account_equity_currency` (pnl_state). Every reader asserts the relevant `<value>_currency` == CHAD_BASE_CURRENCY (CAD).

## 4. Consumers
dynamic_caps.total_equity, pnl_state.account_equity, portfolio_risk_cap, portfolio_var, drawdown_guard, profit_lock, position sizing — all derive from the single CAD source; none apply FX.

## 5. Reconciliation test (rewrite of BOX-034 §4a)
- Compare pnl_state.account_equity vs portfolio_snapshot total in CAD, within max(1.0 CAD, 0.05% of total).
- Assert the relevant per-value currency fields (`account_equity_currency`, snapshot `ibkr_equity_currency`) == CAD, per the §3 convention.
- Bound timing skew: both samples within a defined freshness window (or derive pnl_state from the same snapshot atomically) so residual drift is sampling-lag only, never currency.

## 6. Acceptance criteria (PROVEN TO BIND, not "code exists")
- grep proof: exactly one service writes ibkr_equity.
- portfolio_snapshot / dynamic_caps / pnl_state all carry their per-value currency tags == CAD (`ibkr_equity_currency`, `total_equity_currency`, `account_equity_currency`).
- Reconciliation test passes across a full 300s window (multiple live samples), not one lucky read.
- portfolio_risk_cap shows no CAD<->USD oscillation across a 10-min observation.
- Fail-closed test: NetLiquidation currency tag != CAD -> collector refuses the bare write.

## 7. Out of scope (tracked separately)
USD display reporting (cosmetic); the `manual`/`config_default` seeded positions (Bug B / position-truth); L1-L6 red-team items.

## 8. Implementation gate
Risk mutation: NO code until operator GO on this spec. Then Channel 2 implements with tests; verify §6 on the live box across a full timer window before commit.
- Reconciliation (2026-06-03): the shipped increments to date are NON-MUTATING — additive currency-tag writers (Inc 2, Inc 3 Step 1a/1b) and WARN-ONLY reader assertions (Inc 3 Step 2). None change any equity value, cap, or control flow. This gate continues to bind the still-pending RISK-MUTATING step: the Step 2 enforce flip (warn -> fail-closed), which gets its own operator GO + §6 live verification. See §9 for per-increment status.

## 9. Implementation status (progress tracking)
- Inc 2 — portfolio_snapshot per-value currency tags (`ibkr_equity_currency` / `ibkr_equity_currency_ok`): SHIPPED & live (recorded inline §3).
- Inc 3 Step 1a (commit b6d333f) + Step 1b (commit 40a9c55): writers additively tag `dynamic_caps.total_equity_currency`/`_ok` and `pnl_state.account_equity_currency`/`_ok`. LIVE + verified 2026-06-03 via orchestrator restart — both files read CAD with `_ok=true` (pnl_state auto-flipped on the next profit-lock oneshot). Ref restart PA `BOX-034A_orchestrator_restart_step1a_2026-06-02` (commit fc51872, Completion Record §10).
- Inc 3 Step 2 (commit d0f7a78): WARN-MODE currency assertions across the 5 equity consumers (greppable `CURRENCY_WARN_*` markers: DYNCAPS_PROVIDER, PNL_STATE, SNAPSHOT_LEG, TOTAL_EQUITY_OK_FALSE, RISK_CAP_UNVERIFIED, UNTAGGED_FALLBACK). Warn-only — no enforcement, no control-flow or equity-value change; 10 caplog tests (FIRES + SILENT per consumer). warn-mode LIVE-verified 2026-06-03 (5 consumers armed + silent across cycles; restart PA step2 COMPLETED, commit 76a88f7).
- REMAINING: (a) Step 2 enforce flip — warn -> fail-closed assertion at consumers; RISK-MUTATING, bound by the §8 gate (own operator GO + §6 live verification). (b) Inc 4 — reconciliation-test rewrite per §5.
