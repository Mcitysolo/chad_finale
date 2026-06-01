# BOX-034 — Canonical equity source policy (GAP-007)

> **AMENDMENT STATUS — added 2026-06-01.** This policy is **amended by**
> `ops/pending_actions/BOX-034A_canonical_equity_currency_unification_2026-06-01.md` (commit fe59de5).
> BOX-034A declares one base currency (**CAD**) and one writer for `ibkr_equity`, and
> root-causes the §4a drift as a **CAD↔USD dual-writer race** (confirmed live 2026-06-01) —
> not staleness or dropped legs.
>
> **Supersession is CONDITIONAL and PENDING.** BOX-034A §8 gates implementation on operator GO.
> Until that GO ships: sections **§3b, §3c, §4a, §5** below describe **current (defective)** behavior
> and remain what the live system + `test_canonical_equity_source.py` actually enforce.
> On BOX-034A implementation they are **superseded** by BOX-034A §3–§6.
> The "USD" assertions in §3b/§4a are NOT correct truth — they are the defect BOX-034A fixes.

Generated: 2026-05-20T00:38:00Z (CHAD Box 034)

## 1. Background

CHAD has multiple runtime files that carry an "equity" or "account value"
quantity. Several modules read from different files at different
freshnesses, which historically created the GAP-007 risk that
dashboard/operator truth could diverge from ops/risk artifacts without
the operator being told. Box 034 closes that gap by formally declaring
the canonical sources and the temporal-skew rules.

## 2. Equity-source inventory

> [BOX-034A] The two writers here are not benign interleaving — they write ibkr_equity in different currencies (collector→CAD, publisher→USD). This race is the root cause of the §4a drift.

| # | File | Schema | Key(s) | Writer | TTL | Refresh cadence |
|---|------|--------|--------|--------|-----|-----------------|
| 1 | `runtime/pnl_state.json` | `pnl_state.v1` | `account_equity` | `chad.risk.profit_lock.ProfitLockEngine._compute` (called every live-loop cycle) | 60 s | every live-loop cycle (~60 s) |
| 2 | `runtime/portfolio_snapshot.json` | (no `schema_version`; doc-defined `ibkr_equity`/`kraken_equity`/`coinbase_equity`) | `ibkr_equity`, `kraken_equity`, `coinbase_equity`, `ts_utc`, `ttl_seconds` | `chad/portfolio/ibkr_portfolio_collector_v2.py` (via `chad-ibkr-collector.timer` every ~2 min) and `chad/ops/portfolio_snapshot_publisher.py` (5 min) | 300 s | ~120 s (collector) |
| 3 | `runtime/positions_truth.json` | `positions_truth.v1` | `equity` / `total_equity` / `net_liquidation` (legacy keys; current schema does not always populate them) | `chad/ops/lifecycle_truth_publisher.py` | mtime-based | every lifecycle-publisher cycle |
| 4 | `runtime/positions_snapshot.json` | n/a | `net_liquidation` / `netLiquidation` / `equity` / `total_equity` | legacy collector | varies | varies |
| 5 | `runtime/portfolio_state.json` | `portfolio_state.v1` | (rebalance-only; does NOT carry live equity in the canonical schema) | rebalance state publisher | 3600 s | weekly rebalance |
| 6 | `runtime/dynamic_caps.json` | (caps schema) | `total_equity` | dynamic caps publisher | varies | per dynamic-caps run |
| 7 | `runtime/equity_history.ndjson` | `equity_history.v1` | `ibkr_equity_usd`, `total_equity_usd` (daily snapshots, not live) | `chad/ops/equity_history_publisher.py` | daily | daily |

Reference values captured at 2026-05-20T00:36 UTC:
- `pnl_state.account_equity = 258365.97` (ts 2026-05-20T00:36:01Z)
- `portfolio_snapshot.ibkr_equity = 258181.39` (ts 2026-05-20T00:34:55Z)
- `portfolio_snapshot.kraken_equity = 184.58`
- ibkr+kraken+coinbase = 258365.97 → exactly equals pnl_state.account_equity ✓

## 3. Canonical declarations

The following declarations are the formal CHAD policy as of Box 034.
They MUST be respected by all current and future consumer modules.

### 3a. Operator / dashboard truth — `pnl_state.json::account_equity`

- **Canonical source for the operator-facing dashboard, Telegram/status
  reports, daily ops report's "account_value" header, and any
  human-facing "equity now" claim.**
- File: `runtime/pnl_state.json`
- Schema: `pnl_state.v1`
- Key: `account_equity` (float USD)
- Freshness key: `ts_utc`
- Stale threshold: `ttl_seconds` field (currently 60 s)
- Producer: `chad/risk/profit_lock.py` side-writes from the active
  composite equity provider chain on every live-loop cycle.
- Justification: this is the post-allocator, post-profit-lock view that
  the operator sees on the dashboard. It is the most observed value;
  drift between it and broker truth must be detected, not hidden.

### 3b. Ops / risk truth — `portfolio_snapshot.json` (sum of native equities)

> [BOX-034A — superseded on implementation] The "USD" assertion is incorrect; the v2 collector emits CAD. Canonical currency is CAD (BOX-034A §2).

- **Canonical source for drawdown guard, withdrawal manager, business
  phase tracker, equity-history publisher, health monitor, daily-ops
  report's `inputs.portfolio_snapshot` block, and any non-operator
  artifact that needs the raw broker-reported equity per venue.**
- File: `runtime/portfolio_snapshot.json`
- Keys: `ibkr_equity`, `kraken_equity`, `coinbase_equity`, `ts_utc`,
  `ttl_seconds`.
- Total equity (USD): `ibkr_equity + coinbase_equity + kraken_equity`.
- Stale threshold: `ttl_seconds` (currently 300 s).
- Producer: `chad-ibkr-collector.timer` runs
  `chad/portfolio/ibkr_portfolio_collector_v2.py` every ~2 min and
  reads IBKR `NetLiquidation` from the live account summary. Kraken
  and Coinbase legs are merged in by the same writer (preserved across
  IBKR-only refreshes).
- Justification: this file is the closest authoritative read to broker
  truth (an actual IBKR API NetLiquidation call) and persists per-venue
  breakdowns, which is what risk/ops artifacts need.

### 3c. Broker raw — IBKR `NetLiquidation` (read inside the collector)

> [BOX-034A — superseded on implementation] get_net_liquidation ignores the currency tag; BOX-034A §3 makes it currency-aware and fail-closed.

- **Canonical raw source.** Every other equity number in CHAD
  ultimately derives from this read.
- Producer: `chad/portfolio/ibkr_portfolio_collector_v2.py::get_net_liquidation`.
- Persisted into `portfolio_snapshot.json` (see §3b).

## 4. Temporal-skew policy

### 4a. Acceptable skew between canonical sources

> [BOX-034A — superseded on implementation] Drift is a CAD↔USD currency flip, not staleness; reconciliation must be currency-explicit (BOX-034A §5).

Let `pnl_eq = pnl_state.account_equity` and
`snap_total = portfolio_snapshot.{ibkr_equity + kraken_equity + coinbase_equity}`.

- **Steady state**: `|pnl_eq - snap_total| ≤ max(1.0 USD, 0.0005 × snap_total)`
  (0.05 % of total or $1, whichever is larger — to absorb rounding and
  per-leg refresh interleaving).
- A larger drift indicates one of:
  - one source is stale (see §4b),
  - profit-lock is computing from a non-snapshot equity provider
    (e.g. a stale dynamic_caps.json), or
  - a kraken/coinbase leg dropped from the snapshot due to an upstream
    feed error.

### 4b. Stale thresholds

- `pnl_state.json` MUST be no older than `ttl_seconds` (60 s by default).
  When stale, the dashboard MUST surface the stale label and MUST NOT
  display the cached value as if it were live. (Recommended follow-up
  fix; see §6.)
- `portfolio_snapshot.json` MUST be no older than `ttl_seconds` (300 s
  by default). When stale, ops artifacts MUST log
  `portfolio_snapshot_stale` and pass through with a stale marker;
  risk gates that depend on it (drawdown, withdrawal) MUST short-circuit
  to fail-closed behaviour rather than use the stale value.

### 4c. Non-canonical sources

- `positions_truth.json` equity fields are LEGACY and NOT canonical for
  operator truth; the schema does not always populate them. Profit-lock
  composite provider keeps them as a last-resort fallback ONLY.
- `positions_snapshot.json` (legacy) is also a fallback ONLY.
- `portfolio_state.json` does NOT carry live equity in its canonical
  schema; do NOT add equity reads there.
- `equity_history.ndjson` is a daily-snapshot ledger and MUST NOT be
  used as a live source — it is for trailing/drawdown computation only
  (where stale is by design).

### 4d. Stale must never silently override fresh

The composite equity provider in `chad/risk/profit_lock.py:807-821` is
ordered so the freshest, most-canonical source wins. The provider chain
MUST be reviewed if any new source is added — particularly any source
with a TTL > 300 s must be placed AFTER the canonical chain.

## 5. Enforcement

> [BOX-034A — superseded on implementation] The enforcement test must also assert currency==CAD and bound sampling skew (BOX-034A §6); the current test passes on a lucky same-currency read.

Box 034 adds `chad/tests/test_canonical_equity_source.py` which asserts:

1. `runtime/pnl_state.json` (when present) has `schema_version == pnl_state.v1`, carries an `account_equity` key, a `ts_utc` key, and a `ttl_seconds` ≤ 120 (matches §3a expectation).
2. `runtime/portfolio_snapshot.json` (when present) carries `ibkr_equity`, `kraken_equity`, `coinbase_equity`, `ts_utc`, and `ttl_seconds`; `ttl_seconds` ≤ 600 (matches §3b expectation).
3. When both canonical files exist and both are fresh, the §4a skew
   invariant holds: `|pnl_state.account_equity - (ibkr+kraken+coinbase)| ≤ max(1.0, 0.0005 × snap_total)`.
4. The profit-lock composite provider does NOT include any provider
   that reads from `equity_history.ndjson` (which would be a temporal-
   skew violation per §4c).

If any canonical file is absent (e.g. a fresh repo clone without runtime
state), the relevant test is skipped rather than failing — these are
runtime-invariant checks intended for live and integration environments.

## 6. Recommended consumer-side follow-ups (NOT applied in this box)

These are documented operator-decision items, not Box-034 patches:

- (R1) Dashboard staleness label: `chad/dashboard/api.py:_portfolio()`
  currently reads `pnl.get("account_equity")` without consulting
  `ts_utc + ttl_seconds`. When stale, the dashboard should display the
  stale label (e.g. `—` or `"stale"`) rather than the cached value. A
  minimal patch (2-3 lines) would compute
  `is_fresh = now - parse(pnl["ts_utc"]) < pnl["ttl_seconds"]` and
  surface a `account_value_is_stale` field. Defer to operator approval.
- (R2) Profit-lock provider chain: review whether
  `_build_default_equity_provider` should add `portfolio_snapshot.json`
  ahead of `positions_truth.json` (it has fresher, broker-authoritative
  data). The current ordering predates the portfolio_snapshot publisher
  and may be silently using a less-fresh fallback in some runtime
  configurations.

These items remain pending in this file until an operator decides; the
Box-034 closure does NOT depend on them.

## 7. Status

- Recorded: 2026-05-20T00:38:00Z (Box 034)
- Canonical declarations §3a / §3b / §3c: ACTIVE.
- Parity test: ACTIVE (`chad/tests/test_canonical_equity_source.py`).
- R1 / R2 follow-ups: pending operator decision.
