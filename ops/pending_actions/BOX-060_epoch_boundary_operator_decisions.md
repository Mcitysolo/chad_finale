# BOX-060 — Epoch-boundary operator decisions

**Status:** Pending Action (paper / governance — records deferrals; enables nothing).
**Source:** Official Evidence-Locked Completion Matrix v0.1, Box 060.
**Title:** Epoch-boundary operator decisions resolved
**Acceptance criterion (verbatim):**
> "deferred services like full-cycle refresh / crypto risk-off are enabled
> or formally deferred."

## Decision

**ALL EIGHT DEFERRED CHAD TIMERS ARE FORMALLY DEFERRED.** No operator
authorization to enable any of them is present in this task. This
document codifies the deferral for each — reason, owner, impact,
unlock criteria — so no unit is left undecided. CHAD remains PAPER.
Live trading remains NOT authorized. `ready_for_live=false`.

Two of the eight are named explicitly in the Box-060 acceptance
criterion (**full-cycle refresh** and **crypto risk-off**); the
remaining six were surfaced by an exhaustive `systemctl
list-unit-files | grep chad | grep disabled` scan and are recorded
here for completeness.

## Decision register

For each disabled-and-inactive `chad-*.timer`, the decision is:
**DEFER**. Unlock requires explicit operator approval recorded under
`ops/pending_actions/` plus the per-unit conditions below.

### 1. `chad-full-cycle-refresh.timer`

- **Service:** `chad-full-cycle-refresh.service`
  (`python -m chad.core.full_execution_cycle`, `DRY_RUN`)
- **Description:** "CHAD Full Execution Cycle Refresh (DRY_RUN) —
  keeps price_cache.json fresh"
- **Timer cadence:** OnBootSec=60, OnUnitActiveSec=120 (every ~2 min)
- **Current state:** `disabled` + `inactive` (no recent journal events
  in the last 7 days).
- **Decision:** **DEFER.**
- **Reason:** A separate, narrower price-cache refresh path
  (`chad-price-cache-refresh.timer`, also deferred — see item 5) is
  the canonical price-cache writer when enabled. Running the full
  execution cycle every 2 min as a refresher mixes two concerns
  (cycle exercise + price-cache write) and can create competing
  evidence in `runtime/`. Until the operator decides which of the
  two refresh paths owns `price_cache.json`, neither is enabled.
- **Owner:** Operator + `chad/core/full_execution_cycle.py` author.
- **Impact while deferred:** `runtime/price_cache.json` is currently
  stale (~14 h at audit time). This is one of the contributing
  factors to the Box-059 soak window remaining at 0 trading days.
- **Unlock criteria:**
  1. Operator decides which path owns `price_cache.json` (this
     timer vs `chad-price-cache-refresh.timer` vs the IBKR bar
     publisher's incidental writes).
  2. Conflict-resolution policy documented (last-writer wins?
     dedicated writer? schema-version bump?).
  3. Dry-run cycle for 1 hour confirms no decision-trace
     duplication or `runtime/` thrash.
  4. Recorded approval in `ops/pending_actions/BOX-060_..._enable.md`.

### 2. `chad-crypto-risk-off.timer`

- **Service:** `chad-crypto-risk-off.service`
  (`python -m chad.core.full_execution_cycle && python -m chad.ops.crypto_risk_off_publisher`)
- **Description:** "CHAD Crypto Risk-Off Refresher (writes
  runtime/crypto_risk_off.json)"
- **Timer cadence:** OnBootSec=30s, OnUnitActiveSec=60s
- **Current state:** `disabled` + `inactive`; `runtime/crypto_risk_off.json`
  last modified **2026-04-03** (~48 days old).
- **Decision:** **DEFER.**
- **Reason:** Runs the full execution cycle every 60 s with the
  `CHAD_ALPHA_CRYPTO_*` env block set — this is a heavy refresher
  whose only artifact is a single risk-off boolean. The same
  decision the publisher emits can be derived offline from
  `runtime/crypto_derivatives.json` (Phase-B Item 4, public Kraken
  Futures ticker — currently active). The `chad-crypto-derivatives-refresh.timer`
  is already active and supplies the upstream signal; the
  risk-off classifier itself does not need its own minute-cadence
  service while no live crypto execution lane exists (Kraken Futures
  is Box-057 scoped out spot-only).
- **Owner:** Operator + `chad/ops/crypto_risk_off_publisher.py`
  author.
- **Impact while deferred:** `runtime/crypto_risk_off.json` is stale
  but no consumer hard-blocks on it (alpha_crypto reads
  derivatives-publisher signals as confidence-only — see Box-057 /
  Box-058 evidence). Stale risk-off file is informational only.
- **Unlock criteria:**
  1. Crypto risk-off classifier becomes a hard gate for any
     execution lane (currently none — Kraken Futures is spot-only).
  2. Cadence reduced from 60 s to a value that does not pin a full
     execution cycle every minute (e.g. 5 min) OR the publisher is
     refactored to read `crypto_derivatives.json` without running a
     full cycle.
  3. Recorded approval in `ops/pending_actions/BOX-060_..._enable.md`.

### 3. `chad-crypto-risk-notify.timer`

- **Service:** `chad-crypto-risk-notify.service`
- **Description:** "CHAD Crypto Risk Notifier (Telegram on
  state-change)"
- **Current state:** `disabled` + `inactive`.
- **Decision:** **DEFER.**
- **Reason:** Depends on the crypto risk-off publisher (item 2) to
  produce fresh `runtime/crypto_risk_off.json` writes. With item 2
  deferred, this notifier has no fresh state to react to. Telegram
  channel discipline (Box-042 telegram-dedupe cleanup) is the
  governance owner.
- **Owner:** Operator + notifier maintainer.
- **Impact while deferred:** No automated Telegram crypto risk-off
  alerts. Acceptable while no live crypto execution lane exists.
- **Unlock criteria:** Item 2 unlocks first; then enable this
  notifier with explicit operator approval.

### 4. `chad-ibkr-cash-collector.timer`

- **Service:** `chad-ibkr-cash-collector.service`
- **Description:** "CHAD IBKR Cash Collector (TotalCashValue
  snapshot)"
- **Current state:** `disabled` + `inactive`.
- **Decision:** **DEFER.**
- **Reason:** Equity / cash truth is already published by the IBKR
  portfolio collector (`chad/portfolio/ibkr_portfolio_collector_v2.py`)
  on the `chad-positions-snapshot.timer` cadence (currently active).
  A dedicated cash-collector duplicates that surface and risks two
  competing snapshots of the same broker truth.
- **Owner:** Operator + portfolio collector maintainer.
- **Impact while deferred:** None — cash is covered by the
  positions-snapshot path.
- **Unlock criteria:** Operator decides the portfolio collector is
  insufficient (e.g. needs sub-minute cash freshness), OR the
  positions-snapshot path is decommissioned. Either path requires
  explicit operator approval.

### 5. `chad-price-cache-refresh.timer`

- **Service:** `chad-price-cache-refresh.service`
- **Description:** "CHAD Price Cache Refresh (Polygon NDJSON ->
  runtime/price_cache.json)"
- **Current state:** `disabled` + `inactive`.
- **Decision:** **DEFER.**
- **Reason:** Polygon NDJSON ingestion requires a paid data
  subscription and an active Polygon API key. CHAD does not have one
  in the current operator deployment. The IBKR bar provider supplies
  bar data instead; price_cache.json is currently being maintained
  (when fresh) by the IBKR-side path. See item 1 for the
  competing-writer issue.
- **Owner:** Operator + market-data team.
- **Impact while deferred:** None — IBKR bars provide equivalent
  price feed inputs.
- **Unlock criteria:** Polygon subscription procurement decision
  (similar to Box-058 Coinglass shape); operator approval; conflict
  with item 1 resolved first.

### 6. `chad-stop-refresh.timer`

- **Service:** `chad-stop-refresh.service`
- **Description:** "CHAD STOP state refresher (prevents stale TTL
  fail-closed)"
- **Current state:** `disabled` + `inactive`; nonetheless
  `runtime/stop_state.json` is fresh (mtime within minutes) — written
  by another publisher (governor / heartbeat) on every cycle.
- **Decision:** **DEFER.**
- **Reason:** Stop-state freshness is already maintained by the
  active publishers (`chad-governor.timer`,
  `chad-decision-trace-heartbeat.timer`) — `runtime/stop_state.json`
  TTL is currently within bounds without this dedicated refresher.
  Enabling it would create a second writer.
- **Owner:** Operator + governor maintainer.
- **Impact while deferred:** None observed — TTL-refresh logic is
  already covered by the active path.
- **Unlock criteria:** Governor path is decommissioned OR an audit
  identifies stop_state TTL gaps the dedicated refresher would close.

### 7. `chad-symbol-bench.timer`

- **Service:** `chad-symbol-bench.service`
- **Description:** "CHAD Symbol Bench Refresher (ledger-driven
  global bench list)"
- **Current state:** `disabled` + `inactive`.
- **Decision:** **DEFER.**
- **Reason:** Symbol-bench (the ledger-driven "best universe"
  refresher) is a strategy-discovery / R&D artifact. With the
  Dynamic Universe Scanner (`chad-dynamic-universe-scanner-refresh.timer`,
  currently active) supplying the live universe, the symbol-bench
  refresher is duplicative until the operator decides to use it for
  override / comparison.
- **Owner:** Operator + universe-scanner maintainer.
- **Impact while deferred:** None — universe is supplied by the
  scanner.
- **Unlock criteria:** Operator decides to use ledger-driven bench
  alongside / instead of the scanner; documented selection policy.

### 8. `chad-warmup.timer`

- **Service:** `chad-warmup.service`
- **Description:** "CHAD Warmup Runner (build SCR effective_trades
  safely)"
- **Current state:** `disabled` + `inactive`.
- **Decision:** **DEFER.**
- **Reason:** SCR is currently at `state=CONFIDENT` with
  `effective_trades=196`, `paper_trades=3914`. The warmup runner
  exists to bootstrap `effective_trades` from a cold start; CHAD is
  far past that band. Enabling it now is a no-op at best and a
  source of write contention against SCR snapshots at worst.
- **Owner:** Operator + SCR maintainer.
- **Impact while deferred:** None.
- **Unlock criteria:** A fresh epoch / SCR reset that returns
  effective_trades to a cold range (<50). Currently not applicable.

## Currently-active counterpart timers (no decision needed)

For reference, these epoch-relevant timers are **already active** and
do not require a deferral entry. Each is the canonical writer of its
runtime artifact:

| Active timer | Artifact | Cadence (observed) |
|---|---|---|
| `chad-crypto-derivatives-refresh.timer` | `runtime/crypto_derivatives.json` (public Kraken Futures ticker — funding / OI / crowding) | ~5 min |
| `chad-positions-snapshot.timer` | `runtime/positions_snapshot.json` + `runtime/portfolio_snapshot.json` | ~5 min |
| `chad-reconciliation-publisher.timer` | `runtime/reconciliation_state.json` + `runtime/position_guard_drift.json` | ~5 min |
| `chad-live-readiness.timer` | `runtime/live_readiness.json` | ~10 min |
| `chad-governor.timer` | `runtime/stop_state.json` + governor heartbeat | ~10 min |
| `chad-lifecycle-truth-publisher.timer` | `runtime/trade_lifecycle_state.json` | ~1 min |
| `chad-mutation-state-publisher.timer` | mutation-state freshness | ~1 min |
| `chad-execq-publisher.timer` | `runtime/execution_quality.json` | ~1 min |
| `chad-scr-sync.timer` | `runtime/scr_state.json` | ~1 min |
| `chad-decision-trace-heartbeat.timer` | `runtime/decision_trace_heartbeat.json` | ~5 min |
| `chad-feed-watchdog.timer` | feed staleness alerts | ~2 min |
| `chad-ibkr-watchdog.timer` | IBKR health alerts | ~2 min |
| `chad-ibkr-broker-events.timer` | `data/broker_events/BROKER_EVENTS_IBKR_<date>.ndjson` | ~5 min |
| `chad-ibkr-paper-fill-harvester.timer` | `data/fills/FILLS_<date>.ndjson` (paper) | ~5 min |
| `chad-trade-closer.timer` | `runtime/trade_closer_state.json` | ~1 min |
| `chad-paper-trade-exec.timer` | paper executor | ~5 min |
| `chad-kraken-futures-intel-refresh.timer` | `runtime/kraken_futures_intel.json` (public ticker) | ~5 min |
| `chad-dynamic-universe-scanner-refresh.timer` | dynamic universe scan | ~5 min |
| `chad-options-monitor.timer` | options panel | ~1 min |
| `chad-volume-scan.timer` | volume scan | ~5 min |

This list is provided for context; Box-060 closure does not depend
on enumerating active timers exhaustively. The decision register
above covers the units the acceptance criterion names plus every
other `chad-*.timer` whose `UnitFileState=disabled`.

## Failure semantics

If any of the eight deferred timers is enabled outside this policy
(i.e. without operator approval and without the per-unit unlock
criteria met), it counts as an unauthorized configuration change.
Discovery surfaces:

- `runtime/change_canary_state.json:tamper_detected=true` if the
  systemd unit-file hash changes.
- This file's decision register would no longer match observed
  `UnitFileState`. A future Box-060 re-audit would detect the
  divergence.

## Live trading authorization

This policy **does not** enable any service. This policy **does not**
authorize live trading. This policy **does not** set
`ready_for_live=true`. CHAD remains PAPER. Live activation still
requires the Pre-Live Operator Tasks in `CLAUDE.md`, the Box-059
clean five-day soak, and an explicit operator GO recorded in
`ops/pending_actions/`.

## Verification command (for the policy guard tests)

```
source venv/bin/activate
python3 -m pytest chad/tests/test_box060_official_epoch_boundary_decisions_guard.py -v
```

The guard tests verify only that the policy doc exists, enumerates
every deferred unit, records the two headline decisions (full-cycle
refresh + crypto risk-off), and does not silently authorize live
trading. They do **not** assert any unit has been enabled.

## Pending Action

This document is a **policy** only. There is **no runtime config
change** to apply. No service restart, enable, or disable is
required. No operator approval is needed beyond review. CHAD remains
PAPER. Live trading remains NOT authorized. `ready_for_live=false`.
Box 61+ remain open.
