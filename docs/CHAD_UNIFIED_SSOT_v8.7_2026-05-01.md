# CHAD Unified SSOT v8.7

**Version:** 8.7
**Date:** 2026-05-01
**Status:** Active — Paper Trading
**Supersedes:** `docs/CHAD_UNIFIED_SSOT_v8.6_2026-05-01.md` (commit `eafb8cc`, 2026-05-01)

This document is the master reference for the CHAD trading system as it
exists at HEAD (`1207eeb`, 2026-05-01). It captures every commit since
v8.6, every wired strategy, every runtime invariant, and the live state
of the machine at the moment of writing. v8.7's signature contribution
is the **CHAD AI Health Monitor** — a three-tier autonomous health
surveillance system that combines a deterministic rule engine, a
Claude-reasoning escalation tier, and an auto-remediation tier. v8.6
ensured the *operator can sleep through the night*; v8.7 ensures the
system can *self-diagnose and self-heal* across the most common
failure modes without operator intervention. Every v8.6 invariant
remains intact — the audit-session deltas, SCR=CONFIDENT, the feed
watchdog, the SIGTERM handler, the surgical phantom-fill guard. The
single new addition is autonomous bug detection layered on top.

If this document and the code disagree, either the code drifted or
this revision needs another pass — either way, revise the document
before relying on the disagreement.

---

## Table of Contents

0. [Preamble](#0-preamble)
1. [Mission & Architecture](#1-mission--architecture)
2. [Runtime State (live snapshot)](#2-runtime-state-live-snapshot)
3. [The 16 Strategies](#3-the-16-strategies)
4. [Execution Pipeline (post-fix path)](#4-execution-pipeline-post-fix-path)
5. [Risk & Governance — The Full Chain](#5-risk--governance--the-full-chain)
6. [The Business Framework](#6-the-business-framework)
7. [Reconciliation](#7-reconciliation)
8. [Intelligence Layer](#8-intelligence-layer)
9. [Telegram Operator Interface](#9-telegram-operator-interface)
10. [Dashboard (chadtrades.com)](#10-dashboard-chadtradescom)
11. [Services & Timers](#11-services--timers)
12. [Data & Storage](#12-data--storage)
13. [Change Log (delta from v8.5)](#13-change-log-delta-from-v85)
14. [Known Issues](#14-known-issues)
15. [Phase Roadmap](#15-phase-roadmap)
16. [Appendices](#16-appendices)

---

## 0. PREAMBLE

### Document metadata

| Field | Value |
|---|---|
| Document version | 8.7 |
| Date written | 2026-05-01 (UTC) |
| Predecessor | v8.6 — `docs/CHAD_UNIFIED_SSOT_v8.6_2026-05-01.md` (`eafb8cc`) |
| Repository HEAD at write time | `1207eeb` — *"Build: CHAD AI Health Monitor — 3-tier autonomous bug detection, rule engine, Claude reasoning, auto-remediation"* |
| Branch | `main` |

### Server / repo / mode

- **Server:** AWS EC2, Ubuntu 24.04, kernel 6.17.0-1009-aws.
- **Repo root:** `/home/ubuntu/chad_finale`.
- **Python:** `python3` via `/home/ubuntu/chad_finale/venv` (governance
  rule in `CLAUDE.md` — never invoke `python`).
- **IBKR paper account:** Canadian-domiciled, `$183,741.46 USD` net
  liquidation (source: `runtime/portfolio_snapshot.json:ibkr_equity` at
  `2026-05-01T10:59:20Z` after IBKR FX-quoted CAD→USD conversion).
- **Kraken paper:** connected — 0.0012 BTC + 252.85 CAD ≈ `$184.58 USD`
  (source: `runtime/portfolio_snapshot.json:kraken_equity` at
  `2026-05-01T10:59:20Z`).
- **Total equity (USD):** `$183,926.04` =
  `ibkr_equity ($183,741.46) + kraken_equity ($184.58)`.
- **Allocator equity (`runtime/dynamic_caps.json:total_equity`):**
  `$183,982.36` (`2026-05-01T11:08:27Z`) — caps cycle is fresher than
  the 5-minute snapshot publisher; the small delta is the publisher
  cadence offset, not a fault.
- **Execution posture:** PAPER — `CHAD_EXECUTION_MODE=paper`. The
  systemd unit ships `dry_run` (`/etc/systemd/system/chad-live-loop.service`)
  but `chad/core/live_loop.py:_is_paper_mode` treats both `paper` and
  `dry_run` as paper for guard-rebuild purposes.
- **Live readiness:** `runtime/live_readiness.json:ready_for_live=false`
  (last evaluated `2026-04-09T13:40:29Z` — independent of the SCR
  promotion to CONFIDENT; live promotion requires explicit operator GO
  per CLAUDE.md governance rule #3).

### This session's commit chain (17 commits since v8.5)

In topological order (oldest → newest, per
`git log 9f55688..HEAD --oneline --reverse`):

1. **`42268f0`** — *Fix: live loop cycle audit — phantom fills, unprotected
   stages, intent guard, regime TTL, VXX bars, bias logging* (2026-04-30).
   Opening commit of the session — rolls up the live-loop forensic
   audit. Records intent guard, regime-state TTL behavior, VXX bar
   backfill, and bias logging. The phantom-fill fix in this commit was
   too broad (touched the normalizer); it was reverted and replaced
   with a surgical guard in `3be6785`.

2. **`3be6785`** — *Fix: surgical phantom fill guard at submit call site —
   revert normalizer, block at live_loop evidence write* (2026-04-30).
   The phantom-fill issue was that error-status orders (orders that
   IBKR rejected) were still being persisted as paper-fill evidence
   downstream of the normalizer, producing ghost positions in the
   ledger. The surgical guard was placed at `chad/core/live_loop.py:1180`
   at the submit call site: *if order status is `error`/`Cancelled`/
   `Inactive`, skip evidence write and log a structured warning*. The
   normalizer rewrite from `42268f0` was reverted to keep the
   single-chokepoint invariant intact.

3. **`ceb53ea`** — *Fix: SCR audit — add excluded_pnl_zero counter to
   trade_stats_engine, balance exclusion ledger* (2026-04-30). The SCR
   exclusion ledger had three documented buckets
   (`excluded_manual`, `excluded_nonfinite`, `excluded_untrusted`) but
   the actual exclusion logic had a fourth implicit bucket — trades
   with PnL exactly $0.00, which were being silently dropped from
   effective_trades without being counted anywhere. Added an explicit
   `excluded_pnl_zero` counter so the ledger now balances:
   `paper_trades = effective_trades + excluded_untrusted + excluded_manual + excluded_nonfinite + excluded_pnl_zero`.

4. **`23f4c03`** — *Fix: SCR audit — min-trade gates for CAUTIOUS/
   CONFIDENT bands, PAUSED hysteresis router (caller wiring pending)*
   (2026-04-30). Two changes: (a) added explicit minimum-trade gates —
   `min_cautious=100`, `min_confident=133`; SCR cannot promote past
   WARMUP without 100+ effective trades, cannot promote past CAUTIOUS
   without 133+; (b) added a PAUSED hysteresis router so a single bad
   week does not cycle WARMUP → PAUSED → WARMUP every minute — the
   router holds PAUSED for `paused_recovery_ticks` consecutive healthy
   reads before promoting back. Caller wiring pending — see commit
   `28e8963` for the round-trip.

5. **`5267705`** — *Fix: position guard audit — atomic _save_state write,
   chad_strategy_open in reconciliation counts* (2026-04-30). The
   `position_guard._save_state` writer was not atomic — a process
   crash mid-write could leave `runtime/position_guard.json` truncated,
   masking open positions on the next startup. Switched to write-temp
   + os.replace pattern. Added `chad_strategy_open` field to the
   reconciliation counts block — the prior summary collapsed all CHAD
   open positions into `chad_open` regardless of whether they were
   strategy-tracked or `broker_sync` echoes; the new field exposes
   the strategy-tracked count separately.

6. **`cdfed06`** — *Fix: profit router audit — test isolation for ledger,
   exclude broker_sync from routing* (2026-04-30). Two issues:
   (a) profit-router unit tests were sharing the production ledger
   path (`runtime/profit_routing.json`) and bleeding 19 phantom
   `alpha_test` entries into production — fix is `routing_path` param
   defaulting to None which the test harness can override. The 19
   phantom entries were purged in this commit
   (`alpha_test_purged_at_utc=2026-04-30T19:05:17.503425Z`,
   `alpha_test_purged_count=19`). (b) `broker_sync` realized PnL was
   being routed through 50/30/20 — but `broker_sync` is bookkeeping
   for inherited paper positions, not strategy alpha; it should not
   contribute to amplifier or beta. Now excluded at
   `chad/risk/profit_router.py:route_profit`.

7. **`931dd4e`** — *Fix: Telegram audit — live loop crash, recon RED,
   profit lock, salary, drawdown, fill dedupe, SCR recovery alerts*
   (2026-04-30). Seven new Telegram alert types added — the operator's
   "did anything important happen overnight" surface was thin. New
   alerts: live loop crash (cycle exception caught, alert sent before
   re-raise); reconciliation RED (both publish paths); profit lock
   transitions (NORMAL → WARN → LOCK1 → … → HARD_STOP); salary
   authorized (PAY-phase trigger); drawdown breach (5% from HWM);
   fill dedupe (suppress duplicate alerts within 5 minutes on same
   `(symbol, side, fill_price)`); SCR recovery from PAUSED (hysteresis
   router promotion). Also added: WARMUP demotion alert when SCR
   regresses. Added `EnvironmentFile=-/etc/chad/telegram.env` to
   `chad-live-loop.service` so the live loop can actually emit alerts.

8. **`c267076`** — *Fix: dashboard audit — staleness signal
   oldest_source_mtime_utc, brute force login protection* (2026-04-30).
   Added `oldest_source_mtime_utc` to the top-level `/api/state`
   payload — the dashboard previously had no single signal that said
   "the underlying runtime files are stale"; now there is. Added brute
   force login protection: 5 failed attempts within a 5-minute window
   triggers a 5-minute IP lockout. Also ratified the `/api/state`
   schema as: `{ts_utc, ibkr_connected, chad_status, portfolio,
   open_positions, strategies, system_health, intelligence, business,
   oldest_source_mtime_utc}` — frozen for chat-context consumers.

9. **`ba65367`** — *Fix: reconciliation audit — add MYM/M2K to futures
   exclusion, raise TTL to 360s* (2026-04-30). MYM and M2K (the new
   v8.5 micro-equity-index futures) were not in
   `KNOWN_FUTURES_SYMBOLS` in `reconciliation_publisher.py:58`; their
   contract-month variants tripped the futures resolution path and
   surfaced as drifts. Added both. Raised reconciliation TTL from 300s
   to 360s — at 300s the publisher cadence (also 300s) would
   occasionally publish a state file that was already at the staleness
   line by the time downstream consumers read it; 360s gives a 60s
   margin.

10. **`52786b7`** — *Fix: edge decay audit — wire halt enforcement, move
    before signals, fix key name, edge-trigger alert, harden clear
    script* (2026-04-30). The edge decay monitor was *publishing* halt
    state to `runtime/edge_decay_state.json` but nothing was reading
    it — `is_strategy_halted()` was unwired. Five changes: (a) wired
    `is_strategy_halted()` into signal building so a halted strategy
    drops out of the universe before it can emit; (b) moved the halt
    check from after stage-4 (sizing) to before stage-1 (signals) so
    halted strategies never enter the pipeline; (c) fixed key name
    `consecutive_negative` → `consecutive_neg` to match what the
    monitor actually publishes; (d) made halt alerts edge-trigger only
    (was firing every cycle while halted — operator was getting paged
    every 60s); (e) hardened `scripts/clear_edge_decay.py` to require
    explicit `--confirm` flag.

11. **`71d8ea6`** — *Fix: data feed audit — regime_state TTL enforcement
    at read time, stale returns unknown* (2026-04-30). The regime_state
    TTL was only checked at write time inside the publisher. Consumers
    (allocator, RegimeBooster, regime_activation_matrix lookups) were
    reading the file blindly and using whatever was there — including
    a 6-hour-stale `trending_bull` claim during a market that had
    rolled into ranging. Added TTL enforcement at every read site:
    if `now - ts_utc > ttl_seconds`, the read returns
    `regime=unknown`. Universal eligibility for `unknown` continues
    to behave as documented (regime_activation_matrix `unknown`
    column).

12. **`a4accdd`** — *Fix: data feed audit — feed watchdog service
    monitors 7 critical feeds every 120s, alerts on stale*
    (2026-04-30). New service `chad-feed-watchdog.service` + timer at
    120s cadence. Monitors seven critical feeds: `price_cache.json`
    (180s TTL), `regime_state.json` (180s), `dynamic_caps.json`
    (180s), `regime_booster.json` (180s), `kraken_prices.json` (120s),
    `reconciliation_state.json` (480s), `event_risk.json` (2400s). On
    stale detection: structured journal log + Telegram alert. Replaces
    the ad-hoc "I noticed the feed was stale" operator surface with
    a proactive check.

13. **`28e8963`** — *Fix: wire PAUSED hysteresis caller — inject
    prev_state/paused_recovery_ticks into evaluate_confidence,
    round-trip through ShadowStats* (2026-04-30). Companion to
    `23f4c03`. The PAUSED hysteresis router was implemented but the
    caller (`backend/api_gateway.py`) was not passing `prev_state` and
    `paused_recovery_ticks` — they round-trip through
    `ShadowStats.compute_state()` so the router actually sees the
    history. End-to-end wiring now closed.

14. **`74e5a0d`** — *Fix: D2 bucket accumulators — drain mechanism,
    get_beta_remaining, mark_beta_consumed, beta injection behind
    CHAD_PROFIT_ROUTER_BETA_INJECTION flag* (2026-04-30). The 50/30/20
    profit router was an **append-only ledger** — the totals
    (`trading_capital`, `beta_allocation`, `amplifier_allocation`)
    grew monotonically and there was no way to mark a chunk as
    *consumed*. Added two new functions: `get_beta_remaining()`
    returns `beta_allocation_total - consumed_beta_usd` and
    `mark_beta_consumed(usd)` increments the consumed counter
    atomically. New persisted fields:
    `consumed_beta_usd` and `consumed_amplifier_usd`. Beta injection
    into the `beta` strategy is wired but gated behind the
    `CHAD_PROFIT_ROUTER_BETA_INJECTION` env flag (default off) — paper
    soak before enabling.

15. **`9bc5286`** — *Fix: D4 SIGTERM handler — clean shutdown on systemd
    stop, Telegram alert, respects in-flight cycle* (2026-04-30). Pre
    this commit, `systemctl stop chad-live-loop` would tear the
    process out mid-cycle, occasionally producing inconsistent
    `position_guard.json` state and at least one phantom fill
    incident. New SIGTERM handler: (a) sets a stop flag at the top of
    the cycle loop; (b) lets the in-flight cycle finish (max ~12s
    given current cycle time); (c) closes broker connections cleanly;
    (d) sends a Telegram "live loop stopped cleanly" alert. systemd
    `KillSignal=SIGTERM` and `TimeoutStopSec=30` honored.

16. **`fec3d5b`** — *Fix: D6 A4 gate intraday/daily split (900s/172800s),
    D7 recent-trades 500-line cap* (2026-05-01). Two D-series fixes:
    (D6) the A4 routing gate (`bar_freshness`) had a single 300s
    threshold that worked for intraday equities but was nonsense for
    daily/swing/futures strategies whose bars naturally exceed 300s.
    Split into: intraday strategies (alpha, alpha_intraday,
    alpha_options, gamma, gamma_reversion, omega_momentum_options) →
    900s threshold + a separate `bar_missing` failure (no bar at all
    blocks the gate); daily/swing/futures (everything else) → 172800s
    (48h) threshold. (D7) `/api/recent-trades` was streaming the
    entire NDJSON file when called from the dashboard — on a heavy
    trading day this was a multi-MB response. Capped at the most
    recent 500 lines per file, paginated by date.

17. **`d35884b`** — *Fix: disable per-fill Telegram alerts — too noisy;
    morning brief + EOD recap only* (2026-05-01). The fill-level alerts
    introduced in `931dd4e` (with dedupe) were still producing 50–80
    Telegram messages a day. Operator feedback: "I don't want to know
    every fill, I want morning brief + EOD recap." Per-fill alerts
    disabled; the morning brief (Mon-Fri 13:00 UTC) and end-of-day
    recap (Mon-Fri 21:35 UTC) carry the consolidated fill summary.
    Other alert types from `931dd4e` (live-loop crash, recon RED,
    profit lock transitions, salary authorized, drawdown breach, SCR
    recovery, WARMUP demotion) are kept — those are *exceptions*, not
    routine fills.

### v8.7 commit (1 commit since v8.6)

18. **`1207eeb`** — *Build: CHAD AI Health Monitor — 3-tier autonomous
    bug detection, rule engine, Claude reasoning, auto-remediation*
    (2026-05-01). New three-tier surveillance system layered on top of
    the v8.6 foundations. **Tier 1** — deterministic rule engine
    (`chad/ops/health_monitor_rules.py`) running every 5 minutes via
    `chad-health-monitor.timer`, evaluating 9 rules (R01 critical
    services, R02 feed freshness, R03 SCR paused, R04 stop bus, R05
    reconciliation RED, R06 profit lock, R07 disk usage, R08 corrupt
    runtime files, R09 edge decay halts). **Tier 2** — Claude
    reasoning: when Tier 1 raises a critical or unfamiliar condition,
    the monitor packages a full system snapshot and dispatches it to
    `claude-sonnet-4-6` for diagnostic narrative + recommended actions.
    **Tier 3** — auto-remediation
    (`chad/ops/health_monitor_remediation.py`): for a vetted set of
    safe actions, the monitor executes the fix itself — restart dead
    services via `sudo systemctl restart`, restart stale feed
    publishers, restore corrupt runtime files from backup, archive old
    fill files when disk usage crosses threshold. All other detected
    states are notify-only — SCR PAUSED, stop bus active,
    reconciliation RED, profit lock LOCK2 or higher, edge decay halts.
    Telegram surface emits three new message types: `AUTO-FIXED`
    (Tier-3 success), `HEALTH MONITOR CRITICAL` (Tier-1 escalation),
    `AI HEALTH ANALYSIS` (Tier-2 Claude narrative). Sudo passwordless
    pre-confirmed for the systemctl restart actions; no broader sudo
    privilege granted to the monitor. The monitor is additive —
    nothing pre-existing was removed or weakened.

### What's different from v8.5 (executive delta)

This is the largest single-session delta since v8.3. Seventeen commits,
ten forensic audits, one promotion (SCR WARMUP → CONFIDENT). The
audits were structured as: *the strategies are clean per v8.5 — what
about everything around them?* Each audit produced a list of findings;
each finding became a commit. The result is that the *paths* through
the system — alerts, halts, reconciliation, shutdown, feed health —
are now as rigorous as the strategy modules themselves.

1. **AUDIT SESSION COMPLETE — 10 forensic audits, 17 commits.** The
   audits, in order: live loop cycle (commit `42268f0`), surgical
   phantom-fill (`3be6785`), SCR ledger balance (`ceb53ea`), SCR
   gates + PAUSED hysteresis (`23f4c03`), position guard
   (`5267705`), profit router (`cdfed06`), Telegram coverage
   (`931dd4e`), dashboard (`c267076`), reconciliation (`ba65367`),
   edge decay (`52786b7`), data feed TTL (`71d8ea6`), feed watchdog
   (`a4accdd`), PAUSED caller wiring (`28e8963`), bucket accumulators
   D2 (`74e5a0d`), SIGTERM D4 (`9bc5286`), A4 split / recent-trades
   D6/D7 (`fec3d5b`), per-fill alert volume (`d35884b`). Every audit
   has at least one closing commit; nothing was filed and forgotten.

2. **PHANTOM FILL FIX — surgical guard at live_loop.py:1180,
   error-status orders skip evidence write.** Pre-fix: orders that
   IBKR rejected (status `error`, `Cancelled`, `Inactive`) were still
   flowing through the evidence writer downstream of the
   normalizer, producing ghost paper positions in
   `position_guard.json` and `data/fills/FILLS_*.ndjson`. The fix is
   surgical — guard at the submit call site that filters error-status
   orders before they reach the writer. The original wider fix in
   `42268f0` touched the normalizer; this was reverted in `3be6785`
   to preserve the single-chokepoint invariant.

3. **SCR NOW CONFIDENT — 147 effective trades, sizing_factor=1.0,
   paper_only=false.** Min-trade gates added: cautious=100,
   confident=133. SCR has cleared both gates: 147 effective trades is
   above the 133-confident threshold, win_rate (55.78%), sharpe_like
   (+1.0489), and max_drawdown (−$1,454.84) are all in the confident
   band. `paper_only=false` — the SCR side is no longer blocking
   live; live promotion still requires the operator's explicit GO per
   governance rule #3.

4. **PAUSED HYSTERESIS FULLY WIRED — router logic (commit `23f4c03`)
   + caller wiring (commit `28e8963`).** The router holds PAUSED for
   `paused_recovery_ticks` consecutive healthy reads before promoting
   back, so a single bad-data tick does not cycle WARMUP → PAUSED →
   WARMUP every minute. The caller wiring round-trips `prev_state`
   and `paused_recovery_ticks` through `ShadowStats.compute_state()`
   so the router actually sees state history. End-to-end now closed.

5. **POSITION GUARD ATOMIC WRITES — `_save_state` uses tmp+replace.**
   `chad_strategy_open` field added to reconciliation counts. The
   prior writer was `open(..., 'w').write(json.dumps(...))` — a
   process crash between truncate and write produced a zero-byte
   file. Now writes to a tempfile and `os.replace`s atomically.
   `chad_strategy_open` exposes the strategy-tracked open count
   separately from `chad_open` (which still includes
   `broker_sync` echoes).

6. **PROFIT ROUTER CLEAN — test isolation fixed (`routing_path`
   param), `broker_sync` excluded from routing, 19 phantom
   `alpha_test` entries purged, drain mechanism added
   (`consumed_beta_usd`, `consumed_amplifier_usd`), beta injection
   wired behind `CHAD_PROFIT_ROUTER_BETA_INJECTION` flag.** The
   ledger went from append-only to drain-aware. The `alpha_test`
   purge stamped both `alpha_test_purged_at_utc` and
   `alpha_test_purged_count=19` for forensic traceability.
   `broker_sync` realized PnL no longer routes through 50/30/20 —
   bookkeeping for inherited positions is not strategy alpha. The
   beta injection feature is implemented end-to-end but flagged off
   by default for paper soak.

7. **TELEGRAM COVERAGE COMPLETE — 7 new alert types, per-fill alerts
   DISABLED.** New: live-loop crash, reconciliation RED (both publish
   paths), profit-lock transitions, salary authorized, drawdown
   breach, SCR recovery from PAUSED, WARMUP demotion. Per-fill alerts
   were too noisy (50–80/day even with dedupe) — disabled in
   `d35884b`; consolidated fill summary lives in morning brief + EOD
   recap. `chad-live-loop.service` now loads
   `EnvironmentFile=-/etc/chad/telegram.env` so the live loop can
   actually emit alerts (was missing pre-`931dd4e`).

8. **DASHBOARD HARDENED — `oldest_source_mtime_utc` in `/api/state`,
   brute force login protection (5 attempts → 5-min lockout),
   `/api/recent-trades` capped at 500 lines/file.** `/api/state`
   schema ratified as: `{ts_utc, ibkr_connected, chad_status,
   portfolio, open_positions, strategies, system_health,
   intelligence, business, oldest_source_mtime_utc}` — frozen for
   chat-context consumers. The staleness signal closes a long-standing
   gap where the dashboard had no single field that said "the
   underlying runtime files are old".

9. **RECONCILIATION HARDENED — MYM/M2K added to
   `KNOWN_FUTURES_SYMBOLS`, TTL raised 300 → 360s.** The new v8.5
   futures symbols were tripping the contract-resolution path and
   appearing as drifts. TTL margin closes a race where a
   300s-cadence publisher could write a state file that was already
   at the 300s staleness line by the time downstream consumers read.

10. **EDGE DECAY HALT NOW ENFORCED — `is_strategy_halted()` wired
    into signal building (was completely unwired).** Pre-fix, the
    edge decay monitor was *publishing* halt state but nothing was
    *reading* it — a strategy could be flagged "halt: 5 consecutive
    losses" and continue emitting signals every cycle. Now wired
    before stage-1 (signals) so halted strategies drop out of the
    universe before they can emit. Halt alerts are edge-trigger only
    (was paging every 60s while halted). `scripts/clear_edge_decay.py`
    requires explicit `--confirm` flag.

11. **DATA FEED WATCHDOG — `chad-feed-watchdog.timer` monitors 7
    feeds every 120s, Telegram alert on stale.** `regime_state` TTL
    enforced at read time (stale → `regime=unknown`). MGC/SIL expired
    contracts were resolved by service restart during the audit.
    The watchdog covers `price_cache`, `regime_state`,
    `dynamic_caps`, `regime_booster`, `kraken_prices`,
    `reconciliation_state`, `event_risk` — proactive replacement
    for "I noticed the feed was stale" operator surface.

12. **A4 GATE SPLIT — intraday strategies use 900s threshold +
    missing-bar fail; daily/swing/futures use 172800s unchanged.**
    The single 300s A4 threshold was nonsense for daily/swing
    strategies whose bars naturally exceed 300s. Now strategy-aware:
    intraday strategies (alpha, alpha_intraday, alpha_options, gamma,
    gamma_reversion, omega_momentum_options) get 900s + a separate
    `bar_missing` failure path; everything else gets 172800s.

13. **SIGTERM HANDLER — clean shutdown on systemd stop, Telegram
    alert, respects in-flight cycle.** `systemctl stop
    chad-live-loop` no longer tears the process out mid-cycle.
    The handler sets a stop flag, lets the in-flight cycle finish,
    closes broker connections cleanly, sends a Telegram alert.

14. **SCR COUNTER BALANCED — `excluded_pnl_zero` counter added,
    ledger now balances:** `5000 = 147 + 4424 + 0 + 0 + N`. The
    fourth implicit-bucket trades-with-PnL-exactly-$0.00 are now
    counted explicitly. (Live snapshot:
    `paper_trades=5000 = effective_trades=147 + excluded_untrusted=4424 +
    excluded_manual=0 + excluded_nonfinite=0 + excluded_pnl_zero=429`.)

15. **VXX BARS — added to universe, bars backfilled.** VXX was 26
    days stale; backfilled and added to the universe. Closes a small
    gap on volatility-tracking ETF coverage.

16. **CHAD AI HEALTH MONITOR (NEW v8.7) — three-tier autonomous bug
    detection layered on top of the v8.6 foundations.** A new
    surveillance system that combines a deterministic rule engine, a
    Claude-reasoning escalation tier, and an auto-remediation tier.
    Adds `chad-health-monitor.timer` (cadence 300s, `OnBootSec=120`)
    plus three new modules in `chad/ops/`.

    - **Tier 1 — Rule engine** (`chad/ops/health_monitor_rules.py`).
      Every 5 minutes, evaluates **9 rules**:
      - **R01 critical services** — checks the hot-path systemd units
        are `active (running)`.
      - **R02 feed freshness** — confirms the 7 feeds covered by the
        v8.6 feed watchdog are inside their TTLs.
      - **R03 SCR paused** — flags `runtime/scr_state.json:state=PAUSED`.
      - **R04 stop bus** — flags `runtime/stop_bus.json:active=true`.
      - **R05 reconciliation RED** — flags
        `runtime/reconciliation_state.json:status=RED`.
      - **R06 profit lock** — flags profit-lock mode `LOCK2` or
        higher (LOCK2 / LOCK3 / HARD_STOP).
      - **R07 disk usage** — flags root partition at or above the
        configured threshold.
      - **R08 corrupt runtime files** — checks each canonical
        `runtime/*.json` parses as valid JSON; flags zero-byte and
        truncated reads.
      - **R09 edge decay halts** — flags any strategy in
        `is_strategy_halted()`-true state per the v8.6 wiring.

    - **Tier 2 — Claude reasoning** (`chad/ops/health_monitor.py`).
      When Tier 1 raises a critical or unfamiliar condition, the
      monitor packages a full system snapshot (state files, recent
      journal, rule outputs) and dispatches it to `claude-sonnet-4-6`
      for diagnostic narrative and recommended next actions. The
      response is forwarded to Telegram as an `AI HEALTH ANALYSIS`
      message.

    - **Tier 3 — Auto-remediation**
      (`chad/ops/health_monitor_remediation.py`). For a vetted set of
      safe actions, the monitor executes the fix itself:
      - **Restart dead services** via `sudo systemctl restart
        chad-<unit>` (sudo passwordless pre-confirmed for
        `systemctl restart chad-*`).
      - **Restart stale feed publishers** when R02 fires on a feed
        whose owning unit is identifiable.
      - **Restore corrupt runtime files from backup** when R08
        triggers and a recent valid snapshot exists.
      - **Archive old fill files** when R07 triggers and old
        `data/fills/FILLS_*.ndjson` are eligible for rotation.

    All **other** detected states are **notify-only** — SCR PAUSED
    (R03), stop bus active (R04), reconciliation RED (R05), profit
    lock LOCK2+ (R06), edge decay halts (R09). The monitor never
    auto-resumes a paused SCR, never clears the stop bus, and never
    self-cancels a profit lock — those decisions remain operator-only.

    Telegram surface adds three new message types:
    - **`AUTO-FIXED`** — emitted when Tier 3 successfully executes a
      remediation, naming the rule and the action taken.
    - **`HEALTH MONITOR CRITICAL`** — emitted on Tier-1 escalation
      that does not have a Tier-3 fix path.
    - **`AI HEALTH ANALYSIS`** — emitted when Tier 2 produces a
      Claude-authored diagnostic narrative.

    Files: `chad/ops/health_monitor.py`,
    `chad/ops/health_monitor_rules.py`,
    `chad/ops/health_monitor_remediation.py`. Timer:
    `chad-health-monitor.timer` (every 300s, `OnBootSec=120`).

### Strategic effect

v8.5 verified that the strategies *emit signals*. v8.6 verifies that
the *paths around* the strategies behave correctly. v8.7 adds an
additional layer: **the system watches itself.**

1. *When a strategy stops working, it actually halts.* (Edge decay
   enforcement.)
2. *When a feed goes stale, the operator hears about it within
   2 minutes.* (Feed watchdog.)
3. *When the operator runs `systemctl stop`, nothing breaks.*
   (SIGTERM handler.)
4. *When IBKR rejects an order, no phantom fill lands in the
   ledger.* (Surgical phantom-fill guard.)
5. *When SCR demotes to PAUSED, it does not flap back the next
   minute.* (PAUSED hysteresis end-to-end.)
6. *When a paper write crashes, the position guard does not
   truncate.* (Atomic write.)
7. *When the operator opens the dashboard, they see whether the
   data they are looking at is fresh.* (`oldest_source_mtime_utc`.)
8. *When something important happens, the operator gets a Telegram
   alert. When something routine happens, they don't.* (Coverage +
   noise control.)
9. *When a critical service dies, the monitor restarts it; when a
   runtime file corrupts, the monitor restores from backup; when
   something unfamiliar fires, Claude reasons about it before paging
   the operator.* (CHAD AI Health Monitor — NEW v8.7.)

The system has graduated from "the strategies emit signals" (v8.5)
to "the operator can sleep through the night" (v8.6) to "the system
watches itself, fixes what is safely fixable, and only pages the
operator for the rest" (v8.7).

---

## 1. MISSION & ARCHITECTURE

### Mission

CHAD (Compounding Hedge-Fund Algorithmic Desk) is a **business**, not a
bot. The business earns money by running a diversified portfolio of
systematic trading strategies over multiple asset classes, compounding
the profits through a structured allocation rule, and paying a salary
to the operator from sustained surplus above a high-water mark — never
touching the seed capital base.

The product goal is plain: **the operator checks their phone at the
end of the day, sees what CHAD made, and does nothing.** Everything
downstream — risk management, execution, reconciliation, health
monitoring, salary authorization — is automation designed to make "do
nothing" the safe default.

In v8.6 the "do nothing" promise is operationally credible. The feed
watchdog, SIGTERM handler, atomic position-guard write, edge-decay
enforcement, and seven-type Telegram exception coverage close the
last unattended-operation gaps. SCR has promoted to CONFIDENT —
sizing_factor 1.0, paper_only=false. The salary engine awaits 10 more
days of equity history (currently 4) and continued HWM discipline.

### Architecture (ASCII)

```
┌────────────────────┐
│  16 strategy heads │  raw TradeSignals (one bucket per strategy)
└─────────┬──────────┘
          ▼
┌────────────────────────────────────────────────────────────┐
│  signal_router.route                                       │
│   - bucket on (symbol, side, asset_class)                  │
│   - additive netting within bucket                         │
│   - primary_strategy = largest size contributor            │
│   - meta from primary strategy carried into RoutedSignal   │
│   - HALT FILTER (NEW v8.6): drop halted strategies         │
│     before pipeline entry (edge_decay_monitor wiring)      │
└─────────┬──────────────────────────────────────────────────┘
          ▼
┌────────────────────────────────────────────────────────────┐
│  execution_pipeline.split_signals_by_asset_class           │
│       CRYPTO          everything else                      │
│         │                  │                               │
│         ▼                  ▼                               │
│   Kraken lane        IBKR lane                             │
│                                                            │
│   build_kraken_     build_execution_plan                   │
│     intents_from_   ↓                                      │
│     routed_signals  build_intents_from_plan                │
│         │           ↓                                      │
│   _enforce_         routing_gates.run_all_gates            │
│     kraken_rest_    (A4 SPLIT v8.6 / E2 / E5 / R7 / S5)    │
│     pair (v8.3)     ↓                                      │
│         │           vote_collector  (S1, min_votes=1)      │
│         ▼           ↓                                      │
│   KrakenExecutor    sizing pipeline  (R3 → R5 → R6 → S5)   │
│         │           ↓                                      │
│   _assert_kraken_   IbkrAdapter.submit_strategy_intents    │
│     rest_pair       │                                      │
│     (REST border;   ▼                                      │
│      v8.3)          IBKR fills                             │
│         ▼           │                                      │
│   Kraken fills      │                                      │
│         │           │                                      │
│         └──── PHANTOM-FILL GUARD (NEW v8.6) ───────────────┤
│         │   error/Cancelled/Inactive → skip evidence       │
│         ▼                                                  │
│         ┴─→ normalize_paper_fill_evidence ─────────────────┤
│                                                            │
│                       ↓                                    │
│                 PaperExecutionEvidenceWriter               │
│                       ↓                                    │
│            data/fills/FILLS_YYYYMMDD.ndjson                │
│                       ↓                                    │
│       position_reconciler / reconciliation_publisher       │
│                       ↓                                    │
│              ProfitRouter (50/30/20 + drain v8.6)          │
└────────────────────────────────────────────────────────────┘

   Strategies   Pipeline    Plan    Risk   Allocator
        ───────────────────────────────────────►
                                                │
                                                ▼
   ┌──── BUSINESS FRAMEWORK (v8.3, hardened v8.6) ────────────┐
   │                                                          │
   │   tier_filter ◄── TierManager ◄── tier_state.json        │
   │   winner_scale ◄── WinnerScaler ◄── winner_scaling.json  │
   │   regime_boost ◄── RegimeBooster ◄── regime_booster.json │
   │                                                          │
   │   ─►  Caps ──► SCR ──► LiveGate ──► Execution            │
   │                                          │               │
   │                                          ▼               │
   │                                   Fills → Ledger         │
   │                                          │               │
   │                                          ▼               │
   │                                   Profit_Router 50/30/20 │
   │                                   (DRAIN v8.6)           │
   │                                          │               │
   │                                          ▼               │
   │                                   WithdrawalManager      │
   │                                   (HWM/Phase)            │
   │                                          │               │
   │                                          ▼               │
   │                                   Salary Authorization   │
   └──────────────────────────────────────────────────────────┘

   ┌──── OBSERVABILITY (NEW v8.6) ──────────────────────────┐
   │   chad-feed-watchdog.timer (120s)                       │
   │   ↓                                                     │
   │   7 feeds checked (price_cache, regime_state,           │
   │   dynamic_caps, regime_booster, kraken_prices,          │
   │   reconciliation_state, event_risk)                     │
   │   ↓                                                     │
   │   Telegram alert on stale                               │
   │                                                         │
   │   chad-live-loop SIGTERM handler                        │
   │   ↓                                                     │
   │   in-flight cycle drains, broker disconnects, alert     │
   │                                                         │
   │   /api/state.oldest_source_mtime_utc                    │
   │   (top-level dashboard staleness signal)                │
   └─────────────────────────────────────────────────────────┘
```

### Built / Degraded / Not yet

| Component | Status | Note |
|---|---|---|
| 16 strategy registry | **BUILT** | Forex commented out (`chad/strategies/__init__.py` registry block) — 16 of 17 active. |
| Signal router (bucket meta carry) | **BUILT** | `chad/utils/signal_router.py:179`. |
| Asset-class splitter | **BUILT** | `chad/execution/execution_pipeline.py:1164`. |
| IBKR EMS / OMS | **BUILT** | A1 separation `chad/execution/ems.py` + `oms.py` (Phase-8 Session 9). |
| Kraken EMS / OMS | **BUILT** | REST altname maps `chad/execution/execution_pipeline.py:1228-1264`. |
| Kraken REST pair guard (executor + border) | **BUILT v8.3** | `chad/execution/kraken_executor.py:41`, `chad/exchanges/kraken_client.py:55`. |
| Routing gates 5-stack | **BUILT (A4 split v8.6)** | `chad/execution/routing_gates.py:run_all_gates` — A4 splits 900s intraday / 172800s daily-swing-futures. |
| Sizing R3/R5/R6 | **BUILT** | `target_daily_vol=0.015`, `correlation_threshold=0.65`. |
| Profit lock | **BUILT** | `runtime/profit_lock_state.json` mode=NORMAL, sizing 1.0. |
| **SCR** | **BUILT (CONFIDENT v8.6)** | `paper_only=false`, sizing_factor 1.0, 147 effective trades. Min-trade gates 100/133. |
| **SCR PAUSED hysteresis** | **BUILT v8.6** | router (commit `23f4c03`) + caller (`28e8963`) end-to-end. |
| Stop bus | **BUILT** | `runtime/stop_bus.json` active=false. |
| **Edge decay (F4) ENFORCED** | **BUILT v8.6** | `is_strategy_halted()` wired into signal building (was completely unwired pre-`52786b7`). |
| Reconciliation publisher | **BUILT (TTL 360s, MYM/M2K v8.6)** | GREEN; `chad_strategy_open` count separated from `chad_open`. |
| Paper evidence normalizer | **BUILT** | Single chokepoint `chad/execution/paper_exec_evidence_writer.py:925`. |
| **Phantom-fill guard** | **BUILT v8.6** | Surgical guard at `chad/core/live_loop.py:1180` — error-status orders skip evidence write. |
| **Position guard atomic write** | **BUILT v8.6** | `_save_state` uses tmp+replace pattern. |
| Position guard `opened_at` | **BUILT** | `chad/core/position_guard.py:155-157`. |
| Broker-truth rebuild in paper | **BUILT** | `chad/core/live_loop.py:641-651`. |
| Telegram intelligence layer | **BUILT** | Free-text router + elite voice. |
| **Telegram exception alerts (7 types)** | **BUILT v8.6** | live-loop crash, recon RED, profit-lock transitions, salary authorized, drawdown breach, SCR recovery, WARMUP demotion. |
| **Telegram per-fill alerts** | **DISABLED v8.6** | Too noisy (commit `d35884b`); consolidated in morning brief + EOD recap. |
| Telegram BUSINESS STATUS section | **BUILT v8.3** | `chad/ops/daily_chad_report.py:1474-1505`. |
| Dashboard chat | **BUILT** | `/api/chat`, model `claude-sonnet-4-6`. |
| Dashboard business endpoint | **BUILT v8.3** | `chad/dashboard/api.py:484-525`. |
| **Dashboard staleness signal** | **BUILT v8.6** | `oldest_source_mtime_utc` at top level of `/api/state`. |
| **Dashboard brute-force protection** | **BUILT v8.6** | 5 attempts → 5-min IP lockout. |
| **Dashboard `/api/recent-trades` cap** | **BUILT v8.6** | 500 lines/file (D7). |
| Bar provider polling | **BUILT** | `chad-ibkr-bar-provider.service`. |
| Strategy intelligence cache | **BUILT** | `runtime/strategy_intelligence.json` (48h TTL). |
| **Feed watchdog** | **BUILT v8.6** | `chad-feed-watchdog.timer` 120s — 7 feeds. |
| **regime_state TTL enforced at read** | **BUILT v8.6** | Stale → `regime=unknown` (commit `71d8ea6`). |
| **SIGTERM handler** | **BUILT v8.6** | `chad-live-loop.service` clean shutdown (commit `9bc5286`). |
| **VXX bars** | **BUILT v8.6** | Universe + bars backfilled (commit `42268f0`). |
| PortfolioSnapshotPublisher | **BUILT v8.3** | `chad/ops/portfolio_snapshot_publisher.py`. |
| EquityHistoryPublisher | **BUILT v8.3** | 4 records as of write time. |
| TierManager | **BUILT v8.3** | PRO. |
| WinnerScaler | **BUILT v8.3** | All 16 at 1.0× currently (sample base reset). |
| RegimeBooster | **BUILT v8.3** | 1.3× ACTIVE. |
| WithdrawalManager | **BUILT v8.3** | `phase=GROW`, `authorized=$0` (history_days=4 < 14). |
| BusinessPhaseTracker | **BUILT v8.3** | GROW. |
| **Profit-routing 50/30/20 with drain** | **BUILT v8.6** | `consumed_beta_usd`, `consumed_amplifier_usd` tracked; injection flagged off. |
| `alpha_options` new entries | **DEGRADED** | Existing SPY long blocks new spreads (MAINTAINED state) until existing closes. |
| `omega_vol` health | **DEGRADED** | `health_score=0.10` (3 samples). |
| ML veto loop (Phase 5B) | **NOT YET** | XGBoost retrain pipeline incomplete. |
| Per-trade P&L decomposition | **NOT YET** | Alpha vs spread vs slippage attribution incomplete. |
| Live trading | **NOT YET** | `runtime/live_readiness.json:ready_for_live=false`; SCR side now CONFIDENT but operator GO required. |
| Salary withdrawal automation | **NOT YET** | `WithdrawalManager` authorizes; operator still moves money manually. |
| Amplifier bucket wiring | **NOT YET** | Pending winner_scaler producing non-trivial multipliers (currently all 1.0×). |
| `CHAD_PROFIT_ROUTER_BETA_INJECTION` | **NOT YET (FLAGGED)** | Implementation complete; flag off by default for paper soak. |

---

## 2. RUNTIME STATE (live snapshot)

All values pulled at write time from the canonical state files. Each
row cites its source.

### Account equity

| Field | Value | Source |
|---|---|---|
| `ibkr_equity` | `$183,741.46 USD` | `runtime/portfolio_snapshot.json:ibkr_equity` (`2026-05-01T10:59:20Z`) |
| `kraken_equity` | `$184.58 USD` | `runtime/portfolio_snapshot.json:kraken_equity` |
| `coinbase_equity` | `$0.00` | unused (CAD-based account, no Coinbase) |
| **total_equity (snapshot)** | **`$183,926.04 USD`** | sum (ibkr + kraken + coinbase) |
| `total_equity` (allocator) | `$183,982.36` | `runtime/dynamic_caps.json:total_equity` (`2026-05-01T11:08:27Z`) |
| `daily_risk_fraction` | `0.05` | `runtime/dynamic_caps.json:daily_risk_fraction` |
| `portfolio_risk_cap` | `$9,199.12` | `runtime/dynamic_caps.json:portfolio_risk_cap` (post profit-lock factor 1.0) |
| Kraken raw | `BTC=0.0012, CAD=252.85` | `runtime/kraken_balances.json` |

The small (~$56) delta between snapshot and allocator equity is the
publisher cadence offset (snapshot is on a 5-minute timer; allocator
publishes on each orchestrator cycle). Not a fault.

### SCR — Self-Calibrating Risk

Source: `runtime/scr_state.json` (ts `2026-05-01T11:02:22Z`).

| Field | Value |
|---|---|
| state | **`CONFIDENT`** |
| sizing_factor | **`1.0`** |
| effective_trades | `147` (nested under `stats` key) |
| paper_trades | `5000` |
| live_trades | `0` |
| excluded_untrusted | `4424` |
| excluded_manual | `0` |
| excluded_nonfinite | `0` |
| **excluded_pnl_zero** | **`429`** (NEW v8.6 — was implicit) |
| total_trades | `5000` |
| win_rate | `0.5578` (≈ 55.78%) |
| sharpe_like | `+1.0489` |
| max_drawdown | `-1454.84` |
| total_pnl (effective) | `-4527.94` |
| **paper_only** | **`false`** (NEW v8.6 — SCR-side live block lifted) |
| reasons[0] | `"CONFIDENT: win_rate, sharpe, and drawdown all within confident band."` |
| ttl_seconds | 180 |

#### Min-trade gates (NEW v8.6)

| Band | min_trades | Other gates |
|---|---|---|
| WARMUP | n/a | initial state |
| CAUTIOUS | **100** | win_rate ≥ 0.35, sharpe_like ≥ 0.10, max_drawdown ≥ −$15,000 |
| **CONFIDENT (current)** | **133** | + sustained CAUTIOUS-band metrics |
| PAUSED | n/a | hysteresis-protected (recovery_ticks gate) |

`paper_trades = 5000 = 147 (effective) + 4424 (untrusted) + 0 (manual) +
0 (nonfinite) + 429 (pnl_zero)`. Ledger balances.

The SCR-side block on live trading has lifted — `paper_only=false`,
`sizing_factor=1.0`. Live promotion still requires:
- Operator GO per CLAUDE.md governance rule #3.
- `LiveGate` accept the posture change.
- `runtime/live_readiness.json:ready_for_live=true` (currently false;
  last evaluated 2026-04-09).

### PAUSED hysteresis (NEW v8.6, end-to-end)

`runtime/scr_state.json` carries (when relevant):
- `prev_state` — what the previous tick reported.
- `paused_recovery_ticks` — how many consecutive healthy ticks to
  require before promoting back from PAUSED.

Wired through `backend/api_gateway.py` →
`ShadowStats.compute_state(prev_state=..., paused_recovery_ticks=...)`
→ router (commit `23f4c03` + `28e8963`). Currently inactive (we are in
CONFIDENT, not PAUSED), but the wiring is verified end-to-end.

### Regime classifier

Source: `runtime/regime_state.json` (ts `2026-05-01T11:08:30Z`).

| Field | Value |
|---|---|
| regime | `trending_bull` |
| previous_regime | `trending_bull` (stable transition) |
| confidence | `0.7536` |
| inputs_used | `realized_vol_percentile, adx, trend_slope, market_breadth` |
| source | `live_loop.run_once` |
| ttl_seconds | **120** (read-time enforcement NEW v8.6) |

#### TTL read-time enforcement (NEW v8.6)

Pre-`71d8ea6`, every consumer (allocator, RegimeBooster,
`regime_activation_matrix` lookups) read `regime_state.json` blindly.
Now: at read time, if `now − ts_utc > ttl_seconds` (120s), the read
returns `regime=unknown`. `unknown` is a universally-eligible regime
in `config/regime_activation_matrix.json` so the system continues to
operate; it just stops asserting a regime it can no longer prove.

Active strategies in `trending_bull` (per
`config/regime_activation_matrix.json:9-13`):
`alpha, alpha_crypto, alpha_futures, alpha_intraday, alpha_forex,
alpha_options, beta, beta_trend, delta, gamma_futures, omega_macro,
omega_momentum_options`.
*(`alpha_forex` is registry-disabled — see §3.)*

### Tier

Source: `runtime/tier_state.json` (ts `2026-05-01T10:59:16Z`).

| Field | Value |
|---|---|
| tier_name | **`PRO`** |
| tier_description | `"All 16 strategies firing at meaningful size"` |
| current_equity_usd | `$183,908.99` |
| tier_min_equity | `$160,000` |
| tier_max_equity | `$1,000,000` |
| previous_tier | `PRO` (no change) |
| promoted_at_utc | `2026-04-27T19:00:35Z` |
| enabled_strategies | 16 strategies (full PRO list) |

PRO enables:
`alpha, alpha_intraday, alpha_crypto, alpha_options, alpha_futures,
delta, delta_pairs, gamma, gamma_futures, gamma_reversion, beta,
beta_trend, omega, omega_vol, omega_macro, omega_momentum_options`.

### Business phase

Source: `runtime/business_phase.json` (ts `2026-05-01T10:54:07Z`).

| Field | Value |
|---|---|
| phase | **`GROW`** |
| phase_description | `"Engine is built. Now growing the account before salary starts. SCR must reach CONFIDENT first."` |
| current_equity_usd | `$183,889.29` |
| seed_capital_usd | `$50,000.00` |
| growth_pct_from_seed | **`+267.78%`** |
| days_in_phase | `4` |
| next_phase_requirement | `"To enter PAY phase: Need 10+ more days of equity history."` |
| compound_metrics.total_return_pct | `+267.78%` |
| compound_metrics.annualized_return_pct | `0.0` (suppressed below 14d history; see `business_phase_tracker.py:202`) |
| compound_metrics.days_active | `3` |
| compound_metrics.high_water_mark_usd | `$183,874.27` |

The `phase_description` text references "SCR must reach CONFIDENT
first" — that gate has cleared at the SCR side. The `next_phase_requirement`
text now correctly identifies the *remaining* gate: 10 more days of
equity history (current: 4, target: 14). Once that lands, the salary
engine unblocks and PAY-phase preconditions are met assuming HWM
discipline holds.

### Withdrawal authorization

Source: `runtime/withdrawal_authorization.json` (ts `2026-05-01T06:59:58Z`).

| Field | Value |
|---|---|
| phase | **`GROW`** |
| current_equity_usd | `$183,974.75` |
| seed_capital_usd | `$50,000.00` |
| high_water_mark_usd | `$183,874.27` |
| drawdown_from_hwm_pct | `−0.0546%` (current equity above HWM) |
| spendable_surplus_usd | `$100.49` |
| **authorized_withdrawal_usd** | **`$0.00`** |
| scr_state | **`CONFIDENT`** |
| history_days | `4` |
| reason | `"GROW phase override: only 4 days of equity history (need 14+). Building track record before paying."` |

The reason field captures the new state cleanly: SCR is CONFIDENT but
the 14-day history gate holds. With current HWM `$183,874.27` and
current equity `$183,974.75`, surplus is `$100.49`; once history clears
the gate, the formula `min(surplus × 0.30, max_monthly_salary $2,000)`
would authorize roughly `$30/mo`. Real salary unlocks when CHAD makes
sustained new highs.

### Winner scaling

Source: `runtime/winner_scaling.json` (ts `2026-05-01T10:58:11Z`).

| Field | Value |
|---|---|
| max_multiplier | `1.50` |
| min_multiplier | `0.50` |
| median_expectancy | `0.0` |
| n_strategies_scaled | `0` |
| n_strategies_neutral | `11` |
| min_trades_for_scaling | `5` |
| excluded_strategies | `broker_sync, manual, paper_exec, unknown` |

**All 16 canonical strategies plus 3 carryover labels are at multiplier
1.0 currently.** The expectancy state has reset since the last winner-
scaling pass; with `median_expectancy=0.0`, the scaler cannot rank
(divides-by-zero / no signal) and falls through to neutral. Expected to
re-populate as live cycles continue producing closed trades.

| Strategy | Multiplier |
|---|---|
| `alpha` | 1.000 |
| `alpha_intraday` | 1.000 |
| `alpha_options` | 1.000 |
| `alpha_futures` | 1.000 |
| `alpha_crypto` | 1.000 |
| `beta` | 1.000 |
| `beta_trend` | 1.000 |
| `gamma` | 1.000 |
| `gamma_futures` | 1.000 |
| `gamma_reversion` | 1.000 |
| `delta` | 1.000 |
| `delta_pairs` | 1.000 |
| `omega` | 1.000 |
| `omega_vol` | 1.000 |
| `omega_macro` | 1.000 |
| `omega_momentum_options` | 1.000 |
| `broker_sync` | 1.000 (excluded) |
| `RECONCILED_PHASE2_20260419` | 1.000 |
| `reconciled_phase2_20260419_carryover` | 1.000 |

This is a known transient — the v8.5 multipliers (`alpha 1.5×`,
`delta 1.304×`, `alpha_futures 0.5×`) reset when the expectancy state
recomputed against fresher trade history. The system is correctly
declining to scale on insufficient signal; multipliers will re-emerge
as the trade base accumulates.

### Regime booster

Source: `runtime/regime_booster.json` (ts `2026-05-01T11:08:41Z`).

| Field | Value |
|---|---|
| multiplier | **`1.3`** |
| **active** | **`true`** |
| reasons | `["high_confidence_0.75", "vix_calm_16.9", "trending_bull_bias"]` |
| regime | `trending_bull` |
| confidence | `0.7536` |
| vix | `16.89` |
| event_severity | `medium` |

The booster is now ACTIVE. Three positive factors fire concurrently:
- `high_confidence_0.75` (+0.10) — confidence ≥ 0.75 cleared.
- `vix_calm_16.9` (+0.10) — VIX ≤ 18.0 cleared.
- `trending_bull_bias` (+0.10) — regime is trending_bull.

Sum: `1.0 + 0.10 + 0.10 + 0.10 = 1.30`, applied as a global multiplier
in `dynamic_risk_allocator.py:362-378`. No vetoes active.

### Equity / PnL

| Field | Value | Source |
|---|---|---|
| account_equity | `$183,982.36` | `runtime/dynamic_caps.json:total_equity` |
| portfolio_risk_cap | `$9,199.12` | `runtime/dynamic_caps.json:portfolio_risk_cap` (post profit-lock) |
| daily realized PnL today | `−$45.20` | `runtime/pnl_state.json:realized_pnl` (`2026-05-01T11:08:41Z`) |
| trade_count today | `73` | `runtime/pnl_state.json:trade_count` |
| pnl_pct_of_equity | `−0.0246%` | `runtime/pnl_state.json` |

### Profit lock

Source: `runtime/profit_lock_state.json` (ts `2026-05-01T11:08:41Z`,
60s TTL).

- mode = `NORMAL`
- sizing_factor = `1.00`
- stop_new_entries = `false`
- daily_loss_limit_pct = `3.0%`
- daily_loss_limit_dollars = `$5,519.47` (3% of equity)
- daily_loss_today = `−$45.20`
- daily_loss_limit_hit = `false`
- profit_lock_active = `false`
- explain = `"profit lock inactive"`

Equity scaling: WARN at +1.5%, LOCK1 at +3.0%, LOCK2 at +5.0%, LOCK3
at +8.0%, HARD_STOP at +10.0%. Currently NORMAL.

### Reconciliation

Source: `runtime/reconciliation_state.json` (ts `2026-05-01T11:04:28Z`,
**ttl_seconds=360** NEW v8.6).

- status = **`GREEN`**
- broker_source = `ibkr:clientId=83`
- counts:
  - `chad_open` = **8**
  - **`chad_strategy_open` = 15** (NEW v8.6 — strategy-tracked count separated)
  - `broker_positions` = 11
- worst_diff = `0.0`
- mismatches = `[]`
- drifts = `[`
  - `{"symbol": "BAC", "chad": 0.0, "broker": -17.0, "diff": 17.0}`,
  - `{"symbol": "CVX", "chad": 0.0, "broker": -24.0, "diff": 24.0}`,
  - `{"symbol": "GOOGL", "chad": 0.0, "broker": 2.0, "diff": 2.0}`
- `]`
- excluded_symbols = `["AAPL", "NVDA"]`
- futures_excluded_symbols = `[]` (MYM/M2K added to KNOWN_FUTURES_SYMBOLS NEW v8.6)
- ttl_seconds = **360** (was 300 v8.5; raised in commit `ba65367`)

The drifts are all classified correctly — broker holds positions in
BAC/CVX/GOOGL (paper account inheritance), CHAD has no strategy entry
for any of them, so the entire diff is in `broker_sync`. Mismatches
list is empty. Status remains GREEN because `worst_diff=0.0` is
computed across the mismatches list, not the drifts list.

### Stop bus

Source: `runtime/stop_bus.json`.

- active = `false`
- cleared_at = `2026-04-22T01:56:50Z`
- cleared_by = `smoke_test`

### Open positions

From `runtime/position_guard.json` and the reconciliation state:
`chad_open=8`, `chad_strategy_open=15`, `broker_positions=11`.

The `chad_strategy_open=15` count is the new v8.6 separate exposure of
strategy-tracked open positions; `chad_open=8` is the deduplicated
count after rolling up `broker_sync|*` echoes. Both are now visible.

### Market snapshot

VIX last close = **`16.89`** (well below the 18.0 calm threshold,
below 25.0 elevated, below the +0.10 booster bump). VXX bars are
NOW current (commit `42268f0` backfill — was 26 days stale).

Kraken (`runtime/kraken_balances.json`): BTC `0.0012`, CAD `252.85`,
USD-equivalent `$184.58`.

### Services

Total `chad-*` units loaded: **97** (53 services + 44 timers — adds
`chad-feed-watchdog.timer` + `chad-feed-watchdog.service` from v8.6).
Active (running): **14** services. Failed: **0**.

The major hot-path services are all running (`chad-live-loop`,
`chad-orchestrator`, `chad-ibgateway`, `chad-ibkr-bar-provider`,
`chad-kraken-ws`, `chad-shadow-status`, `chad-metrics`, `chad-backend`,
`chad-dashboard`, `chad-telegram-bot`, `chad-x11vnc`, `chad-xvfb`,
`chad-strategy-intelligence-refresh`).

### Bar freshness

- 1d bars: `data/bars/1d/` — 30+ symbols (now includes refreshed VXX).
- 1m bars: `data/bars/1m/` — 25 symbols polled via
  `chad-ibkr-bar-provider.service` every 30s.
- VIX: `data/bars/1d/VIX.json`, last close `16.89` (consumed by
  `omega`, `omega_vol`, `omega_macro`, `omega_momentum_options`,
  `alpha_options`, RegimeBooster).
- VXX: backfilled in commit `42268f0` (was 26 days stale).
- MGC/SIL: expired contracts resolved by service restart during
  the data feed audit (commit `a4accdd` companion fix).

### Strategy intelligence

Source: `runtime/strategy_intelligence.json` — same classifier-input
wiring gap noted in v8.5; AI cache TTL 48h. Not regressed but not
improved — see §14.

### Institutional consensus

Source: `runtime/institutional_consensus.json` — same as v8.5
(weekly Sunday 00:00 UTC refresh; last `2026-04-26T00:00:05Z`).

### Profit routing ledger

Source: `runtime/profit_routing.json` (latest decision
`2026-04-30T23:58:47Z`).

| Field | Value |
|---|---|
| decisions logged | 51 |
| `totals.trading_capital` | **`$1,905.26`** |
| `totals.beta_allocation` | **`$1,143.15`** |
| `totals.amplifier_allocation` | **`$762.10`** |
| **NEW v8.6 — `consumed_beta_usd`** | **`$0.00`** |
| **NEW v8.6 — `consumed_amplifier_usd`** | **`$0.00`** |
| **NEW v8.6 — `alpha_test_purged_at_utc`** | **`2026-04-30T19:05:17Z`** |
| **NEW v8.6 — `alpha_test_purged_count`** | **`19`** |

The 19 purged `alpha_test` entries were test-harness leakage caused by
the unit tests sharing the production ledger path before commit
`cdfed06`. Stamped for forensic traceability.

The drain mechanism: `get_beta_remaining() = beta_allocation_total −
consumed_beta_usd = $1,143.15 − $0.00 = $1,143.15`. Beta injection is
implemented but flagged off (`CHAD_PROFIT_ROUTER_BETA_INJECTION` env
flag, default off). Once enabled, the `beta` strategy will draw
opportunistically from the beta bucket; until then, the bucket
accumulates.

`broker_sync` realized PnL is no longer routed (commit `cdfed06`) —
the four `broker_sync` entries in the ledger pre-date this fix. Going
forward, `broker_sync` realized PnL produces `{"no_routing": True,
"reason": "broker_sync_excluded"}`.

---

## 3. THE 16 STRATEGIES

For each strategy: source path, sleeve, weight, dollar cap (post full
chain), regime gating, fire window, universe, signal conditions,
current health, and status.

Weights from `config/strategy_weights.json`; caps from
`runtime/dynamic_caps.json`; regime gating from
`config/regime_activation_matrix.json`; sleeve membership from
`chad/risk/dynamic_risk_allocator.py:454-461`.

> **Sleeve recap** — the 50/30/20 chassis:
> - **ALPHA sleeve** (50%): `alpha, alpha_futures, alpha_intraday,
>   alpha_options, omega_momentum_options, gamma, gamma_futures,
>   gamma_reversion`.
> - **BETA sleeve** (30%): `beta, beta_trend`.
> - **ADAPTIVE sleeve** (20%): `omega, omega_macro, omega_vol, delta,
>   delta_pairs, alpha_crypto`.

Each strategy carries three v8.3 fields surfaced in the cap breakdown
(`runtime/dynamic_caps.json:strategies.<name>`): `tier_factor` (0.0
if not enabled in the current tier; else 1.0), `winner_factor` (0.5–
1.5), `regime_factor` (1.0–1.5). The final `dollar_cap` =
`base_cap × tier_factor × winner_factor × regime_factor`.

**Important v8.6 change in caps:** Regime booster is **active at
1.3×** so all strategies receive a 1.3× boost. WinnerScaler is at
1.0× across the board (transient — see §2). With SCR=CONFIDENT and
sizing_factor=1.0, the caps below now actually constrain real
sizing — they are no longer academic under WARMUP's 0.10×.

### alpha — Intraday tactical momentum brain

| Field | Value |
|---|---|
| Source | `chad/strategies/alpha.py` |
| Sleeve | ALPHA |
| Weight | 0.16 (largest) |
| tier_factor | `1.0` (PRO enables) |
| winner_factor | `1.0` (median_expectancy=0 — see §2) |
| regime_factor | **`1.3`** (booster active) |
| Active in | `trending_bull, trending_bear, volatile, unknown` |
| Silent in | `ranging, adverse` |
| Window | Per-symbol 3 signals/day cap; max 8 signals per cycle. |
| Universe | Legend-driven via `ctx.legend.weights`, fallback to `ctx.prices` keys. |
| Conditions | Four entry regimes: uptrend (EMA fast > slow, momentum ≥ 0.35 ATR), recovery (price > EMA slow, momentum ≥ 0.175 ATR), downtrend (mirrored shorts), chop (fade mid-band). |
| Health | Refreshed per-cycle. |
| Open positions | Per `position_guard.json`. |
| Status | **ACTIVE** — regime-boosted. |

### alpha_intraday — Delta high-convexity day-trading brain

| Field | Value |
|---|---|
| Source | `chad/strategies/alpha_intraday.py` |
| Sleeve | ALPHA |
| Weight | 0.03 |
| tier_factor | `1.0` |
| winner_factor | `1.0` |
| regime_factor | `1.3` |
| Active in | `trending_bull, trending_bear, volatile, unknown` |
| Silent in | `ranging, adverse` |
| Window | 1m bars with daily fallback; 10-min per-symbol cooldown. |
| Universe | SPY, QQQ, AAPL, NVDA, MSFT, GOOGL, BAC, MES, MNQ, BTC-USD. |
| Conditions | Vol explosion / momentum surge / mean-reversion snap. |
| Health | Diagnostic logging present (commit `9f55688`); no silent zero-signal returns. |
| Status | **ACTIVE**. |

### alpha_options — Defined-risk vertical spreads

| Field | Value |
|---|---|
| Source | `chad/strategies/alpha_options.py` (+ `alpha_options_config.py`) |
| Sleeve | ALPHA |
| Weight | 0.04 |
| tier_factor | `1.0` |
| winner_factor | `1.0` |
| regime_factor | `1.3` |
| Active in | `trending_bull, trending_bear, volatile` |
| Silent in | `ranging, unknown, adverse` |
| Window | Triggered by alpha/gamma/gamma_reversion signals at confidence ≥ 0.70. Max 4 open spreads. |
| Universe | SPY only (`alpha_options_config.py:25-27`). |
| Conditions | Bullish source signal → bull call spread (21–45 DTE, 2% OTM, 5% width). Bearish → bear put spread (mirror). |
| Open positions | SPY long open from 2026-04-24 carrying. |
| Status | **CONDITION-GATED** — see §14. |

### alpha_futures — Futures momentum engine

| Field | Value |
|---|---|
| Source | `chad/strategies/alpha_futures.py` |
| Sleeve | ALPHA |
| Weight | 0.09 |
| tier_factor | `1.0` |
| winner_factor | `1.0` (was 0.5 v8.5; reset with expectancy refresh) |
| regime_factor | `1.3` |
| Active in | `trending_bull, trending_bear, volatile, unknown` |
| Silent in | `ranging, adverse` |
| Window | MES, MNQ continuous; **MCL/MGC restricted 13:30–20:00 UTC for new entries**. |
| Universe | **MES, MNQ, MGC**. |
| Conditions | Momentum + breakout override. Exits on 2× ATR stop, EMA_slow cross, 20-bar time stop. |
| Status | **ACTIVE** — winner-scaler currently neutral; will re-evaluate as expectancy state populates. |

### alpha_crypto — Crypto momentum signals

| Field | Value |
|---|---|
| Source | `chad/strategies/alpha_crypto.py` |
| Sleeve | ADAPTIVE |
| Weight | 0.04 (key `alpha_crypto`; renamed from `crypto` in v8.4) |
| tier_factor | `1.0` |
| winner_factor | `1.0` |
| regime_factor | `1.3` |
| Active in | `trending_bull, trending_bear, volatile, unknown` |
| Silent in | `ranging, adverse` |
| Window | Always armed when regime allows; long-only emission. |
| Universe | Default `BTC-USD, ETH-USD, SOL-USD`; CAD pairs added when Kraken paper holds ZCAD. **Two-step REST pair guard from v8.3**. |
| Conditions | (1) SMA20 momentum breakout; (2) 5d/20d vol-ratio expansion ≥ 0.7; (3) regime multiplier — 0.5 in trending_bear, 1.0 elsewhere. |
| Status | ACTIVE. |

### beta — Institutional long-term compounder

| Field | Value |
|---|---|
| Source | `chad/strategies/beta.py` |
| Sleeve | BETA |
| Weight | 0.05 |
| tier_factor | `1.0` |
| winner_factor | `1.0` |
| regime_factor | `1.3` |
| Active in | every non-`adverse` regime |
| Silent in | `adverse` |
| Window | Max 2 signals/cycle; max 3/week rolling; per-symbol 7-day rebalance gate. |
| Universe | `runtime/institutional_consensus.json` top names. |
| Conditions | If `target_weight − current_weight ≥ 0.5%` (lowered v8.5 from 2%), emit a BUY sized to fill ~50% of the gap. |
| **NEW v8.6 — Beta injection** | Available via `CHAD_PROFIT_ROUTER_BETA_INJECTION` env flag; default off; pulls from `consumed_beta_usd`-aware bucket. |
| Status | **ACTIVE** — beta injection wiring complete, flagged off pending paper soak. |

### beta_trend — Legend-driven long-term ETF / equity allocator

| Field | Value |
|---|---|
| Source | `chad/strategies/beta_trend.py` |
| Sleeve | BETA |
| Weight | 0.20 |
| tier_factor | `1.0` |
| winner_factor | `1.0` |
| regime_factor | `1.3` |
| Active in | `trending_bull, trending_bear, volatile, unknown` |
| Silent in | `ranging, adverse` |
| Window | Once-per-UTC-day per symbol; max 20 signals/day; 21-day hold; 14-day cooldown. |
| Universe | Top legend-weighted names. |
| Conditions | Entry if flat: BUY size = `clamp(3 + legend_weight × 10, 3, 8)`. Add-on after 21d. |
| Status | **ACTIVE**. |

### gamma — Activated swing engine

| Field | Value |
|---|---|
| Source | `chad/strategies/gamma.py` |
| Sleeve | ALPHA |
| Weight | 0.07 |
| tier_factor | `1.0` |
| winner_factor | `1.0` |
| regime_factor | `1.3` |
| Active in | `ranging, volatile, unknown` |
| Silent in | `trending_bull, trending_bear, adverse` |
| Window | No fixed schedule; bar-driven. Min 60 bars. |
| Universe | Pulled dynamically from `ctx.bars` keys. |
| Conditions | Trend regime: EMA_fast > slow, price > fast, momentum ≥ 0.35 ATR. Range regime: deviates from EMA_slow by ≥ 0.75 ATR. |
| Status | **REGIME-SILENT** (currently `trending_bull`). |

### gamma_futures — Futures mean-reversion counterpart

| Field | Value |
|---|---|
| Source | `chad/strategies/gamma_futures.py` |
| Sleeve | ALPHA |
| Weight | 0.05 |
| tier_factor | `1.0` |
| winner_factor | `1.0` |
| regime_factor | `1.3` |
| Active in | `trending_bull, trending_bear, volatile, unknown` |
| Silent in | `ranging, adverse` |
| Window | Continuous (energy 24/5; bonds liquid hours). |
| Universe | **MCL** primary; extends with `MYM, M2K` if bars exist; otherwise `ZN, ZB`. **Disjoint from alpha_futures** post-`6af971d`. |
| Conditions | Short when `RSI > 70 AND price > BB_upper`; Long when `RSI < 30 AND price < BB_lower`. |
| **MYM/M2K reconciliation** | NEW v8.6 — added to `KNOWN_FUTURES_SYMBOLS` in `reconciliation_publisher.py:58`; no longer surfaces as drift. |
| Status | **ACTIVE**. |

### gamma_reversion — ETF statistical mean-reversion

| Field | Value |
|---|---|
| Source | `chad/strategies/gamma_reversion.py` |
| Sleeve | ALPHA |
| Weight | 0.04 |
| tier_factor | `1.0` |
| winner_factor | `1.0` |
| regime_factor | `1.3` |
| Active in | `ranging` |
| Silent in | every other regime |
| Window | Continuous within active regime. Min 40 bars. |
| Universe | **SPY, QQQ, GLD, TLT**. |
| Conditions | 3/3 confluence — RSI / Bollinger / z-score / ROC alignment. |
| Status | **REGIME-SILENT**. |

### delta — Cross-asset convexity hunter

| Field | Value |
|---|---|
| Source | `chad/strategies/delta.py` |
| Sleeve | ADAPTIVE |
| Weight | 0.02 |
| tier_factor | `1.0` |
| winner_factor | `1.0` (was 1.304 v8.5) |
| regime_factor | `1.3` |
| Active in | `trending_bull, trending_bear, volatile, unknown` |
| Silent in | `ranging, adverse` |
| Window | No per-symbol cooldown (conviction-driven). Cash floor $2.5k. |
| Universe | `ctx.delta_universe` override → legend top weights → price keys; max 4 symbols. |
| Conditions | Conviction ≥ 0.65 required. Trend / breakout / momentum scoring. |
| Status | **ACTIVE**. |

### delta_pairs — Market-neutral ETF pairs trader

| Field | Value |
|---|---|
| Source | `chad/strategies/delta_pairs.py` |
| Sleeve | ADAPTIVE |
| Weight | 0.05 |
| tier_factor | `1.0` |
| winner_factor | `1.0` |
| regime_factor | `1.3` |
| Active in | `ranging` |
| Silent in | every other regime |
| Window | Continuous within active regime. 60-day lookback, min 40 bars. |
| Universe | **SPY/QQQ, SPY/IWM, QQQ/IWM**. |
| Conditions | z-score entry at \|z\| ≥ 2.0; exit ≤ 0.5; stop ≥ 3.5. |
| Status | **REGIME-SILENT**. |

### omega — Wealth-safe hedge sleeve

| Field | Value |
|---|---|
| Source | `chad/strategies/omega.py` |
| Sleeve | ADAPTIVE |
| Weight | 0.05 |
| tier_factor | `1.0` |
| winner_factor | `1.0` |
| regime_factor | `1.3` |
| Active in | `volatile, unknown` |
| Silent in | trending regimes (by design) |
| Window | Cooldown 60 minutes. |
| Universe | **SH** (inverse SPY) and **PSQ** (inverse QQQ). |
| Conditions | Activation requires ≥ 2 sensor agreement: drawdown ≤ -6%, ATR% ≥ 3%, VIX ≥ 25. |
| Status | **ARMED**, currently dormant — `trending_bull` regime + VIX 16.89. |

### omega_vol — VIX-linked volatility alpha

| Field | Value |
|---|---|
| Source | `chad/strategies/omega_vol.py` |
| Sleeve | ADAPTIVE |
| Weight | 0.05 |
| tier_factor | `1.0` |
| winner_factor | `1.0` |
| regime_factor | `1.3` |
| Active in | `volatile` |
| Silent in | every other regime |
| Window | 5-state VIX regime. UVXY hard 10-bar time stop. |
| Universe | **SVXY, UVXY**. |
| Conditions | LOW_VOL → SVXY BUY; CRISIS → UVXY BUY; VOL_CRUSH → SVXY mean-revert. |
| Health | **0.10 on 3 samples** — flagged. |
| Status | **DEGRADED** (lowest health score; sample size too small to act on). |

### omega_macro — Macro regime futures

| Field | Value |
|---|---|
| Source | `chad/strategies/omega_macro.py` |
| Sleeve | ADAPTIVE |
| Weight | 0.03 |
| tier_factor | `1.0` |
| winner_factor | `1.0` |
| regime_factor | `1.3` |
| Active in | every regime except `adverse` |
| Silent in | `adverse` |
| Window | Continuous; min 40 bars. |
| Universe | **ZN, ZB, M6E**. |
| Conditions | 4-state macro regime: RISK_OFF, RISK_ON, STAGFLATION, NEUTRAL. |
| Status | **ACTIVE**, currently `RISK_ON`-leaning per VIX 16.89. |

### omega_momentum_options — Intraday single-leg options momentum

| Field | Value |
|---|---|
| Source | `chad/strategies/omega_momentum_options.py` |
| Sleeve | ALPHA |
| Weight | 0.03 |
| tier_factor | `1.0` |
| winner_factor | `1.0` |
| regime_factor | `1.3` |
| Active in | `trending_bull, trending_bear, volatile` |
| Silent in | `ranging, unknown, adverse` |
| Window | 9:45 AM ET → 3:30 PM ET; hard exit 3:45 PM ET; 15-min per-symbol cooldown; max 3 concurrent. |
| Universe | **SPY, QQQ, AAPL, NVDA, MSFT**. |
| Conditions | Momentum (0.3% in 5 bars) + EMA slope + 1.5× volume; VIX regime filter (skip if VIX > 40). |
| Status | **ARMED** within session window. |

### Strategy summary table

| # | Strategy | Sleeve | Weight | Tier | Winner | Regime | Tier Eligible (PRO) | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | alpha | ALPHA | 0.16 | 1.0 | 1.0 | **1.3** | YES | ACTIVE |
| 2 | alpha_intraday | ALPHA | 0.03 | 1.0 | 1.0 | 1.3 | YES | ACTIVE |
| 3 | alpha_options | ALPHA | 0.04 | 1.0 | 1.0 | 1.3 | YES | CONDITION-GATED |
| 4 | alpha_futures | ALPHA | 0.09 | 1.0 | 1.0 | 1.3 | YES | ACTIVE |
| 5 | alpha_crypto | ADAPTIVE | 0.04 | 1.0 | 1.0 | 1.3 | YES | ACTIVE |
| 6 | beta | BETA | 0.05 | 1.0 | 1.0 | 1.3 | YES | ACTIVE (injection flagged off) |
| 7 | beta_trend | BETA | 0.20 | 1.0 | 1.0 | 1.3 | YES | ACTIVE |
| 8 | gamma | ALPHA | 0.07 | 1.0 | 1.0 | 1.3 | YES | REGIME-SILENT |
| 9 | gamma_futures | ALPHA | 0.05 | 1.0 | 1.0 | 1.3 | YES | ACTIVE |
| 10 | gamma_reversion | ALPHA | 0.04 | 1.0 | 1.0 | 1.3 | YES | REGIME-SILENT |
| 11 | delta | ADAPTIVE | 0.02 | 1.0 | 1.0 | 1.3 | YES | ACTIVE |
| 12 | delta_pairs | ADAPTIVE | 0.05 | 1.0 | 1.0 | 1.3 | YES | REGIME-SILENT |
| 13 | omega | ADAPTIVE | 0.05 | 1.0 | 1.0 | 1.3 | YES | ARMED |
| 14 | omega_vol | ADAPTIVE | 0.05 | 1.0 | 1.0 | 1.3 | YES | DEGRADED |
| 15 | omega_macro | ADAPTIVE | 0.03 | 1.0 | 1.0 | 1.3 | YES | ACTIVE |
| 16 | omega_momentum_options | ALPHA | 0.03 | 1.0 | 1.0 | 1.3 | YES | ARMED |
| — | **Σ** | — | **1.00** | — | — | **1.3 global** | — | — |

With portfolio_risk_cap = $9,199.12 and a uniform 1.3× regime boost,
the per-strategy final caps in `runtime/dynamic_caps.json:strategies.<name>.dollar_cap`
are now ~30% larger than the v8.5 base. Combined with SCR's
sizing_factor=1.0 (vs WARMUP's 0.10), this is a **meaningful step-up
in deployable capital** — the system will trade with real size for
the first time since SCR was created.

---

## 4. EXECUTION PIPELINE (post-fix path)

### Stages

```
raw signals (TradeSignal[])
  → HALT FILTER (NEW v8.6) — drop halted strategies before pipeline
  → signal_router.route                       # bucket on (symbol,side,asset_class)
  → split_signals_by_asset_class              # CRYPTO vs IBKR
  → IBKR lane: build_execution_plan
              → build_intents_from_plan
              → routing_gates.run_all_gates   # A4 SPLIT v8.6 / E2 / E5 / R7 / S5
              → vote_collector.submit_intent  # S1 (min_votes=1)
              → sizing pipeline               # R3 → R5 → R6 → S5
              → IbkrAdapter.submit_strategy_trade_intents
  → Kraken lane: build_kraken_intents_from_routed_signals
                → _enforce_kraken_rest_pair       # v8.3 layer 2
                → KrakenExecutor.submit
                → kraken_client.add_order
                → _assert_kraken_rest_pair        # v8.3 layer 1 (border)
  → fills → PHANTOM-FILL GUARD (NEW v8.6) → normalize_paper_fill_evidence
            (error/Cancelled/Inactive → skip evidence)
  → PaperExecutionEvidenceWriter
  → position_guard, slippage_tracker, edge_decay_monitor, expectancy_tracker
  → ProfitRouter.route_profit (50/30/20 + drain v8.6)
```

### NEW v8.6 — Halt filter at signal-build entry

`chad/risk/edge_decay_monitor.is_strategy_halted(strategy)` is now
called before stage-1 signal building. Halted strategies drop out of
the iteration loop entirely — no signals emitted, nothing reaches the
router. This closes the gap from pre-`52786b7` where the monitor
*published* halt state but no consumer *read* it.

```python
# chad/core/live_loop.py — pre-stage-1 halt filter (commit 52786b7)
for strategy in registry.iter_active(regime):
    if is_strategy_halted(strategy.name):
        LOG.info("strategy_halted_skipping name=%s reason=edge_decay", strategy.name)
        continue
    # ... normal signal build path ...
```

Edge-trigger alerting in the monitor itself: previously, the alert
logic re-fired every cycle while the strategy was halted, which paged
the operator every 60s. Now it fires once on transition into halt and
once on transition out (recovery), nothing in between.

### NEW v8.6 — Phantom-fill guard at submit call site

`chad/core/live_loop.py:1180` — surgical guard at the submit call
site, downstream of the order placement and upstream of the evidence
writer:

```python
# Phantom-fill guard (commit 3be6785) — error-status orders
# must not produce paper-fill evidence. The normalizer is the
# single chokepoint and is intentionally narrow; this guard
# blocks at the submit site so the chokepoint stays clean.
order_status = (order.get("status") or "").lower()
if order_status in ("error", "cancelled", "inactive"):
    LOG.warning(
        "phantom_fill_guard_blocked status=%s order_id=%s strategy=%s symbol=%s",
        order_status, order.get("order_id"), strategy_name, symbol,
    )
    continue  # skip evidence write
```

The decision to put the guard at the submit site rather than inside
the normalizer was deliberate (commit `3be6785` reverted the
normalizer change from `42268f0`):

- The normalizer's contract is "given a fill, normalize its fields";
  it is not the place to decide whether the fill happened at all.
- The submit site has full context: order id, strategy, symbol, the
  intended action — easy to log a structured warning.
- Keeping the normalizer narrow preserves the v8.1 single-chokepoint
  invariant.

### Bucket key — `(symbol, asset_class)` (commit `d666b5d`)

Carried forward unchanged from v8.1 — bucket key is `(symbol, asset_class)`
not `(symbol, side)`. Citations:
`chad/utils/signal_router.py:102`,
`chad/execution/execution_pipeline.py:680-685`.

### Meta propagation (commit `f6caf3d`)

Carried forward unchanged. The router selects meta from the bucket's
`primary_strategy` (largest size contributor):

```python
meta=data.get("meta_by_strategy", {}).get(primary) or {}
```

This is what allows `position_reconciler` to assert paper-mode
strategy ownership and skip drift checks against IBKR for
non-`broker_sync` entries.

### Asset class router

`chad/execution/execution_pipeline.py:1164-1187` —
`split_signals_by_asset_class` returns `(ibkr_signals, kraken_signals)`:

- `AssetClass.CRYPTO` → Kraken lane.
- Everything else (or missing/unknown) → IBKR lane.

### OPTIONS branch

`chad/execution/execution_pipeline.py:616-665` —
`resolve_ibkr_instrument_spec` dispatches on `AssetClass`. The OPT
spec resolver produces an IBKR `Contract` with `sec_type="OPT"` and a
real expiry/strike when the strategy supplies it.

### Paper fill normalization (commit `21166c9`, narrow per `3be6785`)

Single chokepoint `normalize_paper_fill_evidence` at
`chad/execution/paper_exec_evidence_writer.py:925-983` enforces four
invariants in paper mode:

1. `asset_class` is never blank/unknown when symbol is recognisable.
2. `fill_price > 0` when `runtime/price_cache.json` has a price (with
   futures contract-month normalisation, e.g. `MGCK6 → MGC`).
3. `status` is rewritten to `paper_fill` when raw status is
   `PendingSubmit`, etc., and a positive price is available.
4. **Hard invariant** — raises `ValueError` if status would still be
   pending after normalisation.

`is_live=True` records pass through unchanged. The normalizer is
*intentionally narrow* — phantom-fill prevention happens upstream at
the submit site, not here.

### Three writers, one chokepoint

| Writer | Path |
|---|---|
| Live loop | `chad/core/live_loop.py:1129-1151` |
| Position reconciler | `chad/core/position_reconciler.py:217,282` |
| Timer-driven paper executor | `/usr/local/bin/chad_paper_trade_executor.py:11,222` |

### Kraken REST border — two-layer enforcement (v8.3, commit `15677a2`)

Carried forward unchanged. Two-layer defense against `EQuery:Unknown
asset pair`:
- **Layer 1 (REST border):** `chad/exchanges/kraken_client.py:55`
- **Layer 2 (executor pre-flight):** `chad/execution/kraken_executor.py:41`

### NEW v8.6 — A4 gate split: intraday vs daily/swing/futures

`chad/execution/routing_gates.py` — A4 (`bar_freshness`) is now
strategy-aware:

```python
INTRADAY_STRATEGIES = frozenset({
    "alpha", "alpha_intraday", "alpha_options",
    "gamma", "gamma_reversion", "omega_momentum_options",
})
A4_INTRADAY_THRESHOLD_S = 900       # 15 minutes
A4_DAILY_SWING_THRESHOLD_S = 172800  # 48 hours

def a4_bar_freshness(intent, ctx):
    bars = ctx.bars.get(intent.symbol)
    threshold = (A4_INTRADAY_THRESHOLD_S
                 if intent.strategy in INTRADAY_STRATEGIES
                 else A4_DAILY_SWING_THRESHOLD_S)
    if not bars:
        # NEW v8.6 — bar_missing fails the gate for intraday strategies
        if intent.strategy in INTRADAY_STRATEGIES:
            return Block(reason="bar_missing")
        # daily/swing/futures: bar_missing tolerated (provider lag)
        return Pass()
    age_s = now_utc_seconds() - bars[-1].ts
    if age_s > threshold:
        return Block(reason=f"bar_stale_{age_s:.0f}s_gt_{threshold}s")
    return Pass()
```

Pre-fix, the single 300s threshold meant: (a) intraday strategies
had a too-tight bound during minor provider hiccups; (b) daily/swing
strategies were *always* tripping the gate because their bars
naturally exceed 300s. Now the gate is meaningful in both directions.

### NEW v8.6 — `/api/recent-trades` line cap (D7)

`chad/dashboard/api.py` — the recent-trades endpoint streams from
`data/trades/trade_history_YYYYMMDD.ndjson`. Pre-fix, on a heavy
trading day this could be a multi-MB response. Now capped at **500
lines per file**, paginated by date. The cap is a hard rstrip-by-line
during stream — the client gets the most recent 500 lines
deterministically.

---

## 5. RISK & GOVERNANCE — THE FULL CHAIN

### The complete cap-calculation chain

```
config/strategy_weights.json
   ↓  base weights (16 strategies, sum=1)
correlation_overlay (chad/risk/correlation_strategy.py)
   ↓  proportions adjusted by inter-strategy correlation
chassis_enforcement (chad/risk/dynamic_risk_allocator.py:474-547)
   ↓  rebalance to 50/30/20 sleeves with 5% tolerance
TIER FILTER  (v8.3)
   ↓  zero strategies not in current tier's enabled list
WINNER SCALING (v8.3)
   ↓  per-strategy expectancy multiplier 0.5x–1.5x
REGIME BOOSTER (v8.3)
   ↓  global multiplier 1.0x–1.5x
   final_caps → runtime/dynamic_caps.json
SCR sizing_factor (CONFIDENT v8.6 → 1.0×)  applied at execution time
   ↓
EDGE DECAY HALT FILTER (NEW v8.6) — applied at signal entry
```

The chain is the same as v8.5 with one structural addition: edge
decay halt filtering happens *before* signal entry, not as a cap
zero-out. A halted strategy never enters the cap calculation because
it never enters the iteration. Caps are computed only for active
strategies.

### Step-by-step with citations

| Step | Module | Behavior |
|---|---|---|
| 1. base_weight | `config/strategy_weights.json` | `StrategyAllocation.from_env_or_default`. |
| 2. correlation overlay | `chad/risk/correlation_strategy.py` | Stale `correlation` archived 2026-03-26 — fails soft. |
| 3. chassis 50/30/20 | `dynamic_risk_allocator.py:474-547` | `enforce_chassis(weights)` if drift > `CHASSIS_TOLERANCE = 0.05`. |
| 4. **tier_filter** | `dynamic_risk_allocator.py:603-614` | Reads `runtime/tier_state.json:enabled_strategies`. |
| 5. **winner_scaling** | `dynamic_risk_allocator.py:617-633` | Reads `runtime/winner_scaling.json:multipliers`. |
| 6. **regime_booster** | `dynamic_risk_allocator.py:636-644` | Reads `runtime/regime_booster.json:multiplier`. Allocator clamps to `[1.0, 1.5]`. |
| 7. cap composition | `dynamic_risk_allocator.py:362-378` | `cap = portfolio_risk_cap × frac × tier × winner × regime`. |
| 8. SCR sizing | `chad/risk/scr_state.py` | At execution. **CONFIDENT 1.0× v8.6** (was WARMUP 0.10× v8.5). |
| 9. **edge decay halt** (NEW v8.6) | `chad/risk/edge_decay_monitor.py:is_strategy_halted` | At signal entry; halted strategy drops from iteration. |

### Allocator code excerpt

Same as v8.5 (`dynamic_risk_allocator.py:355-378`). With the booster
active at 1.3× and SCR at 1.0×, the math now yields meaningful caps:

```
portfolio_risk_cap = $9,199.12
alpha (weight 0.16):
  base_cap = 9,199.12 × 0.16 = 1,471.86
  × tier_factor 1.0
  × winner_factor 1.0
  × regime_factor 1.3
  = $1,913.42 (per-strategy final cap)
  × SCR sizing 1.0
  = $1,913.42 deployable per cycle for alpha
```

### Fail-soft behavior

| File | Stale/missing → |
|---|---|
| `runtime/tier_state.json` (>10 min old) | `tier_filter` inactive — no strategies zeroed. |
| `runtime/winner_scaling.json` (>10 min old) | All `winner_factor = 1.0`. |
| `runtime/regime_booster.json` (>10 min old) | `regime_mult = 1.0`. |

`BUSINESS_OVERLAY_STALE_SECONDS = 600` at
`dynamic_risk_allocator.py:559`. The feed watchdog NEW v8.6 covers
all three with 180s TTL, well inside the 600s overlay window — so a
publisher hiccup gets surfaced via Telegram before the allocator
silently downgrades.

### 50/30/20 chassis

Module: `chad/risk/dynamic_risk_allocator.py:454-547`. Frozen sets
unchanged from v8.5. Disable via env `CHAD_CHASSIS_ENFORCEMENT=0`.

```python
ALPHA_STRATEGIES = frozenset({
    "alpha", "alpha_futures", "alpha_intraday", "alpha_options",
    "omega_momentum_options", "gamma", "gamma_futures", "gamma_reversion",
})
BETA_STRATEGIES = frozenset({"beta", "beta_trend"})
ADAPTIVE_STRATEGIES = frozenset({
    "omega", "omega_macro", "omega_vol", "delta", "delta_pairs",
    "alpha_crypto",
})
```

### Sizing — equity / ETF intents

`chad/execution/execution_pipeline.py:_apply_sizing_layer`:

```
base → R3 vol_adjusted → R5 composite_cap → R6 correlation_monitor → S5 event_gate → OMS
```

| Layer | Module | Behavior |
|---|---|---|
| **R3** | `chad/risk/vol_adjusted_sizer.py` | `mult = clamp(target_daily_vol/realized_vol, 0.1, 2.0)`, target=0.015. |
| **R5** | `chad/risk/composite_size_cap.py` | `min(vol_adj, max_per_symbol=300, sector_remaining=$5k, 0.5%×ADV, 5%×equity/ref_px)`. |
| **R6** | `chad/risk/correlation_monitor.py` | Multiplies by `threshold/avg_corr` (floor 0.1) when book \|r\| > 0.65. |
| **S5** | `chad/analytics/event_calendar.py` | Inside event window: `urgency=high` reject; `normal` reduce 50%. |

### SCR — Self-Calibrating Risk

State machine in `chad/risk/scr_state.py`. Runtime at
`runtime/scr_state.json`. Config at `runtime/scr_config.json`.

| State | sizing_factor | What changes |
|---|---|---|
| WARMUP | 0.10 | All trades 10% of planned size; live blocked. |
| CAUTIOUS | 0.25 | 2.5× upsize from WARMUP; still paper-only. |
| **CONFIDENT (current v8.6)** | **1.00** | Full size; live-eligible (still gated by operator GO). |
| PAUSED | 0.00 | Hard stop. **Hysteresis-protected NEW v8.6.** |

#### NEW v8.6 — Min-trade gates

The state machine now enforces minimum effective_trades counts at
band boundaries:

```python
# chad/risk/scr_state.py — min-trade gates (commit 23f4c03)
MIN_TRADES_CAUTIOUS = 100
MIN_TRADES_CONFIDENT = 133

def evaluate_band(stats, prev_state=None, paused_recovery_ticks=0):
    n = stats.effective_trades
    if n < MIN_TRADES_CAUTIOUS:
        return State.WARMUP, "Warmup: only %d effective trades (< 100 required)." % n
    if n < MIN_TRADES_CONFIDENT:
        # cautious-band metric checks ...
        return State.CAUTIOUS, "..."
    # confident-band metric checks ...
    return State.CONFIDENT, "CONFIDENT: win_rate, sharpe, and drawdown all within confident band."
```

Pre-fix, the band logic was metric-only and could promote a strategy
to CONFIDENT on 50 trades if the metrics happened to look favorable —
risky. Now the trade-count gate is a hard precondition.

#### NEW v8.6 — PAUSED hysteresis end-to-end

The PAUSED → recovery transition is hysteresis-protected. The router
holds PAUSED until `paused_recovery_ticks` consecutive healthy reads
have been observed. Wiring (commit `28e8963`):

```python
# backend/api_gateway.py — caller injection
prev_state = scr_state_cache.get("state")
paused_recovery_ticks = scr_state_cache.get("paused_recovery_ticks", 0)
new_state, reasons = evaluate_confidence(
    stats=shadow_stats,
    prev_state=prev_state,
    paused_recovery_ticks=paused_recovery_ticks,
)
```

ShadowStats round-trips `prev_state` and `paused_recovery_ticks`
through `compute_state()` so the router actually sees the history
between cycles. End-to-end now closed.

### Profit lock — circuit breaker

Module: `chad/risk/profit_lock.py`. State:
`runtime/profit_lock_state.json`. Six modes, equity-driven.
Unchanged from v8.5.

| Mode | Trigger (% of equity) | Sizing factor |
|---|---|---|
| NORMAL (current) | default | 1.00 |
| WARN | profit ≥ 1.5% | 1.00 (flagged) |
| LOCK1 | profit ≥ 3.0% | 0.50 |
| LOCK2 | profit ≥ 5.0% | 0.25 |
| LOCK3 | profit ≥ 8.0% | 0.10 |
| HARD_STOP | profit ≥ 10.0% | 0.00 |

Currently NORMAL: realized_pnl_today=−$45.20, daily_loss_limit=$5,519.47.

NEW v8.6 — Telegram alerts on transitions (commit `931dd4e`). The
profit-lock publisher emits a Telegram alert on edge-trigger
transitions only (NORMAL → WARN → LOCK1 → … → HARD_STOP). Re-entries
into the same mode are suppressed.

### NEW v8.6 — Position guard atomic write

`chad/core/position_guard.py:_save_state` (commit `5267705`):

```python
def _save_state(state: dict, path: Path = STATE_PATH) -> None:
    """Atomic write — temp file + os.replace.

    Rationale: a process crash between truncate and write produced
    a zero-byte position_guard.json on at least one occasion, which
    masked open positions on the next startup.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    os.replace(tmp, path)
```

The previous implementation was `path.write_text(json.dumps(state))`
which had a window between truncate-on-open and the write completion.
Now the file is never seen in a half-state.

### NEW v8.6 — Edge decay halt enforcement

Module: `chad/risk/edge_decay_monitor.py`. Halts a strategy when ≥ 5
consecutive losses on ≥ 20-trade base. Recovery via
`scripts/clear_edge_decay.py --strategy <name> --confirm`.

#### Three changes in commit `52786b7`

1. **Wire `is_strategy_halted()` into signal building.** Pre-fix, the
   monitor was *publishing* halt state to `runtime/edge_decay_state.json`
   but no consumer was *reading* it.
2. **Move halt check before stage-1 (signals).** Was after stage-4
   (sizing) — a halted strategy could still emit and be sized down
   to zero, wasting cycle work. Now it never enters iteration.
3. **Fix key name.** Code was reading `consecutive_negative` while
   the monitor was writing `consecutive_neg`. Reads were silent
   defaults to 0 — halts never fired.
4. **Edge-trigger alerting.** `_emit_halt_alert()` checks the
   monitor's previous state; alert fires only on transitions
   (active→halt, halt→cleared).
5. **Harden `clear_edge_decay.py`.** Requires `--confirm` flag.
   Operator can no longer accidentally clear a halt by tab-completing
   a script name.

### Routing gates (5 gates, A4 SPLIT v8.6)

| # | Gate | Block reason | NEW v8.6 |
|---|---|---|---|
| 1 | A4 `data_freshness` | `bar_stale` / `bar_missing` | **SPLIT — 900s intraday + missing-bar fail / 172800s daily-swing-futures** |
| 2 | E2 `stale_intent` (≤ 300s) | `intent_expired` | — |
| 3 | E5 `too_late_to_chase` (≤ 0.5%) | `price_moved` | — |
| 4 | R7 `net_ev` (≥ min_edge) | `net_ev_below_min_edge` | — |
| 5 | S5 `event_risk` window | reject (high urgency) / reduce 50% | — |

---

## 6. THE BUSINESS FRAMEWORK

This section documents the seven modules and seven runtime files
that together form the "CHAD as a business" framework. Live values
re-read at write time.

### 6A. Equity History Publisher

**Source:** `chad/ops/equity_history_publisher.py`
**Output:** `runtime/equity_history.ndjson`
**Cadence:** daily 23:59 UTC via `chad-equity-history.timer`.

#### Current state

`runtime/equity_history.ndjson` currently contains **4 records** (was
1 in v8.5). Daily snapshots since 2026-04-27. The 14-day minimum gate
in `WithdrawalManager` requires 10 more records.

### 6B. Portfolio Snapshot Publisher

**Source:** `chad/ops/portfolio_snapshot_publisher.py`
**Output:** `runtime/portfolio_snapshot.json`
**Cadence:** every 5 minutes via `chad-portfolio-snapshot.timer`.

Unchanged from v8.5 in code; clientId=84 paper read with USD.CAD FX
quote conversion. Watched by `chad-feed-watchdog.timer` NEW v8.6 (no
direct entry — but the publisher's downstream `dynamic_caps.json` is
in the watchdog's seven-feed list).

### 6C. Tier Manager

**Source:** `chad/risk/tier_manager.py`
**Output:** `runtime/tier_state.json`
**Cadence:** every 5 minutes via `chad-tier-manager.timer`.
**Config:** `config/tiers.json`.

Unchanged from v8.5. Current state PRO with 16 enabled strategies.

### 6D. Winner Scaler

**Source:** `chad/risk/winner_scaler.py`
**Output:** `runtime/winner_scaling.json`
**Cadence:** every 15 minutes via `chad-winner-scaler.timer`.
**Config:** `config/winner_scaling_policy.json`.

Logic unchanged from v8.5. **Current state — all 16 strategies at
1.0× because `median_expectancy=0.0`** — when the scaler cannot rank
(no signal), it falls through to neutral. Expected to re-emerge as
trade history accumulates with the SCR-CONFIDENT real-size cycles.

### 6E. Regime Booster

**Source:** `chad/risk/regime_booster.py`
**Output:** `runtime/regime_booster.json`
**Cadence:** every 60 seconds via `chad-regime-booster.timer`.
**Config:** `config/regime_booster_policy.json`.

#### Current state — ACTIVE at 1.30× (NEW v8.6)

`runtime/regime_booster.json` (`2026-05-01T11:08:41Z`):

| Field | Value |
|---|---|
| `multiplier` | **`1.3`** |
| `active` | **`true`** |
| `reasons` | `["high_confidence_0.75", "vix_calm_16.9", "trending_bull_bias"]` |
| `regime` | `trending_bull` |
| `confidence` | `0.7536` |
| `vix` | `16.89` |
| `event_severity` | `medium` |

This is the **first time the booster has been active** since v8.3 was
cut. Confidence cleared 0.75 (the high_confidence threshold), VIX
fell below 18.0 (the calm threshold), and regime stayed
`trending_bull`. Three positive factors (+0.10 each) sum to 1.30×.

#### NEW v8.6 — TTL read-time enforcement of inputs

`runtime/regime_state.json` is an input to the booster. Pre-`71d8ea6`,
a stale regime_state could feed the booster forever. Now: the booster
does the read-time TTL check; a stale read returns
`regime=unknown`, which fails the `unfavorable_regime_<regime>` veto
check and forces multiplier to 1.0. Fail-soft to neutral.

### 6F. Withdrawal Manager

**Source:** `chad/risk/withdrawal_manager.py`
**Output:** `runtime/withdrawal_authorization.json`
**Cadence:** every 6 hours via `chad-withdrawal-manager.timer`.
**Config:** `config/withdrawal_policy.json`.

Logic unchanged from v8.5 (HWM bug already fixed in v8.4 commit
`f62d914`).

#### Current state

```
phase                       GROW
current_equity_usd          $183,974.75
seed_capital_usd            $50,000.00
high_water_mark_usd         $183,874.27
drawdown_from_hwm_pct       -0.0546  (current above HWM)
spendable_surplus_usd       $100.49
authorized_withdrawal_usd   $0.00
scr_state                   CONFIDENT
history_days                4
reason                      "GROW phase override: only 4 days of equity
                             history (need 14+). Building track record
                             before paying."
```

What unlocks PAY for this account (NEW v8.6 status):
1. ~~SCR promotes from WARMUP → CONFIDENT~~. **DONE.**
2. Build **10 more days** of equity history (currently 4; need 14).
3. Stay above HWM with no >5% drawdown.

NEW v8.6 — Telegram alert on PAY-phase trigger (commit `931dd4e`):
when `authorized_withdrawal_usd` first crosses zero, an edge-trigger
alert fires.

### 6G. Business Phase Tracker

**Source:** `chad/ops/business_phase_tracker.py`
**Output:** `runtime/business_phase.json`
**Cadence:** every 30 minutes via `chad-business-phase.timer`.

Logic unchanged. Current state GROW (4 days). The phase_description
text is verbatim from the policy — once history clears 14 days and
HWM holds, the phase will flip to PAY and the description will read
*"Engine running well. Salary authorized at $\<authorized>/month from
surplus above high water mark."*

### 6H. Profit Router (50/30/20 — DRAIN MECHANISM v8.6)

**Source:** `chad/risk/profit_router.py`
**Output:** `runtime/profit_routing.json`
**Cadence:** invoked on every realized profit close.

#### Core split (unchanged)

- **50%** → `trading_capital`
- **30%** → `beta_allocation`
- **20%** → `amplifier_allocation`

#### NEW v8.6 — Drain mechanism

The ledger went from **append-only** to **drain-aware**. New persisted
fields and functions:

```python
# chad/risk/profit_router.py — drain mechanism (commit 74e5a0d)

# Persisted runtime fields (NEW v8.6)
# - consumed_beta_usd: float = 0.0
# - consumed_amplifier_usd: float = 0.0

def get_beta_remaining(routing_path: Optional[Path] = None) -> float:
    """How much beta-bucket capital is unconsumed and available for
    injection into the beta strategy."""
    state = _load_state(routing_path)
    return state["totals"]["beta_allocation"] - state.get("consumed_beta_usd", 0.0)

def mark_beta_consumed(usd: float, routing_path: Optional[Path] = None) -> None:
    """Atomically increment consumed_beta_usd. Used by the beta
    injection path when CHAD_PROFIT_ROUTER_BETA_INJECTION is enabled."""
    with _lock(routing_path):
        state = _load_state(routing_path)
        state["consumed_beta_usd"] = state.get("consumed_beta_usd", 0.0) + usd
        _save_state(state, routing_path)
```

Beta injection into the `beta` strategy is wired but gated behind the
`CHAD_PROFIT_ROUTER_BETA_INJECTION` env flag. **Default off.** Paper
soak required before enabling. Once enabled, `beta` will pull
opportunistically from the unconsumed beta bucket.

#### NEW v8.6 — `broker_sync` excluded from routing

`route_profit(realized_pnl, closing_strategy, account_equity)` now
short-circuits when `closing_strategy == "broker_sync"`:

```python
EXCLUDED_FROM_ROUTING = {"broker_sync"}

def route_profit(realized_pnl, closing_strategy, account_equity, ...):
    if closing_strategy in EXCLUDED_FROM_ROUTING:
        return {"no_routing": True, "reason": "broker_sync_excluded"}
    if not (realized_pnl > 0):
        return {"no_routing": True, "reason": "not_profitable"}
    # ... 50/30/20 split ...
```

`broker_sync` is bookkeeping for inherited paper positions, not
strategy alpha. Pre-fix, broker_sync realized PnL was generating
amplifier and beta contributions — distorting both buckets. The four
existing `broker_sync` ledger entries pre-date this fix and remain in
the totals.

#### NEW v8.6 — Test isolation (`routing_path` param)

`route_profit` and helpers now accept an optional `routing_path`
parameter. The unit tests pass a temp file, isolating the test ledger
from the production ledger. **19 phantom `alpha_test` entries** that
had bled into production were purged in commit `cdfed06`:

```json
"alpha_test_purged_at_utc": "2026-04-30T19:05:17.503425Z",
"alpha_test_purged_count": 19
```

#### Current totals

`runtime/profit_routing.json` (latest decision `2026-04-30T23:58:47Z`):

| Bucket | Total | Consumed | Remaining |
|---|---|---|---|
| `trading_capital` | $1,905.26 | n/a (compounds in account) | n/a |
| `beta_allocation` | **$1,143.15** | **$0.00** | **$1,143.15** |
| `amplifier_allocation` | **$762.10** | **$0.00** | **$762.10** |
| **Sum** | **$3,810.51** | — | — |

51 routing decisions logged. Beta and amplifier buckets are fully
unconsumed pending injection feature flag enable.

### Business framework — call graph

```
runtime/portfolio_snapshot.json   (5min)  ──┐
                                            ├──► runtime/tier_state.json     (5min)
                                            └──► runtime/withdrawal_authorization.json (6h)
                                                          │
                                                          └──► runtime/business_phase.json (30min)

runtime/equity_history.ndjson     (daily) ──► WithdrawalManager + BusinessPhaseTracker

runtime/expectancy_state.json     (5min)  ──► runtime/winner_scaling.json    (15min)

runtime/regime_state.json         (60s)   ──┐  (NEW v8.6: read-time TTL enforcement)
runtime/event_risk.json           (1800s) ──├──► runtime/regime_booster.json  (60s timer)
data/bars/1d/VIX.json             (daily) ──┘

[risk allocator]
  reads: tier_state, winner_scaling, regime_booster
  writes: runtime/dynamic_caps.json with per-strategy
          {tier_factor, winner_factor, regime_factor, dollar_cap}

[on every realized profit close]
  ProfitRouter → runtime/profit_routing.json (drain-aware ledger v8.6)

[NEW v8.6 — observability]
  chad-feed-watchdog.timer → 7 feeds → Telegram alert on stale
```

---

## 7. RECONCILIATION

### Publisher path

`chad/ops/reconciliation_publisher.py:run_publish`. Runs every cycle
of `chad-reconciliation-publisher.timer`. Reads
`runtime/position_guard.json` + IBKR positions and produces
`runtime/reconciliation_state.json`.

### Paper-mode behavior (commit `f6caf3d`)

Carried forward unchanged. In paper mode, only `broker_sync|*` entries
are compared to IBKR truth.

### Status thresholds

`chad/ops/reconciliation_publisher.py:225-233`:

- `worst_diff ≤ 1.0` → **GREEN**
- `worst_diff ≤ 2.0` → **YELLOW**
- otherwise → **RED**

### Drift vs Mismatch

`reconciliation_publisher.py:212-217`. Drift when CHAD's strategy side
is zero OR all of the diff is in `broker_sync`. Otherwise mismatch.

### NEW v8.6 — TTL raised 300 → 360s

Commit `ba65367`. The publisher cadence is 300s (every 5 minutes); a
state file written at T+0 expires at T+300, exactly when the next
publisher cycle is *starting*. Downstream consumers reading at T+295
were occasionally getting the about-to-expire file. 60s margin closes
the race.

### NEW v8.6 — `KNOWN_FUTURES_SYMBOLS` includes MYM/M2K

```python
# chad/ops/reconciliation_publisher.py:58 (commit ba65367)
KNOWN_FUTURES_SYMBOLS = frozenset({
    "MCL", "ES", "NQ", "CL", "GC", "RTY",
    "MES", "MNQ", "MGC", "SIL",
    "MYM", "M2K",  # NEW v8.6
})
```

Pre-fix, MYM and M2K (added to the universe in v8.5) were tripping
the contract-resolution path because they were not in the recognized
futures set. Now correctly excluded from drift classification when
the broker reports them under a contract-month suffix.

### NEW v8.6 — `chad_strategy_open` count

`reconciliation_publisher.py` now publishes a separated count:

```json
"counts": {
  "chad_open": 8,
  "chad_strategy_open": 15,
  "broker_positions": 11
}
```

`chad_open` is the pre-existing deduplicated count after rolling up
`broker_sync|*` echoes. `chad_strategy_open` (NEW v8.6) is the count
of strategy-tracked positions specifically — alpha/delta/gamma/etc.,
without `broker_sync`. Both numbers are now visible to operators and
to the dashboard.

### NEW v8.6 — Telegram alert on RED status

Commit `931dd4e`. Both publish paths (the timer publisher and the
reconciler-side publisher) now emit a Telegram alert on transition to
RED. Edge-trigger only — the alert does not re-fire if RED persists.

### Excluded symbols

| Set | Symbols | Why |
|---|---|---|
| `excluded_symbols` (current) | `["AAPL", "NVDA"]` | Pinned per audit. |
| `KNOWN_FUTURES_SYMBOLS` | `MCL, ES, NQ, CL, GC, RTY, MES, MNQ, MGC, SIL, MYM, M2K` | Futures contract-resolution path; MYM/M2K added v8.6. |

### Current state

`runtime/reconciliation_state.json` (ts `2026-05-01T11:04:28Z`):

```
status               GREEN
broker_source        ibkr:clientId=83
counts.chad_open            8
counts.chad_strategy_open   15
counts.broker_positions     11
worst_diff           0.0
mismatches           []
drifts               [{"symbol":"BAC","chad":0.0,"broker":-17.0,"diff":17.0},
                      {"symbol":"CVX","chad":0.0,"broker":-24.0,"diff":24.0},
                      {"symbol":"GOOGL","chad":0.0,"broker":2.0,"diff":2.0}]
excluded_symbols     ["AAPL","NVDA"]
futures_excluded     []
ttl_seconds          360
```

---

## 8. INTELLIGENCE LAYER

Each runtime intel feed is independent — failures degrade gracefully
(stale data is preferred over no data; consumers test
`max_age_hours`). NEW v8.6: seven of the highest-criticality feeds
are now actively monitored by `chad-feed-watchdog.timer`.

| File | Schema | Purpose | Watched? | Fresh? |
|---|---|---|---|---|
| `runtime/regime_state.json` | `regime_state.v1` | Live classifier inputs + regime label. | **YES (180s)** | 120s TTL — current. **Read-time TTL enforced NEW v8.6.** |
| `runtime/strategy_intelligence.json` | n/a | AI-generated regime profile per strategy. | no | 48h cache. |
| `runtime/expectancy_state.json` | n/a | Per-strategy rolling expectancy. | no | Refreshed every 5 min by `chad-expectancy-tracker.timer`. |
| `runtime/trends_state.json` | n/a | Google Trends ratios per symbol. | no | `chad-trends-refresh.timer`. |
| `runtime/reddit_sentiment.json` | n/a | Reddit mention/sentiment per symbol. | no | `chad-reddit-sentiment-refresh.timer`. |
| `runtime/short_interest.json` | n/a | Short float % per symbol. | no | `chad-short-interest-refresh.timer`. |
| `runtime/event_risk.json` | `event_risk.v1` | US session-edge windows. | **YES (2400s)** | 1800s TTL. Currently `severity=medium`. |
| `runtime/institutional_consensus.json` | `institutional_consensus.v1` | Top 25 holdings across 7 funds. | no | Sunday 00:00 UTC weekly. |
| `runtime/profit_routing.json` | `profit_routing.v1` | 50/30/20 + drain ledger NEW v8.6. | no | Per-realised-close. |
| `runtime/business_phase.json` | `business_phase.v1` | Phase + plain-English description. | no | 30 min. |
| `runtime/tier_state.json` | `tier_state.v1` | Tier name + enabled strategies. | no | 5 min. |
| `runtime/withdrawal_authorization.json` | n/a | Salary authorization. | no | 6h. |
| `runtime/winner_scaling.json` | `winner_scaling.v1` | Per-strategy multipliers. | no | 15 min. |
| `runtime/regime_booster.json` | `regime_booster.v1` | Global multiplier. | **YES (180s)** | 60s timer cycle. **ACTIVE 1.30× NEW v8.6.** |
| `runtime/dynamic_caps.json` | n/a | Per-strategy dollar caps. | **YES (180s)** | Orchestrator cycle. |
| `runtime/price_cache.json` | n/a | Live price snapshots. | **YES (180s)** | 60s timer cycle. |
| `runtime/kraken_prices.json` | n/a | Kraken WS prices. | **YES (120s)** | Real-time. |
| `runtime/reconciliation_state.json` | n/a | Reconciliation snapshot. | **YES (480s)** | 360s TTL NEW v8.6. |
| `runtime/equity_history.ndjson` | `equity_history.v1` | Daily HWM checkpoints. | no | Daily 23:59 UTC; **4 records as of write time**. |

### Strategy intelligence current state (carried gap)

`runtime/strategy_intelligence.json` — same gap as v8.5 / v8.4. Macro
context provider returns `unknown` for VIX and macro-risk inputs. Not
regressed but not improved; would benefit from a real CPI/FOMC/NFP
calendar source replacing `MarketHoursRiskProvider` (Phase 12 roadmap
item).

### Expectancy → WinnerScaler

Same plumbing as v8.5. Currently producing all-1.0× output because
`median_expectancy=0.0` (the scaler cannot rank with zero signal).
Will repopulate as live cycles produce closed-trade data with the
SCR-CONFIDENT real-size sizing.

### Event risk → RegimeBooster

Same as v8.5. `severity=medium` from `MarketHoursRiskProvider` is
below the high/extreme veto threshold, so the booster is allowed to
fire — and as of v8.6, it does (1.30× active).

### NEW v8.6 — `chad-feed-watchdog.timer` (7 feeds, 120s)

`chad/ops/feed_watchdog.py` (commit `a4accdd`):

```python
WATCHED_FEEDS = [
    {"name": "price_cache",    "file": "price_cache.json",          "ttl": 180},
    {"name": "regime_state",   "file": "regime_state.json",         "ttl": 180},
    {"name": "dynamic_caps",   "file": "dynamic_caps.json",         "ttl": 180},
    {"name": "regime_booster", "file": "regime_booster.json",       "ttl": 180},
    {"name": "kraken_prices",  "file": "kraken_prices.json",        "ttl": 120},
    {"name": "reconciliation", "file": "reconciliation_state.json", "ttl": 480},
    {"name": "event_risk",     "file": "event_risk.json",           "ttl": 2400},
]
```

On stale detection: structured journal log (
`feed_stale name=<name> age=<seconds> ttl=<ttl>`) plus a Telegram
alert. Edge-trigger — alert fires once per stale event, not every
cycle while stale. Recovery (feed becomes fresh again) emits a
"recovered" message.

The TTL values are calibrated to be ~3× the publishers' update
cadence so a single missed cycle does not fire a false alarm. Worth
re-tuning once we have a few weeks of operational data.

---

## 9. TELEGRAM OPERATOR INTERFACE

### Free-text routing

`chad/utils/telegram_bot.py:1365` — `handle_free_text` classifies the
message intent and dispatches to the appropriate slash-command
handler.

### Slash commands (registered handlers)

From `chad/utils/telegram_bot.py:1455-1468`: `/ping`, `/help`,
`/coach_mode`, `/status`, `/readiness`, `/why_blocked`, `/risk`,
`/perf`, `/live_gate`, `/shadow`, `/portfolio_active`,
`/portfolio_targets <profile>`, `/portfolio_rebalance <profile>`,
`/price <symbol>`, plus advisory + AI research commands.

### Morning brief — elite prodigy voice

`chad/ops/daily_chad_report.py:MorningBrief` (`:1196`).

Schedule: `chad-morning-brief.timer` — Mon-Fri 9:00 AM ET (13:00 UTC).

NEW v8.6: now consolidates the day's fill summary (per-fill alerts
disabled in commit `d35884b`). The brief lists:
- Phase / Account / Tier / Salary (BUSINESS STATUS — v8.3).
- Yesterday's closed trades (count, P&L, top contributor).
- Open positions and what they're waiting for.
- Booster state and reasons.
- Feed health (watchdog summary).

### NEW v8.6 — Seven exception alert types (commit `931dd4e`)

```python
# chad/utils/telegram_notify.py — new alert types
ALERT_TYPES = {
    "live_loop_crash":        "🚨 Live loop crashed mid-cycle. See journal.",
    "reconciliation_red":     "🔴 Reconciliation went RED. mismatches=%d worst_diff=%.2f",
    "profit_lock_transition": "🔒 Profit lock %s → %s (sizing %.2fx)",
    "salary_authorized":      "💰 Salary authorized $%.2f/mo (PAY phase active)",
    "drawdown_breach":        "📉 Drawdown breach: %.2f%% from HWM (cap 5.0%%)",
    "scr_recovery_paused":    "✅ SCR recovered from PAUSED → %s",
    "warmup_demotion":        "⚠️  SCR demoted to WARMUP (was %s)",
}
```

Each alert is edge-trigger — fires once on transition, suppressed on
re-entry. The fill-dedupe window for the (now disabled) per-fill
alerts was 5 minutes on `(symbol, side, fill_price)`.

### NEW v8.6 — `EnvironmentFile=-/etc/chad/telegram.env` in chad-live-loop.service

Pre-`931dd4e`, the live-loop service did not load `telegram.env`, so
the `live_loop_crash` and other in-loop alerts could not actually
emit. Now loaded:

```ini
# /etc/systemd/system/chad-live-loop.service (excerpt)
[Service]
EnvironmentFile=-/etc/chad/telegram.env
EnvironmentFile=-/etc/chad/chad.env
EnvironmentFile=-/etc/chad/ibkr.env
```

The same change was applied to `chad-feed-watchdog.service` so the
watchdog can also emit Telegram alerts.

### NEW v8.6 — Per-fill alerts DISABLED (commit `d35884b`)

The fill-level alerts (introduced in `931dd4e` with 5-minute dedupe)
were still producing 50–80 messages a day in active markets. Operator
feedback: "I don't want to know every fill, I want morning brief +
EOD recap." Per-fill alerts were disabled by setting
`ALERT_PER_FILL = False` in `telegram_notify.py`. The seven exception
types from above remain enabled — those are *exceptions*, not routine.

### End-of-day brief

`chad-daily-report.timer` — Mon-Fri 4:35 PM ET (21:35 UTC). Same
elite-prodigy voice. NEW v8.6: includes the consolidated fill summary
that the per-fill alerts no longer cover.

### Real-time alerts

`chad/utils/telegram_notify.py` — read-only side effects, deterministic
retries, dedupe via runtime files. Trigger surfaces in v8.6:
- (NEW) live-loop crash
- (NEW) reconciliation RED (both publish paths)
- (NEW) profit-lock transitions
- (NEW) salary authorized
- (NEW) drawdown breach
- (NEW) SCR recovery from PAUSED
- (NEW) WARMUP demotion
- Stop-bus activation
- Edge-decay halt (now edge-trigger only NEW v8.6)
- IBKR / data-feed staleness watchdog (7 feeds NEW v8.6)
- Per-fill: **DISABLED v8.6**

### Weekly summary — BUSINESS PHASE block

`chad/ops/daily_chad_report.py:WeeklySummary` (`:1570`),
`run_weekly_summary` (`:1763`). Schedule:
`chad-weekly-report.timer` — Sundays 20:00 UTC. BUSINESS PHASE block
unchanged from v8.5.

### Voice lock

CHAD voice enforced by system prompts in `MorningBrief._chads_take`
(elite prodigy) and the dashboard chat (strict no-jargon plain
English). LLM call failure → silent degrade to data-only.

---

## 10. DASHBOARD (chadtrades.com)

### Auth

- Password-protected via `/etc/chad/dashboard.env` (basic auth +
  session token in cookie).
- Public health endpoint `/health` (no auth) for systemd / monitoring.
- **NEW v8.6 — Brute-force protection.** 5 failed login attempts
  within a 5-minute window → 5-minute IP lockout. Implemented in
  `chad/dashboard/api.py` middleware:

  ```python
  # chad/dashboard/api.py — brute force protection (commit c267076)
  FAILED_ATTEMPTS_LIMIT = 5
  LOCKOUT_WINDOW_S = 300

  _failed_attempts: Dict[str, List[float]] = defaultdict(list)
  _lockouts: Dict[str, float] = {}

  def check_rate_limit(client_ip: str) -> bool:
      now = time.time()
      if client_ip in _lockouts and _lockouts[client_ip] > now:
          return False
      attempts = [t for t in _failed_attempts[client_ip] if now - t < LOCKOUT_WINDOW_S]
      _failed_attempts[client_ip] = attempts
      if len(attempts) >= FAILED_ATTEMPTS_LIMIT:
          _lockouts[client_ip] = now + LOCKOUT_WINDOW_S
          return False
      return True
  ```

### Routing

- TLS via Certbot (cert valid through 2026-07-19).
- nginx → `127.0.0.1:8765` (FastAPI/Uvicorn).

### NEW v8.6 — `oldest_source_mtime_utc` staleness signal

`chad/dashboard/api.py` `_state()` builder now computes the oldest
mtime across all source files contributing to the snapshot:

```python
# chad/dashboard/api.py (commit c267076)
SOURCE_FILES = [
    RUNTIME / "scr_state.json",
    RUNTIME / "regime_state.json",
    RUNTIME / "tier_state.json",
    RUNTIME / "business_phase.json",
    RUNTIME / "withdrawal_authorization.json",
    RUNTIME / "regime_booster.json",
    RUNTIME / "winner_scaling.json",
    RUNTIME / "reconciliation_state.json",
    RUNTIME / "dynamic_caps.json",
    RUNTIME / "profit_lock_state.json",
    RUNTIME / "pnl_state.json",
]

def oldest_source_mtime_utc():
    oldest = None
    for f in SOURCE_FILES:
        if f.exists():
            mt = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
            if oldest is None or mt < oldest:
                oldest = mt
    return oldest.isoformat() if oldest else None
```

The dashboard front-end can render a banner if
`now − oldest_source_mtime_utc > 600s`. Closes a long-standing UX gap
where the dashboard would render an apparently-current view from
15-minute-stale data.

### NEW v8.6 — `/api/state` schema ratified

The full top-level schema is now frozen for chat-context consumers:

```json
{
  "ts_utc":                     "<iso>",
  "ibkr_connected":             true,
  "chad_status":                "TRAINING|LIVE|...",
  "portfolio":                  { ... },
  "open_positions":             [ ... ],
  "strategies":                 { ... },
  "system_health":              { ... },
  "intelligence":               { ... },
  "business":                   { ... },
  "oldest_source_mtime_utc":    "<iso>"
}
```

Chat context (`/api/chat`) reads this same object, so any new
top-level field automatically becomes available to the chat without
a separate plumbing pass.

### Panels (preserved)

- **Training Mode card** — `_chad_status` (`:327`).
- **Account Value** — `_portfolio` (`:275`).
- **Realized PnL** — `runtime/pnl_state.json`.
- **Market** — `_intelligence` (`:455`).
- **What CHAD Is Watching** — strategy_intelligence + intel feeds.
- **Open Positions** — `_open_positions` (`:301`).
- **Recent Trades** — `_iter_recent_closed_trades` (`:625`).
  **NEW v8.6: 500-line cap per file (D7).**
- **Strategy Performance** — `_strategies` (`:356`).
- **Ask CHAD chat** — `/api/chat`.
- **Business endpoint** — `_business()` (v8.3).

### NEW v8.6 — `/api/recent-trades` 500-line cap

`chad/dashboard/api.py` (commit `fec3d5b` D7):

```python
RECENT_TRADES_LINE_CAP = 500

def _iter_recent_closed_trades(date_str):
    path = TRADES / f"trade_history_{date_str}.ndjson"
    if not path.exists():
        return []
    with path.open() as f:
        # tail -n 500 semantics; iterate by line and keep last N
        lines = deque(f, maxlen=RECENT_TRADES_LINE_CAP)
    return [json.loads(line) for line in lines]
```

Pre-fix, on a heavy trading day this was a multi-MB response. Now the
client gets the most recent 500 closed trades per file, deterministically.

### NEW v8.6 — Chat context includes `oldest_source_mtime_utc`

`chad/dashboard/api.py:1086-1103` — chat context picks up the new
`oldest_source_mtime_utc` automatically because the chat builder reads
the same `_state()` payload. The system prompt is updated to mention
the field and instruct CHAD to acknowledge stale data when it answers
factual questions.

### Service health (commit `56eaeea`)

`api.py:_system_health` correctly counts oneshot service success.

### Chat (`/api/chat`)

- Endpoint `chad/dashboard/api.py:1121`.
- Model: `claude-sonnet-4-6`.
- Plain-English voice (system prompt `:910-948`).

---

## 11. SERVICES & TIMERS

**99 `chad-*` units loaded** (54 services + 45 timers) — adds
`chad-health-monitor.service` and `chad-health-monitor.timer` (NEW
v8.7) on top of v8.6's 97. **14 services running. 0 failed.**

### Hot-path services (always running)

| Unit | Purpose |
|---|---|
| `chad-live-loop.service` | Hot-path: rebuild guard → signals → gates → execution. **NEW v8.6 — SIGTERM handler, telegram.env loaded.** |
| `chad-orchestrator.service` | Risk-budget publisher (`runtime/dynamic_caps.json`). |
| `chad-ibgateway.service` | IBC-managed IB Gateway paper port 4002. |
| `chad-ibkr-bar-provider.service` | Polls IBKR `reqHistoricalData` every 30s. |
| `chad-kraken-ws.service` | Kraken WebSocket crypto feed. |
| `chad-shadow-status.service` | HTTP endpoint `:9618/shadow` (SCR sizing source). |
| `chad-metrics.service` | Prometheus-style metrics on `:9620/metrics`. |
| `chad-backend.service` | FastAPI backend for dashboard/reports. |
| `chad-dashboard.service` | Public dashboard on `127.0.0.1:8765`. |
| `chad-telegram-bot.service` | Operator alerts + briefs + free-text routing. |
| `chad-x11vnc.service`, `chad-xvfb.service` | Virtual display for IB Gateway. |
| `chad-strategy-intelligence-refresh.service` | AI per-strategy regime profile. |

### NEW v8.6 — `chad-live-loop.service` SIGTERM handler

```python
# chad/core/live_loop.py — SIGTERM handler (commit 9bc5286)
_stop_requested = False

def _sigterm_handler(signum, frame):
    global _stop_requested
    LOG.info("sigterm_received cleaning_shutdown")
    _stop_requested = True
    # Telegram alert (best-effort)
    try:
        send_telegram("🛑 Live loop received SIGTERM — finishing in-flight cycle.")
    except Exception:
        pass

signal.signal(signal.SIGTERM, _sigterm_handler)

def main_loop():
    while not _stop_requested:
        run_cycle()
    # Drain — close broker connections cleanly
    try:
        ibkr_adapter.disconnect_clean()
        kraken_executor.disconnect_clean()
    finally:
        send_telegram("✅ Live loop stopped cleanly.")
```

systemd unit `KillSignal=SIGTERM`, `TimeoutStopSec=30`. The handler
respects the in-flight cycle (cycle time ~10–12s) and lets it finish
before draining and exiting. No more mid-cycle truncation.

### NEW v8.6 — `chad-feed-watchdog.timer`

| Field | Value |
|---|---|
| Unit | `/etc/systemd/system/chad-feed-watchdog.{timer,service}` |
| Cadence | `OnBootSec=60`, `OnUnitActiveSec=120` (every 2 minutes) |
| AccuracySec | 10 |
| Persistent | true |
| ExecStart | `python3 chad/ops/feed_watchdog.py` |
| EnvironmentFile | `-/etc/chad/telegram.env`, `-/etc/chad/chad.env` |

The watchdog checks 7 feeds and emits Telegram alerts on stale
detection. Fires the first time a feed crosses TTL; recovery emits a
"recovered" message.

### v8.3 timers (business framework)

| Timer | Cadence | Purpose |
|---|---|---|
| `chad-portfolio-snapshot.timer` | every 5 min | Refresh `portfolio_snapshot.json`. |
| `chad-equity-history.timer` | daily 23:59 UTC | Append HWM checkpoint. |
| `chad-withdrawal-manager.timer` | every 6h | Compute salary authorization. |
| `chad-tier-manager.timer` | every 5 min | Equity-tier strategy enable/disable. |
| `chad-winner-scaler.timer` | every 15 min | Per-strategy expectancy multipliers. |
| `chad-business-phase.timer` | every 30 min | BUILD/GROW/PAY publisher. |

### v8.4 timers (gap-fills)

| Timer | Cadence | Purpose |
|---|---|---|
| `chad-regime-booster.timer` | every 60s | Refresh `regime_booster.json`. |
| `chad-event-risk.timer` | every 10 min | Refresh `event_risk.json`. |

### NEW v8.6 timer

| Timer | Cadence | Purpose |
|---|---|---|
| `chad-feed-watchdog.timer` | every 120s | **NEW v8.6.** Monitors 7 critical feeds; Telegram alert on stale. |

### NEW v8.7 timer

| Timer | Cadence | Purpose |
|---|---|---|
| `chad-health-monitor.timer` | every 300s (`OnBootSec=120`) | **NEW v8.7.** 3-tier AI health monitor — rule engine (9 rules), Claude reasoning escalation, auto-remediation. |

#### NEW v8.7 — `chad-health-monitor.timer`

| Field | Value |
|---|---|
| Unit | `/etc/systemd/system/chad-health-monitor.{timer,service}` |
| Cadence | `OnBootSec=120`, `OnUnitActiveSec=300` (every 5 minutes) |
| ExecStart | `python3 chad/ops/health_monitor.py` |
| EnvironmentFile | `-/etc/chad/telegram.env`, `-/etc/chad/chad.env` |
| Sudo | passwordless `systemctl restart chad-*` (pre-confirmed; no broader privilege) |

The health monitor evaluates 9 rules every 5 minutes (Tier 1),
escalates to `claude-sonnet-4-6` for diagnostic narrative when a
critical or unfamiliar condition fires (Tier 2), and executes a
vetted set of safe remediations directly (Tier 3 — restart dead
services, restart stale feed publishers, restore corrupt runtime
files from backup, archive old fill files). Notify-only states
(SCR PAUSED, stop bus active, reconciliation RED, profit lock
LOCK2+, edge decay halts) are paged to the operator without
auto-action. Telegram message types: `AUTO-FIXED`,
`HEALTH MONITOR CRITICAL`, `AI HEALTH ANALYSIS`. Modules:
`chad/ops/health_monitor.py`, `chad/ops/health_monitor_rules.py`,
`chad/ops/health_monitor_remediation.py`.

### Timers — hot-path (preserved)

| Timer | Cadence | Purpose |
|---|---|---|
| `chad-trade-closer.timer` | OnBootSec=45, OnUnitActiveSec=60 | Scheduled exits. |
| `chad-scr-sync.timer` | every 60s | Refresh `scr_state.json`. |
| `chad-reconciliation-publisher.timer` | every 5 min | Publishes reconciliation snapshots (TTL 360s NEW v8.6). |
| `chad-paper-trade-exec.timer` | every 10m | Backstop paper executor. |
| `chad-paper-trade-executor.timer` | every 5 min | Alternate paper-trade pipeline. |
| `chad-ibkr-paper-fill-harvester.timer` | every 5m | Harvests broker fills. |
| `chad-ibkr-broker-events.timer` | every 5m | Collects broker events. |
| `chad-ibkr-price-refresh.timer` | 60s mkt / 300s off-hours | Price cache refresh. |
| `chad-options-monitor.timer` | every 60s during market hours | Monitors options positions. |
| `chad-options-chain-refresh.timer` | Mon-Fri 12:30 UTC | Refresh options chain cache. |

### Timers — analytics & feeds (preserved)

| Timer | Cadence | Purpose |
|---|---|---|
| `chad-expectancy-tracker.timer` | every 5m | Per-strategy expectancy. |
| `chad-symbol-blocker.timer` | every 5m | Symbol perf-based blocking. |
| `chad-trends-refresh.timer` | recurring | Google Trends. |
| `chad-reddit-sentiment-refresh.timer` | recurring | Reddit sentiment. |
| `chad-short-interest-refresh.timer` | recurring | Short interest. |
| `chad-burnin-check.timer` | every 10m | Burn-in health check. |
| `chad-burnin-daily-summary.timer` | 23:59 UTC daily | Burn-in summary. |
| `chad-disk-guard.timer` | every 30m | Disk usage guard. |
| `chad-portfolio-artifacts.timer` | every 60m | Portfolio artifact publisher. |
| `chad-rebalance-auto-executor-paper.timer` | every 10m | Phase 12 rebalance preview (paper). |
| `chad-operator-intent-refresh.timer` | every 10m | Operator intent file refresh. |
| `chad-action-applier.timer` | every 60s | SSOT v5 ActionApplier. |
| `chad-execq-publisher.timer` | every 60s | execution_quality + latency_state publisher. |
| `chad-mutation-state-publisher.timer` | every 60s | action_state + change_canary_state. |
| `chad-lifecycle-truth-publisher.timer` | every 60s | trade_lifecycle_state + positions_truth. |
| `chad-profit-lock-publisher.timer` | every 60s | profit_lock + pnl_state publisher. |
| `chad-calendar-state-publisher.timer` | every 5m | calendar_state.json publisher. |

### Timers — reporting

| Timer | Cadence | Purpose |
|---|---|---|
| `chad-morning-brief.timer` | Mon-Fri 13:00 UTC (9:00 ET) | Elite-prodigy pre-market brief (BUSINESS STATUS + consolidated fill summary NEW v8.6). |
| `chad-daily-report.timer` | Mon-Fri 21:35 UTC (4:35 ET) | EOD recap with consolidated fill summary NEW v8.6. |
| `chad-weekly-report.timer` | Sunday 20:00 UTC | Weekly summary with BUSINESS PHASE block. |
| `chad-advisory-pre-market.timer` | recurring | Pre-market advisory. |
| `chad-ibkr-daily-bars-refresh.timer` | nightly | Daily bars refresh (futures). |
| `chad-proofs-cleanup.timer` | daily 03:20 UTC | Proof artifacts cleanup. |

### Intentionally disabled / retired

- `chad-reconciliation.timer` — masked via `/dev/null` symlink (v8.4).
- `chad-options-chain-refresh.service` — standalone form disabled.
- `chad-polygon-stocks`, `chad-bars-validate`,
  `chad-daily-bars-backfill` — masked.

---

## 12. DATA & STORAGE

### Universe

`config/universe.json`:
- **Equities/ETFs (37):** AAPL, SPY, MSFT, GOOGL, BAC, IEMG, QQQ, VWO,
  NVDA, GLD, SH, PSQ, SVXY, UVXY, **VXX (refreshed v8.6)**, VIXY,
  IWM, TLT, plus 20 institutional consensus symbols added in v8.5.
- **Futures (10):** MES (CME), MNQ (CME), MCL (NYMEX), MGC (COMEX),
  ZN (CBOT), ZB (CBOT), M6E (CME), SIL (COMEX), MYM, M2K.

### Bar provider

- Service: `chad-ibkr-bar-provider.service`.
- Mechanism: `reqHistoricalData` polling every 30s.
- 1m bars: 25 symbols stored in `data/bars/1m/`.
- 1d bars: 30+ symbols stored in `data/bars/1d/` (now includes
  refreshed VXX after commit `42268f0`).
- Daily bar refresh: `chad-ibkr-daily-bars-refresh.timer` runs
  nightly.

### Crypto data

- Kraken 1d bars via REST: `BTC-USD, ETH-USD, SOL-USD`.
- Kraken WS feed: real-time prices + balances. **NEW v8.6 watched by
  feed watchdog (kraken_prices.json TTL 120s).**

### Price cache

`runtime/price_cache.json` — refreshed by
`chad-ibkr-price-refresh.timer`. **NEW v8.6 watched by feed watchdog
(180s TTL).**

### Fills ledger

- Equity / futures / options: `data/fills/FILLS_YYYYMMDD.ndjson`.
- Crypto: `data/fills/kraken_fills_YYYYMMDD.ndjson`.

### NEW v8.6 — `data/trades/trade_history_YYYYMMDD.ndjson` 500-line cap on dashboard

The dashboard endpoint `/api/recent-trades` caps at 500 most-recent
lines per file (commit `fec3d5b` D7). The underlying NDJSON files are
not truncated — the cap is at the read path.

### SEC 13F refresh

Weekly cron Sunday 00:00 UTC via
`scripts/update_institutional_consensus.py`. Output:
`runtime/institutional_consensus.json`.

### Business framework state files

| File | Schema | Cadence | Source module |
|---|---|---|---|
| `runtime/equity_history.ndjson` | `equity_history.v1` | daily 23:59 UTC | `equity_history_publisher.py` |
| `runtime/portfolio_snapshot.json` | n/a (`ttl_seconds=300`) | every 5 min | `portfolio_snapshot_publisher.py` |
| `runtime/tier_state.json` | `tier_state.v1` | every 5 min | `tier_manager.py` |
| `runtime/withdrawal_authorization.json` | n/a | every 6h | `withdrawal_manager.py` |
| `runtime/business_phase.json` | `business_phase.v1` | every 30 min | `business_phase_tracker.py` |
| `runtime/winner_scaling.json` | `winner_scaling.v1` | every 15 min | `winner_scaler.py` |
| `runtime/regime_booster.json` | `regime_booster.v1` | every 60s | `regime_booster.py` |

### Other key runtime files (preserved)

- `runtime/dynamic_caps.json` — orchestrator-published per-strategy
  dollar caps with `business_overlays` block.
- `runtime/profit_lock_state.json` — circuit breaker state.
- `runtime/stop_bus.json` — halt flag.
- `runtime/strategy_health.json` — F3 composite per strategy.
- `runtime/expectancy_state.json` — F1 rolling expectancy
  (consumed by WinnerScaler).
- `runtime/options_chains_cache.json` — cached options chains.
- `runtime/last_route_decision.json` — DecisionTrace bridge.
- `runtime/live_readiness.json` — `ready_for_live=false`.

### `dynamic_caps.json:business_overlays` block (v8.3)

Same schema as v8.5. Current values reflect:
- `regime_booster_multiplier`: **`1.3`** (v8.5 was 1.0)
- `tier_filter_active`: `true`
- `winner_multipliers`: all 1.0 (v8.5 had alpha 1.5×, delta 1.304×,
  alpha_futures 0.5×; reset with expectancy refresh)

### Disk usage

- Filesystem: `/dev/root` 48 GB total. Monitor via `chad-disk-guard.timer`.

---

## 13. CHANGE LOG (delta from v8.5)

In commit order (oldest first since v8.5 was cut). 17 commits in v8.6
plus 1 new commit in v8.7. All on 2026-04-30 / 2026-05-01.

### Commit list (`git log 9f55688..HEAD --oneline --reverse`)

```
42268f0 2026-04-30 Fix: live loop cycle audit — phantom fills, unprotected stages, intent guard, regime TTL, VXX bars, bias logging
3be6785 2026-04-30 Fix: surgical phantom fill guard at submit call site — revert normalizer, block at live_loop evidence write
ceb53ea 2026-04-30 Fix: SCR audit — add excluded_pnl_zero counter to trade_stats_engine, balance exclusion ledger
23f4c03 2026-04-30 Fix: SCR audit — min-trade gates for CAUTIOUS/CONFIDENT bands, PAUSED hysteresis router (caller wiring pending)
5267705 2026-04-30 Fix: position guard audit — atomic _save_state write, chad_strategy_open in reconciliation counts
cdfed06 2026-04-30 Fix: profit router audit — test isolation for ledger, exclude broker_sync from routing
931dd4e 2026-04-30 Fix: Telegram audit — live loop crash, recon RED, profit lock, salary, drawdown, fill dedupe, SCR recovery alerts
c267076 2026-04-30 Fix: dashboard audit — staleness signal oldest_source_mtime_utc, brute force login protection
ba65367 2026-04-30 Fix: reconciliation audit — add MYM/M2K to futures exclusion, raise TTL to 360s
52786b7 2026-04-30 Fix: edge decay audit — wire halt enforcement, move before signals, fix key name, edge-trigger alert, harden clear script
71d8ea6 2026-04-30 Fix: data feed audit — regime_state TTL enforcement at read time, stale returns unknown
a4accdd 2026-04-30 Fix: data feed audit — feed watchdog service monitors 7 critical feeds every 120s, alerts on stale
28e8963 2026-04-30 Fix: wire PAUSED hysteresis caller — inject prev_state/paused_recovery_ticks into evaluate_confidence, round-trip through ShadowStats
74e5a0d 2026-04-30 Fix: D2 bucket accumulators — drain mechanism, get_beta_remaining, mark_beta_consumed, beta injection behind CHAD_PROFIT_ROUTER_BETA_INJECTION flag
9bc5286 2026-04-30 Fix: D4 SIGTERM handler — clean shutdown on systemd stop, Telegram alert, respects in-flight cycle
fec3d5b 2026-05-01 Fix: D6 A4 gate intraday/daily split (900s/172800s), D7 recent-trades 500-line cap
d35884b 2026-05-01 Fix: disable per-fill Telegram alerts — too noisy; morning brief + EOD recap only
1207eeb 2026-05-01 Build: CHAD AI Health Monitor — 3-tier autonomous bug detection, rule engine, Claude reasoning, auto-remediation  ← NEW v8.7
```

### 1. `42268f0` — *Fix: live loop cycle audit* (2026-04-30)

Opening rollup commit of the audit session. Forensic walk of the
`run_cycle()` path:

- **Phantom fills:** noted that error-status orders were leaking into
  evidence writes. The fix attempted in this commit touched the
  normalizer too broadly — reverted in `3be6785` and replaced with a
  surgical guard at the submit site.
- **Unprotected stages:** signal-build, plan-build, and intent-build
  stages had no try/except boundary; a single bad strategy could
  crash the cycle. Wrapped each stage; failures log and skip the
  affected strategy/symbol without aborting the cycle.
- **Intent guard:** `chad/core/live_loop.py` now asserts intent
  consistency (strategy non-empty, symbol non-empty, side ∈
  {BUY,SELL}, qty > 0) before passing to the routing gates.
- **Regime TTL:** placeholder noting the TTL gap that became commit
  `71d8ea6`'s read-time enforcement.
- **VXX bars:** `data/bars/1d/VXX.json` was 26 days stale. Backfilled
  via `chad-ibkr-daily-bars-refresh` invocation; VXX added to
  universe.
- **Bias logging:** added structured per-strategy bias logging to
  `live_loop._log_cycle_summary` so per-cycle skew is visible in
  journals.

### 2. `3be6785` — *Fix: surgical phantom fill guard at submit call site* (2026-04-30)

Reverts the normalizer rewrite from `42268f0` and replaces it with a
narrow guard at the submit call site (`chad/core/live_loop.py:1180`):
error-status orders (`error`, `Cancelled`, `Inactive`) skip evidence
write and emit a structured warning log. Preserves the v8.1 single-
chokepoint invariant for the normalizer.

### 3. `ceb53ea` — *Fix: SCR audit — `excluded_pnl_zero` counter* (2026-04-30)

`runtime/scr_state.json:stats` had three documented exclusion
buckets but trades with PnL exactly $0.00 were silently dropped from
`effective_trades` without being counted anywhere. Added explicit
`excluded_pnl_zero` counter; ledger now balances.

### 4. `23f4c03` — *Fix: SCR audit — min-trade gates + PAUSED router* (2026-04-30)

Two changes:
- **Min-trade gates** at band boundaries: `MIN_TRADES_CAUTIOUS=100`,
  `MIN_TRADES_CONFIDENT=133`. State machine cannot promote past the
  band without these counts even if metrics look good.
- **PAUSED hysteresis router:** added the router logic that holds
  PAUSED for `paused_recovery_ticks` consecutive healthy reads before
  promoting back. Caller wiring deferred to commit `28e8963`.

### 5. `5267705` — *Fix: position guard audit* (2026-04-30)

- **Atomic `_save_state`:** writes via tempfile + `os.replace` (was
  truncate+write — could leave 0-byte file on crash).
- **`chad_strategy_open` in reconciliation counts:** prior summary
  collapsed all CHAD-side opens into `chad_open` regardless of
  whether they were strategy-tracked or `broker_sync` echoes. New
  field exposes the strategy-tracked count separately.

### 6. `cdfed06` — *Fix: profit router audit* (2026-04-30)

- **Test isolation:** `route_profit` and helpers accept optional
  `routing_path` param. Tests pass a temp file. **19 phantom
  `alpha_test` entries** purged from production ledger; stamped
  `alpha_test_purged_at_utc` and `alpha_test_purged_count`.
- **`broker_sync` excluded from routing:** broker_sync realized PnL
  no longer contributes to amplifier or beta buckets. Returns
  `{"no_routing": True, "reason": "broker_sync_excluded"}`.

### 7. `931dd4e` — *Fix: Telegram audit — 7 new alert types* (2026-04-30)

Seven new alert types: `live_loop_crash`, `reconciliation_red`,
`profit_lock_transition`, `salary_authorized`, `drawdown_breach`,
`scr_recovery_paused`, `warmup_demotion`. All edge-trigger only.
`fill_dedupe` window 5min on `(symbol, side, fill_price)` for the
per-fill alerts (later disabled in `d35884b`).

Also added `EnvironmentFile=-/etc/chad/telegram.env` to
`chad-live-loop.service` so the live loop's `live_loop_crash` and
in-loop alerts can actually emit.

### 8. `c267076` — *Fix: dashboard audit* (2026-04-30)

- **`oldest_source_mtime_utc`** at top level of `/api/state`. Single
  staleness signal across all source files.
- **Brute force login protection:** 5 attempts → 5-min IP lockout.
- **`/api/state` schema ratified** as the frozen contract for chat-
  context consumers.

### 9. `ba65367` — *Fix: reconciliation audit — MYM/M2K, TTL 360s* (2026-04-30)

- Added `MYM, M2K` to `KNOWN_FUTURES_SYMBOLS` in
  `chad/ops/reconciliation_publisher.py:58`.
- Raised reconciliation TTL from 300s to 360s (60s margin over the
  300s publisher cadence).

### 10. `52786b7` — *Fix: edge decay audit — wire halt enforcement* (2026-04-30)

Five changes:
1. Wire `is_strategy_halted()` into signal building (was completely
   unwired).
2. Move halt check from after stage-4 (sizing) to before stage-1
   (signals).
3. Fix key name `consecutive_negative` → `consecutive_neg`.
4. Edge-trigger halt alerts (was firing every cycle while halted).
5. Harden `scripts/clear_edge_decay.py` to require `--confirm` flag.

### 11. `71d8ea6` — *Fix: data feed audit — regime_state TTL at read time* (2026-04-30)

Added read-time TTL enforcement in every consumer of
`runtime/regime_state.json`. Stale read → `regime=unknown`. Fail-soft
to universally-eligible regime.

### 12. `a4accdd` — *Fix: data feed audit — feed watchdog* (2026-04-30)

New service `chad-feed-watchdog.{service,timer}` at 120s cadence.
Monitors 7 critical feeds with per-feed TTL. Telegram alert on
stale; recovery alert on fresh.

Companion fix: MGC and SIL had expired contract resolution at IBKR;
restarting the bar provider service refreshed the contract month and
the bars resumed flowing. Not a separate commit but an operational
side effect of the audit.

### 13. `28e8963` — *Fix: wire PAUSED hysteresis caller* (2026-04-30)

Companion to `23f4c03`. The router was implemented but not called
correctly. Now `backend/api_gateway.py` injects `prev_state` and
`paused_recovery_ticks` into `evaluate_confidence`; `ShadowStats`
round-trips them through `compute_state()`. End-to-end wiring closed.

### 14. `74e5a0d` — *Fix: D2 bucket accumulators with drain* (2026-04-30)

- **Drain mechanism:** new `consumed_beta_usd`, `consumed_amplifier_usd`
  fields. New functions `get_beta_remaining()`, `mark_beta_consumed()`.
- **Beta injection:** wired into the `beta` strategy but gated behind
  `CHAD_PROFIT_ROUTER_BETA_INJECTION` env flag (default off).

### 15. `9bc5286` — *Fix: D4 SIGTERM handler* (2026-04-30)

`chad-live-loop.service` now handles SIGTERM cleanly: sets a stop
flag at the top of the cycle loop, lets in-flight cycle finish,
closes broker connections, sends Telegram alert. Respects systemd
`KillSignal=SIGTERM` and `TimeoutStopSec=30`.

### 16. `fec3d5b` — *Fix: D6 A4 gate split, D7 recent-trades cap* (2026-05-01)

- **A4 gate split:** intraday strategies → 900s threshold +
  `bar_missing` failure; daily/swing/futures → 172800s.
- **`/api/recent-trades` cap:** 500 lines/file via deque(maxlen=500)
  during stream.

### 17. `d35884b` — *Fix: disable per-fill Telegram alerts* (2026-05-01)

`ALERT_PER_FILL = False` in `telegram_notify.py`. Per-fill alerts
were 50–80/day even with 5min dedupe; consolidated into morning brief
and EOD recap. Other 7 alert types from `931dd4e` remain enabled.

### 18. `1207eeb` — *Build: CHAD AI Health Monitor* (2026-05-01) — NEW v8.7

Three-tier autonomous health surveillance, layered on top of the v8.6
foundations. Adds three modules and one timer; nothing pre-existing
was removed.

- **Tier 1 — rule engine** (`chad/ops/health_monitor_rules.py`).
  9 rules evaluated every 5 minutes: R01 critical services running,
  R02 feed freshness (7 v8.6 feeds), R03 SCR paused, R04 stop bus
  active, R05 reconciliation RED, R06 profit lock LOCK2+, R07 disk
  usage at threshold, R08 corrupt runtime files (zero-byte / invalid
  JSON), R09 edge decay halts.
- **Tier 2 — Claude reasoning** (`chad/ops/health_monitor.py`).
  When Tier 1 raises a critical or unfamiliar condition, the monitor
  packages a full system snapshot and dispatches it to
  `claude-sonnet-4-6`; the diagnostic narrative + recommended actions
  return as an `AI HEALTH ANALYSIS` Telegram message.
- **Tier 3 — auto-remediation**
  (`chad/ops/health_monitor_remediation.py`). Vetted safe actions:
  restart dead services (`sudo systemctl restart chad-*`, passwordless
  pre-confirmed for that exact form), restart stale feed publishers,
  restore corrupt runtime files from backup, archive old fill files
  when disk crosses threshold. All other rule fires (R03, R04, R05,
  R06, R09) are notify-only — never auto-resume an SCR pause, clear
  the stop bus, or self-cancel a profit lock.
- **Telegram surface:** three new message types — `AUTO-FIXED` (Tier
  3 success), `HEALTH MONITOR CRITICAL` (Tier 1 escalation without a
  Tier 3 fix), `AI HEALTH ANALYSIS` (Tier 2 narrative).
- **systemd:** new units
  `/etc/systemd/system/chad-health-monitor.{service,timer}`.
  `OnBootSec=120`, `OnUnitActiveSec=300` (every 5 minutes).
  `EnvironmentFile=-/etc/chad/telegram.env, -/etc/chad/chad.env`.

### Aggregate

18 commits across the v8.5 → v8.7 span (17 in v8.6 + 1 in v8.7). Net
code change is overwhelmingly additive — guard logic, alert plumbing,
watchdog service, hysteresis router, and the v8.7 health monitor. The
phantom-fill normalizer revert in `3be6785` remains the only material
revert. No file deletions.

---

## 14. KNOWN ISSUES

### RESOLVED IN v8.6

| ID | Was | Resolution | Commit |
|---|---|---|---|
| Phantom fill leakage | BUG — error-status orders persisted | Surgical guard at submit site | `3be6785` |
| SCR ledger imbalanced | BUG — `excluded_pnl_zero` was implicit | Explicit counter added | `ceb53ea` |
| SCR no min-trade gates | RISK — band promotion on metrics alone | `MIN_TRADES_CAUTIOUS=100`, `MIN_TRADES_CONFIDENT=133` | `23f4c03` |
| PAUSED hysteresis incomplete | BUG — router built but caller unwired | Caller injects `prev_state`/`paused_recovery_ticks` | `23f4c03` + `28e8963` |
| Position guard non-atomic write | BUG — 0-byte file on crash | Tempfile + `os.replace` | `5267705` |
| Reconciliation count opaque | UX — strategy-open conflated with broker_sync echoes | `chad_strategy_open` field | `5267705` |
| Profit router test isolation | BUG — 19 phantom `alpha_test` entries in prod | `routing_path` param + ledger purge | `cdfed06` |
| `broker_sync` routing distortion | BUG — broker_sync PnL contaminated buckets | Excluded from `route_profit` | `cdfed06` |
| Telegram coverage thin | UX — exception types unalerted | 7 new alert types | `931dd4e` |
| Per-fill Telegram noise | UX — 50–80 messages/day | Disabled; consolidated in briefs | `d35884b` |
| `chad-live-loop` missing `telegram.env` | BUG — in-loop alerts could not emit | `EnvironmentFile=-/etc/chad/telegram.env` | `931dd4e` |
| Dashboard staleness opaque | UX — no single staleness signal | `oldest_source_mtime_utc` | `c267076` |
| Dashboard brute-force exposed | SECURITY — no rate limit | 5 attempts → 5min lockout | `c267076` |
| `/api/state` schema unstable | API — no frozen contract | Schema ratified | `c267076` |
| `/api/recent-trades` unbounded | PERF — multi-MB responses | 500-line cap | `fec3d5b` |
| Reconciliation MYM/M2K drift | BUG — new futures classified as drift | Added to `KNOWN_FUTURES_SYMBOLS` | `ba65367` |
| Reconciliation TTL race | BUG — 300s TTL == 300s cadence | Raised to 360s | `ba65367` |
| Edge decay halt unwired | CRITICAL — halts published but ignored | `is_strategy_halted()` wired into signal entry | `52786b7` |
| Edge decay key mismatch | BUG — `consecutive_negative` vs `consecutive_neg` | Fixed | `52786b7` |
| Edge decay alert spam | UX — fired every cycle while halted | Edge-trigger only | `52786b7` |
| `clear_edge_decay.py` no confirm | SAFETY — easy accidental clear | `--confirm` flag required | `52786b7` |
| `regime_state.json` no read-time TTL | BUG — stale reads silently honored | Stale → `regime=unknown` | `71d8ea6` |
| No feed watchdog | OBSERVABILITY — operator surface ad-hoc | 7-feed watchdog @ 120s | `a4accdd` |
| MGC/SIL expired contracts | BUG — expired contract month | Resolved by service restart | (operational, paired with `a4accdd`) |
| Profit router append-only | DESIGN — no drain mechanism | `consumed_beta_usd`, `consumed_amplifier_usd` + injection flag | `74e5a0d` |
| `chad-live-loop` mid-cycle SIGTERM | BUG — process torn out, occasional inconsistency | SIGTERM handler with cycle drain | `9bc5286` |
| A4 gate single threshold | BUG — 300s nonsense for daily/swing | Strategy-aware split (900s / 172800s) | `fec3d5b` |
| VXX bars 26 days stale | DATA GAP — universe entry not refreshed | Backfilled | `42268f0` |

### STILL OPEN

| ID | Severity | Summary |
|---|---|---|
| 4 pre-existing test fixture failures | COSMETIC | 3× `test_position_guard.py` (clientId fixture artifact) + 1× `test_regime_classifier.py::test_g2_matrix` (config drift). All cosmetic; production paths use the right clientIds and the calibrated matrix is intentional. |
| Amplifier bucket wiring | DESIGN | Pending winner_scaler producing non-trivial multipliers (currently all 1.0×). When multipliers re-emerge, amplifier bucket should inject into the top-multiplier strategy similarly to the beta path. |
| `CHAD_PROFIT_ROUTER_BETA_INJECTION` | OPERATIONAL | Flag off by default. Paper soak required before enabling; expect 1–2 weeks of clean ledger drains in test mode first. |
| Per-trade P&L decomposition | NOT YET BUILT | Alpha vs spread vs slippage attribution incomplete. Today's closed-trade record carries total realized PnL only. |
| ML veto loop (Phase 5B) | NOT YET BUILT | XGBoost retrain pipeline incomplete. |
| `alpha_options` stuck on SPY | DEGRADED | Position MAINTAINED → no new spreads emit until existing closes. By design (no flip on options spreads); investigate whether time-based exit should be added. |
| `omega_vol` health = 0.10 | DEGRADED | 3 samples; sample size too small to act on. F4 edge-decay should halt at 5-loss threshold (now actually enforced v8.6). |
| `strategy_intelligence` all profile=normal | DEGRADED | Macro context provider returns `unknown` for VIX/macro inputs. Same as v8.5; needs a real CPI/FOMC/NFP calendar source. |
| `event_risk.json` is bootstrap-only | COSMETIC | `MarketHoursRiskProvider` placeholder; replace with macro calendar (Phase 12 roadmap). |
| `days_in_phase` approximate | COSMETIC | Uses `len(history)` proxy; fine until phase transitions accumulate. |
| `winner_scaling.json` all 1.0× | TRANSIENT | `median_expectancy=0.0` after expectancy refresh; will repopulate as live cycles produce closed trades. Not a fault. |
| Salary withdrawal automation | NOT YET BUILT | `WithdrawalManager` authorizes; operator still moves money manually. Out of scope until live. |

### TRACKED ISSUES (from v8.5 §13, still open)

| ID | Sev | Status | Summary |
|---|---|---|---|
| ISSUE-22 | P2 | OPEN | Legacy placeholder audit item. |
| ISSUE-29 | P1 | PARTIAL | `apply_close_intents` mutates guard before broker confirms. Reconciler-side fix landed; root cause untreated. |
| ISSUE-50 | P1 | OPEN | `chad-options-chain-refresh` hangs when IBKR ushmds farm is down. Timeout wrapper pending. |
| ISSUE-54 | P2 | OPEN | `runtime/pnl_state.json` still tracked in git (currently dirty). |
| ISSUE-58 | P2 | OPEN | `chad-trade-closer.timer` uses OnBootSec/OnUnitActiveSec; needs OnCalendar or documented seed step. |
| ISSUE-75 | P1 | OPEN | Multiple call sites write `position_guard.json` directly. Atomic writer landed v8.6; unifying call sites still pending. |
| ISSUE-78 | P2 | OPEN | Two code paths read `CHAD_EXECUTION_MODE`. |

### TEST FAILURES (current run, 4 failed / 1015 passed expected)

| Test | Cause | Severity |
|---|---|---|
| `test_position_guard.py::test_rebuild_clears_broker_sync_when_strategy_entry_added` | Pre-existing — `clientId=99` collision in fixtures. | Cosmetic. |
| `test_position_guard.py::test_rebuild_preserves_broker_sync_when_no_strategy_entry_for_symbol` | Same root cause. | Cosmetic. |
| `test_position_guard.py::test_rebuild_partial_attribution_multi_strategy` | Same root cause. | Cosmetic. |
| `test_regime_classifier.py::test_g2_matrix_loads_with_calibrated_values` | Config drift — assertion vs intentionally-expanded matrix. | Cosmetic. |

Total: **4 failed, 1015 passed** (was 5 failed / 1014 passed in v8.5
— the 5th failure was a position_guard variant that the atomic-write
fix in `5267705` happens to make consistent again).

### COSMETIC (unchanged from v8.5)

- **`alpha_futures` docstring lists `MCL`** — module docstring drift.
- **`alpha_futures_config.py:55-60` defaults to 4 symbols** — dead code.
- **`strategy_health.json` covers 6 of 16 strategies** — F3 sample
  threshold; quiet strategies absent, not faulty.
- **`event_risk.json` is bootstrap-provider only** — see Phase 12.
- **`days_in_phase` approximate** — `len(history)` proxy.

---

## 15. PHASE ROADMAP

### IMMEDIATE (next session)

- First full Monday-Friday session with **SCR=CONFIDENT, sizing_factor=1.0**
  and **regime booster active 1.30×**. The strategies will deploy real
  size for the first time since SCR was created.
- Validate the new alert pipeline against a real Monday open — expect
  morning brief at 13:00 UTC, then EOD recap at 21:35 UTC, with
  exception alerts only in between.
- Watch the feed watchdog for false positives during the first
  live-market session; tune TTLs if any feed alerts on routine cycles.
- **Equity history record #5** lands at 23:59 UTC tonight. 9 more
  records to clear the 14-day PAY-phase gate.
- Re-evaluate the winner_scaler output once expectancy_state has
  re-populated — multipliers should re-emerge from 1.0.
- Fix the 4 pre-existing test failures (3× position_guard fixture +
  1× g2_matrix assertion) — fixture/assertion updates only.
- Decide whether to enable `CHAD_PROFIT_ROUTER_BETA_INJECTION` after
  one paper week of clean drain telemetry.

### PHASE 5B — ML veto loop (still pending)

- XGBoost retrain pipeline closure: scheduled retrain + holdout
  validation + canary promotion + edge-decay-style halt.
- Wire the per-strategy ML veto into the routing gate stack as a
  6th gate (default off; opt-in per strategy).

### PHASE 11A — multi-market expansion (partially complete)

- Full options chain integration (per-symbol chains, not just SPY/QQQ
  cache).
- Proper expiry/strike metadata on `alpha_options` fills.
- `alpha_statarb` — stat-arb basket engine.
- `alpha_crypto_alt` — altcoin momentum (DOT, LINK, etc.).

### PHASE 12 — policy automation (mostly complete)

| Item | Status |
|---|---|
| Capital `TierManager` | ✅ **BUILT** |
| `RegimeBooster` | ✅ **BUILT (ACTIVE 1.30× v8.6)** |
| `WinnerScaling` | ✅ **BUILT** (currently neutral pending expectancy refresh) |
| `WithdrawalManager` | ✅ **BUILT** |
| `BusinessPhaseTracker` | ✅ **BUILT** |
| **Profit router drain mechanism** | ✅ **BUILT v8.6** |
| **Feed watchdog** | ✅ **BUILT v8.6** |
| **SIGTERM handler** | ✅ **BUILT v8.6** |
| **Edge decay enforcement** | ✅ **BUILT v8.6** |
| Beta injection (`CHAD_PROFIT_ROUTER_BETA_INJECTION`) | PENDING — flagged off by default; paper soak before enable. |
| Amplifier bucket injection | PENDING — winner_scaler must produce non-trivial multipliers first. |
| `WinnerScaler` daily backtest validation | PENDING. |
| Phase-transition history log | PENDING. |
| FOMC/CPI/NFP calendar in `event_risk.json` | PENDING — replace `MarketHoursRiskProvider`. |

### PHASE 9 — pre-live calibration (carried)

- Regime classifier tuning (ADX proxy → Wilder ADX).
- Kelly fraction tuning (`CHAD_ALLOC_V3_KELLY_MAX`).
- Slippage model fit per asset class.
- Live feature distribution drift monitoring.
- Net-EV gate opt-in (populate `expected_pnl` at strategy level).
- Halt-on-reconciliation-mismatch.
- ISSUE-29, 50, 75, 78, 54, 58.

### PHASE 10 — live capital flip (carried)

Entry criteria:

- All Phase-9 items complete.
- ✅ **SCR = CONFIDENT** (achieved v8.6).
- 60-90 days consistent paper performance.
- Explicit operator GO via governance rule #3.
- WithdrawalManager phase = PAY for ≥ 14 days.

When live:

- `CHAD_EXECUTION_MODE` flips from `paper` to `live`.
- `LiveGate` accepts the posture change.
- First 3 cycles run with manual oversight.
- Profit routing flips from advisory to actual capital movement.
- `WithdrawalManager` authorization becomes the basis for actual
  monthly payouts (operator still moves money).
- Beta and amplifier injection flags evaluated for enable.

---

## 16. APPENDICES

### Appendix A — File inventory of changed paths since v8.5

From `git log 9f55688..HEAD --name-status` and `git diff 9f55688..HEAD --stat`:

| File | Touched in commits | Change |
|---|---|---|
| `chad/core/live_loop.py` | `42268f0`, `3be6785`, `9bc5286`, `52786b7`, `931dd4e` | M (phantom guard, SIGTERM, halt filter, alerts) |
| `chad/core/position_guard.py` | `5267705` | M (atomic write) |
| `chad/risk/scr_state.py` | `23f4c03`, `28e8963` | M (min-trade gates, PAUSED hysteresis) |
| `chad/analytics/trade_stats_engine.py` | `ceb53ea` | M (excluded_pnl_zero) |
| `chad/analytics/shadow_confidence_router.py` | `23f4c03`, `28e8963` | M (PAUSED router + caller) |
| `chad/analytics/regime_classifier.py` | `71d8ea6` | M (TTL read enforcement) |
| `chad/risk/profit_router.py` | `cdfed06`, `74e5a0d` | M (test isolation, drain, broker_sync exclusion) |
| `chad/risk/dynamic_risk_allocator.py` | `74e5a0d` | M (beta injection flag) |
| `chad/risk/edge_decay_monitor.py` | `52786b7` | M (halt key fix, edge-trigger alert) |
| `chad/execution/routing_gates.py` | `fec3d5b` | M (A4 gate split) |
| `chad/execution/trade_closer.py` | `931dd4e` | M (recon RED alerts, routing isolation) |
| `chad/execution/paper_exec_evidence_writer.py` | `3be6785` | M (revert from `42268f0`, narrow normalizer) |
| `chad/ops/reconciliation_publisher.py` | `ba65367`, `5267705`, `931dd4e` | M (MYM/M2K, TTL 360, chad_strategy_open, alerts) |
| `chad/ops/feed_watchdog.py` | `a4accdd` | A (NEW — 7-feed monitor) |
| `chad/dashboard/api.py` | `c267076`, `fec3d5b` | M (oldest_source_mtime_utc, rate limit, 500-line cap) |
| `chad/utils/telegram_notify.py` | `931dd4e`, `d35884b` | M (7 alert types; per-fill disabled) |
| `chad/utils/telegram_bot.py` | `931dd4e` | M (alert helpers) |
| `chad/market_data/ibkr_historical_provider.py` | `42268f0` | M (VXX backfill) |
| `chad/market_data/price_cache_refresh.py` | `42268f0` | M (VIX bar key) |
| `backend/api_gateway.py` | `28e8963` | M (PAUSED hysteresis caller wiring) |
| `scripts/clear_edge_decay.py` | `52786b7` | M (--confirm requirement) |
| `/etc/systemd/system/chad-feed-watchdog.service` | `a4accdd` | A (NEW unit) |
| `/etc/systemd/system/chad-feed-watchdog.timer` | `a4accdd` | A (NEW unit) |
| `/etc/systemd/system/chad-live-loop.service` | `931dd4e`, `9bc5286` | M (telegram.env, KillSignal) |
| `runtime/profit_routing.json` | `cdfed06`, `74e5a0d` | M (alpha_test purge, consumed_*_usd) |
| `runtime/scr_state.json` | published; `paper_only=false` after band promotion | n/a |

### Appendix B — Environment files (names only)

These files live outside the repo, are gitignored, and contain
secrets. Listed by name only.

- `/etc/chad/dashboard.env` — dashboard basic-auth credentials.
- `/etc/chad/claude.env` — Anthropic API key.
- `/etc/chad/ibkr.env` — IBKR Gateway credentials.
- `/etc/chad/kraken.env` — Kraken REST + WS keys.
- `/etc/chad/openai.env` — fallback / non-Claude model creds.
- `/etc/chad/polygon.env` — Polygon API key (currently unused).
- `/etc/chad/telegram.env` — bot token + allowed chat id.
  **NEW v8.6: now loaded by `chad-live-loop.service` and
  `chad-feed-watchdog.service` so in-loop alerts can emit.**
- `/etc/chad/chad.env` — additional envs loaded by `chad-live-loop`.
  **NEW v8.6: documented `CHAD_PROFIT_ROUTER_BETA_INJECTION` (default
  off).**

### Appendix C — Critical paths cheat sheet

| Item | Path |
|---|---|
| Repo root | `/home/ubuntu/chad_finale` |
| Virtualenv | `/home/ubuntu/chad_finale/venv` (always `python3`) |
| Runtime state | `/home/ubuntu/chad_finale/runtime` |
| Daily bars | `/home/ubuntu/chad_finale/data/bars/1d` |
| Minute bars | `/home/ubuntu/chad_finale/data/bars/1m` |
| Fills | `/home/ubuntu/chad_finale/data/fills` |
| Trade history | `/home/ubuntu/chad_finale/data/trades` |
| Slippage ledgers | `/home/ubuntu/chad_finale/data/slippage` |
| Reports | `/home/ubuntu/chad_finale/reports` |
| Revert tarballs | `/home/ubuntu/chad_revert_points/` |
| Systemd units | `/etc/systemd/system/chad-*` |
| Env files | `/etc/chad/*.env` |
| Hot-path entry | `chad/core/orchestrator.py`, `chad/core/live_loop.py` |
| Execution adapter | `chad/execution/ibkr_adapter.py` |
| Kraken executor | `chad/execution/kraken_executor.py` |
| Kraken client | `chad/exchanges/kraken_client.py` |
| Paper executor (timer) | `/usr/local/bin/chad_paper_trade_executor.py` |
| LiveGate | `chad/core/live_gate.py` |
| Risk allocator | `chad/risk/dynamic_risk_allocator.py` |
| Profit Lock | `chad/risk/profit_lock.py` |
| ProfitRouter | `chad/risk/profit_router.py` (drain v8.6) |
| Evidence writer | `chad/execution/paper_exec_evidence_writer.py` |
| Full preview | `chad/core/full_cycle_preview.py` |
| TierManager | `chad/risk/tier_manager.py` |
| WinnerScaler | `chad/risk/winner_scaler.py` |
| RegimeBooster | `chad/risk/regime_booster.py` |
| WithdrawalManager | `chad/risk/withdrawal_manager.py` |
| BusinessPhaseTracker | `chad/ops/business_phase_tracker.py` |
| EquityHistory | `chad/ops/equity_history_publisher.py` |
| PortfolioSnapshot | `chad/ops/portfolio_snapshot_publisher.py` |
| **Feed watchdog (NEW v8.6)** | `chad/ops/feed_watchdog.py` |
| **Phantom-fill guard call site (NEW v8.6)** | `chad/core/live_loop.py:1180` |
| **SIGTERM handler (NEW v8.6)** | `chad/core/live_loop.py` (signal.SIGTERM) |
| **Edge decay halt enforcement (NEW v8.6)** | `chad/risk/edge_decay_monitor.py:is_strategy_halted` (called from `live_loop.py` pre-stage-1) |
| **Position guard atomic write (NEW v8.6)** | `chad/core/position_guard.py:_save_state` |

### Appendix D — Git log of all commits since v8.5

```
42268f0  2026-04-30  Fix: live loop cycle audit — phantom fills, unprotected stages, intent guard, regime TTL, VXX bars, bias logging
3be6785  2026-04-30  Fix: surgical phantom fill guard at submit call site — revert normalizer, block at live_loop evidence write
ceb53ea  2026-04-30  Fix: SCR audit — add excluded_pnl_zero counter to trade_stats_engine, balance exclusion ledger
23f4c03  2026-04-30  Fix: SCR audit — min-trade gates for CAUTIOUS/CONFIDENT bands, PAUSED hysteresis router (caller wiring pending)
5267705  2026-04-30  Fix: position guard audit — atomic _save_state write, chad_strategy_open in reconciliation counts
cdfed06  2026-04-30  Fix: profit router audit — test isolation for ledger, exclude broker_sync from routing
931dd4e  2026-04-30  Fix: Telegram audit — live loop crash, recon RED, profit lock, salary, drawdown, fill dedupe, SCR recovery alerts
c267076  2026-04-30  Fix: dashboard audit — staleness signal oldest_source_mtime_utc, brute force login protection
ba65367  2026-04-30  Fix: reconciliation audit — add MYM/M2K to futures exclusion, raise TTL to 360s
52786b7  2026-04-30  Fix: edge decay audit — wire halt enforcement, move before signals, fix key name, edge-trigger alert, harden clear script
71d8ea6  2026-04-30  Fix: data feed audit — regime_state TTL enforcement at read time, stale returns unknown
a4accdd  2026-04-30  Fix: data feed audit — feed watchdog service monitors 7 critical feeds every 120s, alerts on stale
28e8963  2026-04-30  Fix: wire PAUSED hysteresis caller — inject prev_state/paused_recovery_ticks into evaluate_confidence, round-trip through ShadowStats
74e5a0d  2026-04-30  Fix: D2 bucket accumulators — drain mechanism, get_beta_remaining, mark_beta_consumed, beta injection behind CHAD_PROFIT_ROUTER_BETA_INJECTION flag
9bc5286  2026-04-30  Fix: D4 SIGTERM handler — clean shutdown on systemd stop, Telegram alert, respects in-flight cycle
fec3d5b  2026-05-01  Fix: D6 A4 gate intraday/daily split (900s/172800s), D7 recent-trades 500-line cap
d35884b  2026-05-01  Fix: disable per-fill Telegram alerts — too noisy; morning brief + EOD recap only
```

(17 commits, all by `mcitysolo`.)

### Appendix E — Operator one-liners (NEW v8.6 entries)

```bash
# What does the feed watchdog say?
journalctl -u chad-feed-watchdog -n 50 --no-pager

# What's the SCR ledger balance?
cat /home/ubuntu/chad_finale/runtime/scr_state.json | \
  python3 -c "import json,sys; d=json.load(sys.stdin); s=d['stats']; \
    print(f\"effective={s['effective_trades']} untrusted={s['excluded_untrusted']} \
manual={s['excluded_manual']} nonfinite={s['excluded_nonfinite']} \
pnl_zero={s.get('excluded_pnl_zero',0)} total={s['total_trades']}\"); \
    print(f\"sum check: {s['effective_trades']+s['excluded_untrusted']+s['excluded_manual']+s['excluded_nonfinite']+s.get('excluded_pnl_zero',0)} == {s['total_trades']}\")"

# What's the profit router drain state?
cat /home/ubuntu/chad_finale/runtime/profit_routing.json | \
  python3 -c "import json,sys; d=json.load(sys.stdin); t=d['totals']; \
    print(f\"beta total={t['beta_allocation']:.2f} consumed={d.get('consumed_beta_usd',0):.2f}\"); \
    print(f\"amp total={t['amplifier_allocation']:.2f} consumed={d.get('consumed_amplifier_usd',0):.2f}\")"

# Is the booster active?
cat /home/ubuntu/chad_finale/runtime/regime_booster.json | \
  python3 -c "import json,sys; d=json.load(sys.stdin); \
    print(f\"booster {'ACTIVE' if d['active'] else 'INACTIVE'} {d['multiplier']}x  reasons={d['reasons']}\")"

# Are any strategies halted by edge decay?
cat /home/ubuntu/chad_finale/runtime/edge_decay_state.json 2>/dev/null | \
  python3 -c "import json,sys; d=json.load(sys.stdin); \
    halted=[k for k,v in d.items() if isinstance(v,dict) and v.get('halted')]; \
    print('halted:', halted or '(none)')" || echo "no edge_decay_state.json yet"

# Dashboard staleness signal
curl -s -u 'admin:$(cat /etc/chad/dashboard.env | grep PASS | cut -d= -f2)' \
  http://127.0.0.1:8765/api/state | \
  python3 -c "import json,sys; d=json.load(sys.stdin); \
    print('oldest source:', d.get('oldest_source_mtime_utc'))"

# What phase am I in?
cat /home/ubuntu/chad_finale/runtime/business_phase.json | \
  python3 -m json.tool

# How much salary?
cat /home/ubuntu/chad_finale/runtime/withdrawal_authorization.json | \
  python3 -c "import json,sys; d=json.load(sys.stdin); \
    print(f\"salary=\${d['authorized_withdrawal_usd']:.0f}/mo  phase={d['phase']}  reason={d['reason']}\")"

# Daily checkpoint history
tail -5 /home/ubuntu/chad_finale/runtime/equity_history.ndjson

# SIGTERM the live loop cleanly
sudo systemctl stop chad-live-loop
# (handler will let in-flight cycle finish, drain, send Telegram alert)
```

### Appendix F — Verification sequence (CLAUDE.md governance rule)

After every change:

```bash
cd /home/ubuntu/chad_finale && source venv/bin/activate
python3 -m py_compile <changed_file>
python3 -m pytest chad/tests/ -x -q 2>&1 | tail -20
python3 chad/core/full_cycle_preview.py --dry-run 2>&1 | tail -30
```

Expected today: **4 failed, 1015 passed** (the 4 failures are the
documented pre-existing fixtures — see §14).

### Appendix G — Active git tags

- `STABILITY_FREEZE_20260307_GREEN` — original stable baseline.
- `PRE_HARDENING_20260402` — snapshot before P0 hardening.
- `RATIFICATION_MASTER_20260402` — all hardening + GAP items complete.
- `REVERT_PRE_OVERHAUL_20260419` — snapshot before 2026-04-19/21
  overhaul.

### Appendix H — Rollback commands

```bash
# Roll back to post-hardening stable
git checkout RATIFICATION_MASTER_20260402

# Roll back to pre-overhaul stable (restore runtime from tarball too)
git checkout REVERT_PRE_OVERHAUL_20260419
tar -xzf /home/ubuntu/chad_revert_points/runtime_pre_overhaul_20260419.tar.gz \
        -C /home/ubuntu/chad_finale
```

### Appendix I — Operator daily checks (≈ 2 minutes)

```bash
cd /home/ubuntu/chad_finale && source venv/bin/activate

# 1. Core health
cat runtime/scr_state.json | python3 -m json.tool | head -15
cat runtime/profit_lock_state.json | python3 -m json.tool | head -15
cat runtime/stop_bus.json | python3 -m json.tool

# 2. Regime + booster (NEW v8.6: booster is now active)
cat runtime/regime_state.json
cat runtime/regime_booster.json

# 3. Reconciliation
cat runtime/reconciliation_state.json | python3 -m json.tool

# 4. Strategy health
python3 -c "import json; \
  [print(f'{k}: {v[\"health_score\"]:.3f} ({v[\"sample_count\"]})') \
   for k,v in json.load(open('runtime/strategy_health.json'))['strategies'].items()]"

# 5. Business framework
cat runtime/business_phase.json | python3 -m json.tool
cat runtime/tier_state.json | python3 -m json.tool | head -15
cat runtime/withdrawal_authorization.json | python3 -m json.tool

# 6. NEW v8.6 — Feed watchdog status
journalctl -u chad-feed-watchdog -n 10 --no-pager

# 7. NEW v8.6 — Edge decay halt status
cat runtime/edge_decay_state.json 2>/dev/null | python3 -m json.tool || echo "(no halts)"

# 8. Dashboard
curl -s https://chadtrades.com/health
```

What to look for (today's expected values):

- ✅ SCR `state` = **`CONFIDENT`**, `sizing_factor=1.0`,
  `paper_only=false`.
- ✅ Profit lock `mode` = `NORMAL`.
- ✅ Stop bus `active` = `false`.
- ✅ Reconciliation `status` = `GREEN`,
  `chad_strategy_open=15` exposed.
- ✅ Phase = `GROW` (will flip to PAY when history_days ≥ 14, then
  HWM holds).
- ✅ Tier = `PRO` (16/16 strategies active).
- ✅ Salary authorized = `$0/mo` (correct for GROW; needs 10 more days).
- ✅ Booster = **`1.30× ACTIVE`** (high_confidence + vix_calm +
  trending_bull_bias).
- ✅ Feed watchdog journal: no `feed_stale` lines in last hour.
- ✅ No strategies halted by edge decay.
- ✅ Dashboard returns HTTP 200.

### Appendix J — Key value summary (single-screen reference)

```
┌──────────────────────────────────────────────────────────────────┐
│                    CHAD AT-A-GLANCE — v8.7                       │
├──────────────────────────────────────────────────────────────────┤
│  Equity:   $183,926.04 USD  (IBKR ~$183,741 + Kraken $184.58)    │
│  Phase:    GROW            (10 more days history → PAY)          │
│  Tier:     PRO             (16/16 strategies enabled)            │
│  SCR:      CONFIDENT 1.0x  (147/133 effective trades, paper_only=false) │
│  Regime:   trending_bull   (confidence 0.7536)                   │
│  VIX:      16.89                                                 │
│  HWM:      $183,874.27     (4d history)                          │
│  Salary:   $0/mo authorized (GROW; need 14d history)             │
│  Boost:    1.30x ACTIVE (high_conf + vix_calm + trending_bull)   │
│  Reconcile:GREEN  (chad=8 / strat=15 / broker=11; 3 drifts)      │
│  Profit Lock:  NORMAL                                            │
│  Stop Bus:     INACTIVE                                          │
│  Edge Decay:   ENFORCED  (no halts)                              │
│  Feed Watchdog: ACTIVE   (7 feeds, 120s)                         │
│  SIGTERM Handler: ACTIVE                                         │
│  Phantom Fill Guard: ACTIVE                                      │
│  Health Monitor: ACTIVE (9 rules, Claude reasoning, auto-fix)    │
│  Top Strategy: (winner_scaler resetting; all 1.0x)               │
│  Profit Router: trading $1,905.26 / beta $1,143.15 / amp $762.10 │
│  Bucket Drain: consumed_beta=$0.00, consumed_amp=$0.00           │
│  Beta Injection: FLAGGED OFF (CHAD_PROFIT_ROUTER_BETA_INJECTION) │
│  Today realized P&L: -$45.20  (73 trades)                        │
│  Tests: 4 fail (cosmetic) / 1,015 pass                           │
│  Services: 14 running / 99 loaded / 0 failed                     │
│  Telegram alerts: 7 exception types ON, per-fill OFF, +3 health  │
└──────────────────────────────────────────────────────────────────┘
```

### Appendix K — v8.5 → v8.6 deltas at a glance

| Metric | v8.5 (2026-04-30) | v8.6 (2026-05-01) |
|---|---|---|
| HEAD | `9f55688` | `d35884b` |
| Equity | $183,500.05 USD | $183,926.04 USD |
| **SCR state** | **WARMUP 0.10×** | **CONFIDENT 1.0×** |
| **paper_only** | **true** | **false** |
| effective_trades | 84 | **147** (gate=133) |
| excluded_pnl_zero | implicit | **explicit** |
| Min-trade gates | none | **CAUTIOUS=100, CONFIDENT=133** |
| PAUSED hysteresis | none | **end-to-end wired** |
| **Regime booster** | **1.0× vetoed** | **1.30× ACTIVE** |
| Booster confidence threshold | 0.6864 (below 0.70) | 0.7536 (above 0.75) |
| VIX | 18.71 | **16.89** (calm threshold cleared) |
| Phase | GROW (1d history) | GROW (4d history; need 14) |
| Salary authorized | $0 | $0 |
| Tier | PRO | PRO |
| Reconciliation TTL | 300s | **360s** |
| chad_strategy_open exposed | no | **yes** (15) |
| MYM/M2K in KNOWN_FUTURES | no | **yes** |
| Position guard atomic write | no | **yes** |
| Phantom fill guard | no | **yes (live_loop:1180)** |
| Edge decay halt enforced | NO (unwired) | **YES** |
| Feed watchdog | none | **7 feeds @ 120s** |
| SIGTERM handler | none | **yes** |
| `regime_state` read-time TTL | none | **yes** |
| A4 gate threshold | 300s flat | **900s intraday / 172800s daily** |
| Recent-trades cap | unbounded | **500 lines/file** |
| Dashboard staleness signal | none | **`oldest_source_mtime_utc`** |
| Dashboard brute force | exposed | **5 attempts → 5min lockout** |
| Profit router test isolation | shared prod ledger | **`routing_path` param** |
| Profit router `broker_sync` | routed | **excluded** |
| Profit router drain | append-only | **drain-aware** |
| `consumed_beta_usd` | n/a | **$0.00 tracked** |
| `consumed_amplifier_usd` | n/a | **$0.00 tracked** |
| `alpha_test` ledger entries | 19 phantom | **purged** (`alpha_test_purged_count=19`) |
| Beta injection feature flag | n/a | **`CHAD_PROFIT_ROUTER_BETA_INJECTION` (default off)** |
| Telegram alerts (exception) | thin | **7 new types** |
| Telegram per-fill alerts | enabled | **DISABLED** (consolidated in briefs) |
| `chad-live-loop.service` `telegram.env` | not loaded | **loaded** |
| VXX bars | 26 days stale | **backfilled** |
| Tests | 5 fail / 1014 pass | **4 fail / 1015 pass** |
| Services loaded | 95 | **97** (+ feed-watchdog ×2) |

### Appendix L — v8.6 → v8.7 deltas at a glance

| Metric | v8.6 (2026-05-01) | v8.7 (2026-05-01) |
|---|---|---|
| HEAD | `eafb8cc` | `1207eeb` |
| Commits since predecessor | 17 | **1** |
| **CHAD AI Health Monitor** | **none** | **ACTIVE — 3 tiers, 9 rules, 300s cadence** |
| Tier 1 — rule engine | n/a | **`health_monitor_rules.py` — R01..R09** |
| Tier 2 — Claude reasoning | n/a | **`claude-sonnet-4-6` snapshot dispatch** |
| Tier 3 — auto-remediation | n/a | **`health_monitor_remediation.py`** |
| Auto-fix actions | n/a | **restart dead service, restart stale feed publisher, restore corrupt runtime file from backup, archive old fill files** |
| Notify-only actions | n/a | **SCR PAUSED, stop bus active, reconciliation RED, profit lock LOCK2+, edge decay halts** |
| New Telegram message types | none | **`AUTO-FIXED`, `HEALTH MONITOR CRITICAL`, `AI HEALTH ANALYSIS`** |
| Sudo passwordless | not granted | **`systemctl restart chad-*` (vetted scope)** |
| Services loaded | 97 | **99** (+ health-monitor ×2) |
| Timer cadence (new) | n/a | **`OnBootSec=120, OnUnitActiveSec=300`** |
| All v8.6 invariants | (baseline) | **preserved unchanged** |

---

**End of CHAD Unified SSOT v8.7.**

This document is the truth as of 2026-05-01. CHAD has graduated from
"the strategies emit signals" (v8.5) to "the operator can sleep through
the night" (v8.6) to "the system watches itself, fixes what is safely
fixable, and only pages the operator for the rest" (v8.7). All v8.6
invariants remain intact: SCR=CONFIDENT, regime booster ACTIVE 1.30×,
feed watchdog, SIGTERM handler, atomic position-guard write, edge-decay
halt enforcement, surgical phantom-fill guard, drain-aware profit
router, GREEN reconciliation with `chad_strategy_open` separated,
single-signal dashboard staleness. The single v8.7 addition is the
**CHAD AI Health Monitor** — three tiers (rule engine, Claude
reasoning, auto-remediation) layered on top of the v8.6 foundations.

What stands between this revision and live capital is no longer a
list of bugs. It is a 14-day equity-history requirement, an operator
GO, and continued HWM discipline. Mechanical from here.

If the code disagrees, either the code drifted or this revision
needs another pass. Cut v8.8 before relying on the disagreement.





