# BOX-050 (Official Matrix) — GAP-029 price-cache writer policy

- **Box number (Official Matrix):** 050
- **Box title (Official Matrix):** NEW-GAP-054 GAP-029 proof hardened — unused second price-cache writer is deleted/deprecated or covered by a test
- **Stage:** Stage 3 — Engineering, tests, SSOT, and hidden-gap closure
- **Cut timestamp (UTC):** 2026-05-20T21:07:56Z
- **HEAD at cut:** `bbe7525`
- **Branch:** `main`

---

## 0. Scope and safety statement

- **CHAD remains PAPER.** `CHAD_EXECUTION_MODE=paper`.
- **live trading not authorized.** This policy does not flip
  `ready_for_live`, does not restart any service, does not modify
  `runtime/price_cache.json`, and does not authorize live trading.
- **No code patch.** The deprecation is already in place at the
  systemd-unit layer; no in-place edit required.

---

## 1. The two price-cache writers (factual inventory)

### 1.1 CANONICAL writer — `chad-ibkr-price-refresh.service` (active)

```
Unit file       : /etc/systemd/system/chad-ibkr-price-refresh.service
Timer           : /etc/systemd/system/chad-ibkr-price-refresh.timer
Timer cadence   : OnBootSec=45 / OnUnitActiveSec=60 (60-second tick)
Timer state     : active + enabled + WantedBy=timers.target
Last trigger    : 2026-05-20 21:21:03 UTC (current; tick within last minute)
ExecStart       : /home/ubuntu/chad_finale/venv/bin/python3 -m chad.market_data.price_cache_refresh --provider ibkr --runtime-dir /home/ubuntu/chad_finale/runtime --ttl-seconds 300
Service state   : activating start (oneshot mid-flight at audit)
Provenance      : IBKR snapshots (equities + futures) via IBKRPriceProvider.get_batch_snapshots
Target          : runtime/price_cache.json (atomic_write_json)
```

### 1.2 SECOND (deprecated) writer — `chad-price-cache-refresh.service` (inactive, disabled)

```
Unit file       : /etc/systemd/system/chad-price-cache-refresh.service
Timer           : /etc/systemd/system/chad-price-cache-refresh.timer
Timer cadence   : OnBootSec=30 / OnUnitActiveSec=120
Timer state     : inactive + DISABLED + UnitFileState=disabled (NOT in timers.target.wants/)
Service state   : inactive (UnitFileState=static)
Last trigger    : (empty — never triggered in current systemd session)
ExecStart       : /home/ubuntu/chad_finale/venv/bin/python3 -m chad.market_data.price_cache_refresh --feed-dir "/home/ubuntu/chad_finale/data/feeds" --runtime-dir "/home/ubuntu/chad_finale/runtime" --tail-lines 50000 --ttl-seconds 300
Provenance      : Polygon NDJSON feed (default fallback path in price_cache_refresh.py when `--provider` flag absent)
Target          : runtime/price_cache.json (same path as canonical)
```

**Status: DEPRECATED at the systemd-unit layer.** The unit file
remains on disk under `/etc/systemd/system/` (per CLAUDE.md
governance rule #6, systemd unit files are not modified without
explicit instruction), but the timer is not enabled and the service
will not run on its own. The Polygon code path in
`chad/market_data/price_cache_refresh.py` (lines 365–376) is the
default-fallback branch and is only reachable via manual operator
invocation or by re-enabling the disabled timer.

### 1.3 Tertiary code surface — `IBKRPriceProvider.write_price_cache`

```
Path  : chad/market_data/ibkr_price_provider.py:337 (method on the IBKRPriceProvider class)
Role  : alternative writer API; not directly invoked by any systemd service.
Usage : the canonical chad-ibkr-price-refresh service calls
        IBKRPriceProvider.get_batch_snapshots(...) plus a local
        atomic_write_json(out_path, payload) in price_cache_refresh.py
        rather than calling this method.
```

**Not a third runtime writer** — it is an internal API surface kept
as a utility; covered by `chad/tests/test_polygon_residue_guards.py`
indirectly (which asserts IBKR is the active provider). No additional
deprecation needed.

---

## 2. Consumers (read-only) — they all read `runtime/price_cache.json`

| Consumer code path                                      | Role                                                                       |
| ------------------------------------------------------- | -------------------------------------------------------------------------- |
| `chad/core/live_loop.py:215`                             | live-loop reads `pc_path = runtime/price_cache.json`                       |
| `chad/core/orchestrator.py`                              | orchestrator reads price cache                                              |
| `chad/core/position_reconciler.py:55` (`_PRICE_CACHE_PATH`) | position reconciler                                                      |
| `chad/execution/ibkr_client_ids.py`                       | execution path reads                                                       |
| `chad/execution/net_exposure_gate.py`                     | net-exposure gate                                                           |
| `chad/execution/paper_exec_evidence_writer.py`            | paper exec evidence writer                                                  |
| `chad/execution/tier_instrument_gate.py`                  | tier instrument gate                                                       |
| `chad/execution/trade_closer.py`                          | trade closer                                                                |
| `chad/dashboard/api.py:547,666,1006,1007,1199`            | dashboard / surface API                                                     |
| `chad/intel/advisory_engine.py`                           | intel layer                                                                 |
| `chad/intel/strategy_intelligence.py`                     | intel layer                                                                 |
| `chad/intel/short_interest_provider.py`                   | intel layer                                                                 |
| `chad/intel/synthetic_analyst.py`                         | intel layer                                                                 |
| `chad/market_data/ibkr_price_provider.py:152-155`         | NaN-fallback read (loads last-known prices from cache)                       |
| `chad/market_data/options_greeks_publisher.py:369-372`    | options greeks publisher                                                    |
| `chad/portfolio/options_position_monitor.py:201-203`      | options position monitor                                                    |
| `chad/portfolio/coinbase_portfolio_collector.py`          | coinbase collector                                                          |
| `chad/risk/portfolio_var.py:146`                          | portfolio VaR                                                              |
| `chad/strategies/omega_vol.py`                            | omega_vol strategy                                                          |
| `chad/ops/feed_watchdog.py:18`                            | feed watchdog (ttl=180)                                                     |
| `chad/ops/health_monitor_remediation.py:26`               | maps `price_cache.json → chad-ibkr-price-refresh.timer` for auto-remediation |
| `chad/ops/health_monitor_rules.py:31,133`                 | health-rule maps `price_cache.json → chad-ibkr-price-refresh.timer` + freshness 180s |
| `chad/ops/daily_chad_report.py:558,566,1362,1372`         | daily ops report                                                            |

**Consumers all have a single authority source (`runtime/price_cache.json`)**;
they don't care which writer produced it as long as the schema +
freshness contract is met. The schema is: `{"prices": {symbol: price},
"ts_utc": "...", "ttl_seconds": 300}`.

---

## 3. Canonical-writer decision

| Concern                                                                   | Canonical                                                                                          |
| ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **Active production writer of `runtime/price_cache.json`**                 | **`chad-ibkr-price-refresh.service`** (timer-driven 60s, `--provider ibkr` IBKR snapshots)         |
| **Health-monitor remediation target** (per `chad/ops/health_monitor_*.py`) | `chad-ibkr-price-refresh.timer` (same)                                                              |
| **Operator override** (manual one-shot)                                   | `python -m chad.market_data.price_cache_refresh --provider ibkr` (operator-only)                  |
| **Polygon fallback path**                                                 | **DEPRECATED** — `chad-price-cache-refresh.timer` disabled at systemd layer; Polygon subscription cancelled per GAP-012/GAP-013 (`test_polygon_residue_guards.py`). |
| **IBKRPriceProvider.write_price_cache method**                            | internal utility API; not directly wired into any service; orphan but not harmful (does not run unless explicitly invoked from code that does not exist in current production paths) |

**Canonical writer:** `chad-ibkr-price-refresh.service` invoking
`chad.market_data.price_cache_refresh --provider ibkr`.

---

## 4. Second-writer classification

**Status:** `DEPRECATED`

| Layer                              | Deprecation evidence                                                                                                  |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| systemd timer                      | `chad-price-cache-refresh.timer` — `ActiveState=inactive`, `UnitFileState=disabled`, NOT in `timers.target.wants/`.    |
| systemd service                     | `chad-price-cache-refresh.service` — `ActiveState=inactive`, `UnitFileState=static`.                                  |
| `LastTriggerUSec`                   | empty (timer never triggered in current systemd session).                                                              |
| Polygon provider                   | subscription cancelled per GAP-012/GAP-013 (`test_polygon_residue_guards.py` covers — 5/5 PASS).                       |
| `chad/market_data/polygon_daily_bars_backfill.py` | already removed (per `test_polygon_residue_guards.py` module docstring).                              |
| `backend/polygon_stocks_stream.py`  | kept as gated/optional tool only; raises `SystemExit` if invoked without `CHAD_BAR_PROVIDER=polygon`.                  |
| IBKR canonical posture              | `chad/market_data/ibkr_price_provider.py` is the active provider; docstrings updated per GAP-012/13.                  |
| Unit-file presence                  | files remain on disk under `/etc/systemd/system/` (per CLAUDE.md rule #6 — no systemd modification without explicit instruction). Inert without an operator-issued `systemctl enable` + `daemon-reload`. |

**Net effect:** the second writer cannot execute in the current
production configuration. Re-activation requires an operator-issued
`systemctl enable chad-price-cache-refresh.timer` + `systemctl
daemon-reload` + `systemctl start` — all of which are explicit
operator actions outside Box 050's scope.

---

## 5. Closure path

**Selected:** `PASS_SECOND_WRITER_ALREADY_DEPRECATED` — the
unused second writer is already deprecated at the systemd layer; no
code change required.

**Why this is sufficient under the Box-050 acceptance criterion ("deleted/deprecated or covered by a test"):**

1. **Deprecated at runtime** — `chad-price-cache-refresh.timer` is
   `disabled` + `inactive`; cannot fire without operator re-enable.
2. **Covered by test (the deprecation principle is locked):**
   `chad/tests/test_polygon_residue_guards.py` (GAP-012/GAP-013, 5/5
   pass) already enforces: (a) Polygon backfill module removed, (b)
   Polygon stocks-stream raises `SystemExit` without
   `CHAD_BAR_PROVIDER=polygon`, (c) IBKR is the active provider in
   the running daemon, (d) the active provider does not require
   `polygon-api-client` to be importable.
3. **Health-monitor remediation** in `chad/ops/health_monitor_rules.py:31`
   and `chad/ops/health_monitor_remediation.py:26` both map
   `price_cache.json → chad-ibkr-price-refresh.timer` — so if the
   cache goes stale, auto-remediation re-triggers the canonical
   timer, not the deprecated one.
4. **No DELETE recommended** — per CLAUDE.md governance rule #6, no
   systemd unit files are modified without explicit operator
   instruction. The disabled-state IS the deprecation.

**What this box does NOT do:**

- Does NOT `systemctl daemon-reload`.
- Does NOT delete `chad-price-cache-refresh.service` or `.timer` unit files.
- Does NOT remove the Polygon code branch from `chad/market_data/price_cache_refresh.py` (the branch is only reachable via explicit operator invocation and is harmless when not triggered).
- Does NOT modify `chad/ops/health_monitor_*.py` (which already point at the canonical timer).

---

## 6. Optional follow-ups (NOT executed by Box 050)

| Item | Notes |
| ---- | ----- |
| Operator-approved DELETE of `chad-price-cache-refresh.service` + `.timer` | Requires CLAUDE.md rule #6 explicit operator instruction. Would also require `systemctl daemon-reload`. **Not in scope** for this box. |
| Remove the Polygon branch from `chad/market_data/price_cache_refresh.py` | Code-only change; would simplify the module. Operator-decision deferred. **Not in scope.** |
| Add a static test asserting `chad-price-cache-refresh.timer` remains disabled | Would be an environment-dependent test (reads systemd). Could be added to `chad/ops/systemd_wants_lint.py` lineage. **Not in scope** — operator decision required. |

---

## 7. Patches summary

| Patch class            | Action                                                                                                                  |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Production code        | **None** — deprecation is at systemd layer; no in-place code edit.                                                       |
| Live config            | **None** — `runtime/price_cache.json` NOT mutated (mtime 2026-05-20 12:30 UTC preserved).                                |
| systemd unit files     | **None** — per CLAUDE.md rule #6.                                                                                        |
| Documentation          | **Added** `ops/pending_actions/BOX-050_GAP029_price_cache_writer_policy.md` (this file).                                  |
| Evidence               | **Added** `runtime/completion_matrix_evidence/BOX-050_OFFICIAL_GAP-029_proof_hardened.md` (paired).                       |
| Tests                  | **None added.** `chad/tests/test_polygon_residue_guards.py` already covers the canonical-provider principle (5/5 PASS).   |
| Frozen historical SSOT | **Unchanged** — forward-only.                                                                                            |
| Staged / committed     | **None.**                                                                                                                |

---

## 8. False-closure guardrails

- Does NOT enable `chad-price-cache-refresh.timer`.
- Does NOT restart `chad-ibkr-price-refresh.service`.
- Does NOT write to `runtime/price_cache.json`.
- Does NOT modify any systemd unit file.
- Does NOT change `CHAD_EXECUTION_MODE` or `ready_for_live`.
- Does NOT authorize live trading.

**live trading not authorized. CHAD remains PAPER. `ready_for_live=false`.**
