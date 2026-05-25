# PR-02 — Delta Upstream Abstain (No Valid Live/Cached Price)

**Date:** 2026-05-25
**Scope:** chad/strategies/delta.py, chad/tests/test_delta_abstains_when_no_live_price.py
**Trading posture impact:** NONE (paper). No live enablement. No service restarts.
**Authorization required to apply:** N/A — strategy hardening; behavior change is
strictly "do less" (abstain on invalid price). Cannot widen anything.

---

## Root cause

`chad/strategies/delta.py` had a price gate (`if px <= 0.0: return []`) that
abstained only when the caller-supplied `prices.get(sym)` and `ctx.ticks[sym].price`
were both non-positive. It did NOT consult `ctx.bars[sym]` (last close) or the
broker `runtime/price_cache.json`, and it did NOT cross-check the candidate
against the broker price_cache. When upstream price construction handed delta a
positive-but-stale or positive-but-placeholder value (e.g. the `$100` "no-live-price"
fingerprint), delta accepted it, and the resulting `TradeSignal` propagated to
the planner → IBKR intent builder → live_loop, where it was caught only at the
paper_exec_evidence_writer's deviation guard (P0-1 / commit `ddbc79f`). The
writer correctly tagged the row `pnl_untrusted` / `rejected` / `placeholder` and
preserved an audit trail under `extra.placeholder_*`, but the row still landed
in `data/fills/FILLS_*.ndjson` as a rejected fill. PR-02's mandate is to fix the
emit upstream so the placeholder pattern never reaches the writer.

**Observed evidence (pre-fix):**
- 64 delta IWM/SPY rejected rows in FILLS_20260521..20260525.
- Date breakdown: 05-21=4, 05-22=40, 05-23=10, 05-24=0, 05-25=12.
- Canonical fingerprint per row:
  - `tags=[paper, rejected, delta, equity, pnl_untrusted, placeholder, broker_rejected]`
  - `extra.source_strategies=["delta"]`
  - `extra.placeholder_fill_price=100.0`
  - `extra.placeholder_price_cache=283.8` (real IWM price)
  - `extra.pnl_untrusted_reason="placeholder_no_broker_confirmed_fill_price (deviation~65%; …)"`

---

## Fix approach

Added a single price-resolution helper, `_resolve_positive_price(ctx, sym, prices)`,
in `chad/strategies/delta.py`. It is called at the top of `_propose_for_symbol`
so BOTH the entry (BUY) path and the exit (SELL) path gate through it. The
helper:

1. **Resolves** a candidate price from the first of four trusted sources that
   yields a finite-positive value:
   - `prices[sym]` (caller-supplied map)
   - `ctx.ticks[sym].price` (live tick)
   - last close in `ctx.bars[sym]`
   - `runtime/price_cache.json::prices[sym]` (broker cache)

2. **Validates**: rejects `None`, non-numeric, `NaN`, `±Inf`, `0`, negative.

3. **Cross-checks** for the liquid equity/ETF allowlist
   (`SPY`/`QQQ`/`IWM`/`DIA` — mirrors the writer's `_LIQUID_PRICED_EQUITIES`):
   if the candidate deviates by `>50%` from the broker `price_cache.json`
   ground truth, treat it as a placeholder/stale value and abstain. This is
   the same threshold the paper_exec_evidence_writer's deviation guard uses,
   so PR-02 upstream and P0-1 downstream are aligned.

4. **Logs** one `INFO`-level `DELTA_ABSTAIN_NO_VALID_PRICE symbol=<sym>` line
   on abstain so the reason is discoverable in journalctl.

5. **Returns** `(price, source_label)` on success, `None` on abstain.

The helper resolves `CHAD_PRICE_CACHE_PATH` at call time so tests can monkey-patch
the path without re-importing the module.

P0-1 writer defense at `chad/execution/paper_exec_evidence_writer.py` is left
untouched: it remains the last line of defense for any bad record that escapes
this upstream gate (regression-tested in PR-02 test suite).

---

## Files changed

- `chad/strategies/delta.py` — added module-level imports (`json`, `logging`,
  `math`, `os`, `Path`), `_PRICE_CACHE_PATH`, `_LIQUID_PRICED_ETFS`,
  `_PRICE_CACHE_DEVIATION_THRESHOLD`, helpers `_is_finite_positive`,
  `_load_price_cache_entry`, `_resolve_positive_price`; replaced the
  hand-rolled `prices.get → ctx.ticks` gate inside `_propose_for_symbol`
  with a call to the new helper plus the abstain log line.
- `chad/tests/test_delta_abstains_when_no_live_price.py` — new (23 tests).

No edits to runtime/, data/, reports/, config/, ops/ services, or systemd.

---

## Tests added

23 tests in `chad/tests/test_delta_abstains_when_no_live_price.py`:

Unit tests for `_resolve_positive_price` (13):
- happy path from prices_map
- ctx.ticks fallback
- ctx.bars last-close fallback
- price_cache.json fallback
- abstains when all sources missing
- abstains on 0
- abstains on negative
- abstains on NaN
- abstains on Inf
- abstains on string / non-numeric
- abstains on $100 placeholder when cache disagrees
- accepts within 50% deviation
- skips cross-check for non-ETF symbols

End-to-end via `delta_handler` (8):
- abstains on IWM no-price
- abstains on SPY no-price
- abstains when price is 0 / negative / NaN
- abstains on $100 placeholder vs cache=283.8 (the exact PR-02 reproducer,
  with delta holding IWM, exercising both entry and exit gates)
- emits abstain log line with symbol
- still emits on valid price + valid setup (no false-positive abstain)

Regression guards (2):
- P0-1 writer defense still neutralizes placeholders
- PR-02 changes do not touch unrelated strategy modules

All 23 PR-02 tests pass.

Targeted regression on `-k "delta or placeholder or p0_1 or fill_price or upstream_exclusion"`:
**105 passed**.

Full regression (excluding pre-existing unrelated live-equity-skew flake
`test_canonical_sources_agree_within_skew_tolerance`):
**2483 passed**.

---

## Runtime / artifact evidence

**Pre-patch FILLS audit (immutable historical rows):**
- 64 delta IWM/SPY rejected rows across the last 5 daily FILLS files.
- Breakdown: 05-21=4, 05-22=40, 05-23=10, 05-24=0, 05-25=12 (count as of
  2026-05-25T18:51Z).

**In-process simulation proof (Tier 4):**
- Scenario A (no-price IWM/SPY universe): `delta_handler` produced **0 signals**;
  `DELTA_ABSTAIN_NO_VALID_PRICE symbol=IWM` logged.
- Scenario B ($100 placeholder caller map vs cache=$283.8/$720): **0 signals**.
- Scenario C (valid price, no bars): **0 signals** (correct — no false-positive
  abstain; entry path simply has no OHLC to score).
- Scenario D (zero/negative/NaN/+Inf/-Inf variants): all **0 signals**.
- No broker calls, no runtime JSON edits, no service restarts.

**Runtime observation still needed:**
The chad-live-loop service is a long-running systemd daemon that imports
`chad.strategies.delta` at startup. The daemon currently running on this host
predates this commit and will continue to use the old code path until the next
authorized restart. The patched code is correct and verified by tests + in-process
simulation; **observation lock** that confirms zero new delta/IWM/SPY rejected
rows in FILLS over a full trading-day window will be available only after
operator-authorized restart of `chad-live-loop` (out of PR-02 scope — see safety
rules in this prompt).

Historical rejected rows in `data/fills/FILLS_*.ndjson` are immutable; PR-02
does not (and must not) rewrite them. The P0-1 writer defense continues to tag
any placeholder record that escapes the upstream gate.

---

## Current status

| Requirement | Status |
|---|---|
| Delta does not emit IWM/SPY signal when no positive live/cached price | ✅ verified (tests + simulation) |
| Abstain on None / 0 / negative / NaN / Inf / missing / invalid | ✅ verified |
| Abstain on $100 placeholder when broker cache disagrees | ✅ verified |
| Valid signals preserved when price is valid | ✅ verified |
| P0-1 writer defense preserved | ✅ regression test passes |
| Trust boundary not weakened | ✅ helper only abstains; never widens |
| Unrelated strategies untouched | ✅ only delta.py modified |
| Service restart performed | ❌ NOT performed (out of scope; operator action) |
| Runtime observation lock | ⏳ pending operator restart + next trading-day window |

**PR-02 status: VERIFIED** (code + tests + simulation prove the upstream
abstain works as specified; observation lock awaits operator restart of
chad-live-loop, which is outside the safety envelope of this remediation).

---

## No-live confirmation

- Trading posture remains PAPER. `ready_for_live=false`. `paper_only=true`.
- No broker orders. No order cancellation. No live enablement.
- No service restarts. No systemd unit modifications.
- No runtime/, data/, config/, .env, or secret edits.
- No `git add .`. Only the two explicit files staged: `chad/strategies/delta.py`,
  `chad/tests/test_delta_abstains_when_no_live_price.py`, plus this Pending Action.

---

## Verification commands

```bash
source venv/bin/activate
export PYTHONPATH=/home/ubuntu/chad_finale
export CHAD_SKIP_IB_CONNECT=1

# 1. Compile-check
python3 -m py_compile chad/strategies/delta.py

# 2. PR-02 targeted tests
python3 -m pytest chad/tests/test_delta_abstains_when_no_live_price.py -v

# 3. Targeted regression
python3 -m pytest chad/tests/ -k "delta or placeholder or p0_1 or fill_price or upstream_exclusion" -v

# 4. Full regression (excluding pre-existing live-equity-skew flake)
python3 -m pytest chad/tests/ -q --deselect \
  chad/tests/test_canonical_equity_source.py::test_canonical_sources_agree_within_skew_tolerance

# 5. Inline simulation (no service restart needed)
CHAD_PRICE_CACHE_PATH=/tmp/pr02_check.json python3 - <<'PY'
import json
from pathlib import Path
from datetime import datetime, timezone

# Cache the real IWM price as broker ground truth.
Path("/tmp/pr02_check.json").write_text(json.dumps({"prices": {"IWM": 283.8}}))

from chad.strategies.delta import DEFAULT_PARAMS, delta_handler
from chad.types import PortfolioSnapshot

class _Ctx:
    now = datetime.now(timezone.utc)
    ticks = {}
    legend = None
    bars = {}
    event_risk = None
    delta_universe = ["IWM"]
    portfolio = PortfolioSnapshot(now, 50_000.0, {}, {})

# $100 placeholder vs cache=283.8 → expect 0 signals
print(delta_handler(_Ctx(), DEFAULT_PARAMS, prices={"IWM": 100.0}))
PY
```

## Definition of done

- [x] `_resolve_positive_price` helper added and wired into `_propose_for_symbol`.
- [x] All seven invalid-price classes (None / 0 / negative / NaN / +Inf / -Inf / non-numeric) cause abstain.
- [x] $100 placeholder caught upstream via 50% deviation cross-check against `runtime/price_cache.json`.
- [x] `DELTA_ABSTAIN_NO_VALID_PRICE` log line emitted on abstain.
- [x] PR-02 tests: 23/23 pass.
- [x] Targeted regression: 105/105 pass.
- [x] Full regression: 2483/2483 pass (1 pre-existing unrelated flake deselected).
- [x] P0-1 writer defense intact and regression-tested.
- [x] In-process simulation proves zero signals for every invalid-price scenario.
- [x] Pending Action document written.
- [ ] **Observation lock (operator-driven, out of PR-02 scope):** after the
      next authorized `chad-live-loop` restart, observe one full trading-day
      window with zero new delta/IWM/SPY rejected rows in `data/fills/FILLS_*.ndjson`.
      Old historical rejected rows remain immutable by design.
