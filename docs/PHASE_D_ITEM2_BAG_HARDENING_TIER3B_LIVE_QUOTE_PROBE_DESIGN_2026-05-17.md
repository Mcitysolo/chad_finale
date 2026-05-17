# Phase D Item 2 — BAG Hardening Tier 3B Live Quote Probe Design

## 1. Title
Phase D Item 2 — BAG Hardening Tier 3B Live Quote Probe Design
(authored 2026-05-17; supersedes nothing; extends the Tier 3 design doc
`docs/PHASE_D_ITEM2_BAG_HARDENING_TIER3_QUOTE_CHECK_DESIGN_2026-05-17.md`).

## 2. Status
**DESIGN ONLY — NO LIVE QUOTE REQUESTS YET.**

* No source code changes.
* No runtime mutation.
* No deploy / systemd changes.
* No order placement; no IBKR connection in this task.
* Live BAG execution remains **NOT AUTHORIZED**.
* Tier 3B implementation will land as a separate, dedicated change in a
  subsequent task, gated by operator GO.

## 3. Tier 3A baseline (offline quote-check module)

Committed in `a5c5b35 — Add offline BAG quote check module`. Surface:

* Module: `chad/options/quote_check.py` (12,579 bytes).
* Test:   `chad/tests/test_options_quote_check.py` (17,864 bytes).
* Design: `docs/PHASE_D_ITEM2_BAG_HARDENING_TIER3_QUOTE_CHECK_DESIGN_2026-05-17.md`.

Public API (re-exported via `chad/options/__init__.py`):

| Symbol                       | Kind        | Purpose |
|------------------------------|-------------|---------|
| `OptionLegQuote`             | dataclass   | Per-leg bid/ask/last/theo snapshot for one option contract. |
| `BagComboQuote`              | dataclass   | Optional direct combo bid/ask/last snapshot. |
| `SpreadQuoteCheckInput`      | dataclass   | spec + limit_price + leg quotes + combo quote + tolerances. |
| `SpreadQuoteCheckResult`     | dataclass   | ok / source / mid_debit / deviation / warnings / errors. |
| `compute_combo_mid_debit`    | function    | combo bid/ask preferred over combo.last. |
| `compute_leg_mid_debit`      | function    | long_mid − short_mid, with `theo_only_quote_check` warning. |
| `check_spread_limit_price`   | function    | End-to-end validator with combo→leg→theo→error decision tree. |

Hard safety contract confirmed by grep:

* `chad/options/quote_check.py` contains zero `ib_async` / `ib_insync` /
  `placeOrder` / `connectAsync` / `IbkrAdapter` / `chad.execution` /
  `chad.strategies` references (only docstring mentions).
* `chad/tests/test_options_quote_check.py` enforces the same invariant
  via regex assertions on module source (lines 460–472).

A future live quote probe must therefore produce **only**:

* `OptionLegQuote` instances for each leg, and/or
* a `BagComboQuote` instance for the combo,

then assemble those into `SpreadQuoteCheckInput` and call
`check_spread_limit_price()` for the verdict. The probe must not bypass
or duplicate this validator.

## 4. Current IBKR quote capability audit

### 4.1 `chad/market_data/ibkr_price_provider.py`

* Class `IBKRPriceProvider` supports `secType` ∈ {STK, FX, FUT} only.
* `_make_contract(symbol, sec_type)` switches on `sec_type` and builds
  `Stock` / `Forex` / `Future` — no `OPT` or `BAG` branch.
* Quote path: `reqMarketDataType(3)` (delayed OK) → `reqMktData(contract,
  "", False, False)` (streaming, snapshot=False) → poll `ticker.last` /
  `ticker.close` / `ticker.bid` / `ticker.ask` for ≤ 5s → `cancelMktData`.
* Returns `PriceSnapshot(symbol, last, close, bid, ask, ts_utc, source,
  delayed)`. Bid/ask are surfaced as fields but the fallback chain prefers
  `last → close → cached`; bid/ask are not the primary output and zeros
  are written when NaN, which is **incompatible** with `OptionLegQuote`'s
  contract (the checker requires positive bids/asks or `None`, never `0`).

### 4.2 `chad/market_data/options_chain_refresh.py`

* Builds `Option` template (line 218) with empty strike/right to feed
  `reqContractDetails` — used purely for chain discovery, not quoting.
* No leg-level `reqMktData(Option(...))` exists anywhere in the tree.

### 4.3 `chad/market_data/options_greeks_publisher.py`

* Computes theoretical greeks via Black-Scholes; does **not** request
  broker option quotes. Cannot be reused as a quote source.

### 4.4 `chad/execution/ibkr_adapter.py`

* `_resolve_option` (line 953) builds qualified `Option` contracts with
  conId. `_resolve_combo` (line 1030) builds qualified leg `Option`s plus
  a `BAG` `Contract` with `ComboLeg` references. Both paths only run as
  part of `submit_intent` order placement; there is no exposed
  read-only quote helper.
* Adapter uses ib_async patterns identical to what a probe would need
  (`qualifyContracts` of `Option`, then build BAG with conIds).

### 4.5 Capability verdict

| Quote target | Existing reusable wrapper? | Reuse plan |
|--------------|----------------------------|------------|
| STK / FUT / FX | Yes — `IBKRPriceProvider.get_snapshot` | Not relevant to BAG quoting. |
| OPT leg bid/ask | **No** | New probe script must build `Option` + call `reqMktData` itself. |
| BAG combo bid/ask | **No** | New probe script must build BAG `Contract` + `ComboLeg`s itself. |

`IBKRPriceProvider` is **not** appropriate to extend. Its semantics
(last-preferred, zero-fill on NaN, write-through `price_cache.json`) are
wrong for option leg quoting and would risk cross-contamination of the
equity price cache. The probe should be a **standalone script** that
mirrors the chain refresh job's qualify-then-snapshot pattern.

### 4.6 Subscription / market-data-type risk

* `reqMarketDataType(3)` (delayed) is the only safe default. Live OPRA
  subscriptions are not assumed; an unqualified live request can return
  IBKR error `354 ("Requested market data is not subscribed")`.
* Delayed option quotes are **delayed by ~15 minutes** and may have
  wider effective spreads than realtime. The probe must surface delayed
  status in JSON output so the operator can interpret quote staleness
  before any adapter wiring decision.

## 5. Recommended probe architecture

### 5.1 File

`scripts/probe_bag_quotes.py` — new script (not implemented in this task).

### 5.2 Modes

* `--dry-run-fake` — synthesize `OptionLegQuote` / `BagComboQuote` from
  CLI-provided bid/ask numbers; no IBKR import, no network. Default mode.
  Lets CI and operator dry-runs exercise the JSON output shape and the
  `check_spread_limit_price()` integration without any broker access.
* `--live-readonly` — explicit flag required to attempt IBKR connection.
  Refuses to run if `CHAD_EXECUTION_MODE` is set to `live` (probe is a
  paper / dry-run tool, never live-trade adjacent).

### 5.3 CLI (mirrors `scripts/preview_bag_intent.py`)

```
python3 scripts/probe_bag_quotes.py \
    --symbol SPY \
    --expiry 20260618 \
    --long-strike 737 \
    --short-strike 744 \
    --long-right C \
    --short-right C \
    --limit-price 3.50 \
    [--max-abs-deviation 0.05] \
    [--max-pct-deviation 0.10] \
    [--probe-mode legs|combo|both]        # default: legs
    [--dry-run-fake]                       # default mode when --live-readonly absent
        [--fake-long-bid 1.40 --fake-long-ask 1.50 \
         --fake-short-bid 0.85 --fake-short-ask 0.95 \
         --fake-combo-bid 1.55 --fake-combo-ask 1.65]
    [--live-readonly]
        [--ib-host 127.0.0.1 --ib-port 4002 --client-id <new-id>]
        [--market-data-type 3]            # 3=delayed (default), 1=live, 2=frozen, 4=delayed-frozen
        [--quote-timeout-s 8.0]
```

### 5.4 Live-readonly execution sequence (when implemented)

1. Import `ib_async.IB`, `ib_async.Option`, `ib_async.Contract`,
   `ib_async.ComboLeg`. Imports deferred to inside `--live-readonly`
   branch so dry-run mode is hermetic.
2. Connect: `ib = IB(); ib.connect(host, port, clientId=<probe-id>)`.
3. `ib.reqMarketDataType(market_data_type)` — default 3 (delayed).
4. Build long-leg `Option(symbol, expiry, long_strike, long_right,
   exchange="SMART", currency="USD")` and short-leg `Option(...)`.
5. Qualify both legs: `qualified = ib.qualifyContracts(long_opt,
   short_opt)`. Refuse to continue if either leg returns `conId <= 0`
   (mirrors `_resolve_combo` invariant).
6. For each leg: `ticker = ib.reqMktData(leg, "", snapshot=True,
   regulatorySnapshot=False)`; loop `ib.sleep(0.2)` until bid+ask both
   present or `quote_timeout_s` expires; cancel via `cancelMktData`.
7. Build `OptionLegQuote(symbol, expiry, strike, right, bid, ask, last,
   theo_price=None, ts_utc=now, source="ibkr_delayed"|"ibkr_live")`.
   Convert IBKR NaN → `None` (do **not** zero-fill).
8. If `--probe-mode combo` or `both`: build BAG `Contract` per
   `_resolve_combo` lines 1217–1247 (secType=BAG, conId legs, ratio=1,
   action BUY/SELL); request another `reqMktData` snapshot; convert to
   `BagComboQuote`.
9. Assemble `SpreadQuoteCheckInput(spec, limit_price=args.limit_price,
   long_quote, short_quote, combo_quote, max_abs_deviation,
   max_pct_deviation)`.
10. Call `check_spread_limit_price(input)`.
11. Disconnect cleanly: `ib.disconnect()`.
12. Print one JSON document and exit 0 (`ok=True`) or 2 (`ok=False`).

### 5.5 JSON output shape

```json
{
  "schema_version": "probe_bag_quotes.v1",
  "ts_utc": "2026-05-17T00:00:00Z",
  "mode": "dry-run-fake" | "live-readonly",
  "market_data_type": 3,
  "delayed": true,
  "spec": {
    "symbol": "SPY",
    "expiry": "20260618",
    "long_strike": 737.0,
    "short_strike": 744.0,
    "long_right": "C",
    "short_right": "C",
    "spread_type": "BULL_CALL"
  },
  "long_quote":  {"bid": 1.40, "ask": 1.50, "last": null, "ts_utc": "..."},
  "short_quote": {"bid": 0.85, "ask": 0.95, "last": null, "ts_utc": "..."},
  "combo_quote": {"bid": 1.55, "ask": 1.65, "last": null, "ts_utc": "..."},
  "limit_price": 3.50,
  "check": {
    "ok": false,
    "source": "combo_mid",
    "mid_debit": 1.60,
    "limit_price": 3.50,
    "deviation_abs": 1.90,
    "deviation_pct": 1.1875,
    "max_allowed_deviation": 0.16,
    "warnings": [],
    "errors": ["limit_price_too_far_above_mid"]
  },
  "broker_errors": []
}
```

### 5.6 Integration with `check_spread_limit_price()`

The probe is a *thin adapter*: IBKR ticker → `OptionLegQuote` /
`BagComboQuote` → `SpreadQuoteCheckInput` → `check_spread_limit_price()`.
It contains **no** mid arithmetic, deviation logic, or tolerance code of
its own. The Tier 3A validator is the single source of truth.

### 5.7 Probe-mode default rationale (answers Step 4 question A)

* Default `--probe-mode legs`: leg quotes always exist when the legs are
  individually tradable; BAG combo quotes from IBKR are frequently
  empty or wide for retail-spread combos, especially in delayed mode.
* `combo` mode lets the operator empirically test whether a combo
  ticker yields a usable bid/ask for the specific spread.
* `both` mode lets the operator compare combo mid against leg mid —
  large divergence is itself a hardening signal.

### 5.8 Snapshot vs streaming (answers Step 4 question B)

Use `snapshot=True`. The probe is a one-shot query; we do not want a
persistent subscription, and snapshots auto-cancel after delivery, which
removes a class of leak bug in a script the operator runs interactively.

### 5.9 Delayed vs live (answers Step 4 question C)

Default `--market-data-type 3` (delayed). Live (`1`) is reachable only
with OPRA subscription and risks `Error 354`. The probe **must** record
which mode was used in the JSON output so the operator interprets the
mid correctly.

### 5.10 Quote wait time (answers Step 4 question D)

Poll every 200 ms up to `--quote-timeout-s` (default 8.0). Mirrors
`IBKRPriceProvider.SNAPSHOT_TIMEOUT_S=5.0` but adds margin for the
delayed-data first-tick latency observed in options chain refresh.

### 5.11 Missing bid/ask, last-only behavior (answers Step 4 question E)

* If a leg returns no bid and no ask but a positive `last`, surface
  `bid=None, ask=None, last=last_value` in the `OptionLegQuote`. The
  Tier 3A `_mid` helper will fall back to `last` and the result will
  include source markers downstream. **Do not synthesize bid/ask from
  last**; that defeats the validator's `theo_only_quote_check` style
  warnings.
* If a leg returns nothing at all, the probe must report
  `bid=None, ask=None, last=None` and let the validator emit
  `no_quote_mid_available`. The probe exits non-zero.
* Refuse to invent a bid/ask. Never zero-fill.

### 5.12 Crossed-quote handling (answers Step 4 question F)

Forward raw bid/ask to `OptionLegQuote` / `BagComboQuote` as observed.
The validator (`_is_crossed`, `compute_combo_mid_debit`) already drops
crossed pairs and emits `crossed_quote_ignored`. The probe should not
silently mask the crossed state.

### 5.13 No-order safety (answers Step 4 question G — output)

* Probe imports `IB`, `Option`, `Contract`, `ComboLeg` only.
* Probe must **not** import `chad.execution.ibkr_adapter` (which exposes
  `placeOrder` via `submit_intent`).
* Probe must **not** import `chad.strategies` (which would pull intent
  generation surface).
* A test asserts the probe module source contains zero `placeOrder`,
  `submit_intent`, `IbkrAdapter`, or `chad.strategies` substrings.

### 5.14 Tests without IBKR (answers Step 4 question H)

* `test_probe_bag_quotes_dry_run_fake_legs_ok` — `--dry-run-fake` with
  fake leg bid/ask inside tolerance → exit 0, JSON `check.ok=true`,
  `source="leg_mid"`.
* `test_probe_bag_quotes_dry_run_fake_combo_preferred` — fake combo +
  fake legs → JSON `source="combo_mid"`.
* `test_probe_bag_quotes_dry_run_fake_limit_too_high` →
  `errors=["limit_price_too_far_above_mid"]`, exit 2.
* `test_probe_bag_quotes_dry_run_fake_no_quotes` — no fake quotes
  supplied → `errors=["no_quote_mid_available"]`, exit 2.
* `test_probe_bag_quotes_dry_run_fake_crossed` — crossed combo
  bid/ask → `warnings=["crossed_quote_ignored"]` and validator falls
  through to leg mid.
* `test_probe_bag_quotes_refuses_live_in_live_mode` — sets
  `CHAD_EXECUTION_MODE=live` → script exits non-zero before connecting.
* `test_probe_bag_quotes_no_placeorder_imports` — regex-assertions on
  module source (mirror of test 20 in `test_options_quote_check.py`).
* `test_probe_bag_quotes_no_strategy_imports` — module source contains
  no `chad.strategies` / `chad.execution.ibkr_adapter` references.

### 5.15 Manual live validation (answers Step 4 question I)

Sequenced operator validation, performed **after** the probe lands and
**before** any adapter enforcement is contemplated:

1. `python3 scripts/probe_bag_quotes.py --symbol SPY --expiry <near>
   --long-strike <atm> --short-strike <atm+5> --long-right C
   --short-right C --limit-price <est> --dry-run-fake \
   --fake-long-bid ... --fake-long-ask ... --fake-short-bid ...
   --fake-short-ask ...`
   → must print JSON with `mode=dry-run-fake`, `check.ok=true|false`.
2. `python3 scripts/probe_bag_quotes.py ... --live-readonly` against
   IB Gateway 4002, paper account.
   Success criteria:
   * No `placeOrder`-related log lines (operator greps `journalctl`).
   * Both legs qualify (printed JSON includes long_conId and short_conId
     both `> 0`; or, if the implementation hides conIds, the
     `broker_errors` list is empty and `long_quote`/`short_quote` are
     populated).
   * Either bid+ask or last on at least one leg.
   * `check.source` ∈ {`combo_mid`, `leg_mid`, `theo_mid`}.
   * `check.mid_debit` is finite and positive.
   * No `Error 354` ("market data not subscribed") in `broker_errors`.
   * No new IBKR connection persists after exit (operator inspects
     `journalctl -u chad-* | grep clientId=<probe-id>`).
3. Repeat with `--probe-mode combo` and `--probe-mode both` and compare
   combo vs leg mid for the same spread.

Only after at least three independent operator sessions show stable,
plausible mids may Tier 3C (adapter enforcement) be designed.

## 6. Safety rules

* No order placement; probe must never call `placeOrder`, `placeOrderAsync`,
  `submit_intent`, or any adapter method that wraps them.
* No adapter enforcement; `chad/execution/ibkr_adapter.py` is untouched
  by Tier 3B.
* No strategy changes; `chad/strategies/*` untouched.
* No runtime mutation; the probe writes **nothing** to `runtime/`. JSON
  is printed to stdout only.
* No systemd registration; the probe is operator-invoked from CLI only,
  not added to any service unit.
* `--live-readonly` is the only path that imports `ib_async`. Default
  dry-run-fake mode is hermetic.
* `CHAD_EXECUTION_MODE=live` causes the probe to refuse before any
  import / connection.
* The probe uses a **new, unique client ID** (proposed: `9050`) that is
  added to `chad/execution/ibkr_client_ids.py` in the Tier 3B
  implementation task — picked outside any existing range to avoid
  colliding with `LIVE_LOOP=99`, `PRICE_PROVIDER=9030`,
  `HISTORICAL_PROVIDER=9034`, `POSITIONS_SNAPSHOT=9041`,
  `LEDGER_WATCHER=9040`, etc.
* Probe never touches `price_cache.json`, `positions_snapshot.json`, or
  any other runtime cache.

## 7. Required future tests

* `--dry-run-fake` mode JSON shape and exit codes (5 cases per §5.14).
* Static assertion: no `placeOrder`, `submit_intent`, `IbkrAdapter`,
  `chad.strategies` substrings in probe source.
* `CHAD_EXECUTION_MODE=live` refusal test (subprocess invocation,
  expects exit 2 and `error="probe_refused_live_mode"` in JSON).
* Argparse coverage: missing `--symbol`, missing `--limit-price`,
  invalid `--long-right`, etc. → non-zero exit with clear stderr.
* Integration test that builds `OptionLegQuote` / `BagComboQuote` from
  CLI args and re-runs `check_spread_limit_price()` directly (asserts
  the probe-internal conversion matches the validator's own behavior).
* No new dependency on `chad.execution` or `chad.strategies` in the
  probe's import graph (asserted via `import` regex sweep, like
  `test_options_quote_check.py` line 460–472).

## 8. Manual live validation

Future command shapes and acceptance criteria — see §5.15. Probe must
exit cleanly, leave no persistent subscription, and produce one of the
following `check.source` values: `combo_mid`, `leg_mid`, `theo_mid`. A
`source="none"` result is an automatic FAIL for the validation session.

## 9. Remaining blockers after Tier 3B

Tier 3B closes blocker #1 (live IBKR quote probe). Still outstanding:

2. **Adapter enforcement using quote_check** (Tier 3C). The probe is a
   diagnostic; it is not on the order path. Adapter integration must
   wire `check_spread_limit_price()` into `IbkrAdapter.submit_intent`
   for BAG intents and fail-closed in live mode.
3. **Bracket/OCA or fail-safe exit protection** (Tier 4). BAG fills
   need server-side exit protection so a paper fill that escapes
   trade_closer cannot bleed indefinitely.
4. **spread_id-aware reconciliation / position tracking** (Tier 5).
   Current trade_closer FIFO is symbol-oriented; BAG fills need a
   composite identifier so legs reconcile as one position.
5. **Live BAG fill harness** (Tier 6). End-to-end paper fill capture
   with assertion that quote_check passed before the fill posted.

## 10. Next implementation prompt outline

**Recommendation: Option A — Build `scripts/probe_bag_quotes.py` with
`--dry-run-fake` and `--live-readonly` modes.**

Rationale:

* Option B (more offline tests first) — Tier 3A already ships 18+
  offline tests including malformed-input, crossed-quote, theo-only,
  and combo-preferred cases; further offline coverage hits diminishing
  returns without a live quote source.
* Option C (modify ibkr_adapter to enforce quote_check now) — explicitly
  unsafe: enforcement before we know whether IBKR delivers usable
  delayed option bid/ask snapshots for SPY verticals would either
  (a) silently never fire (if `no_quote_mid_available` becomes the
  default outcome) or (b) start blocking legitimate intents the moment
  live BAG execution is authorized. Empirical probe data must precede
  enforcement design.
* Option D (defer to bracket / failsafe) — premature; bracket design
  presupposes a fill path that can submit, and the unauthorized-live
  status of BAG execution means no fills are imminent. Quote probe is
  cheap (read-only, paper-account-safe) and clears the next blocker.

Implementation outline for the next task:

1. **Baseline.** Confirm tree clean; tests at 2096 passed.
2. **Add client ID** `PROBE_BAG_QUOTES: int = 9050` to
   `chad/execution/ibkr_client_ids.py` and update `client_id_map()` /
   `all_client_ids()`.
3. **Create** `scripts/probe_bag_quotes.py`:
   * Top-level: `CHAD_EXECUTION_MODE` refusal check (mirror
     `preview_bag_intent.py` lines 56–67).
   * Argparse per §5.3.
   * `--dry-run-fake` path: build `OptionLegQuote` / `BagComboQuote` from
     fake-bid / fake-ask args, no IBKR imports inside this branch.
   * `--live-readonly` path: deferred `ib_async` imports, connect with
     `PROBE_BAG_QUOTES` client id, qualify legs, snapshot quotes,
     optional BAG snapshot, disconnect.
   * Shared path: build `SpreadQuoteCheckInput`, call
     `check_spread_limit_price`, print JSON per §5.5, exit 0/2.
4. **Create** `chad/tests/test_probe_bag_quotes.py` with the eight tests
   listed in §5.14 / §7. Tests must not connect to IBKR; the
   `--live-readonly` branch is exercised via mock IB injection or
   skipped behind `pytest.mark.skipif(os.getenv("CHAD_PROBE_LIVE") !=
   "1", reason=...)`.
5. **Verify.** Full suite → 2096 + new tests passing; `git status`
   clean apart from the three new / modified files; commit as
   `Add Phase D BAG live quote probe script (Tier 3B)`.
6. **Operator manual validation.** Run §5.15 sequence on IB Gateway
   paper account; capture JSON outputs into the operator log. No commit
   on this step.

Tier 3B classification (per Step 5): **A — READY FOR READ-ONLY LIVE
QUOTE PROBE SCRIPT.** Evidence: Tier 3A validator is broker-free and
complete; `_resolve_combo` already proves the OPT + BAG contract build
pattern works against ib_async; `IBKRPriceProvider` proves the
delayed-data snapshot pattern works; no missing infrastructure blocks
the script.

## 11. Risk / gap table (Step 5)

| Probe concern                             | Current evidence                                                                                       | Risk                                                                              | Recommended design                                                                                  |
|-------------------------------------------|--------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| OPT contract qualification                | `_resolve_option` (ibkr_adapter.py:953) + `_resolve_combo` qualifies legs (lines 1163–1187)            | If conId=0, BAG build is unsafe                                                   | Mirror `_resolve_combo` invariant: refuse to proceed if either leg returns `conId<=0`               |
| BAG contract qualification                | `_resolve_combo` builds BAG from leg conIds (lines 1217–1247)                                          | BAG with conId=0 legs triggers IBKR Error 321                                     | Probe builds BAG only after legs qualify; allow `--probe-mode legs` to skip BAG entirely            |
| snapshot vs streaming                     | `IBKRPriceProvider` uses streaming (snapshot=False); `options_chain_refresh` likewise                  | Streaming subscription leak if probe crashes mid-poll                             | Use `snapshot=True`; subscription auto-cancels after delivery                                       |
| live vs delayed market data               | `IBKRPriceProvider` defaults to `reqMarketDataType(3)` (delayed)                                       | Live (`1`) fails with Error 354 absent OPRA subscription                          | Default `--market-data-type 3`; surface `delayed=true` in JSON output                               |
| leg bid/ask availability                  | No existing OPT quote helper; chain refresh only fetches strikes/expiries, not leg quotes              | Delayed option bid/ask may be empty for far-OTM or low-volume contracts           | Convert IBKR NaN → `None`, never `0`; allow `last`-only fallback via Tier 3A validator              |
| combo bid/ask availability                | No existing BAG quote helper anywhere in tree                                                          | BAG combo quotes are frequently empty / wide in delayed mode for retail spreads   | Treat combo as optional; default `--probe-mode legs`; `combo`/`both` are operator-opt-in            |
| missing quote handling                    | Tier 3A `check_spread_limit_price` emits `no_quote_mid_available` error and `ok=False`                 | Probe could mask missing quotes by synthesizing values                            | Probe must never synthesize bid/ask from last; forward observed values verbatim                     |
| quote freshness / wait time               | `IBKRPriceProvider.SNAPSHOT_TIMEOUT_S=5.0`; chain refresh waits ~5s after `reqMktData`                 | Too short → routine timeouts; too long → operator pain                            | Default `--quote-timeout-s 8.0`; poll every 200ms                                                    |
| conversion to quote_check dataclasses     | `OptionLegQuote` / `BagComboQuote` accept `bid/ask/last` as `Optional[float]`; validator handles None  | Field-type mismatch (e.g. float NaN instead of None) breaks `_positive_float`     | Convert IBKR NaN → `None` in probe before constructing dataclass; preserve `source="ibkr_delayed"`  |
| no-order safety                           | `preview_bag_intent.py` lines 1–32 demonstrate import-isolation pattern                                | Accidental `chad.execution.ibkr_adapter` import would expose `placeOrder` surface | Static-grep test asserts probe contains no `placeOrder/submit_intent/IbkrAdapter/chad.strategies`   |
| testability without IBKR                  | `test_options_quote_check.py` is hermetic; `preview_bag_intent.py` test pattern shows subprocess style | Live-readonly path is the hard-to-test branch                                     | Default mode is `--dry-run-fake`; live path covered by optional `CHAD_PROBE_LIVE=1` gated test      |

### Classification

**A — READY FOR READ-ONLY LIVE QUOTE PROBE SCRIPT.**

Evidence:

* Tier 3A offline validator (`chad/options/quote_check.py`) is complete
  and broker-free; ready to consume probe output.
* OPT and BAG contract build patterns are already proven by
  `ibkr_adapter._resolve_combo` (lines 1030–1247).
* Delayed-data snapshot pattern is already proven by
  `IBKRPriceProvider` and `options_chain_refresh.py`.
* No missing infrastructure; the probe is incremental on existing
  ib_async usage in the tree.
* All blockers below classification A (offline harness, subscription,
  enforcement) are either already cleared or out of scope for Tier 3B.

Not B: an offline-only probe is what Tier 3A already provides
(synthetic-quote tests in `test_options_quote_check.py` cover the
shape). Adding another offline harness duplicates that work.
Not C: market data subscription is **not** a blocker because the probe
defaults to delayed data (Tier 3 of OPRA, no subscription required).
Not D: deferring would push Tier 3C design without empirical data on
whether IBKR delivers usable delayed option quotes for the spreads the
strategy actually constructs — that is the question the probe exists to
answer.

## 12. Recommended next implementation task

**Option A** — see §10 for justification and step-by-step outline.
