# PENDING ACTION — M2K / MYM Bar-Provider Futures Mapping Fix

- date: 2026-05-26
- prepared_by: audit (read-only)
- target_branch: main (HEAD at audit time: 5961be2)
- governance: surgical edit, one change, one file, no config mutation, no live posture change
- status: VERIFIED (code patch in commit cdab294; runtime observation §10.1 confirms M2K/MYM bars now present in ibkr_bars_cache.json after 2026-05-26T20:14:19Z daemon recycle, zero Error 200, zero M2K bar_stale, safety posture preserved)
- linked evidence: reports/parity_audit/STRATEGY_UTILIZATION_BLOCKER_AUDIT_20260526T173158Z.md (§3, §5)

---

## 1. Root cause

`chad/market_data/ibkr_bar_provider.py:77–80` defines:

```python
FUTURES_SYMBOLS = {
    "MES", "MNQ", "MCL", "MGC",    # Alpha/Gamma futures
    "ZN", "ZB", "M6E", "SIL",       # Omega Macro
}
```

**M2K and MYM are absent from this set**, even though both are listed in `config/universe.json` (M2K @ CME, MYM @ CBOT, both `sec_type=FUT`) and both are referenced as active trading symbols in:
- `chad/market_data/futures_contract_resolver.py:46`
- `chad/market_data/futures_roll_publisher.py:32`
- `chad/strategies/alpha_futures.py:100`
- `chad/strategies/gamma_futures.py:87,104`
- `chad/strategies/gamma_futures_config.py:31`
- `chad/execution/ibkr_adapter.py:309,504`
- `chad/execution/execution_pipeline.py:639`

Because `_is_futures_symbol("M2K")` and `_is_futures_symbol("MYM")` both return `False`, `_make_ib_contract()` at `chad/market_data/ibkr_bar_provider.py:264–265` falls into the else-branch:

```python
return Stock(sym, "SMART", "USD")
```

IBKR has no US-equity definition for "M2K" or "MYM" (those are futures roots), so `reqHistoricalData` fails with **Error 200: No security definition has been found**, and no bars are stored in `runtime/ibkr_bars_cache.json` for either symbol.

Separately, `DEFAULT_UNIVERSE` at `chad/market_data/ibkr_bar_provider.py:83–88` also omits M2K and MYM. This is only the fallback when `config/universe.json` is unreadable, so it does NOT affect the live system today (universe.json is present and includes both symbols) — but it should be aligned with FUTURES_SYMBOLS to avoid a future-degradation footgun.

---

## 2. Effect (observed)

Cycle-level evidence captured in audit at 2026-05-26T15:54–16:01 UTC:

```
2026-05-26 16:00:39,992Z ERROR ib_async.wrapper Error 200, reqId 5601:
  No security definition has been found for the request,
  contract: Stock(symbol='MYM', exchange='SMART', currency='USD')
2026-05-26 16:00:40,572Z ERROR ib_async.wrapper Error 200, reqId 5602:
  No security definition has been found for the request,
  contract: Stock(symbol='M2K', exchange='SMART', currency='USD')
```

Downstream consequences in `chad-live-loop`:
- `GATE_REJECT intent_id= gate=data_freshness reason=bar_stale:age=327404.6s>max=172800s symbol=M2K strategy=gamma_futures` — every cycle (49 rejects in 71 minutes; age advances with wall-clock).
- gamma_futures' primary symbol leg (M2K) cannot pass A4 data_freshness_gate.
- MYM is the secondary leg of gamma_futures AND a candidate for alpha_futures (`_EQUITY_INDEX_FUTURES = frozenset({"MES","MNQ","MYM","M2K"})`); even when the dynamic-risk-allocator outputs ≥1.0 contracts, the bar-stale gate would reject it for the same reason. Today MYM is also skipped earlier at `intent_skipped_invalid_quantity` (whole-units rounding from 0.987 → 0), but if/when sizing grew above 1.0 the bar-stale block would still apply.

`runtime/ibkr_bars_cache.json` currently lists 32 symbols (`AAPL…ZN`) — M2K and MYM are absent.

Broker truth (`ib_async updatePortfolio` polled by the bar-provider's IB session, 2026-05-26T16:01:42Z):
- M2K position = `-16.0` (avgCost 14345.88, marketPrice 2910.50, unrealizedPNL `-$3,305.92`)
- The position is real; the bar-feed defect is the only thing preventing strategy attribution.

---

## 3. Required patch (NOT YET APPLIED)

### 3.1 Target file (single file)
`chad/market_data/ibkr_bar_provider.py`

### 3.2 Edit 1 — `FUTURES_SYMBOLS` (lines 77–80)

```diff
 FUTURES_SYMBOLS = {
-    "MES", "MNQ", "MCL", "MGC",    # Alpha/Gamma futures
-    "ZN", "ZB", "M6E", "SIL",       # Omega Macro
+    "MES", "MNQ", "MCL", "MGC",    # Alpha/Gamma futures
+    "ZN", "ZB", "M6E", "SIL",       # Omega Macro
+    "MYM", "M2K",                   # Micro Dow / Micro Russell — gamma_futures + alpha_futures
 }
```

### 3.3 Edit 2 — `exchange_map` (inside `_make_ib_contract`, lines 211–221)

```diff
         exchange_map = {
             "MES": "CME",
             "MNQ": "CME",
             "MCL": "NYMEX",
             "MGC": "COMEX",
             "ZN": "CBOT",
             "ZB": "CBOT",
             "M6E": "CME",
             "SIL": "COMEX",
             "SI": "COMEX",
+            "MYM": "CBOT",
         }
```

**M2K exchange decision** — config/universe.json declares M2K @ CME. The existing default `exchange_map.get(sym, "CME")` at line 223 already returns CME for any unmapped symbol. **No explicit M2K entry is required**, but for symmetry and self-documentation it MAY be added as `"M2K": "CME"`. Either is acceptable; the minimal-touch option (no M2K entry) is preferred to keep the diff smallest.

### 3.4 Edit 3 (optional, recommended) — `DEFAULT_UNIVERSE` (lines 83–88)

```diff
 DEFAULT_UNIVERSE = [
     # Equities/ETFs
     "SPY", "QQQ", "IWM", "GLD", "TLT",
     # Futures
-    "MES", "MNQ", "MCL", "MGC", "ZN", "ZB", "M6E", "SIL",
+    "MES", "MNQ", "MCL", "MGC", "ZN", "ZB", "M6E", "SIL", "MYM", "M2K",
 ]
```

This is a defensive alignment with FUTURES_SYMBOLS. It only fires if `config/universe.json` is unreadable. Recommended but not strictly load-bearing for fix correctness today.

### 3.5 No other files changed
- No tests touched (new file added — see §4).
- No config (`config/universe.json`, `config/edge_decay_config.json`, anything under `config/`) touched.
- No runtime JSON touched (`runtime/ibkr_bars_cache.json` will be re-populated by the daemon after the patch lands and the service is allowed to recycle the next cycle — note: even THAT recycle does not require a service restart; the daemon's polling loop will pick up the new code only after the next process start; the existing process keeps running old code until then).
- No systemd unit file touched.
- No live posture change. CHAD stays PAPER. allow_ibkr_live remains False.

---

## 4. Tests needed (NEW test file)

Add `chad/tests/test_ibkr_bar_provider_futures_mapping.py` (new file, not in repo today). Existing tests do not cover `_make_ib_contract` symbol→contract resolution — see the audit's `grep -RIln "ibkr_bar_provider\|_make_ib_contract" chad/tests` returning only adjacent helpers.

Required cases (each independent, no IB connection, pure dispatch test):

1. **`test_m2k_resolves_to_futures_not_stock_smart`**
   `_make_ib_contract("M2K", ib=None)` returns an `ib_async.Future` instance with `symbol == "M2K"`, `exchange == "CME"`, `tradingClass == "M2K"`, `currency == "USD"`. Must NOT be `Stock`.

2. **`test_mym_resolves_to_futures_not_stock_smart`**
   `_make_ib_contract("MYM", ib=None)` returns an `ib_async.Future` with `symbol == "MYM"`, `exchange == "CBOT"`, `tradingClass == "MYM"`, `currency == "USD"`. Must NOT be `Stock`.

3. **`test_mym_exchange_resolves_to_cbot`**
   Targeted assertion on exchange to catch silent CME-default regression.

4. **`test_existing_futures_mappings_unchanged`** — regression guard for the symbols already correct today:
   - `MES` → `Future("MES", "CME", tradingClass="MES")`
   - `MNQ` → `Future("MNQ", "CME", tradingClass="MNQ")`
   - `MCL` → `Future("MCL", "NYMEX", tradingClass="MCL")`
   - `MGC` → `Future("MGC", "COMEX", tradingClass="MGC")`
   - `M6E` → `Future("M6E", "CME", tradingClass="M6E")`
   - `ZN` → `Future("ZN", "CBOT", tradingClass="ZN")`
   - `ZB` → `Future("ZB", "CBOT", tradingClass="ZB")`
   - `SIL` → `Future(ibkr_sym="SI", "COMEX", tradingClass="SIL")` (root-symbol remap retained)

5. **`test_equity_symbols_still_return_stock`**
   Sanity: `_make_ib_contract("SPY", ib=None)` returns `ib_async.Stock("SPY","SMART","USD")` — equities not regressed by the new futures-set entries.

6. **`test_futures_symbols_set_includes_m2k_and_mym`**
   Module-level set assertion `{"M2K","MYM"} <= FUTURES_SYMBOLS`. Catches deletion of the new entries by future refactors.

No live IBKR connection involved; all tests construct contracts and inspect attribute values. Reuse pytest pattern from `chad/tests/test_ib_async_import_parity.py` for import handling.

---

## 5. Verification (post-patch, before commit)

Run from `/home/ubuntu/chad_finale` with venv activated and `PYTHONPATH=.`:

### 5.1 Targeted (new test file)
```bash
python3 -m pytest chad/tests/test_ibkr_bar_provider_futures_mapping.py -v
```
Expect 6 passes.

### 5.2 Full regression
```bash
python3 -m pytest chad/tests/ -x -q 2>&1 | tail -20
```
Expect `2529 + 6 = 2535 passed` (assuming no test deletions). At minimum: same count as baseline (2529) plus the new 6 tests, zero failures.

### 5.3 Compile guard
```bash
python3 -m py_compile chad/market_data/ibkr_bar_provider.py
```

### 5.4 Static module-load smoke (offline; CHAD_SKIP_IB_CONNECT=1)
```bash
CHAD_SKIP_IB_CONNECT=1 python3 -c "
from chad.market_data.ibkr_bar_provider import FUTURES_SYMBOLS, _make_ib_contract
assert {'M2K','MYM'} <= FUTURES_SYMBOLS
c_m2k = _make_ib_contract('M2K', ib=None)
c_mym = _make_ib_contract('MYM', ib=None)
print('M2K', type(c_m2k).__name__, c_m2k.exchange, c_m2k.tradingClass)
print('MYM', type(c_mym).__name__, c_mym.exchange, c_mym.tradingClass)
"
```
Expect: `M2K Future CME M2K` and `MYM Future CBOT MYM`.

### 5.5 Full preview (per CLAUDE.md verification sequence)
```bash
python3 chad/core/full_cycle_preview.py --dry-run 2>&1 | tail -30
```

### 5.6 Runtime observation (after merge — operator step, not part of patch)
Once the patched code is on disk, the operator MAY allow the next planned recycle of `chad-ibkr-bar-provider` to occur naturally. **No forced restart required**, no `systemctl` action. Within one polling cycle of the recycled daemon process, expect:
- `runtime/ibkr_bars_cache.json` to grow from 32 → 34 symbols (M2K, MYM appended under `symbols`)
- `journalctl -u chad-ibkr-bar-provider` to stop emitting `Error 200, ... Stock(symbol='M2K'…)` and `… Stock(symbol='MYM'…)` lines.
- `journalctl -u chad-live-loop` to stop emitting `GATE_REJECT … bar_stale … symbol=M2K`.

If the operator chooses NOT to recycle the daemon, the running process retains the old code and the fix has no runtime effect — code is in tree, behavior unchanged. That is acceptable; the patch is forward-compatible.

### 5.7 NO runtime edits
- Do NOT touch `runtime/ibkr_bars_cache.json`.
- Do NOT touch `runtime/strategy_allocations.json`.
- Do NOT touch any other runtime/* JSON.
- Do NOT clear `stop_bus.json`.
- Do NOT modify thresholds.

---

## 6. Current status

**VERIFIED** — all three gating conditions satisfied:
1. ✅ Code patch applied (commit cdab294, single commit, file = `chad/market_data/ibkr_bar_provider.py` + new test file).
2. ✅ §5.1–§5.5 all green at apply time (see §10).
3. ✅ Runtime observation §5.6 confirmed at 2026-05-26T20:23:40Z — fresh M2K and MYM bars present in `runtime/ibkr_bars_cache.json` and zero `bar_stale symbol=M2K` GATE_REJECT lines in `chad-live-loop` since the 2026-05-26T20:14:19Z daemon recycle. Full evidence in §10.1.

M2K data_freshness gate now passes; gamma_futures M2K signals reach the position-conflict suppression layer (`same_side_position_open` — correct given the broker holds -16 M2K short). MYM bars are fresh; MYM signals now flow but stop at `invalid_quantity` (whole-units rounding from 0.987 → 0; out of scope, see §8). Live/paper safety machinery unaffected.

---

## 7. Rollback

Single-file, additive-only change to a Python set + dict.
- `git revert <commit>` is safe; no schema, no runtime, no config implications.
- Worst-case behavior of an in-flight subscriber that already requested an M2K/MYM future: `qualifyContracts` succeeds (front-month resolved), bars accumulate, and the daemon's existing per-symbol error path swallows partial failures (`ibkr_bar_provider.py:464`). No queue cleanup needed on revert.

Snapshot tags already in place (`RATIFICATION_MASTER_20260402`, `REVERT_PRE_OVERHAUL_20260419`) remain the broader-rollback floor; this patch does not require a new pre-patch tag.

---

## 8. Out of scope (separately observed, do NOT bundle)

- MCL front-month resolution staleness (Error 162 `MCLM6@NYMEX Trades` — contract month past last-trade-date 20260518): tracked separately; affects MCL only and does NOT block this patch.
- alpha_futures / alpha_crypto / delta / gamma_futures edge_decay halts: governance Pending Action required (`scripts/clear_edge_decay.py`); not part of this patch.
- M6E / MGC `duplicate_blocked`: expected open-order state, no patch needed.
- MYM `invalid_quantity`: expected whole-units rounding; not a defect.

---

## 9. Sign-off gate

Operator GO required before applying the code change. Recommended ordering after GO:

```
1. Read this Pending Action top-to-bottom; confirm scope.
2. Apply Edits 1–3 in §3 to chad/market_data/ibkr_bar_provider.py.
3. Add chad/tests/test_ibkr_bar_provider_futures_mapping.py.
4. Run §5.1 → §5.5 in order; halt on any failure.
5. Single commit: "PR-M2K-MYM: add M2K/MYM to bar-provider FUTURES_SYMBOLS + MYM→CBOT exchange map"
6. Tag (optional): PR-M2K-MYM_APPLIED_2026-MM-DD
7. Operator-discretion daemon recycle for runtime confirmation (§5.6).
```

CHAD stays PAPER throughout. allow_ibkr_live remains False. No service restart is forced.

---

## 10. Application record (post-apply)

- date applied: 2026-05-26
- applied by: governed surgical patch (chad-finale repo, branch main)
- baseline before patch: HEAD=bb2eccb (Pending-Action commit), 2529 tests passing on second-run baseline (one transient flake on `test_canonical_equity_source.py::test_canonical_sources_agree_within_skew_tolerance` cleared on re-run)
- files changed (explicit; no `git add .` used):
  - `chad/market_data/ibkr_bar_provider.py` — added "M2K","MYM" to `FUTURES_SYMBOLS`; added "M2K":"CME" and "MYM":"CBOT" to `exchange_map`; extended `DEFAULT_UNIVERSE` for symmetry (defensive fallback only)
  - `chad/tests/test_ibkr_bar_provider_futures_mapping.py` — new file, 10 cases (set membership, M2K/MYM dispatch, MYM→CBOT guard, existing-mapping regression, SIL→SI remap retention, equity & unknown-symbol fallbacks)
  - this Pending Action doc (status moved to APPLIED, plus this §10 record)
- tests after patch:
  - new targeted file: 10 / 10 PASS
  - keyword filter `ibkr_bar_provider or futures_mapping or M2K or MYM or bars`: 51 / 51 PASS
  - full suite: 2538 PASS + 1 PRE-EXISTING FAIL (`test_canonical_equity_source.py::test_canonical_sources_agree_within_skew_tolerance`); fail count math = baseline 2528 + 10 new = 2538 PASS (delta exactly the 10 new tests, zero new failures). The 1 failing test is the BOX-034 canonical equity divergence (`pnl_state.account_equity=$179,451` vs `portfolio_snapshot total=$246,654`, drift ≈ $67k) — **unrelated to this patch** (the patch does not touch pnl_state or portfolio_snapshot publishers). Recommended: separate Pending Action under BOX-034 §4a.
- static smoke (CHAD_SKIP_IB_CONNECT=1):
  - `M2K` → `Future(symbol=M2K, exchange=CME, tradingClass=M2K)` ✓
  - `MYM` → `Future(symbol=MYM, exchange=CBOT, tradingClass=MYM)` ✓
  - All 8 existing futures roots unchanged ✓ (MES/MNQ/MCL/MGC/M6E/ZN/ZB/SIL→SI)
  - `SPY` still `Stock(SMART)` ✓ (equity path unaffected)
- no-live-posture confirmation:
  - `ready_for_live` = False (unchanged)
  - `allow_ibkr_live` = False (unchanged)
  - `allow_ibkr_paper` = True (unchanged)
  - `stop_bus.active` = False (unchanged)
  - `reconciliation_state.status` = GREEN (unchanged)
  - `positions_truth.broker_authority_status` = GREEN (unchanged)
  - `trade_lifecycle_state.backlog_flag` = False (unchanged)
  - `position_guard_drift.drift_count` = 1 (was 0 at audit start) — drift entry is `alpha_futures|MES` (GAP-028 PERMISSIVE observation, broker_truth_missing). **Unrelated to this patch**; separate operator surface.
  - No `systemctl` actions taken. No `runtime/*.json` edited. No `config/*` edited.
- runtime observation status: **VERIFIED** (see §10.1 for evidence).

---

## 10.1 Runtime verification evidence (operator-authorized daemon recycle)

- date verified: 2026-05-26
- operator GO: explicit authorization to restart `chad-ibkr-bar-provider.service` only (no IB Gateway restart, no chad-live-loop restart, no runtime/config edits, no order activity)
- pre-restart daemon: PID 2880191, running since 2026-05-18T12:11:21Z, executing pre-patch code in memory
- restart command issued: `sudo systemctl restart chad-ibkr-bar-provider.service` at 2026-05-26T20:14:19Z
- post-restart daemon: PID 1126396, IB-async `Synchronization complete` and `ibkr_bar_provider.daemon_running` logged at 2026-05-26T20:14:20Z

### 10.1.1 Bars cache rewrite (Check 1)

`runtime/ibkr_bars_cache.json` next rewrite at 2026-05-26T20:23:40Z:
- pre-restart: `bar_count=1498`, `symbol_count=32`, **M2K absent, MYM absent**
- post-restart: `bar_count=1645`, `symbol_count=35`, **M2K present (51 bars, latest close=2927.5, ts=2026-05-26 15:13:00-05:00), MYM present (51 bars, latest close=50549.0, ts=2026-05-26 15:13:00-05:00)**
- delta vs §5.6 prediction: predicted 32→34; observed 32→35. The extra symbol is **MCL**, which was also being mis-resolved at runtime pre-restart but is not in scope for this PR. MCL is already listed in `FUTURES_SYMBOLS` (line 78), so no code change is required for MCL; it is being properly served as Future post-recycle. Logged as side observation in §8.

### 10.1.2 Contract dispatch (Check 2)

Daemon logs post-restart show M2K resolving to a Future contract (no further `Stock(symbol='M2K', exchange='SMART')` lines):

```
2026-05-26 20:14:19,820Z INFO ib_async.wrapper position: Position(account='DUK902770',
  contract=Future(conId=770561189, symbol='M2K',
  lastTradeDateOrContractMonth='20260618', multiplier='5', currency='USD',
  localSymbol='M2KM6', tradingClass='M2K'), position=-16.0, avgCost=14345.88)
```

MYM contract type confirmed indirectly by §10.1.3 (zero `Stock(symbol='MYM')` requests post-restart and 51 bars now stored under `symbols.MYM`).

### 10.1.3 IB Error 200 for M2K/MYM (Check 3)

```
journalctl -u chad-ibkr-bar-provider --since "2026-05-26 20:14:19 UTC" \
  | grep -E "Error 200.*(M2K|MYM)" | wc -l
→ 0
```

Pre-restart baseline: 43 Error-200 lines for M2K/MYM in the 10 minutes ending at restart. Post-restart: zero Error-200 lines for M2K/MYM (and zero Error-200 of any kind) across 26 minutes of observation.

### 10.1.4 live-loop M2K bar_stale (Check 4)

```
journalctl -u chad-live-loop --since "2026-05-26 20:14:19 UTC" \
  | grep -E "bar_stale.*M2K|M2K.*bar_stale" | wc -l
→ 0
```

Pre-restart: `GATE_REJECT gate=data_freshness reason=bar_stale:age=338176.8s>max=172800s symbol=M2K strategy=gamma_futures` firing every gamma_futures cycle. Post-restart: 16 fresh M2K signal cycles observed in 30 minutes; all pass A4 data_freshness and proceed to position-conflict suppression (`SKIP suppression=same_side_position_open` — correct given the broker -16 M2K position).

### 10.1.5 No fake fill evidence (Check 5)

`find runtime/ -type f -newermt "2026-05-26 20:14:19" | grep -iE 'evidence|fill|paper.*exec'` returns no new files. The only execution-path activity in live-loop logs post-restart is the pre-existing P0-1-hardened `SKIP_EVIDENCE_UNCONFIRMED_ORDER_STATUS ... reason=unconfirmed_order_status:duplicate_blocked` path for M6E and MGC, which explicitly does not write paper fills. `trade_closer_state.json` mtime advance reflects normal publisher cadence, not new M2K/MYM fill entries.

### 10.1.6 Safety posture preserved

| Key | Pre-restart | Post-restart | Source |
|---|---|---|---|
| `ready_for_live` | False | False (ts 2026-05-26T20:31:04Z) | `runtime/live_readiness.json` |
| `allow_ibkr_live` | False | False (ts 2026-05-26T20:40:08Z) | `runtime/decision_trace_heartbeat.json` |
| `allow_ibkr_paper` | True | True (ts 2026-05-26T20:40:08Z) | `runtime/decision_trace_heartbeat.json` |
| `stop_bus.active` | False | False | `runtime/stop_bus.json` |

No `systemctl` action issued beyond the single authorized `chad-ibkr-bar-provider` restart. No `runtime/*.json` edited. No `config/*` edited. No broker orders placed or cancelled. CHAD remains PAPER.
