# WKF — Weekend-Fix Pre-Open Deployment Record (2026-07-11 → activated 2026-07-12)

**Fix work:** 2026-07-11 (commits landed) · **Activation:** 2026-07-12 ~00:14:47Z (pre-open live-loop recycle)
**Scope:** chad/core/position_guard.py, chad/core/live_loop.py, chad/execution/rth_gate.py,
chad/utils/market_hours.py, chad/execution/ibkr_adapter.py, chad/ops/reconciliation_publisher.py, conftest.py
**Trading posture impact:** NONE (paper). No live enablement. No config mutation. No order path change beyond the RTH block.
**HEAD at activation:** `dbf1c22` (main in sync with origin/main).

---

## 1. What shipped (4 commits on HEAD)

| Unit | Commit | Change |
|------|--------|--------|
| U1 | `0b4c8a9` | `position_guard` books on **CONFIRMED FILL only** + boot phantom-reconcile sweep. |
| U2 | `8b39730` | Market-hours (**RTH**) gate on the equity/ETF submit path. |
| U3 | `d7db150` | Drift semantics **v2** — like-with-like symbol totals + atomic snapshot + `drift_kind` vocab. |
| U2-HF | `dbf1c22` | conftest defaults `CHAD_RTH_GATE=0` so equity-submit tests stay deterministic. |

## 2. Pre-open restart / activation proof

The pre-open window recycled `chad-live-loop.service` at **2026-07-12 00:14:46Z**; the new process
(`python3[2824862]`) came up at 00:14:47Z and immediately exercised all three behaviors, confirming the
restart loaded the WKF code (not a stale daemon):

- `00:14:47Z` — `BOOT_PHANTOM_RECONCILE cleared=3` (U1)
- `00:14:58Z` onward — `RTH_GATE_BLOCK … status=market_closed` (U2)
- `position_guard_drift.json` now emits `schema_version=position_guard_drift.v2` (U3)

## 3. RTH gate contract (U2, live)

`chad/execution/rth_gate.py`: gates asset classes `{equity, etf}` only; **DEFAULT ON**
(`CHAD_RTH_GATE`, only an explicit falsy value disables). A blocked intent places **no order and writes
no idempotency row** — status `market_closed`, then `SKIP_EVIDENCE_UNCONFIRMED_ORDER_STATUS` and
`COOLDOWN_NOT_REARMED_UNCONFIRMED_STATUS`. Futures/other classes are exempt (their own gate).

## 4. Drift v2 semantics (U3, live)

`position_guard_drift.v2`: like-with-like per-symbol guard-vs-broker totals, atomic snapshot
(`snapshot_generation`), and an explicit `drift_kind` vocab (`phantom_guard_entry`,
`broker_untracked_position`, `qty_mismatch`). Observability-only — surfaces drift, does not net or mutate.

---

## 5. Pre-open window outcomes

Recorded **verbatim** from the operator addendum (2026-07-12), as directed:

> ADDENDUM — PRE-OPEN WINDOW OUTCOMES (2026-07-12 00:14Z, verify in journal/runtime):
> - Phantom reconcile swept THREE ghosts at boot: gamma|BAC, gamma|SPY, gamma|MSFT
>   (GUARD_PHANTOM_RECONCILED markers 00:14:47Z).
> - RTH gate proven live: 6 Saturday equity intents blocked, zero orders created.
> - BindAddress=127.0.0.1 applied but INERT for the API socket (4002 still *:4002);
>   GAP-013 remains OPEN-shielded (SG + TrustedIPs), disposition = manual GUI.
> - Drift v2 settled picture: 5 qty_mismatch rows, guard ≈ 2x broker on
>   TLT/IWM/UNH/LLY/V — FIFO entry-accumulation with exits never netted.
>   Surfaced-not-fixed; requires its own reconciliation PA. Record verbatim in §5.

### 5.1 Verification against journal/runtime (2026-07-12)

| # | Claim | Evidence | Verdict |
|---|-------|----------|---------|
| 1 | Phantom sweep of gamma\|BAC, gamma\|SPY, gamma\|MSFT at 00:14:47Z | journald: 3× `GUARD_PHANTOM_RECONCILED … reason=no_broker_position_no_confirmed_fill` @ 00:14:47.380Z + `BOOT_PHANTOM_RECONCILE cleared=3 keys=['gamma\|BAC','gamma\|MSFT','gamma\|SPY']` @ 00:14:47.381Z | **CONFIRMED** (exact set) |
| 2 | RTH gate blocked Saturday equity intents, zero orders created | journald: `RTH_GATE_BLOCK symbol=… status=market_closed` each with `SKIP_EVIDENCE_UNCONFIRMED_ORDER_STATUS` — no order, no idempotency row. First cycles: BAC/gamma, MSFT/beta, SPY/gamma (00:14:58Z), then BAC/MSFT/SPY (00:16:10Z) → the operator's **6**. | **CONFIRMED** (see 5.2 note) |
| 3 | BindAddress applied but INERT; 4002 still `*:4002` | `/etc/chad/ibc/config.ini` has `BindAddress=127.0.0.1` **and** `TrustedIPs=127.0.0.1`; `ss -tulpn` still shows `*:4002` (java pid) → applied but not honored by the socket | **CONFIRMED** (see 5.2 note) |
| 4 | 5 qty_mismatch rows, guard ≈ 2× broker on TLT/IWM/UNH/LLY/V | `runtime/position_guard_drift.json` v2, `drift_count=5`, all `qty_mismatch`: IWM 392/200 (1.96×), LLY 250/130 (1.92×), TLT 780/420 (1.86×), UNH 399/207 (1.93×), V 301/154 (1.95×) | **CONFIRMED** |

### 5.2 Annotations (do not alter the §5.0 verbatim block)

- **RTH "6"** is a point-in-time count. `RTH_GATE_BLOCK` recurs every live-loop cycle while the market is
  closed (15 blocks in the 00:14–00:20Z window; the strategies keep re-proposing on a Saturday). The
  invariant that matters — **zero orders created / zero idempotency rows** — holds on every block.
- **GAP-013 cross-reference.** The operator cited `GAP-013` for the 4002 socket. In
  `docs/CHAD_GAPS_TO_CLOSE.md`, `GAP-013` is the **ports 9618/9619/9620 → 0.0.0.0** exposure item; the
  **4002 BindAddress maintenance** PA is tracked as **`GAP-020`** (`ops/pending_actions/GAP-020_ibgateway_bindaddress_maintenance.md`).
  Both are the same shape (wildcard bind, shielded by SG + `TrustedIPs=127.0.0.1` + `ApiOnly`). Operator
  text preserved verbatim; flagged here only so the routing id is unambiguous.

---

## 6. Follow-ups (open)

- **Drift-v2 reconciliation PA (new).** The 5 `qty_mismatch` rows are surfaced-not-fixed:
  guard FIFO accumulates entries while exits are never netted, so guard ≈ 2× broker. Needs its own
  reconciliation PA (candidate remedy: `scripts/close_guard_entry.py` under the GAP-028 permissive policy,
  or a netting rebuild). **Do not** silently mutate the guard — this is a reconciliation decision.
- **4002 socket disposition** — GAP-020 (BindAddress inert; socket bind is a manual IB Gateway GUI action);
  GAP-013 (metrics ports) tracked separately. Both remain OPEN-shielded, not emergencies.

## 7. No-live confirmation

- Posture remains PAPER (`CHAD_EXECUTION_MODE=paper`); `ready_for_live` unchanged.
- No broker orders placed (the RTH blocks are the opposite — they suppress placement).
- No runtime JSON hand-edits; all artifacts produced by the normal publishers.
- Only service action was the scheduled pre-open live-loop recycle (00:14:46Z); IB Gateway untouched.
