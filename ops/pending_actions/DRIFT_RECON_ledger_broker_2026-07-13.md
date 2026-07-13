# PENDING ACTION — Ledger↔broker reconciliation (DRIFT-RECON)

- **Filed:** 2026-07-13
- **Type:** Forensic baseline + governed disposition + guarded tool. Tool BUILT + tested; runs only
  on typed operator GO. **NOT executed against runtime; NOT deployed.**
- **Author:** DRIFT-RECON (read-only forensic + tooling)
- **Status:** PROPOSED. `--execute` awaits typed operator GO.
- **Depends on / gates:** this is the pre-registered ACTIVE-flip gate (criterion 3) for the exit
  overlay — see `ops/pending_actions/EXIT_AUDIT_equity_roundtrip_close_2026-07-13.md` §4.
- **Governance:** dry-run default · typed confirm token · back-up-before-mutate · never hard-delete ·
  broker_sync (IBKR truth) never written · nothing synthetic enters `data/fills`/scoring.

---

## 1. Root cause (re-verified; EXIT-AUDIT citations still hold)

The 2026-05-27 Epoch-3 start (commit `4c5cd3b`, "Paper Epoch 3: start full-production paper soak")
operationally flattened the broker book while CHAD's ledgers retained state. Every divergence since
is that unrecorded flatten **plus post-reset one-sided accumulation** — entries confirm in the paper
SIM but exits never fire (strategy exits are runtime-blind per the EXIT-AUDIT; the only close path,
the reconciler, is incidental and its SELLs are rejected as phantom). Result: CHAD's paper-SIM
ledger (`trade_closer` FIFO book) **over-counts the authoritative IBKR paper account**, and the
per-strategy guard tracks **none** of the real broker longs.

Live picture (drift v2 `runtime/position_guard_drift.json`, 2026-07-13T16:16Z): `drift_count=7`,
all `broker_untracked_position`, `qty_mismatch=0`, `phantom_guard_entry=0`, reconciliation
`status=GREEN`. Fill trail confirms the mechanism — e.g. IWM confirmed net-BUY 399 with **2,676
rejected SELLs**; TLT net 1,332; SVXY 246 — against broker truth far smaller.

**Durable-source fact (verified, reshapes the fix):** the guard is rebuilt every cycle **from the
`trade_closer` FIFO queues** by `chad/core/live_loop._rebuild_guard_from_paper_ledger:755` (faithful
sum, no broker clamp, overwrites wholesale), and `broker_sync|<sym>` is rewritten every cycle from
live IBKR by `_rebuild_guard_from_broker:987`. Therefore editing the guard alone is **not durable**
(reverts next cycle); the only durable target is `runtime/trade_closer_state.json`, and `broker_sync`
must never be written by hand.

## 2. D1 — forensic baseline (three-way decomposition, live @ 2026-07-13T16:32Z)

| symbol | broker (broker_sync) | guard open (strat) | FIFO lot book | excluded? | divergence provenance |
|--------|------:|------:|------:|---|---|
| TLT  | 420 | 0 | 1260 (gamma, 63 lots) | no (CHAD)  | post-reset SIM over-accumulation; broker holds 420, SIM counts 1260; exits rejected |
| IWM  | 200 | 0 | 392 (gamma, 49)  | no (CHAD)  | same; 2,676 rejected SELL qty (phantom exits) |
| UNH  | 207 | 0 | 399 (gamma, 50)  | no (CHAD)  | same; 867 rejected SELL |
| V    | 154 | 0 | 301 (gamma, 43)  | no (CHAD)  | same |
| SVXY | 16  | 0 | 256 (gamma, 26)  | no (CHAD)  | same; largest SIM/broker ratio |
| LLY  | 130 | 0 | 306 (gamma, 65)  | **yes** (operator, exclusion_policy 05-06) | operator pre-existing broker position; CHAD FIFO lots are stray SIM |
| MSFT | 15  | 0 | 27 (alpha 23 + gamma 4) | **yes** (operator, 04-01) | operator pre-existing; stray CHAD SIM lots |
| SPY  | 0   | 0 | 312 (gamma, 24)  | **yes** | broker holds nothing (phantom); excluded |
| BAC  | 0   | 0 | 253 (gamma, 23)  | **yes** | phantom; excluded |

Provenance summary: the pre-reset longs were already cleared from the FIFO (oldest current lots are
07-08+); the **current** divergence is entirely post-reset SIM-vs-broker. "Which fills the ledger
counts that the flatten erased" → the ongoing SIM over-count vs the real account. "Which broker
holdings the guard missed" → the 5 CHAD longs the broker actually holds (broker_sync
`closed_by=strategy_ownership_assumed`) with **zero** guard attribution. (FIFO counts rise between
runs — the accumulation is live.)

## 3. D2 — disposition (CHOSEN: HYBRID)

**Chosen: HYBRID = rebaseline the SIM over-count/phantoms + adopt broker truth for the CHAD symbols.**

Per symbol:
- **CHAD, broker_qty>0** (IWM, SVXY, TLT, UNH, V) → **ADOPT**: replace the over-counted FIFO lots
  with ONE reconciliation seed lot at broker truth (dominant FIFO strategy = gamma), marked
  `provenance=UNATTRIBUTED_EPOCH3_ACCUMULATION` + `pnl_untrusted`, and open the matching guard entry.
- **CHAD, broker_qty==0** (phantom) → **REBASELINE**: remove the FIFO lots.
- **operator-EXCLUDED** (LLY, MSFT, SPY, BAC) → **REBASELINE**: purge stray CHAD FIFO lots; **never
  adopt** (the broker position is the operator's; `provenance=OPERATOR_FLATTEN`).

Argued against the three constraints:
- **Evidence preservation** — every mutated file is `.bak`-copied first; nothing hard-deleted;
  `processed_fill_ids` retained (old fills never re-open); all removed lot bodies preserved in a
  reconciliation ledger + the signed report.
- **Scoring integrity** — **nothing synthetic is written into `data/fills`** (the scored input), so
  no reconciliation row can enter `effective_trades` by construction. Belt-and-suspenders, every
  synthetic record and the adoption seed lot carry `pnl_untrusted=true` — the canonical, fail-closed
  Stage-2 exclusion (`chad/analytics/trade_stats_engine._is_untrusted:111`; provenance strings alone
  exclude nothing).
- **Exit-overlay correctness** — after recon, the guard shows `gamma|<sym>` open at broker truth for
  the 5 CHAD symbols, so the exit overlay's broker-confirmation (`broker_sync`) passes and it can
  manage true per-position state; the 2 operator symbols correctly stay unmanaged.

**Rejected — (a) pure REBASELINE:** clears the FIFO but leaves the broker-held CHAD longs
un-adopted → guard stays empty → the exit overlay stays blind to the real positions (correctness
constraint fails).

**Rejected — (b) pure ADOPT:** seeds the guard from `broker_sync` but leaves the FIFO over-counted
(1260) → the next-cycle rebuild (`_rebuild_guard_from_paper_ledger`) **reverts** the guard to the
FIFO sum, re-creating `qty_mismatch`, and future SELLs pair against phantom lots (corrupt realized
PnL). Unstable — verified against the rebuilder.

## 4. D3 — the guarded tool

`scripts/reconcile_ledger_to_broker.py` — **dry-run by default** (prints the full plan, mutates
nothing). `--execute` requires the typed token `--confirm RECONCILE-LEDGER-TO-BROKER` **and** passes
the fail-closed gates below. It backs up every touched file to `*.bak_ledger_recon_<stamp>`, emits a
`LEDGER_RECON_APPLIED` marker per row, writes a reconciliation ledger
`runtime/ledger_reconciliation_<stamp>.ndjson` and a signed report `reports/ledger_recon_<stamp>.json`
(git HEAD, gate results, per-symbol plan, backup SHA256s). **Idempotent** — a symbol already carrying
a reconciliation seed lot at broker truth is a no-op. It mutates only `runtime/trade_closer_state.json`
(durable source) and `runtime/position_guard.json` (immediate consistency, exact rebuild shape);
`broker_sync` is never written; no broker I/O; no orders.

### 4a. Fail-closed gate table (`--execute` only; dry-run never gates)

| gate | PASS condition | REFUSE (exit 2) | rationale |
|------|----------------|-----------------|-----------|
| `exec_mode` | execution mode ∈ {paper, dry_run} | live, or the mode check raises | never reconcile a book that could place live orders |
| `scr` | SCR state ∈ **{CONFIDENT, CAUTIOUS, WARMUP}** | UNKNOWN, PAUSED, or any other state | see WARMUP disposition below |
| `reconciliation` | `reconciliation_state.status` ≠ RED | status = RED (broker truth unavailable) | never mutate the book against unverifiable broker truth |

**SCR = WARMUP is a PERMITTED state for this tool** (disposition 2026-07-13, code
`_SAFE_SCR_STATES`, `scripts/reconcile_ledger_to_broker.py`). Live is currently WARMUP
(`sizing_factor≤0.1`, `paper_only`, effective trades below the graduation floor). In WARMUP the SCR
has **not** graduated to trusted live sizing, and — decisively — this tool writes **nothing** into
`data/fills` or `effective_trades` (every synthetic record and the adoption seed lot carry
`pnl_untrusted=true`, the canonical Stage-2 exclusion), so there is **no scored/effective edge for a
reconciliation to corrupt**. Blocking on WARMUP would only prevent a safe, evidence-preserving book
cleanup. **UNKNOWN and PAUSED remain REFUSED:** UNKNOWN signals SCR telemetry failure (state
indeterminate — the control plane is blind) and PAUSED signals an explicit operator halt; reconciling
under either would be acting against a broken or halted control plane.

> **Invocation note (F1):** the documented `python3 scripts/reconcile_ledger_to_broker.py …` now
> works from **any** CWD. The script prepends the repo root (resolved from `__file__`) to `sys.path`
> at import time, so the lazy `chad.*` imports inside the gates resolve even though `chad` is not
> pip-installed and `scripts/` (not the repo root) is `sys.path[0]` under the documented invocation.
> Previously the `exec_mode` gate caught the resulting `ModuleNotFoundError` and fail-closed, so the
> documented command refused before it could reach broker truth.

> **Freshness note (F3):** the plan is derived at **run time** from live inputs on every invocation —
> guard + FIFO re-read from disk, exclusions + marks re-loaded, broker truth + queues recomputed,
> `compute_plan()` a pure recompute; no plan is persisted and reloaded. A post-close dry-run therefore
> reflects **that session's** broker truth. NB: symbols the operator owns (`exclusion_policy` /
> `_EFFECTIVE_NON_CHAD_SYMBOLS` — currently includes BAC and SPY) classify as `REBASELINE_EXCLUDED`
> even when newly broker-held; they are **never** adopted (the broker position is the operator's).

## 5. Execution plan (awaiting typed operator GO — Channel 1)

```
# 1. Inspect the plan (safe, read-only):
python3 scripts/reconcile_ledger_to_broker.py

# 2. Apply (only after review):
python3 scripts/reconcile_ledger_to_broker.py --execute --confirm RECONCILE-LEDGER-TO-BROKER
```

No service restart is required; the next live-loop cycle rebuilds the guard from the reconciled FIFO.

## 6. Rollback

Restore the timestamped backups the tool wrote:
```
cp runtime/trade_closer_state.json.bak_ledger_recon_<stamp> runtime/trade_closer_state.json
cp runtime/position_guard.json.bak_ledger_recon_<stamp>     runtime/position_guard.json
```
`broker_sync` was never touched, so broker truth needs no restore.

## 7. Verification (post-execute)

- `reports/ledger_recon_<stamp>.json` shows `lots_before → lots_after` and per-symbol ADOPT/REBASELINE.
- Next cycle: drift v2 `broker_untracked_position` → **0 for the 5 CHAD symbols** (LLY/MSFT remain,
  benign — operator-excluded); `qty_mismatch` stays 0; reconciliation not RED.
- Exit-overlay ACTIVE-flip **criterion 3** satisfied: `qty_mismatch==0` (already true) AND the guard
  now reflects broker truth so the overlay sees true per-position state for the CHAD longs.
- Synthetic records excluded from `effective_trades` (SCR `excluded_untrusted`), never in `data/fills`.

## 8. Standing rule this PA establishes

**Any future operator broker flatten MUST write a matching CHAD ledger event in the same
transaction** (a `trade_closer` rebaseline + guard close for each flattened symbol), so the SIM
ledger can never again silently diverge from broker truth. Absent that, this reconciliation is a
recurring manual chore; with it, the drift class is closed at the source.

**No code deployed. Tool not executed against runtime. `--execute` awaits typed operator GO.**
