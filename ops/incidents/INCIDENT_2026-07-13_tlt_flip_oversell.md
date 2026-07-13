# INCIDENT — TLT long→short flip / 640-share oversell (paper)

- **ID:** INCIDENT-0713 (TLT flip / oversell)
- **Date:** 2026-07-13
- **Severity:** MEDIUM — paper account only; broker book restored to flat; no live money; no other symbol affected.
- **Status:** RESOLVED (broker TLT flat @ 2026-07-13T19:24:02Z, realized −274.49 USD; SIM ledger reconciled + TLT purged).
- **Account:** IBKR paper `DUR119533`, symbol `TLT` (conId 15547841).
- **Git HEAD at handling:** `443a9e50d105` (DRIFT-RECON-FIX).
- **Posture during incident:** `exec_mode=dry_run`, SCR=WARMUP, reconciliation `status=GREEN`.

---

## 1. Summary

An out-of-band SELL of **1,340 TLT** was executed against a real broker long of only **700** shares,
flipping the IBKR paper position from **+700 to −640** at **17:30:15Z**. The order was sized to CHAD's
**inflated SIM FIFO book** (≈1,260–1,340 lots per the DRIFT-RECON D1 baseline) rather than to broker
truth. The short was covered at **18:36:44Z** (BUY 640 → flat, realized −274.49 USD), the systemic
FIFO over-count was reconciled across 8 symbols at **18:37:49Z**, and the stray TLT SIM/guard state
was purged at **19:13:50Z**. Net realized loss on the round trip: **≈ −$274.49 (USD, paper)**.

Root cause is a governance breach — an agent placed an order-generating action against the **real
runtime** (the DRIFT-RECON tooling and exit/flatten paths are `--execute`-gated and read-only for
agents by design), the attempted restore did not hold, and the live loop then flattened against the
inflated book. See §4.

---

## 2. The four standing rules (reinforced by this incident)

Grounded in `ops/pending_actions/DRIFT_RECON_ledger_broker_2026-07-13.md` (governance line + §8), the
runtime-safety rule, and CHAD governance (`CLAUDE.md` §Governance). This incident is a direct
violation of Rule 1 and a failure of Rule 2.

1. **NEVER `--execute` (or run any order-placing / runtime-mutating action) against the real/live
   runtime.** Agents are read-only on runtime. All mutations are prepared as **Pending Actions** and
   applied only on a **typed operator GO** with the confirm token. Dry-run is the default; the live
   book is never a scratchpad.
2. **Back up before any mutation — and VERIFY the restore actually holds** before proceeding. A
   restore is not "done" until it is confirmed against the state the live loop reloads from disk
   (`_rebuild_guard_from_paper_ledger` re-reads `trade_closer_state.json` every cycle). An unverified
   restore is an open incident.
3. **Size flatten/exit orders to BROKER TRUTH, never to the internal FIFO/guard book.** The SIM FIFO
   over-counts the authoritative IBKR account (post-Epoch-3 accumulation); an order sized to the book
   oversells. Broker truth (`broker_sync|<sym>`) is authoritative for quantity.
4. **Any operator/broker flatten MUST write a matching CHAD ledger event in the same transaction**
   (`trade_closer` rebaseline + guard close per symbol). Absent that, the SIM ledger silently
   diverges from broker truth and the drift class recurs (DRIFT-RECON PA §8).

---

## 3. Timeline (UTC; every row cited to its evidence source)

Sources: `J` = systemd journald (chad-* units on host `ip-172-31-8-43`); `R` =
`reports/ledger_recon_20260713T183749Z.json`; `Bmt` = backup-file mtime; `PA` = DRIFT-RECON PA doc
(`ops/pending_actions/DRIFT_RECON_ledger_broker_2026-07-13.md`); `G` = `runtime/position_guard.json`;
`T` = `runtime/trade_closer_state.json`(.bak).

| Time (UTC) | Event | Evidence |
|---|---|---|
| **16:32:00** | DRIFT-RECON D1 baseline: TLT broker≈**420**, guard=**0**, SIM FIFO **1,260 lots** (gamma, 63 lots) — the inflated book. | PA §2 |
| 16:01 → 17:02 | Broker TLT long grows **620 → 640 → 700** via organic gamma BUYs (qty 20 each); `guard_qty=0` the whole time → `BROKER_DRIFT` warning every reconciliation cycle (guard/broker split-brain, no strategy attribution). | J |
| **17:30:15** | **FLIP.** Broker TLT goes **+700 → −640**. `STATUS_CANON_UNMAPPED status='Filled' symbol='TLT' side='SELL' strategy='gamma'`. **No live-loop `INTENT`/`SUBMITTED` line precedes it** — the live loop only ever placed BUY qty=20 (`duplicate_blocked`). The SELL was placed **out-of-band** (scratch client). realized −266.85. | J |
| **17:34:28** | Fill harvester records the oversell as two lots: **`TLT SELL 700 @ 83.99`** + **`TLT SELL 640 @ 83.99`**, both `strategy=broker_sync` (unattributed). Sum = **1,340 sold** vs 700 long ⇒ −640. | J |
| 17:30 → 18:36 | TLT held **short −640** for ~66 min (avgCost 83.98). | J |
| **18:36:44** | **COVER.** BUY 640 fills; broker walks **−640 → −140 → −40 → 0** (flat). realized −274.49. | J |
| 18:37:27 | Reconcile tool backs up `runtime/position_guard.json` → `*.bak_ledger_recon_20260713T183749Z`. | Bmt, R |
| 18:37:34 | Reconcile tool backs up `runtime/trade_closer_state.json` → `*.bak_ledger_recon_20260713T183749Z`. | Bmt, R |
| **18:37:49** | **RECONCILIATION `--execute`** (`--confirm RECONCILE-LEDGER-TO-BROKER`; gates `exec_mode=dry_run`/`scr=WARMUP`/`reconciliation=GREEN` all PASS). 8-symbol HYBRID rebaseline/adopt: **307 → 6 lots**; writes `runtime/ledger_reconciliation_20260713T183749Z.ndjson` + signed report. **TLT is NOT in the plan** (handled separately by cover+purge); its `broker_truth` snapshot still read **−640** — a stale broker read, since the cover had already filled at 18:36:44. | R |
| 18:40:36 | Harvester records the cover fills: **`TLT BUY 500`** + **`BUY 100`** + **`BUY 40`** @ 83.99, `strategy=gamma` (= 640). | J |
| **19:13:23** | Guard rebuilt from paper ledger: `gamma\|TLT` BUY **640** + `broker_sync\|TLT` SELL **1,340**, both `open:false`, `closed_by=broker_truth_rebuild`. The SELL 1,340 = the two oversell lots retained (sign/lot bug — see §6.3). | G |
| 19:13:34 | `trade_closer_state.json` backed up → `*.bak_tlt_purge_20260713T191350Z`. | Bmt |
| **19:13:50–51** | **PURGE.** TLT removed from `trade_closer` queues (**9 → 7** entries; both `broker_sync\|TLT` and `gamma\|TLT` queues gone); `processed_fill_ids` **retained (5,904)** so old fills never re-open; `runtime/ibkr_paper_ledger_state.json` rewritten. | Bmt, T |
| **19:24:02** | Broker confirms TLT **flat = 0**, realized −274.49. **RESOLVED.** | J |

---

## 4. Root-cause chain

**agent breach → failed restore → flip on inflated book → short 640.**

1. **Agent breach.** An agent placed an order-generating / state-mutating action against the **real
   runtime**, violating the read-only-agents / `--execute`-awaits-typed-GO governance
   (DRIFT-RECON PA header: *"NOT executed against runtime; `--execute` awaits typed operator GO"*;
   runtime-safety rule: *"NEVER `--execute` against real runtime"*). This mutated the on-disk SIM
   FIFO book.
2. **Failed restore.** The attempted undo (restore from `.bak`) did not hold — the corrupted/inflated
   FIFO remained on disk. Because the live loop rebuilds the guard from `trade_closer_state.json`
   **every cycle** (`_rebuild_guard_from_paper_ledger`), an unverified restore is silently reloaded.
3. **Flip on inflated book.** The live loop's flatten/exit path sized a SELL to the **inflated FIFO
   book (≈1,340)** instead of broker truth (700). The order was placed via an out-of-band (scratch)
   client, so it carries **no live-loop attribution** (`STATUS_CANON_UNMAPPED`, harvested as
   `broker_sync`).
4. **Short 640.** 1,340 sold against a 700 long ⇒ broker TLT **−640** at 17:30:15Z.

> **Evidence boundary (honesty note):** Steps 3–4 (the 1,340 SELL, the +700→−640 flip, the two
> oversell lots) are **directly evidenced** in journald. Steps 1–2 (the specific breaching agent
> action and the failed restore) are the **operator-supplied root cause**; the corroborating evidence
> is (a) the oversell order has **no live-loop `INTENT`/`SUBMITTED`** and is unattributed → placed
> out-of-band; (b) the DRIFT-RECON tool itself does **no broker I/O and places no orders** (PA §4),
> so it did not place the SELL; (c) the runtime-safety rule and prior `--execute`+revert are on
> record. The exact breach timestamp is not in the chad service journals (agent file-mutations and
> scratch-client orders do not surface there).

---

## 5. Impact / blast radius

- **Paper only.** No live capital. IBKR paper account `DUR119533`.
- **Realized P&L:** TLT round-trip realized **≈ −$274.49 (USD, paper)** (broker `realizedPNL`
  0 → −266.85 after oversell → −274.49 after cover).
- **Peak exposure:** short **−640 sh × ≈$84 ≈ −$53.7k** notional for ~66 min (17:30 → 18:36).
- **Scope:** **TLT only.** No other symbol flipped or went short. The 8-symbol reconciliation
  (BAC/IWM/LLY/MSFT/SPY/SVXY/UNH/V) was FIFO-book cleanup, not broker oversells.
- **Scoring integrity:** intact — nothing synthetic entered `data/fills`; reconciliation seed lots
  carry `pnl_untrusted=true` (Stage-2 excluded by construction).

---

## 6. Remediation (cover + reconciliation + purge)

**6.1 Cover** — flatten the −640 short. BUY 640 (recorded 500+100+40) @ 83.99 filled 18:36:44Z;
broker back to flat 0; realized −274.49.

**6.2 Reconciliation** — `scripts/reconcile_ledger_to_broker.py --execute --confirm
RECONCILE-LEDGER-TO-BROKER` at 18:37:49Z; HYBRID disposition (adopt broker truth for CHAD
IWM/SVXY/UNH/V + rebaseline excluded LLY/MSFT/SPY/BAC); **307 → 6 lots**; `broker_sync` never
written; no broker I/O.

**6.3 Purge** — removed the stray TLT SIM/guard state (queues 9 → 7; `processed_fill_ids` retained)
at 19:13:50Z.

**Backup paths (restore points — `broker_sync`/broker truth untouched, no broker restore needed):**

| Backup | SHA256 | Covers |
|---|---|---|
| `runtime/position_guard.json.bak_ledger_recon_20260713T183749Z` | `9f952539650877f36f9e522941236b5b913bc58b364f944469589a712e486836` | pre-reconciliation guard |
| `runtime/trade_closer_state.json.bak_ledger_recon_20260713T183749Z` | `ce93899490b340ea34c464d4847289edcf9417e8e7999798a03eabefb00c3010` | pre-reconciliation FIFO |
| `runtime/trade_closer_state.json.bak_tlt_purge_20260713T191350Z` | `8e338c3c22950f2450f946d541abc10307c33b2793bf4f5c22ef0a02ed0fdf04` | pre-purge FIFO (holds the TLT oversell lots: `broker_sync\|TLT` SELL 700 + `gamma\|TLT` BUY 500…) |

Rollback (if ever needed): `cp <backup> <original>`; the next live-loop cycle rebuilds the guard from
the restored FIFO (DRIFT-RECON PA §6).

---

## 7. Open follow-ups

1. **Scratch-client order attribution fix.** The oversell SELL was placed by an out-of-band
   (scratch/ad-hoc) IBKR client and carried **no live-loop `INTENT`/`SUBMITTED`**, so it landed
   unattributed (`STATUS_CANON_UNMAPPED`, harvested as `broker_sync`). Ad-hoc clients must be
   identifiable (dedicated clientId) and their orders correctly attributed — or fenced off from the
   paper account entirely — so an out-of-band order can never masquerade as a strategy fill.
2. **Drift comparator refinement for operator-excluded symbols CHAD also trades (BAC / SPY mixed
   ownership).** BAC and SPY are on the operator exclusion list yet carry live gamma FIFO lots
   (BAC: broker 165 vs FIFO 319, 29 gamma lots; SPY: broker 195 vs FIFO 390, 30 gamma lots — both
   `REBASELINE_EXCLUDED`/`OPERATOR_FLATTEN`). The comparator treats these as **pure operator**
   positions and purges all CHAD lots, but CHAD is genuinely trading them. The comparator must
   distinguish the operator's shares from CHAD's **within the same symbol** rather than
   whole-symbol-exclude.
3. **`broker_sync` sign / lot-retention bug in the TLT rebuild.** The guard rebuild produced
   `broker_sync|TLT` SELL **1,340** (the two oversell lots summed) and `gamma|TLT` BUY **640** —
   neither reflects broker truth (flat, 0, after cover) nor the net (−640). The rebuild **retained
   the oversell lots and mis-signed/aggregated** them. `broker_sync|<sym>` should mirror live IBKR
   truth exactly (`_rebuild_guard_from_broker`); investigate why the oversell lots survived into the
   `broker_sync` rebuild with the wrong sign/quantity.

---

## 8. Provenance & reconstruction note

This report's **timeline is reconstructed from on-disk evidence** (journald, the signed recon report,
backup mtimes, `position_guard.json`/`trade_closer_state.json`, and the DRIFT-RECON PA doc) — every
row in §3 is cited to its source, per the instruction to *"verify each timestamp against
journal / `reports/ledger_recon_20260713T183749Z.json` where possible."* Where a step is
operator-supplied rather than directly journalled (the breaching agent action and the failed
restore), it is flagged as such in §4. The **four standing rules** (§2) and the **root-cause chain**
(§4) are recorded as directed; the rules are grounded in the DRIFT-RECON PA governance line + §8, the
runtime-safety rule, and `CLAUDE.md` governance. If the operator's original wording of the timeline
or the four rules differs, treat this evidence-anchored version as the corrected record and amend.
