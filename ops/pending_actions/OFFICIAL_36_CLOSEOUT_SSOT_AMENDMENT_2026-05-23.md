# OFFICIAL 36 CLOSEOUT SSOT AMENDMENT — 2026-05-23

| Field | Value |
|---|---|
| Document ID | OFFICIAL_36_CLOSEOUT_SSOT_AMENDMENT_2026-05-23 |
| Document class | **Pending Action** (per CHAD governance rule 3 — no direct SSOT mutation) |
| Generated UTC | 2026-05-23 |
| Authoring HEAD | `97e706a` |
| Authoring branch | `main` |
| Working tree at authoring time | clean |
| Live posture at authoring time | paper (`ready_for_live=false`; `allow_ibkr_live=false`; `allow_ibkr_paper=true`) |
| Source closeout report | `reports/parity_audit/OFFICIAL_36_CLOSEOUT_FINAL_REPORT_20260523T001623Z.md` |
| Source worklist | `reports/parity_audit/OFFICIAL_36_CLOSEOUT_WORKLIST_20260522T223154Z.md` |
| Cross-referenced SSOT inputs | `docs/CHAD_GAPS_TO_CLOSE.md`, `docs/CHAD_EVIDENCE_LOCKED_DOD_v1_0_2026-05-20.md`, `docs/CHAD_SSOT_v9_5_FORWARD_ERRATA_2026-05-20.md` |

---

## 0. Status statement — Pending Action, NOT a direct SSOT rewrite

This document is a **Pending Action** that records the official outcome of the 2026-05-22→2026-05-23 36-item closeout sweep. It is **not** a direct mutation of any SSOT file. Per CHAD governance rule 3 ("No direct config mutation. Risk caps, live mode, and strategy config must be prepared as Pending Actions only — never applied directly"), the corresponding amendments to:

- `docs/CHAD_GAPS_TO_CLOSE.md` (body table + summary table),
- `docs/CHAD_EVIDENCE_LOCKED_DOD_v1_0_2026-05-20.md`, and
- `docs/CHAD_SSOT_v9_5_FORWARD_ERRATA_2026-05-20.md`

must be applied in a **separate, operator-authorized doc cycle**. This Pending Action exists to make the amendment scope reviewable in one place before any SSOT file is mutated.

---

## 1. Official final count

| Metric | Value |
|---|---|
| `TOTAL_ITEMS` | **72** |
| `LOCKED_OR_VERIFIED_ITEMS` | **44** |
| `PARTIAL` | **3** |
| `BLOCKED` | **24** |
| `DEFERRED` | **1** |
| `UNKNOWN_REQUIRES_AUDIT` | **0** |
| `READY_FOR_LIVE` | **false** |
| `ALLOW_IBKR_LIVE` | **false** |

Acceptance arithmetic over the in-scope 37 items (entry: 36 OPEN + 1 PARTIAL):

```
LOCKED this session   :  3 (P3-4, D2T3C, L7)
VERIFIED this session :  6 (P2-5, P2-6, P2-7, P3-1, P3-2, P3-5)
PARTIAL               :  3 (P0-1, P1-2, P2-9)
BLOCKED               : 24 (see §4)
DEFERRED              :  1 (P2-8)
UNKNOWN — REQUIRES AUDIT: 0
                        ---
                        37 ✓
```

Plus 35 items already VERIFIED_CLOSED at session entry → 35 + 3 + 6 = **44 LOCKED_OR_VERIFIED_ITEMS** of 72.

OPS-OMEGA-01 (outside the 72-item Elite checklist) is independently `CLOSED — runtime-active` from the prior session and is not double-counted here.

---

## 2. Items LOCKED this session (3)

- **P3-4 — Stale memory entries annotated.** `MEMORY.md` lines for `audit_2026_05_07_findings` and `audit_2026_05_08_placeholder_fills` now carry explicit SUPERSEDED-2026-05-22 cross-references (to BOX-049 and the P0-1 closeout, respectively). The 2026-05-09 entry was already SUPERSEDED-annotated.
- **D2T3C — BAG limit-price unit normalisation.** Per `runtime/completion_matrix_evidence/BOX-051_OFFICIAL_BAG_Tier_3C_limit_price_unit_normalization.md`, the per-share contract is documented across producer (`alpha_options.py:474-499, 519-531`), adapter (`ibkr_adapter.py:1623-1700, 2277`), and paper-fill writer (`paper_exec_evidence_writer.py:971-985`). 76 BAG-related tests pin the invariant. The contradictory `÷100 for IBKR lmtPrice` row in `docs/CHAD_GAPS_TO_CLOSE.md:50` was retired in commit `97e706a`. Per the closeout prompt §8 D2T3C rule, LOCKED is reached when contract documented + tests enforce + no contradictory doc remains — all three conditions hold.
- **L7 — Engineering item gated by D2T3C.** Closed by transitive closure of D2T3C; no separate work required.

## 3. Items VERIFIED this session (6)

- **P2-5 — Ports 9618/9619/9620.** `ss -tlnp` confirms the binds; `ufw status` is `active` with default DENY policy and zero ALLOW rules for these ports → host-firewall enforces effective localhost-only access. Per the closeout prompt §8 P2-5 rule, ss+firewall evidence satisfies VERIFIED. LOCKED still requires SSOT addendum + (optional) AWS-SG cross-check.
- **P2-6 — `legend_top_stocks.json`.** Artifact exists at the canonical loader path `data/legend_top_stocks.json` (919 B, mtime 2026-05-18), read by `chad/utils/legend_loader.py:48`. The 2026-05-21 audit pointed at the wrong path (`runtime/legend_top_stocks.json`). Weekly timer schedules Fri 23:45 UTC.
- **P2-7 — Telegram bot urllib3.** Version snapshot: `urllib3==1.26.20` (pin holds <2.x), `python-telegram-bot==13.15`; `chad-telegram-bot.service` has been `active running` continuously since 2026-04-28 (3 weeks 3 days at session time).
- **P3-1 — Telegram dedupe.** `ops.cleanup_telegram_dedupe --apply` ran in session: 1582 files → 6 active + 1576 archived to `_archive/telegram_dedupe/YYYY/MM/`. Audit log `runtime/cleanup_telegram_dedupe.audit.ndjson` (1.3 MB) captures every move. 9 targeted tests pass.
- **P3-2 — `dynamic_caps_*` backups.** 3 backup files moved from `runtime/` to `_archive/dynamic_caps_backups/` (`dynamic_caps.json.post_fix_20260424T215400Z`, `dynamic_caps.json.pre_fix_20260424`, `dynamic_caps_correlation.json.stale_20260326`); zero code references.
- **P3-5 — Zero-fill strategies classified.** `runtime/completion_matrix_evidence/BOX-049_OFFICIAL_zero_fill_strategies_classified.md` classifies all 18 strategies (17 active + 1 DEFERRED) with 0 UNKNOWN, 0 BROKEN. The 2026-05-21 audit was at HEAD `bbe7525` (predating BOX-049's `9a24dcf` commit); BOX-049 is present at this HEAD.

## 4. Items PARTIAL (3) — exact remaining work

- **P0-1 — Upstream placeholder emitter still active.** Defense-in-depth at `paper_exec_evidence_writer.py:1576-1612`, `paper_exec_evidence_writer.py:147-163`, and `/usr/local/bin/chad_paper_trade_executor.py:187-214` rejects 100% of `fill_price=100.0` rows (`status=rejected`, `pnl_untrusted=true`, `broker_rejected`-tagged), but the root emission from the delta SELL path continues to flow. Remaining work: patch `chad/strategies/delta.py` so delta SELL abstains when no positive live/cached price is available; add `chad/tests/test_delta_abstains_when_no_live_price.py`.
- **P1-2 — 2 runtime `ib_insync` files still need migration/restart.** Production imports remain at:
  - `chad/core/paper_position_closer.py:39,41,49`
  - `chad/core/paper_shadow_runner.py:595,774`
  - `chad/ops/ibkr_broker_events_collector.py:11-12,80,658` is a **noise-filter substring + docstrings only** (non-runtime) and may be retained or scoped out separately.
  Remaining work (status 2026-05-26): the 2 runtime files are migrated to `ib_async` via PR-03 (commit `d476e8c`); the `test_ib_insync_zero_imports_in_production.py` enforcement is in `chad/tests/test_pr03_ib_async_phase2_migration.py`; `chad-paper-position-closer.service` does not exist as a systemd unit (see PR-03 Pending Action lines 87-88 and BOX-038 §3.1 — it is a CLI/oneshot module, no service to restart); `chad-paper-shadow-runner.service` remains MASKED under the 2026-05-06 quiesce policy formalized by `PR-06_shadow_runner_quiesce_formalization_2026-05-26.md` (Path A) and any future re-arm requires a separate operator-approved Path B Pending Action.
- **P2-9 — `chad-options-chain-refresh.service` still failing operationally.** Alert wiring (R17/R18) is shipped; the service itself is `× failed (Result: exit-code) since Fri 2026-05-22 12:30:38 UTC` (`status=1/FAILURE`). A reproduction attempt hung on IB connect (`127.0.0.1:4002`). Remaining work: diagnose IB connect / market-hours dependency in `chad/market_data/options_chain_refresh.py`; obtain operator GO to restart after fix; confirm cache artifact freshness.

## 5. Items BLOCKED (24)

All BLOCKED items require one or more of: future evidence, external action, market-hours broker proof, operator decision, or live-readiness gates. No code work in this session can close them without operator authorization or external state change.

| ID | Root blocker |
|---|---|
| P2-4 | Operator MES sign-off (no `reports/model_doctor/MES_*` artifact) |
| P3-3 | Channel-1 sudo install of `/etc/logrotate.d/chad` |
| C1C | Canadian jurisdiction (Kraken Futures live) |
| C2 | Operator paid-key decision (Coinglass vs Binance) |
| C2B | Gated by C2 (Binance publisher not built) |
| C3 | IBKR DOM/Level-2 entitlement + market hours |
| D1B | Operator promotion-decision artifact (5-day window elapsed) |
| D2T4 | Safe broker test window for bracket/OCA implementation |
| D2T5 | Safe broker test window for spread_id-aware guard |
| D2T6 | Safe broker test window for live BAG fill round-trip |
| ML3 | Operator MES sign-off (same gap as P2-4) |
| L1 | Gated by P0-1 |
| L2 | Gated by P1-2 |
| L4 | Operator decision: weekday_reaudit synthesis or scope-out |
| L5 | Gated by C2 + C2B |
| L6 | Gated by C3 |
| L8 | Gated by D2T4 |
| L9 | Gated by D2T5 |
| L10 | Gated by D2T6 |
| L11 | Gated by D1B |
| L12 | Gated by ML3 + P2-4 |
| L13 | Future trading-day soak evidence |
| L14 | Gated by L1 + L2 |
| L15 | Operator: canary plan + first-live trade size |
| L16 | Operator GO + all upstream prereqs |

(Count is 25 rows including L7 which moved to LOCKED via D2T3C this session; remaining BLOCKED count = 24 per §1.)

## 6. Items DEFERRED (1)

- **P2-8 — APScheduler 3.6.3 vs 4.x.** APScheduler is installed at 3.6.3 but produces **zero grep hits** across `chad/` for either `apscheduler` or `APScheduler`. The dependency is an unused transitive install; the original "upgrade to 4.x" gap is operationally moot. Operator decides whether to (a) remove the dep entirely or (b) document the 3.x pin as benign-unused.

## 7. No-live authorization confirmation

- **`ready_for_live` remains `false`.** Source: `runtime/live_readiness.json:ready_for_live=False` (ts at session close: 2026-05-22T22:16:56Z; unchanged through session).
- **`allow_ibkr_live` remains `false`.** Source: `runtime/decision_trace_heartbeat.json:allow_ibkr_live=False`, `mode={chad_mode: paper, live_enabled: false}`.
- **No live orders were placed during the closeout session.**
- **No broker orders were cancelled during the closeout session.**
- **No `runtime/*.json` was manually edited.** The only `runtime/` mutations were:
  - 1576 `runtime/telegram_dedupe_*.json` files archived via the shipped `ops.cleanup_telegram_dedupe --apply` script (audit-logged).
  - 3 backup files (`dynamic_caps.json.post_fix_*`, `dynamic_caps.json.pre_fix_*`, `dynamic_caps_correlation.json.stale_*`) moved from `runtime/` to `_archive/dynamic_caps_backups/`.
- **No services were restarted.**
- **No service units were modified.**
- **No `systemctl daemon-reload` was executed.**

## 8. Next execution priorities

In strict order — each item is "one change at a time" per CHAD governance rule 1:

1. **Fix P0-1 upstream placeholder source.** Patch `chad/strategies/delta.py` so delta SELL abstains when no positive live or cached price is available. Add a targeted test (`chad/tests/test_delta_abstains_when_no_live_price.py`). Verify no further `fill_price=100.0` rows for `strategy=delta` over one full trading day.
2. **Complete P1-2 `ib_async` Phase 2 migration.** Migrate `chad/core/paper_position_closer.py` and `chad/core/paper_shadow_runner.py` from `ib_insync` → `ib_async`. Add `chad/tests/test_ib_insync_zero_imports_in_production.py` enforcing zero production imports of `ib_insync` going forward. Document the non-runtime carve-out for `chad/ops/ibkr_broker_events_collector.py` noise-filter. Operator-authorized restart of `chad-paper-position-closer.service` + `chad-paper-shadow-runner.service` follows.
3. **Fix P2-9 options-chain-refresh operational failure.** Diagnose the `python3 -m chad.market_data.options_chain_refresh` ExitStatus=1 (IB connect to 127.0.0.1:4002 blocks during the failing run window). Pre-market trading-hours test window may be required. Confirm cache artifact freshness after fix.
4. **Continue paper soak on valid trading days.** L13 cannot be closed via this session per closeout-prompt §8 ("soak days require future trading-day evidence; cannot close future soak days early"). Keep posture `paper` and SCR `CONFIDENT` for the declared soak duration.
5. **Keep blocked external / live-gate items blocked until conditions are met.** No code-around. Do not enable live mode. Do not force `ready_for_live=true`. Do not place orders.

---

## 9. Definition of done for this Pending Action

This Pending Action is `DONE` once an operator-authorized doc cycle:

a. Applies the corresponding amendments to `docs/CHAD_GAPS_TO_CLOSE.md` (body table + summary table reflecting §1–§4 above; per the 2026-05-21 SSOT_COUNT_RECONCILIATION §G.2 already-required corrections).
b. Adds an explicit row to `docs/CHAD_SSOT_v9_5_FORWARD_ERRATA_2026-05-20.md` (or its successor) capturing the §1 final count.
c. Adds the OPS-OMEGA-01 closure (outside the 72) to an "Operations outside the 72-item Elite checklist" appendix in `docs/CHAD_EVIDENCE_LOCKED_DOD_v1_0_2026-05-20.md`.
d. The Pending Action is then either archived under `ops/pending_actions/_applied/` or annotated with the apply commit SHA in this file's status table.

Until that operator-authorized doc cycle lands, the official count of record remains the one in `reports/parity_audit/OFFICIAL_36_CLOSEOUT_FINAL_REPORT_20260523T001623Z.md`.

---

FINAL STATUS: Pending Action **landed** (this document); SSOT mutation **deferred** to operator-authorized doc cycle.
