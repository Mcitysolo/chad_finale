# CHAD Unified SSOT v9.6
# Institutional Closeout — Forensic Audit Remediation Lock

**Version:** 9.6
**Date:** 2026-05-27 (UTC); locked 2026-05-28
**Supersedes (for forward reading):** `docs/CHAD_SSOT_v9_5_FORWARD_ERRATA_2026-05-20.md` and `docs/CHAD_UNIFIED_SSOT_v9_4_2026-05-18.md`. All prior cuts (v9.0 / v9.1 / v9.2 / v9.3 / v9.4 / v9.5-forward-errata) remain **FROZEN** historical artifacts; this document states forward corrections only and does not edit them.
**Lineage note:** The previous full SSOT was **v9.4 (2026-05-18)**; a **v9.5 forward-errata (2026-05-20)** sits between v9.4 and this cut. This document is numbered **v9.6** to avoid colliding with the existing v9.5 forward-errata. (Operator decision, 2026-05-28.)
**Predecessor reference commit:** `72f361fac544c82cd231d100b2d0bc86d428a990` (short `72f361f` — "Docs: close pre-SSOT elite upgrade gaps", 2026-05-17; the v9.3-era HEAD).
**HEAD commit at v9.6 lock:** `1626f2069fdf54656d399212fb36a132a00bb640` (short `1626f20`)
**Branch:** `fix/forensic-high-closeout-20260527`
**Test baseline:** 2639 collected; **2638–2639 passing** under `CHAD_SKIP_IB_CONNECT=1` (one known-intermittent test, `test_canonical_equity_source::test_canonical_sources_agree_within_skew_tolerance`, passes in isolation but flaps under full-suite parallelism — see §2.8).
**Lock type:** Institutional closeout — forensic audit remediation
**Live order posture:** No new live execution authorization granted by this SSOT.

---

## 0. Preamble and Version Delta

### 0.1 What the prior cuts were
- **v9.3 (2026-05-17)** — pre-SSOT elite upgrade gap closure; documented the Phase C max-buildable workstream, FMP intel integration, Phase D Item 1 observation-only veto, XGB veto governance, and Phase D Item 2 ladder Tiers 1–3B. Test baseline then: **2114 passing**.
- **v9.4 (2026-05-18)** — Remediation-Batch Consolidation forward-errata: GAP-001/002 behavioral pin, GAP-032 lint guard, baseline erratum (corrected a v9.3 test-count claim). HEAD `a5311d27`.
- **v9.5 forward-errata (2026-05-20)** — Stage-3 Completion-Matrix index (Boxes 001→044), carrying deferred operator decisions; explicitly a forward errata, not a full rewrite. HEAD `bbe7525`.

### 0.2 What v9.6 captures
v9.6 is the **Institutional Closeout SSOT**. It captures:

- The 2026-05-27 forensic full-system audit (HIGH=6, MED listed, LOW documented).
- The morning's HIGH-severity multi-phase fix (8 commits).
- The afternoon's institutional closeout pass (4 commits):
  - **Decision 2:** Mechanical evidence-gated soak rule (`ac04e16`)
  - **Decision 3:** Systemd alert template + port-binding invariant + pre-commit hook (`1e00568`)
  - **Decision 1:** position_truth_v2 engine design + schema + stub + validator + tests (`1626f20`)
  - **This document:** v9.6 SSOT cut (the commit that lands this file)

**Commit accounting (corrected to repo reality):** `git rev-list --count 72f361f..HEAD` = **69 commits**. Of those, the **most recent 12** are this audit-remediation batch (morning 8 + afternoon 4, detailed in §6). The other **57** are the v9.4 cut, the v9.5 forward-errata cut, the Stage-3 Completion-Matrix box work (Boxes 001–062), the PR-02…PR-12 series, the ib_async Phase-2 migration, and the Paper Epoch 3 start — all already documented in their own cuts. (The forensic audit's `DOC-RECONCILE-1` MED item flagged exactly this 57-commit documentation drift; this v9.6 cut closes it.)

### 0.3 What v9.6 is NOT
v9.6 is NOT a live-trading authorization. Specifically:
- It does NOT flip `ready_for_live`.
- It does NOT wire `position_truth_v2.json` into production (migration Step 1 of the design §12 plan is the next operator decision).
- It does NOT install the systemd alert template (the Channel 1 install runbook is the next operator decision).
- It does NOT close PORT-BINDING-1 for port 9618 (requires the Channel 1 `chad-backend` service restart).
- It does NOT implement the soak evaluator (the rule is locked; the evaluator is a follow-on PA).

### 0.4 Forensic audit context
Reference: `reports/forensic_audits/CHAD_FORENSIC_FULL_SYSTEM_AUDIT_20260527T142951Z.md`.
6 HIGH items identified; all 6 addressed by either landed code, landed Pending Actions, or both (§3.1).

### 0.5 Governance rules (unchanged from CLAUDE.md)
1. One change at a time; baseline → change → verify → proceed.
2. No full rewrites; surgical changes only.
3. No direct config mutation; Pending Actions only.
4. Verification sequence after every code change.
5. Never modify `runtime_FREEZE_*` or `data_FREEZE_*`.
6. Never modify systemd unit files without explicit instruction.
7. Never restart live services without explicit instruction.
8. Commit and tag git after each completed P0/P1/P2 item.
9. Always use `python3` via `/home/ubuntu/chad_finale/venv`.
10. `pytest` requires `CHAD_SKIP_IB_CONNECT=1`.

---

## 1. Mission and Architecture

### 1.1 Mission (unchanged)
CHAD (Compounding Hedge-fund Algorithmic Desk) is a multi-strategy, multi-broker
algorithmic trading desk operating in **PAPER** posture. Its mission is to compound a
paper book through a governed, fail-closed execution stack until a clean multi-session
soak and explicit operator GO authorize a live transition. Safety and auditability take
precedence over throughput at every layer.

### 1.2 Architecture stack (v9.6 — additions only over v9.3/v9.4/v9.5-errata)
The established hot-path stack (carried forward):

```
  1. Signal Layer            (strategies → RoutedSignal)
  2. Routing Layer           (signal_router, signal_guard, dedupe)
  3. Risk & Allocation Layer (dynamic_risk_allocator, profit_lock, tier_state)
  4. Execution Layer         (ibkr_adapter, paper_trade_executor, LiveGate)
  5. Attribution Layer       (paper_exec_evidence_writer, fills ledger)
  6. Reconciliation Layer    (reconcile_positions, position_guard, drift detector)
  7. Dashboard/Operator Layer(status server, metrics, Telegram)
  8. Governance Layer        (Pending Actions, SCR, stop_bus, epoch trackers)
```

**NEW v9.6 — Position Truth v2 Layer (DESIGN + STUB, NOT YET WIRED)** — inserted
logically between the Reconciliation Layer (6) and the Dashboard/Operator Layer (7):

```
  POSITION TRUTH V2 LAYER (DESIGN + STUB, NOT YET WIRED)
    chad/core/position_truth_engine.py
    chad/schemas/position_truth_v2.py
    chad/validators/position_truth_v2.py
    docs/design/POSITION_TRUTH_V2_ENGINE_DESIGN_2026-05-27.md
```

**NEW v9.6 — Service Reliability Layer (CHANNEL 2 LANDED; CHANNEL 1 INSTALL PENDING)** —
a horizontal layer crossing all hot-path layers:

```
  SERVICE RELIABILITY LAYER (CHANNEL 2 LANDED; CHANNEL 1 INSTALL PENDING)
    chad/ops/service_failure_alert.py
    ops/systemd_templates/chad-service-alert@.service
    chad/validators/port_binding.py
    config/port_binding_allowlist.json
    .git/hooks/pre-commit (installed)
```

**NEW v9.6 — Soak Governance (under Governance Layer):**

```
  SOAK GOVERNANCE
    ops/pending_actions/SESSION_SOAK_MECHANICAL_EVIDENCE_GATE_RULE_2026-05-27.md
    Evaluator: NOT YET IMPLEMENTED (default classification on RED = FAIL / NOT COUNTABLE)
```

### 1.3 Cross-cutting safety promises (v9.6 additions)
Carried forward from v9.3/v9.4:
- No live BAG execution authorization.
- No IBKR DOM authorization.
- No Kraken Futures live for Canadian deployment.
- No Coinglass dependency.
- No dynamic-scanner writeback.

New in v9.6:
- **No production wiring of `position_truth_v2.json`** — the engine reads its two
  sources but does not write to `runtime/` in any operator-unauthorized phase. The
  CLI is `--check`-only; `write_truth_v2()` is exercised only against test temp dirs.
- **No new `chad-*` service may bind to `0.0.0.0`** without an entry in
  `config/port_binding_allowlist.json` (enforced by the installed pre-commit hook).
- **No new `chad-*` service is considered complete** without an
  `OnFailure=chad-service-alert@%n.service` directive in its unit file (enforced by
  the Channel 1 install runbook checklist).

---

## 2. Current Runtime Truth — 2026-05-27 / lock 2026-05-28

All values below are read directly from `runtime/` at the v9.6 lock moment.

### 2.1 Equity, risk cap, weights
From `runtime/dynamic_caps.json` (ts 2026-05-28T02:30:30Z):
- `total_equity` = **$265,577.40**
- `portfolio_risk_cap` = **$13,278.87**
- `daily_risk_fraction` = **0.05**
- 16 normalized strategy weights present (alpha, alpha_crypto, alpha_futures,
  alpha_intraday, alpha_options, beta, beta_trend, delta, delta_pairs, gamma,
  gamma_futures, gamma_reversion, omega, omega_macro, omega_momentum_options,
  omega_vol). Two strategies carry `halt_clamp_applied=true` (delta, gamma_futures);
  regime_factor 1.25 across the book.

(Note: equity has grown from the ~$183,264 recorded in CLAUDE.md as of 2026-05-09 to
$265,577 at this lock — 19 days of continued paper compounding.)

### 2.2 SCR posture
From `runtime/scr_state.json` (ts 2026-05-28T02:30:40Z):
- `state` = **CONFIDENT**
- `sizing_factor` = **1.0**
- `paper_only` = **false**
- stats: `effective_trades`=199, `sharpe_like`=2.938, `win_rate`=0.7588,
  `total_pnl`=$18,138.88, `paper_trades`=4031, `max_drawdown`=-$290.00,
  `excluded_manual`=12, `excluded_untrusted`=321, `live_trades`=0.
- reason: "CONFIDENT: win_rate, sharpe, and drawdown all within confident band."

**Known intermittent UNKNOWN flutter:** during this closeout work, the SCR band was
observed flapping to UNKNOWN at least twice (≈02:00Z and ≈02:21Z reads) before
self-resolving to CONFIDENT. Per the saved operator memory *broker_latency_flutter_pattern*,
this is the stop_bus broker-latency rolling-average tripping (>2000ms) and auto-recovering
via `broker_latency_clean_streak=5` (~5 min); the SCR shadow may flap UNKNOWN concurrently.
It is **not** caused by Decisions 1/2/3, and SCR correctly fails closed (no decisions are
made on an UNKNOWN band). See §9.1.

### 2.3 Live readiness
From `runtime/live_readiness.json` (schema `live_readiness_state.v1`, ts 2026-05-28T02:25:01Z):
- `ready_for_live` = **false** ✓

### 2.4 Tier state
From `runtime/tier_state.json` (schema `tier_state.v2`, ts 2026-05-28T02:29:55Z):
- `tier_name` = **SCALE** ("Full stack — all 16 strategies, existing sizing pipeline unchanged")
- `current_equity_usd` = $265,577.40
- `tier_min_equity` = $160,000 / `tier_max_equity` = $10,000,000
- `previous_tier` = SCALE; `demotion_pending` = false

### 2.5 Portfolio snapshot
From `runtime/portfolio_snapshot.json` (ts 2026-05-28T02:29:51Z):
- `ibkr_equity` = **$265,392.82**
- `kraken_equity` = **$184.58**
- `coinbase_equity` = **$0.00**

### 2.6 Reconciliation
From `runtime/reconciliation_state.json` (ts 2026-05-28T02:29:59Z):
- `status` = **GREEN**, `drifts` = `[]`
- `counts` = `{chad_open: 19, chad_strategy_open: 18, broker_positions: 19}`
- `broker_source` = `ibkr:clientId=83`
- `chad_state_source` = `position_guard.json`
- `positions_truth.json` (schema `positions_truth.v1`, ts 02:30:40Z):
  `broker_authority_status`=**GREEN**, `truth_ok`=true.

### 2.7 Position truth v2 (NEW v9.6 — observation-only via CLI)
Engine CLI (`python3 -m chad.core.position_truth_engine --check`) at lock moment:
- `schema_version` = `position_truth_v2.v1`
- `engine_version` = `1.0.0`
- `global_authority_health` = **RED**
- `positions_count` = **19**
- merge-rule distribution = **19× M5** (all symbols fail-closed)
- `fail_closed_symbols` = all 19: AAPL, BAC, CVX, GLD, GOOGL, IEMG, KO, M2K, M6E,
  MCL, MES, MSFT, NVDA, PEP, QQQ, SPY, TLT, UNH, VWO
- source ages: ledger **533.4s**, snapshot **69.5s**

**Interpretation (important honesty note):** the RED reading is the
**`ledger_ttl` tuning sensitivity flagged in design §14**, not a real broker-truth
divergence. The ledger is event-driven; during the quiescent overnight window there were
no new fills for ~9 minutes, so the derived ledger timestamp (`max(opened_at_utc)`) aged to
533s — well past the *deliberately aggressive* proposed `ledger_ttl=60s`. Rule M5 therefore
fires `stale_source` for every symbol. An earlier read during Decision-1 work (ledger age
~31s, within TTL) classified all **19 symbols M1 (perfect agreement) → GREEN**. The
reconciliation layer (§2.6) independently confirms GREEN broker authority at this same
moment. This is precisely the kind of "routine race / quiescent-source" condition the engine
is designed to surface **explicitly** rather than silently guess — and it is the leading
input to the operator's `ledger_ttl` tuning decision (design §14, open questions).

No `runtime/position_truth_v2.json` artifact was written to production — this read came
from the CLI only.

### 2.8 Full test suite baseline
`CHAD_SKIP_IB_CONNECT=1 python3 -m pytest chad/tests/ -q`:
- **2639 collected.** The Decision-1 full run reported **2639 passed, 0 failed**; the
  v9.6-preflight run reported **2638 passed, 1 failed**. The single flapping test is
  `test_canonical_equity_source::test_canonical_sources_agree_within_skew_tolerance`,
  which **passes in isolation** and is a pre-existing intermittent (the Session-1 tracker §3
  baseline already records a separate intermittent pytest failure as a PARTIAL observation,
  not a session disqualifier). Net new tests in this closeout batch: **34** (14 Decision-3 +
  20 Decision-1).
- Test-count delta from v9.3: **2114 → 2639 (+525 tests)**.

### 2.9 Git baseline
- HEAD: `1626f2069fdf54656d399212fb36a132a00bb640` (`1626f20`)
- Branch: `fix/forensic-high-closeout-20260527`
- Tag applied at v9.6 lock (annotated, local only, not pushed): `SSOT_v9_6_2026-05-27`

---

## 3. Forensic Audit Findings (2026-05-27)

Source: `reports/forensic_audits/CHAD_FORENSIC_FULL_SYSTEM_AUDIT_20260527T142951Z.md`.

### 3.1 HIGH-severity items and disposition

| HIGH_ID | Title | Disposition | Evidence |
|---|---|---|---|
| TRUTH-RECONCILE-1 | Canonical paper-position authority gap | PARTIALLY FIXED → ARCHITECTURE LANDED | Decision 1 engine + design (`1626f20`); R1 PA status-updated |
| OPTIONS-CHAIN-1 | Silent `chad-options-chain-refresh` failure | FIXED | Health-monitor freshness rule (`b759cb6`) + Decision 3 OnFailure patch staged |
| STOP-BUS-RECOVERY-1 | IBKR sustained-latency no auto-recovery | PARTIALLY FIXED | Observability + alert path landed (`804abdd`); auto-recovery design PA raised |
| FUTURES-ROLL-1 | Expired SILK6-class contract polling spam | FIXED | Bar-provider futures-expiry gate (`7a0c100`) |
| PORT-BINDING-1 | Ports 9618/9619/9620 bind `0.0.0.0` | PARTIALLY FIXED | 9619/9620 defaults flipped to localhost (`ae614bb`); 9618 Channel-1 pending; invariant codified + allowlist + pre-commit (`1e00568`) |
| HISTORICAL-PLACEHOLDER-1 | 14 fake $100 fills on disk | FIXED | Shared placeholder-rejection predicate + read-only quarantine tool (`29e7197`) |

Summary: **3 FIXED** (OPTIONS-CHAIN-1, FUTURES-ROLL-1, HISTORICAL-PLACEHOLDER-1),
**3 PARTIALLY FIXED with architecture/observability landed** (TRUTH-RECONCILE-1,
STOP-BUS-RECOVERY-1, PORT-BINDING-1 — each with a clearly-scoped remaining
operator/Channel-1 step).

### 3.2 MED items (DEFERRED to post-v9.6)
- **SCHEMA-VERSION-1:** 78 runtime files lack a `schema_version` field — backfill is a
  post-v9.6 hygiene PA.
- **DOC-RECONCILE-1:** v9.3 documentation was 57 commits stale — **CLOSED by this v9.6 cut**
  (see §0.2 commit accounting).

### 3.3 LOW items
Listed in the audit §18; no v9.6 action.

---

## 4. Institutional Closeout — Decisions 1/2/3 Detail

### 4.1 Decision 2 — Mechanical Evidence-Gated Soak Rule
Commit `ac04e16`.

- Path: `ops/pending_actions/SESSION_SOAK_MECHANICAL_EVIDENCE_GATE_RULE_2026-05-27.md`
- sha256: `800480a570e8b2e50d115b9a62d52d6b7022cd5dc0a4421a740528061ef60723`

Authoritative grading contract for any future `broker_authority_RED` window during a soak
session (reproduced verbatim from §5 of the rule document):

> A broker_authority_RED window may be classified as EXPLAINED during a soak session only
> when ALL of the following are mechanically verifiable from artifacts written during the
> window:
>
> 1. Duration: the RED window duration is ≤ 1× ledger publish cadence (15 minutes wall-clock).
> 2. No fresh signals: zero RoutedSignal emissions during the window, verified from the signal_router audit log.
> 3. No fresh entries: zero intent_type=ENTRY items in the execution pipeline, verified from execution_plan_audit.ndjson (or equivalent audit artifact).
> 4. No sizing decisions: runtime/dynamic_caps.json ts_utc lies outside the RED window OR the per-strategy sizing_factor values are unchanged across the window.
> 5. No SCR transitions: runtime/scr_state.json state band is unchanged across the window.
> 6. No risk-state mutation: ts_utc on runtime/profit_lock_state.json, runtime/stop_bus.json, and runtime/tier_state.json are all unchanged across the window.
> 7. No PnL realization: zero new closed-trade rows in data/trades/trade_history_<date>.ndjson during the window.
> 8. Reconciler self-resolved: the next chad-reconciliation-publisher cycle after the window returned broker_authority_status=GREEN without operator intervention or manual JSON edit.
>
> If any of (1)–(8) cannot be verified or fails, the window is FAIL / NOT COUNTABLE for soak
> purposes. No exception. EXPLAINED classifications are recorded in
> runtime/session_explanations.json with provenance and per-artifact sha256. Operators cannot
> mark a window EXPLAINED. Only the soak evaluator can, mechanically. If the evaluator is not
> available, the default classification is FAIL / NOT COUNTABLE.

**Evaluator status:** NOT YET BUILT. Until it exists there is no manual override path; every
RED window defaults to FAIL / NOT COUNTABLE. Decision 2 superseded the prior
`SESSION_1_tracker_amendment_evidence_gated_2026-05-27.md` PA and is referenced by the
Session-1 tracker.

### 4.2 Decision 3 — Service Reliability Infrastructure
Commit `1e00568`.

Components:
- `chad/ops/service_failure_alert.py` — Python alert module (journal tail + metadata-only
  runtime snapshot + Telegram via `chad.utils.telegram_notify`; exit codes 0/2/3/4;
  refuses non-`chad-*` units).
- `ops/systemd_templates/chad-service-alert@.service` — template alert unit.
- `ops/systemd_templates/patches/chad-options-chain-refresh.patch` — OnFailure directive.
- `ops/systemd_templates/patches/chad-backend.patch` — OnFailure directive + uvicorn
  `--host 0.0.0.0 → 127.0.0.1`.
- `config/port_binding_allowlist.json` — schema `port_binding_allowlist.v1`; single allowed
  entry `chad-dashboard:8765` (nginx-fronted public surface).
- `chad/validators/port_binding.py` — extended to overlay the allowlist file.
- `scripts/install_precommit_hooks.sh` — idempotent installer.
- `.git/hooks/pre-commit` — installed; enforces the port-binding invariant + a
  `fill_price=100` heuristic on staged strategy code.
- `ops/runbooks/INSTALL_chad_service_alert_template_2026-05-27.md` — Channel 1 runbook
  (operator-authorized; **not yet executed**).

Tests: **14** (7 `service_failure_alert` + 7 `port_binding_validator`).

### 4.3 Decision 1 — Position Truth v2 Engine (Design + Stub)
Commit `1626f20`.

Components:
- `docs/design/POSITION_TRUTH_V2_ENGINE_DESIGN_2026-05-27.md`
  (sha256 `6370c6353fed069fa448a27055ca1254519b8c7667d7f7e80fe7181bc0eef7fd`)
- `chad/schemas/position_truth_v2.py`
- `chad/core/position_truth_engine.py`
- `chad/validators/position_truth_v2.py`
- `chad/tests/test_position_truth_engine.py`
- `chad/tests/test_position_truth_v2_validator.py`
- `chad/tests/fixtures/position_truth_v2/` (7 fixture directories)

Tests: **20** (12 engine + 8 validator).

The 5 merge rules (M1–M5) are codified in `chad/core/position_truth_engine.py` with
deterministic precedence **M5 → M4 → M1 → M2 → M3**:
- **M1** both agree → GREEN, trust value.
- **M2** ledger newer, integer delta within 2× cadence → YELLOW, trust ledger.
- **M3** snapshot newer, integer delta (missed fill) → DEGRADED, trust snapshot, fail-closed.
- **M4** fractional/side/sign mismatch → RED, DISAGREEMENT, fail-closed.
- **M5** stale source or one-sided absence → RED, FAIL_CLOSED.

Migration plan (design §12), each separately operator-authorized:
- Step 1 (timer-wired write of `runtime/position_truth_v2.json`) — Pending Action TBD.
- Step 2 (7-calendar-day shadow comparison vs `positions_truth.json`) — Pending Action TBD.
- Step 3 (consumer repoint + legacy publisher retirement) — Pending Action TBD.

---

## 5. Pending Actions Register (as of v9.6 lock)

| PA | Status | Owner channel | Blocks |
|---|---|---|---|
| `R1_canonical_position_authority_gap_2026-05-27` | STATUS UPDATE 2026-05-27 — design landed | Operator (canonical decision) → Channel 2 (migration steps) | Production v2 wiring |
| `R3_silent_unit_state_and_market_data_failures_2026-05-27` | PROPOSED | Channel 1 (unit edits via runbook) | OnFailure routing |
| `IBKR_RELIABILITY_*` (socket backpressure) | PARTIALLY ADDRESSED — observability landed | Channel 2 (auto-recovery design) | Auto-recovery behavior |
| `FUTURES_CONTRACT_EXPIRY_*` (roll mapping) | FIXED in code (`7a0c100`) | n/a | — |
| `PORT_BINDING_localhost_only_hardening_2026-05-27` | STATUS UPDATE 2026-05-27 — patches staged | Channel 1 (unit edit + restart) | Port 9618 closeout |
| `HISTORICAL_PLACEHOLDER_*` (fills quarantine) | FIXED in code (`29e7197`) | Operator decision on `--apply` quarantine | Historical row quarantine |
| `OPTIONS_CHAIN_OnFailure_directive_2026-05-27` | STATUS UPDATE 2026-05-27 — patch staged | Channel 1 (unit edit) | Failure visibility |
| `IBKR_AUTO_RECOVERY_*` (design) | PROPOSED | Channel 2 design | Recovery behavior |
| `PORT_BINDING_systemd_unit_edits_2026-05-27` | STATUS UPDATE 2026-05-27 — patches staged | Channel 1 (unit edit + restart) | Port 9618 closeout |
| `SESSION_1_tracker_amendment_evidence_gated_2026-05-27` | SUPERSEDED by SESSION_SOAK_MECHANICAL_EVIDENCE_GATE_RULE | n/a | — |
| `SESSION_SOAK_MECHANICAL_EVIDENCE_GATE_RULE_2026-05-27` | LOCKED 2026-05-27 — evaluator pending | Channel 2 evaluator implementation | Soak grading mechanism |
| `PAPER_EPOCH_3_SESSION_TRACKER_2026-05-28` | Active; references mechanical rule | n/a | — |

---

## 6. Commit Chain Since v9.3 (`72f361f..HEAD`)

`git rev-list --count 72f361f..HEAD` = **69 commits**. This institutional-remediation batch
is the most recent **12**; the prior **57** are the v9.4 cut, v9.5 forward-errata cut, Stage-3
Completion-Matrix box work, the PR-02…PR-12 series, the ib_async Phase-2 migration, and the
Paper Epoch 3 start — all documented in their own cuts (this section indexes only the
remediation batch; the other 57 are intentionally not re-listed here as they are not new).

### 6.1 Morning HIGH-severity multi-phase closeout (8 commits)
```
8583373  Pending Action: evidence-gated Session 1 tracker amendment (no auto-pass on RED)
29e7197  Fill validation: strict trusted-fake placeholder rejection + read-only quarantine tool (HISTORICAL-PLACEHOLDER-1)
ae614bb  Validators + defaults: localhost-only binding (PORT-BINDING-1)
7a0c100  Bar provider: futures expiry gate; stop SILK6-class polling spam (FUTURES-ROLL-1)
804abdd  IBKR reliability: sustained-latency observability + alert path (STOP-BUS-RECOVERY-1)
b759cb6  Health monitor: options-chain freshness rule + failure artifact path (R3 / OPTIONS-CHAIN-1)
6574dab  Validators: add fail-closed position authority validator (R1 / TRUTH-RECONCILE-1)
11f45a5  Pending Actions: add 2 missing HIGH forensic items (PORT-BINDING-1, HISTORICAL-PLACEHOLDER-1)
```
(Immediately preceded by `f854f54` — "Pending Actions: raise high-severity forensic audit
findings; preserve audit evidence in reports/forensic_audits/" — the audit-evidence
preservation commit.)

### 6.2 Afternoon institutional closeout (4 commits)
```
ac04e16  Decision 2: lock mechanical evidence-gated soak rule for transient broker_authority_RED windows (supersedes prior amendment PA)
1e00568  Decision 3: institutional systemd alert template + port-binding invariant + pre-commit enforcement
1626f20  Decision 1: position_truth_v2 engine design + schema + stub + validator + tests (no production wiring)
<this commit>  Docs: CHAD Unified SSOT v9.6 — institutional closeout (D1+D2+D3); supersedes v9.4 + v9.5-errata
```

---

## 7. Live Readiness (Unchanged)

The prior live-promotion statement remains authoritative: live promotion requires operator
GO, a clean multi-session paper soak, and clean equity/performance history. v9.6 does not
grant or imply any live execution authorization.

Soak status at v9.6 lock:
- Paper Epoch 3 active.
- Day-0 (2026-05-27) classified **DIRTY / NOT COUNTABLE** (IBKR latency >7h stop_bus halt;
  operator-authorized Gateway restart).
- Session 1 candidate window opens **2026-05-28T00:00:00Z**.
- Clean-soak count: **0 / 5**.
- Session 1 RED-window evaluation per the mechanical rule (§4.1); evaluator not yet built →
  any RED defaults to FAIL / NOT COUNTABLE.

---

## 8. Channel 1 Install Queue (Operator-Authorized)

Three Channel 1 tasks are queued for operator authorization. v9.6 does not authorize any of
them; they are listed for visibility:

1. Install `chad-service-alert@.service` template
   (runbook: `ops/runbooks/INSTALL_chad_service_alert_template_2026-05-27.md`).
2. Edit `chad-options-chain-refresh.service` to add `OnFailure=chad-service-alert@%n.service`.
3. Edit `chad-backend.service` to add `OnFailure=…` and flip uvicorn `--host` to
   `127.0.0.1` (closes PORT-BINDING-1 on the next service restart).

Recommended timing: after Session 1 evaluation closes (2026-05-29T00:00:00Z) to minimize
Session-1 noise (a `chad-*` restart would disqualify the session under tracker §3.17).

---

## 9. Known Issues and Residual Risks (v9.6)

Carried forward from prior cuts (live BAG / DOM / Kraken-CA / Coinglass / scanner-writeback
prohibitions remain) plus the following:

### 9.1 RISK-V9-6-01 — SCR intermittent UNKNOWN state
- Observed flapping to UNKNOWN twice during v9.6 closeout work; self-resolved to CONFIDENT.
- Cause: broker-latency flutter (rolling-avg >2000ms trips stop_bus; auto-recovers via
  `broker_latency_clean_streak=5`). Not caused by Decisions 1/2/3.
- Mitigation: SCR fails closed to UNKNOWN; no decisions are made on an unknown band.
- Action: separate post-soak hardening workstream (relates to STOP-BUS-RECOVERY-1 /
  IBKR auto-recovery design PA).

### 9.2 RISK-V9-6-02 — position_truth_v2 not yet wired
- Design + stub landed; production wiring deferred to the operator-authorized migration plan.
- The legacy `positions_truth.json` continues to serve consumers (GREEN at lock; §2.6).
- The engine's CLI-only RED reading at lock (§2.7) is a `ledger_ttl` tuning sensitivity, not
  a divergence — feeds the operator's `ledger_ttl` decision before any wiring.

### 9.3 RISK-V9-6-03 — Soak evaluator not yet implemented
- Mechanical rule locked; evaluator code pending.
- Default behavior on RED = FAIL / NOT COUNTABLE (no operator override).
- Action: follow-on Pending Action to build the evaluator before any RED window can be
  classified EXPLAINED.

### 9.4 RISK-V9-6-04 — Channel 1 install pending
- Port-9618 closeout, two OnFailure directives, and the alert-template install are all
  pending operator authorization.
- AWS Security Groups continue to block external exposure in the interim.
- Action: operator-authorized runbook execution (§8).

---

## 10. Phase Roadmap (post-v9.6)

### Immediate (operator decisions)
1. Authorize the Channel 1 install runbook (post-Session-1).
2. Choose the canonical-writer trajectory: pick a date to begin migration Step 1 of the
   position_truth_v2 design; tune `ledger_ttl` first (see §2.7 / §9.2).

### Near-term (Channel 2 PAs)
3. Build the soak evaluator implementing the mechanical rule.
4. Build the position_truth_v2 timer-driven publisher (migration Step 1).
5. Address IBKR auto-recovery design (separate PA).
6. Backfill `schema_version` on the 78 runtime files (MED, SCHEMA-VERSION-1).

### Mid-term
7. Run position_truth_v2 in shadow mode for 7 calendar days (migration Step 2).
8. Repoint consumers to position_truth_v2; retire the legacy publisher (migration Step 3).
9. Cut the next full SSOT (v9.7) after migration completes.

### Long-term
10. Complete Paper Epoch 3 soak (5 clean sessions).
11. Operator GO for live promotion.
12. Phase D Item 2 Tier 3C BAG limit-price unit normalisation.
13. Phase D Item 2 Tiers 3D–7 for live BAG.

---

## 11. Definition of Done (this SSOT cut)

- [x] v9.6 SSOT file exists at the declared path.
- [x] Document accounts for all 69 commits since v9.3 (12 remediation-batch detailed in §6;
      57 prior attributed to existing cuts).
- [x] All HIGH audit findings have a disposition recorded (§3.1).
- [x] All Pending Actions listed (§5).
- [x] All Channel 1 queued tasks listed (§8).
- [x] Residual risks explicit (§9).
- [x] Live posture explicitly preserved (no flip).
- [ ] File committed with the message in Tier 4 (lands with this commit).
- [ ] Tag `SSOT_v9_6_2026-05-27` applied (annotated; not pushed).

---

## 12. No-Live Confirmation

This SSOT does not authorize live trading.
`ready_for_live` remains false.
`allow_ibkr_live` remains false.
`allow_ibkr_paper` remains true.
No broker orders may be placed or cancelled under this SSOT.
