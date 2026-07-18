# PENDING ACTION — Exit-overlay flip criteria amendment (FU1-B7)

- **Filed:** 2026-07-18
- **Type:** Criteria amendment (doc-only). Amends §4 of
  `ops/pending_actions/EXIT_AUDIT_equity_roundtrip_close_2026-07-13.md`.
- **Author:** FU1 (flip-unblock follow-up to `docs/ULTRA_CLOSE_AUDIT_2026-07-17.md`).
- **Status:** PROPOSED — requires typed operator GO. **No code, no config, no runtime state is
  changed by this document.** The overlay remains DEPLOYED IN SHADOW (submits nothing).
- **Governance:** One change at a time · no direct config mutation (a `mode` flip is its own
  authorization PA) · no flip without pre-registered written criteria · this doc does not itself
  authorise any flip.

---

## 0. Why this amendment exists

The ULTRA-CLOSE pre-flip audit (`docs/ULTRA_CLOSE_AUDIT_2026-07-17.md`) graded the SHADOW→ACTIVE
flip against the seven pre-registered criteria in EXIT_AUDIT §4 and returned **FLIP_BLOCKED** on
five independent blockers. Four are code defects, now fixed under FU1:

| Blocker | Fix | Commit |
|---|---|---|
| B-2 seed-lot untrust does not propagate | meta→extra/tags + fill_ids backstop | `217a3dc` |
| B-3 anchor wipe / max_hold dead / no cost basis | merge-not-replace + FIFO truth + ratchet | `e3192f7` |
| B-4 regime-reduction double-sell (INCIDENT-0713) | broker-truth clamp on the reduction leg | `81c07bd` |
| B-5 unbounded rejection storm | bounded retry (backoff + stand-down + coach alert) | `1dd8fa7` |
| B-6 overlay not exclusion-aware (PA §3 false) | in-module `SKIP_EXCLUDED` + 5 doc fixes | `98ea551` |

Three of the seven criteria are **not code defects** — they are defects in the *criteria
themselves*. The audit flagged each as needing an operator ruling, not a code change:

- **Criterion 2** (`SCR ∈ {CONFIDENT, CAUTIOUS}`) sits on a **WARMUP paradox** (audit §A.2, B-1).
- **Criterion 5** (confirmation proof) is **structurally unsatisfiable in shadow** — a bootstrap
  paradox with a proxy (audit §A.5).
- **Criterion 7** requires a rollback procedure that **does not exist** (audit §A.7).

This PA amends criteria 2, 5, and 7. It does not touch criteria 1, 3, 4, or 6.

---

## 1. Amendment to Criterion 2 — the WARMUP paradox

### The paradox

Criterion 2 requires `runtime/scr_state.json` `state ∈ {CONFIDENT, CAUTIOUS}` at flip time. Live
now (`2026-07-18T02:00:47Z`, ttl 180, fresh): **`state = WARMUP`, `sizing_factor = 0.1`**;
`effective_trades` is below the CONFIDENT/CAUTIOUS threshold (the audit read 67 vs a 100 floor on
07-17). WARMUP exit requires accumulating honest scoreable round-trips. But:

1. The overlay — the *systematic* close engine that would generate those round-trips — is barred
   from ACTIVE until SCR leaves WARMUP. **It cannot contribute to the maturity it is gated on.**
2. The only working close path meanwhile is the *incidental* reconciler (EXIT_AUDIT §1, Fault B),
   which produces round-trips slowly and unsystematically.
3. Pre-FU1-B2, seed-lot closes banked **fabricated** alpha into `effective_trades` (audit §D, the
   first ~810 shares were fake) — so the WARMUP counter was being advanced by contaminated
   evidence. **B-2 (`217a3dc`) retires that**: from now on WARMUP progress is honest — truer, and
   likely slower.

Net: a real tension. The gate can be slow to clear, and the overlay cannot help clear it.

### Ruling (operator picks ONE)

**It is NOT a true deadlock** — the reconciler and strategy round-trips mature SCR independently of
the overlay, so WARMUP *does* progress without the overlay. The choice is about speed vs. caution.

- **Option 2-A — HARD gate, conservative (RECOMMENDED).** Keep Criterion 2 exactly as written:
  no flip until SCR is CONFIDENT/CAUTIOUS. Accept slower, honest maturation (B-2 removed the fake
  accelerant) as the price of not flipping on thin evidence. Re-grade only after SCR exits WARMUP,
  **and** after ≥5 shadow sessions carrying proposed closes on **more than one symbol** (audit
  §G.5 — the current 744-row proof is one decision on one symbol, UNH), **and** after the phantom
  defense and reduce-only clamp have each fired at least once in production.
- **Option 2-B — supervised WARMUP bootstrap (higher risk, requires explicit GO).** Permit a
  bounded ACTIVE flip *during* WARMUP under strict extra guards, purely to bootstrap honest
  evidence, then re-evaluate. Guards, ALL required: single symbol only; WARMUP `sizing_factor 0.1`
  left untouched (it already caps size to 10%); manual oversight on the first 3 live cycles (§3
  protocol); the phantom defense + reduce-only clamp + B-5 bounded-retry all exercised at least
  once; **not on a Friday** (audit §C-extra weekend hazard). Any guard breach → immediate inert
  (§3).

**Recommendation: 2-A.** 2-B is available if the operator judges the maturation stall
unacceptable, but it spends the very safety margin the criterion exists to protect.

### WARMUP re-grade protocol (applies under either option)

Re-grading is a fresh audit pass, not an automatic pass: re-read `scr_state.json` (honor TTL),
re-count qualifying multi-symbol shadow sessions, confirm the phantom/reduce-only/retry defenses
each fired in production, and re-run the full §4 checklist. A WARMUP→CONFIDENT transition alone
does not arm the overlay.

---

## 2. Amendment to Criterion 5 — the confirmation-proof proxy ruling

### The bootstrap paradox (audit §A.5)

Criterion 5 as written — "≥1 shadow-window equity close that, if it had been live, would have
confirmed at the broker" — is **structurally unsatisfiable in shadow**. Shadow records a *mark*
(`exit_overlay.v1` carries `price`/`atr`/`atr_stop` from `price_cache`), **not a fill**: there is
no submit, no broker round-trip, no fill price. Read as "the overlay must demonstrate it," C5
requires the first ACTIVE close as its own proof — unsatisfiable without the flip it gates.

### The proxy (audit §A.5)

The operative intent is narrower: *"Fault C no longer **universally** rejects equity closes."*
"Universally" falls to a single counterexample from **any** path. It exists, in-window and in
Epoch-3 — `data/fills/FILLS_20260713.ndjson` (verified present, 34 TLT rows):

- `c1f1242…` TLT SELL **1340 @ 84.03**, `status=Filled`, `reject=false`, 17:30:14Z
- `e6d8242…` TLT SELL **700 @ 83.99**, `status=paper_fill`, `extra.ibkr_exec_id` present
- `4694fbf…` TLT SELL **640 @ 83.99**, broker-confirmed

Real, non-placeholder, non-`error` fill prices against a genuinely broker-held 700 TLT long. Both
structural fixes have landed: **PR-02b `5c5507e`** removed the $100 placeholder fallback and the
**RTH gate `8b39730`** (deployed 07-12) which the overlay inherits via `apply_close_intents` —
result: **zero rejected equity SELLs 07-13..07-17.** Fault C's *universal* reject is retired.

### Ruling

**Rewrite Criterion 5** to its operative intent and **formally accept the TLT proxy** as
satisfying it:

> **Criterion 5 (amended).** Fault C no longer *universally* rejects equity closes. Satisfied by a
> single in-window, in-Epoch broker-confirmed equity SELL with a non-placeholder, non-`error` fill
> price (the TLT 1340/700/640 fills of 2026-07-13, `FILLS_20260713.ndjson`), together with the
> landed structural fixes (`5c5507e` placeholder removal, `8b39730` RTH gate) and the observed
> zero rejected equity SELLs across 07-13..07-17. The overlay is **not** required to produce its
> own live fill in shadow — that is the bootstrap paradox and is impossible by construction.

This is the honest reading; C5-as-written can never be met and would silently block the flip
forever. Criterion 5 does **not** block the flip once amended.

---

## 3. Amendment to Criterion 7 — a real rollback procedure + oversight protocol

Audit §A.7: C7's "rollback documented" matched only the line restating the requirement — **no
procedure existed**. This section is that procedure. (C7's kill-switch half already PASSES:
`CHAD_POSITION_EXIT_OVERLAY` via `resolve_mode`, verified live.)

### 3.1 Rollback (ACTIVE → inert), fastest-first

1. **Emergency inert (seconds, no code/config change).** Set the env kill-switch
   `CHAD_POSITION_EXIT_OVERLAY=off` (or `shadow`) — it **overrides** the config `mode` per
   `config/position_exit_overlay.json` and `resolve_mode` — via the live-loop systemd drop-in, then
   restart the live-loop (operator action; §6/§7 of CLAUDE.md — never restart without GO). The next
   cycle evaluates in OFF/SHADOW and submits nothing. This is the primary revert; it needs no PA.
2. **Durable revert (config).** Prepare a Pending Action to set
   `config/position_exit_overlay.json` `mode` back to `shadow`. **Never edit the config directly** —
   the flip and the revert are each their own authorization commit (git history is the enforcement
   log, per the config's own `_comment`). Once merged, the env override from step 1 can be removed.
3. **Code revert is NOT normally needed.** The flip is a `mode` change, not a code change; the
   overlay code is safe in shadow. Only if a flip landed alongside a code change would
   `git revert <commit>` apply — followed by the §4 verification sequence.

### 3.2 State cleanup after inerting

- Confirm no standing ACTIVE close is mid-flight: inspect `runtime/exit_overlay_heartbeat.json`
  (`mode` should read off/shadow), the B-5 submit ledger
  (`runtime/position_exit_overlay_state_submit_ledger.json`), and
  `runtime/exit_overlay_stand_down.json`.
- Run reconciliation; confirm `runtime/position_guard_drift.json` `qty_mismatch=0` and
  `reconciliation_state.json` not RED.
- If an overlay-originated close left a phantom, reconcile it with the DRIFT-RECON tool
  **dry-run first** (per memory `drift_recon_ledger_broker_2026_07_13` — NEVER `--execute` against
  real runtime without GO).

### 3.3 Manual-oversight protocol for the first 3 live cycles (C7 second half)

- **Owner:** the operator performing the flip (named in the flip commit).
- **Watch-list, each cycle:** `runtime/exit_overlay_heartbeat.json` (mode/evaluated/would_close),
  the day's `data/exit_overlay/exit_overlay_*.ndjson` (verdicts), overlay-tagged rows in
  `data/fills/FILLS_*.ndjson` (reason `exit_overlay_*`), `runtime/position_guard_drift.json`, and
  `runtime/exit_overlay_stand_down.json`.
- **Hard abort triggers (any one → inert via §3.1 step 1 immediately):** a submitted close qty
  exceeding broker-held same-side (reduce-only breach / any short flip); a `SKIP_UNCONFIRMED` that
  nonetheless reached the broker; a regime-reduction that oversold (B-4 regression); an
  `EXIT_OVERLAY_STAND_DOWN` marker (B-5); or reconciliation flipping RED.
- **Not on a Friday** — the weekend RTH-block hazard (audit §C-extra) pollutes the evidence base.

---

## 4. What this amendment does NOT change

- Criteria 1, 3, 4, 6 stand unchanged.
- The frozen overlay thresholds (`config/position_exit_overlay.json`) are untouched — changing a
  threshold is a separate deliberate logged event / new trial.
- No `mode` flip is authorised here. Arming is still its own PA against the amended §4.
- No runtime state, service file, or restart is touched by this document.

---

## 5. Operator GO block (typed acceptance required)

Arming the overlay after this amendment requires the operator to record, in writing:

- [ ] Criterion 2 ruling selected: **2-A (hard gate)** ☐ or **2-B (supervised bootstrap)** ☐.
- [ ] Criterion 5 amended text accepted (TLT proxy) ☐.
- [ ] Criterion 7 rollback procedure + oversight protocol accepted ☐.
- [ ] Re-graded against the amended §4 with SCR, multi-symbol shadow count, and the
      phantom/reduce-only/retry production-exercise checks all satisfied ☐.
- [ ] Flip is not on a Friday ☐.

Until every box is checked in a GO commit, the overlay stays in SHADOW (evidence-only).
