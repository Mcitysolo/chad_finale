# Box 053 — BAG bracket/OCA or fail-safe exit policy

**Status:** Pending Action — DOCUMENTATION ONLY. No production code change.
No runtime mutation. No order placement. No live authorization.

**Scope:** Pins the contract that **a live BAG position cannot sit
unmanaged if `live_loop` fails**. Closure path =
`PASS_LIVE_BAG_BLOCKED_UNTIL_PROTECTED` (live BAG entry is structurally
blocked today; bracket/OCA wiring and out-of-band fail-safe closer
remain pending).

---

## 1. Risk being mitigated (RISK-BAG-03)

Per `docs/CHAD_UNIFIED_SSOT_v9_3_2026-05-17.md:1511-1521` and
`docs/PHASE_D_ITEM2_BAG_HARDENING_TIER1.md:98-103`:

> Today the only close mechanism for a live BAG is the
> `alpha_options.max_hold_seconds=3600` exit, which is emitted by
> `live_loop`. If `live_loop` is down (crash, restart, redeploy,
> systemd unit stopped), a live BAG would sit unmanaged — no bracket,
> no OCA child, no out-of-band closer.

The Box 053 acceptance is to prove that **today this cannot happen** —
because live BAG entry is structurally blocked at three independent
layers — and to document the explicit policy for when bracket/OCA or
fail-safe wiring lands.

---

## 2. Three-layer structural block on live BAG entry

### Layer A — Adapter dry_run default
File: `chad/execution/ibkr_adapter.py:258` (`IbkrConfig.dry_run: bool = True`)
- Default `IbkrConfig.dry_run=True`. Under dry_run the adapter's
  `_submit_intent` returns a synthetic `SubmittedOrder` with
  `status="dry_run"` and **never invokes `placeOrder`**.
  (`chad/execution/ibkr_adapter.py:2287-2334`.)
- Environment: `IBKR_DRY_RUN` defaults to `True` in
  `chad/core/live_gate.py:498`.

### Layer B — LiveGate fail-closed
File: `chad/core/live_gate.py:776` (`evaluate_live_gate`)
- `runtime/live_readiness.json::ready_for_live=false` → DENY_ALL at
  gate 3 (line 885-889).
- `exec_cfg.exec_mode != "live"` → DENY_ALL at gate 8 (line 1090-1094).
- `operator.operator_mode != "ALLOW_LIVE"` → DENY_ALL at gate 2
  (line 863-873).
- Final `allow_ibkr_live = exec_cfg.ibkr_enabled AND NOT
  exec_cfg.ibkr_dry_run AND operator.allow_ibkr_live` (line 1099).
- Each gate is a sufficient block.

### Layer C — No bracket/OCA, no fail-safe closer wiring
- Grep over `chad/execution/`, `chad/risk/`, `chad/core/`, `ops/` for
  `parentId|transmit|ocaGroup|ocaType|Bracket|brackets` returns
  **zero hits** in production execution code (only documentation
  references). Confirmed by Box 053 test
  `test_ibkr_adapter_source_has_no_bracket_or_oca_wiring`.
- No `chad.risk.bag_failsafe_closer` /
  `chad.execution.bag_failsafe_closer` /
  `chad.ops.bag_failsafe_closer` / `ops.bag_failsafe_closer` module
  is importable. Confirmed by Box 053 test
  `test_no_bag_failsafe_closer_module_exists`.

The only BAG exit signal source today is
`chad.strategies.alpha_options.AlphaOptionsTuning.max_hold_seconds`
(default 3600 s) emitted by `live_loop` once the configured age is
exceeded. Confirmed by Box 053 test
`test_alpha_options_max_hold_seconds_is_the_documented_bag_exit`.

---

## 3. Hard rules

1. **Live BAG entry MUST remain blocked** until at least one of the
   following lands:
   - Native IBKR bracket (parent BAG + child stop / target with
     `parentId` and `transmit` chain), OR
   - An out-of-band fail-safe BAG closer (cron-driven, independent
     of `live_loop`).
2. **No relaxation of Layer A/B/C** without an Official Matrix box
   update. The Box 053 anchor tests fail by design when wiring
   lands, forcing a deliberate refresh of this policy.
3. **No paper-only relaxation that leaks to live.** Paper
   simulators (e.g. `paper_exec_evidence_writer.simulate_bag_paper_fill`)
   must NOT depend on broker-side bracket / OCA semantics.
4. **`max_hold_seconds=3600` is the documented paper-mode exit
   default.** Changing it requires a deliberate Box 053 refresh — the
   anchor test pins the value.

---

## 4. When bracket/OCA OR fail-safe closer lands

Triggering failure modes for Box 053 anchor tests (refresh signals):

| Anchor test                                                                 | What it asserts                                                            | When it fails (next refresh signal)                                   |
| --------------------------------------------------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `test_ibkr_adapter_source_has_no_bracket_or_oca_wiring`                      | Adapter source has no parentId / transmit / ocaGroup / ocaType / Bracket    | When native IBKR bracket wiring is added to `ibkr_adapter.py`           |
| `test_order_factory_does_not_set_bracket_or_oca_attributes_on_bag`           | `_OrderFactory.build` does not stamp bracket fields on the BAG Order         | When `_OrderFactory` is taught to attach a child stop / target          |
| `test_no_bag_failsafe_closer_module_exists`                                  | No `bag_failsafe_closer` module is importable                                | When an out-of-band cron closer is created                              |
| `test_alpha_options_max_hold_seconds_is_the_documented_bag_exit`             | `AlphaOptionsTuning.max_hold_seconds == 3600`                                | When the default changes (paper hardening or live-promotion calibration) |
| `test_default_ibkr_config_is_dry_run`                                       | `IbkrConfig.dry_run` defaults to `True`                                      | When the adapter default flips to live (would require explicit promotion) |

When any of these fails, Box 053 evidence and this policy MUST be
refreshed to reflect the new protection layer.

---

## 5. Acceptance criteria — Box 053

| Criterion                                                                                       | Status |
| ----------------------------------------------------------------------------------------------- | ------ |
| Live BAG entry is structurally blocked under PAPER posture (Layer A)                              | PASS   |
| LiveGate fail-closes on `ready_for_live=false` and on `ibkr_dry_run=true` (Layer B)                | PASS   |
| Adapter source contains no bracket / OCA wiring (Layer C anchor)                                  | PASS   |
| `_OrderFactory.build` does not stamp bracket / OCA fields on BAG (Layer C anchor)                  | PASS   |
| No out-of-band BAG fail-safe closer module exists (Layer C anchor)                                 | PASS   |
| `alpha_options.max_hold_seconds=3600` is the documented BAG exit driver                            | PASS   |
| Runtime posture artifacts (`live_readiness.json`, `execution_environment.json`) confirm PAPER block | PASS   |
| Tests cover all blocking layers + future-wiring refresh anchors                                    | PASS (9 tests)  |

---

## 6. Runtime / live invariants

- No `runtime/*.json` mutation.
- No SQLite mutation.
- No order placement.
- No BAG order built / previewed / submitted.
- No live authorization.
- No `systemctl daemon-reload` / restart / start / stop.
- `chad-live-loop.service` remains `active (running)`.
- `runtime/live_readiness.json` `ready_for_live` remains `false`.
- HEAD invariant `bbe7525` before/after.

---

## 7. Cross-references

- Box 051 (Official) — BAG per-share / contract-dollar unit normalization:
  `runtime/completion_matrix_evidence/BOX-051_OFFICIAL_BAG_Tier_3C_limit_price_unit_normalization.md`
- Box 052 (Official) — BAG adapter quote enforcement:
  `runtime/completion_matrix_evidence/BOX-052_OFFICIAL_BAG_adapter_quote_enforcement.md`
- RISK-BAG-03 (SSOT v9.3):
  `docs/CHAD_UNIFIED_SSOT_v9_3_2026-05-17.md:1511-1521`
- Phase D Item 2 Tier 1 brief:
  `docs/PHASE_D_ITEM2_BAG_HARDENING_TIER1.md:98-103`
- Phase D Item 2 Tier 2 design:
  `docs/PHASE_D_ITEM2_BAG_HARDENING_TIER2_DESIGN_2026-05-17.md:57-59`
