# BOX-048 (Official Matrix) — alpha_intraday_micro classification

- **Box number (Official Matrix):** 048
- **Box title (Official Matrix):** NEW-GAP-045 alpha_intraday_micro classified — Strategy emits in proper conditions or is explicitly documented dormant/deferred
- **Stage:** Stage 3 — Engineering, tests, SSOT, and hidden-gap closure
- **Cut timestamp (UTC):** 2026-05-20T19:26:32Z
- **HEAD at cut:** `bbe7525`
- **Branch:** `main`
- **Status (this doc):** classification declared; operator weight decision still pending

> **Note on Box 033 supersession.** This Box-048 (Official) supersedes
> the Box-033 framing "SOFT_DEFERRED_PENDING_CONTEXTBUILDER" with the
> stricter classification **ACTIVE_CONDITIONALLY_SILENT** — the
> ContextBuilder tier-profile wiring prerequisite has been verified
> present (see §3 below). Box 033's operator-decision options
> A / B / C remain available; the recommendation is unchanged (default
> to Option C — hold — until operator weight decision arrives).
>
> The earlier Box-033 pending action
> (`ops/pending_actions/BOX-033_alpha_intraday_micro_weight_policy.md`)
> remains valid as the operator-decision artifact; this Box-048 doc
> adds the runtime-verified classification and a tighter wiring proof.

---

## 0. Scope and safety statement

- **CHAD remains PAPER.** `CHAD_EXECUTION_MODE=paper`.
- **live trading not authorized.** This classification does not flip
  `ready_for_live` and does not authorize live trading.
- **No silent enable.** Box 048 does **not** add an
  `alpha_intraday_micro` weight to `config/strategy_weights.json`; the
  operator's Box-033 Options A / B / C remain the only paths.
- **No runtime mutation.** No production code / config / runtime
  state modified.

---

## 1. Reference inventory (27 files)

Verified by `grep -rln 'alpha_intraday_micro'` (excluding venv/.git/caches/backups/data/reports/logs):

| File                                                                                                          | Role                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `chad/types.py:67`                                                                                            | `StrategyName.ALPHA_INTRADAY_MICRO = "alpha_intraday_micro"` (enum entry).                                                              |
| `chad/strategies/__init__.py:93-95, 240-243`                                                                  | imports `alpha_intraday_micro_handler` + `build_alpha_intraday_micro_config`; registers `StrategyRegistration` in `_REGISTRY`.        |
| `chad/strategies/__init__.py:63-` (`DEFERRED_STRATEGIES`)                                                      | `alpha_intraday_micro` **NOT** in DEFERRED_STRATEGIES (only `alpha_forex` is).                                                          |
| `chad/strategies/alpha_intraday_micro.py`                                                                     | Handler module — 5 setup families (ORB, VWAP_RECLAIM, VWAP_REJECTION, PULLBACK_CONTINUATION, SWEEP_REVERSAL) on MES/MNQ; tier-aware. |
| `chad/risk/tier_manager.py:90, 95, 119`                                                                       | `_CANONICAL_STRATEGY_NAMES` list includes `"alpha_intraday_micro"`.                                                                    |
| `chad/utils/context_builder.py:90-95, 175-178, 506, 521-522, 761-800`                                          | ContextBuilder wires `tier_profile` + `tier_name` onto MarketContext via `_load_tier_profile()` (reads `runtime/tier_state.json`).     |
| `chad/utils/session.py`                                                                                       | session window helpers used by alpha_intraday_micro setup gates.                                                                     |
| `chad/analytics/setup_family_expectancy_updater.py`                                                            | expectancy stats updater (consumes per-strategy results).                                                                              |
| `chad/execution/trade_closer.py`                                                                              | trade_closer references the strategy name.                                                                                            |
| `config/tiers.json:10, 30, 56`                                                                                | `alpha_intraday_micro` in `enabled_strategies` of multiple tier profiles.                                                              |
| `config/strategy_weights.json`                                                                                | **NOT in `weights` dict** (only 16 weights; alpha_intraday_micro absent). Allocator gate drops the strategy here.                     |
| `chad/tests/test_alpha_intraday_micro.py`                                                                     | 18 unit tests for the handler (all pass).                                                                                              |
| `chad/tests/test_strategy_registry_parity.py:65-70, 134-149, 153-159, 165-170`                                  | `WEIGHT_DEFERRED_ALLOWLIST` includes `"alpha_intraday_micro"`; parity tests assert (a) allowlist covers missing-weight strategies, (b) every allowlist entry must still be in active registry, (c) every allowlist entry must point to a real pending-action file. |
| `chad/tests/test_meta_forwarding.py`                                                                          | meta-forwarding test references the strategy name.                                                                                    |
| `chad/tests/test_setup_family_expectancy_updater.py`                                                          | expectancy test.                                                                                                                       |
| `chad/tests/test_workstream3.py`                                                                              | end-to-end workstream test.                                                                                                            |
| `deploy/chad-micro-eod-flatten.service`                                                                       | systemd unit for micro EOD flatten.                                                                                                    |
| `deploy/chad-setup-expectancy.service`                                                                        | systemd unit for setup expectancy.                                                                                                     |
| `ops/micro_eod_flatten.py`                                                                                    | EOD flatten utility for micro positions.                                                                                              |
| `ops/pending_actions/BOX-033_alpha_intraday_micro_weight_policy.md`                                            | Box-033 prior policy + operator decision options A/B/C.                                                                                |
| `ops/pending_actions/BOX-052_final_open_gaps_reconciled.md`                                                    | Box-052 G-08 row (`alpha_intraday_micro weight policy — operator decision`).                                                            |
| `ops/pending_actions/BOX-054_final_deployment_actions_list_ready.md`                                            | Box-054 D-09 action (operator weight policy / ratification).                                                                          |
| `docs/CHAD_UNIFIED_SSOT_v9.1_2026-05-13.md` (line 1285)                                                        | residual line: "alpha_intraday_micro registered but conditionally inactive at runtime ... REQUIRES CONTEXT-BUILDER WIRING VERIFICATION before MICRO/STARTER/PRO_GROWTH testing." |
| `docs/CHAD_UNIFIED_SSOT_v9.2_2026-05-15.md`, `v9_3_2026-05-17.md`                                              | carry-forward of the same residual.                                                                                                   |
| `docs/CHAD_SSOT_v9_5_FORWARD_ERRATA_2026-05-20.md`                                                            | references the GAP-033/045 lineage; does not retract the residual.                                                                    |
| `docs/CHAD_EVIDENCE_LOCKED_DOD_v1_0_2026-05-20.md`                                                            | DoD references Box-033 G-08 follow-up.                                                                                                |
| `docs/CHAD_OPERATOR_CHANGELOG_BOXES_001_050_2026-05-20.md`                                                    | changelog Box-033 entry.                                                                                                              |
| `docs/PHASE_D_DYNAMIC_UNIVERSE_SCANNER_DESIGN_2026-05-16.md`                                                  | Phase-D design reference.                                                                                                              |

---

## 2. Registry / weights / tier / deferred state

| Surface                                          | State                                                                                            |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| `StrategyName.ALPHA_INTRADAY_MICRO`              | ✅ present (`chad/types.py:67`).                                                                  |
| Active registry (`chad/strategies/__init__.py`)  | ✅ present (`_REGISTRY` entry at line 240).                                                       |
| `DEFERRED_STRATEGIES`                            | ❌ NOT included (only `ALPHA_FOREX` is formally deferred).                                         |
| `_CANONICAL_STRATEGY_NAMES` (`chad/risk/tier_manager.py:95`) | ✅ present.                                                                            |
| `config/tiers.json` `enabled_strategies`         | ✅ present in multiple tier profiles (lines 10, 30, 56).                                          |
| `runtime/tier_state.json` `enabled_strategies`   | ✅ present (current tier=SCALE; alpha_intraday_micro in enabled_strategies list).                  |
| `config/strategy_weights.json` `weights` dict    | ❌ **NOT present** (16 weights; alpha_intraday_micro absent).                                      |
| `WEIGHT_DEFERRED_ALLOWLIST` (parity test)        | ✅ present (`chad/tests/test_strategy_registry_parity.py:70`); references Box-033 pending action. |

`registered_in_enum: true`; `registered_in_tier_manager: true`;
`weighted_in_strategy_weights: false`;
`deferred_or_dormant_policy_present: true` (this doc + Box-033 doc +
parity-test allowlist).

---

## 3. ContextBuilder tier-profile wiring (verified at audit)

The SSOT v9.1 residual claim was: "ContextBuilder must populate
tier_profile and tier_name on MarketContext for this strategy to emit
signals. **Status: REQUIRES CONTEXT-BUILDER WIRING VERIFICATION.**"

This box verifies the wiring **is in place** at audit time:

### 3.1 Code path (verified by grep)

`chad/utils/context_builder.py`:

- **Lines 90-95**: `from chad.risk.tier_manager import TierRiskProfile`
  (fail-closed import — `TierRiskProfile = None` on `ImportError`).
- **Lines 175-178**: MarketContext dataclass declares
  `tier_profile: Optional[Any] = None` + `tier_name: str = ""`.
- **Line 506**: `tier_profile_obj, tier_name_str = self._load_tier_profile()`.
- **Lines 521-522**: passes `tier_profile=tier_profile_obj` and
  `tier_name=tier_name_str` into MarketContext constructor.
- **Lines 761-800**: `_load_tier_profile()` reads
  `runtime/tier_state.json` and constructs a `TierRiskProfile` from it,
  fail-closed `(None, "")` on any failure (missing file, malformed JSON,
  TierRiskProfile import unavailable).

### 3.2 Runtime evidence (verified by `cat runtime/tier_state.json`)

```
mtime 2026-05-20 19:25 UTC  (current — refreshed today)
{
  "schema_version": "tier_state.v2",
  "tier_name": "SCALE",
  "tier_description": "Full stack — all 16 strategies, existing sizing pipeline unchanged",
  "current_equity_usd": 179614.68,
  "tier_min_equity": 160000.0,
  "tier_max_equity": 10000000.0,
  "enabled_strategies": [
    "alpha", "alpha_crypto", "alpha_futures", "alpha_intraday",
    "alpha_intraday_micro",  // ← present
    "alpha_options", "beta", "beta_trend", "delta", "delta_pairs",
    "gamma", "gamma_futures", "gamma_reversion", "omega", "omega_macro",
    "omega_momentum_options", "omega_vol"
  ],
  "risk_profile": { … all null caps; primary_session_only=false; flatten_before_eod=false; … }
}
```

`tier_state.json` is present, current, valid SCALE tier with
`alpha_intraday_micro` enabled. ContextBuilder's `_load_tier_profile()`
will produce a non-None `tier_profile` and a non-empty `tier_name`
("SCALE") on every cycle.

**Conclusion (supersedes SSOT v9.1 residual):** the ContextBuilder
wiring prerequisite is **VERIFIED**. The strategy's handler will
receive a populated `tier_profile` on every cycle that ContextBuilder
runs. The SSOT v9.1 residual claim is **superseded by this evidence**
(forward-only, not retroactively edited per CHAD governance).

### 3.3 Why the strategy is still silent in runtime

Despite wiring verification:

- `journalctl -u chad-live-loop.service --since 2026-05-19T17:10:47` for
  `alpha_intraday_micro` → **0 hits** over ~26 hours.
- `data/fills/FILLS_*.ndjson` for `"strategy":"alpha_intraday_micro"` →
  **0 hits** across last 5 daily files.
- `data/trades/trade_history_*.ndjson` for `"strategy":"alpha_intraday_micro"` →
  **0 hits** across last 5 daily files.

**Root cause:** `config/strategy_weights.json` does not contain an
`alpha_intraday_micro` weight. The allocator (`chad.risk.dynamic_risk_allocator`
+ `chad.risk.allocator_v3`) filters "eligible strategies … present in
base weights, base weight > 0" — so even if the handler emits a
TradeSignal, the allocator drops it at sizing time.

**Secondary contributing factor:** the handler emits only when intraday
session windows + setup conditions (ORB, VWAP_RECLAIM, VWAP_REJECTION,
PULLBACK_CONTINUATION, SWEEP_REVERSAL) actually trigger; absent that,
the handler returns `[]` even without the allocator gate.

The **primary gate** is the missing weight entry. The journal silence is
consistent with both factors and does not contradict the
ACTIVE_CONDITIONALLY_SILENT classification.

---

## 4. Classification

**`ACTIVE_CONDITIONALLY_SILENT`**

| Possible classification                              | Verdict | Why                                                                                                                                            |
| ---------------------------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| ACTIVE_EMITTING                                       | NO      | 0 journal mentions / 0 fills / 0 trades observed in last 26 h.                                                                                  |
| **ACTIVE_CONDITIONALLY_SILENT**                       | **YES** | Handler is registered + tier-wired + receives populated tier_profile every cycle. Allocator gate drops it because no `strategy_weights.json` entry; setup-condition gates also non-triggering in current window. |
| SOFT_DEFERRED_PENDING_CONTEXTBUILDER                  | NO (superseded) | ContextBuilder wiring **verified present** (see §3); the SSOT-v9.1 "wiring not verified" prerequisite is satisfied.                  |
| FORMALLY_DORMANT                                      | NO      | Not in `DEFERRED_STRATEGIES`; handler is in the active registry; tier_state.json includes it in `enabled_strategies`.                          |
| BROKEN_REQUIRES_FIX                                   | NO      | All 18 `test_alpha_intraday_micro.py` tests pass; 8/8 parity tests pass; 12/12 tier_manager tests pass; no import error or runtime exception.   |

---

## 5. Operator decision (Box-033 options remain open)

This box does **not** make the operator decision. The Box-033 options
remain available; this Box-048 documentation tightens the classification
but does not pre-empt the operator's choice:

| Option                                | Effect                                                                                                                                | Recommendation                                                                                                                                                |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A — Formally activate** with weight  | Add `alpha_intraday_micro` to `config/strategy_weights.json`; re-normalise; remove from `WEIGHT_DEFERRED_ALLOWLIST`; restart live-loop. | Requires deliberate allocation decision + operator GO. Live trading remains gated by `live_readiness.json` regardless.                                         |
| **B — Formally defer**                  | Add `ALPHA_INTRADAY_MICRO` to `DEFERRED_STRATEGIES`; comment out the `_REGISTRY` entry (mirror `alpha_forex` pattern at lines 195–198); remove from `WEIGHT_DEFERRED_ALLOWLIST`; restart live-loop. | Lowest-risk path — makes the silent-state deterministic.                                                                                                       |
| **C — Hold (status quo)** [DEFAULT]    | Keep ACTIVE_CONDITIONALLY_SILENT state; this Box-048 closure + Box-033 pending action + parity-test allowlist capture the audited soft-deferred state. | **Recommended default** until operator chooses A or B. Box 048 closes on this status. No code/config change required.                                          |

**Note vs Box 033:** Box 033 recommended Option C "until operator
confirms whether ContextBuilder tier_profile wiring is in place".
Box 048 verifies the wiring **is** in place (§3); this removes the
wiring-uncertainty argument for staying at Option C. The remaining
arguments for Option C are (a) no operator allocation decision has
been made and (b) staying at C preserves the status quo with full
audit trail. Box 048 maintains the Option C default on those grounds.

---

## 6. Patches summary

| Patch class            | Action                                                                                                                  |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Production code        | **None** — registry / handler / ContextBuilder wiring already correct; Box 048 does not change behaviour.               |
| Config                 | **None** — `config/strategy_weights.json` NOT mutated (no weight added; operator decision required).                     |
| Documentation          | **Added** `ops/pending_actions/BOX-048_alpha_intraday_micro_classification.md` (this file).                              |
| Evidence               | **Added** `runtime/completion_matrix_evidence/BOX-048_OFFICIAL_alpha_intraday_micro_classified.md` (paired with this doc). |
| Tests                  | **None added** — existing parity / alpha_intraday_micro / tier_manager tests already enforce the audited soft-deferred state. All pass. |
| `WEIGHT_DEFERRED_ALLOWLIST` | **Unchanged** — `alpha_intraday_micro` remains in the allowlist; will be removed when Option A or B is selected.   |
| Frozen historical SSOT | **Unchanged** — forward-only.                                                                                            |
| Staged / committed     | **None** — `git diff --cached` empty.                                                                                    |

---

## 7. Tests run

| Test                                                                  | Result            |
| --------------------------------------------------------------------- | ----------------- |
| `chad/tests/test_strategy_registry_parity.py`                          | **8 / 8 passed**   |
| `chad/tests/test_alpha_intraday_micro.py`                              | **18 / 18 passed** |
| `chad/tests/test_tier_manager.py`                                      | **12 / 12 passed** |
| Full suite (cited Supplemental-Annex Box 047 GREEN baseline 2361/2361) | not re-run; baseline still GREEN per `runtime/completion_matrix_evidence/BOX-047_TEST_BASELINE_FULL_GREEN_OR_CLASSIFIED.md` |

---

## 8. False-closure guardrails

- Does **not** add `alpha_intraday_micro` to `config/strategy_weights.json`.
- Does **not** silently enable the strategy.
- Does **not** modify `DEFERRED_STRATEGIES`.
- Does **not** authorize live trading.
- Does **not** make a current-positions claim.
- Does **not** preempt the operator's Box-033 A / B / C decision.

**live trading not authorized. CHAD remains PAPER. `ready_for_live=false`.**
