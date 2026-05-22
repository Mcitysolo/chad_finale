# BOX-036 — XGB Veto Documentation (active model, threshold, shadow flag, promotion workflow)

- **Box:** 036 — GAP-019 XGB veto documented
- **Stage:** 3 — Engineering, tests, SSOT, hidden-gap closure
- **Document timestamp (UTC):** 2026-05-20T01:29:43Z
- **Document type:** documentation/policy attestation (no live runtime mutation, no service restart, no live trading authorization)

This document consolidates the active XGB veto inventory required by Completion Matrix Box 036. It is sourced exclusively from in-tree code, in-tree tests, the tracked baseline manifest, the live systemd unit, and read-only runtime evidence from the journal. It does not change behaviour and does not flip enforcement on.

---

## 1. Active model inventory

| Field | Value | Source |
| --- | --- | --- |
| Active model path (preferred) | `runtime/models/xgb_veto/current/xgb_veto_model.json` *(absent — runtime/current/ dir does not exist on this host)* | `chad/analytics/ml_veto_predictor.py:38-40` |
| Active model path (fallback, current source-of-truth) | `shared/models/xgb_veto_model.json` | `chad/analytics/ml_veto_predictor.py:31-32`; CLI: `python3 scripts/promote_xgb_veto.py --status` → `source=baseline` |
| Model version | `xgb_veto_20260510_020007` | `shared/models/xgb_veto_manifest.json` field `model_version`; reproduced by promote CLI status |
| Manifest path (active) | `shared/models/xgb_veto_manifest.json` | predictor `_load_manifest` (runtime path missing → baseline) |
| Manifest hash (short, logged) | `sha256:74afb9e1c8f3e0fe4` | journal `ML_SHADOW … manifest_hash=sha256:74afb9e1c8f3e0fe4` |
| Manifest hash (full) | `sha256:74afb9e1c8f3e0fe4a9c2012dd09b1fa78fab984f1c2308cc6cd839ebc6aa3fb` | manifest `model_sha256` field |
| Trained-at (UTC) | `2026-05-10T02:00:07.712729+00:00` | manifest `trained_at_utc` |
| Training samples | 1091 | manifest `training_samples` |
| Validation accuracy | 0.7534 | manifest `metrics.accuracy` |
| Validation logloss | 0.5364 | manifest `metrics.logloss` |
| Dataset hash | `sha256:68b35d6f442b2f18ef507a14d0298e0d45ecfc713b2d47028382db55fa37a715` | manifest `dataset_hash` |
| Prior model backup | `shared/models/xgb_veto_model_20260510_020007.json` | manifest `prior_model_backup` |
| Feature set (10) | `strategy_encoded`, `regime_encoded`, `vix_level`, `hour_of_day`, `day_of_week`, `scr_sizing_factor`, `is_buy`, `equity_normalized`, `recent_win_rate`, `regime_confidence` | predictor `FEATURE_NAMES`; manifest `feature_names` |

The on-disk dirty model from the 2026-05-17 retrain (`xgb_veto_20260517_020001`, accuracy 0.7112) is **not** the active model — `shared/models/xgb_veto_manifest.json` still names version `xgb_veto_20260510_020007`. The two timestamped model blobs in `shared/models/xgb_veto_model_2026051{0,7}*.json` are untracked rollback backups, not active artefacts. The 2026-05-17 trainer dirty state is the subject of `docs/XGB_VETO_MODEL_ARTIFACT_HYGIENE_PLAN_2026-05-17.md` (Phase 5 — pending operator decision).

## 2. Threshold

- **Active value:** `0.65` (default, set by env-var fallback in predictor source).
- **Source of truth:** `chad/analytics/ml_veto_predictor.py:44` — `VETO_THRESHOLD = float(os.environ.get("CHAD_ML_VETO_THRESHOLD", "0.65"))`.
- **Runtime override:** `CHAD_ML_VETO_THRESHOLD` is **not** set in the live systemd environment (`systemctl show chad-live-loop.service --property=Environment`), so the default applies.
- **Logged value (per cycle):** `threshold=0.65` in the journal `ML_SHADOW` line.

## 3. Shadow-only flag (enforcement gating)

The predictor never blocks trades unless **all** of the following are simultaneously true:

| Gate | Env var / signal | Live state | Source |
| --- | --- | --- | --- |
| Master enforcement switch | `CHAD_ML_VETO_ENABLED` ∈ {1, true, yes, on} | **UNSET → OFF** | `ml_veto_predictor.py:405-411`; live `systemctl show chad-live-loop.service --property=Environment` does not contain it |
| Canary strategy allowlist (must be non-empty AND match `intent.strategy`) | `CHAD_ML_VETO_CANARY_STRATEGIES` (CSV) | **UNSET → no canary, no enforcement** | `ml_veto_predictor.py:392-402`; env not set live |
| Intent class must be ENTRY | `classify_intent(intent, ctx)` returns `INTENT_ENTRY` | enforced by predictor and re-checked in `live_loop.py:1938` | `ml_veto_predictor.py:304-389`, `live_loop.py:1932-1939` |
| Manifest must be valid AND fresh | `_manifest_is_valid` AND `not _manifest_is_stale` (≤ 30 d) | manifest is valid; freshness check at request time | `ml_veto_predictor.py:192-241` |
| Manifest sample count | `training_samples >= CHAD_ML_VETO_MIN_SAMPLES` (default 100) | manifest reports 1091 — OK | `ml_veto_predictor.py:53, 219-225` |

**Net live behaviour: shadow_only.** Every entry intent is scored, the decision is INFO-logged to journald as `ML_SHADOW …`, and **no intent is vetoed**. Protective intents (exit / reduction / hedge / liquidation) are never even eligible for enforcement (`is_protective_intent` short-circuit on `ml_veto_predictor.py:529-533`).

### Live evidence (read-only, 2026-05-20)

```
ML_SHADOW symbol=M6E strategy=omega_macro intent_class=entry \
  model_version=xgb_veto_20260510_020007 manifest_hash=sha256:74afb9e1c8f3e0fe4 \
  loss_prob=0.310 threshold=0.65 would_veto=False final_action=shadow_only \
  reason=loss_probability_below_threshold

ML_SHADOW symbol=MCL strategy=gamma_futures intent_class=entry \
  model_version=xgb_veto_20260510_020007 manifest_hash=sha256:74afb9e1c8f3e0fe4 \
  loss_prob=0.865 threshold=0.65 would_veto=True final_action=shadow_only \
  reason=loss_probability_below_threshold
```

Note the second line — `would_veto=True` but `final_action=shadow_only`: the model would have vetoed this entry if enforcement were on, the live system shadow-logged the intended action and let the intent through. This is the desired Batch-9 shadow soak behaviour.

## 4. Strategies / lane affected

- Lane: signal → intent path in `live_loop.run_once`, applied to every routed entry intent immediately before submission (`chad/core/live_loop.py:1891-1942`).
- Strategies scored: all strategies whose intents flow through this path (every Stage-4 entry intent). The 10-dim `_STRATEGY_MAP` (`ml_veto_predictor.py:82-89`) covers alpha / alpha_intraday / alpha_futures / alpha_options / alpha_crypto / beta / beta_trend / delta / delta_pairs / gamma / gamma_futures / gamma_reversion / omega_vol / omega_momentum_options / omega_macro / omega.
- Strategies eligible for enforcement: **none on this host** — the canary allowlist is empty.

## 5. Promotion workflow (current implementation)

Source: `docs/XGB_VETO_MODEL_ARTIFACT_HYGIENE_PLAN_2026-05-17.md`, Phases 1-3 (implemented), Phase 5 (pending operator decision). CLI: `scripts/promote_xgb_veto.py`. Tests: `chad/tests/test_xgb_promotion_workflow.py` (14 tests, all passing).

### Stages

1. **Train (automated, Sunday 02:00 UTC).**
   - Unit: `/etc/systemd/system/chad-xgb-train.service`, triggered by `chad-xgb-train.timer` (`OnCalendar=Sun *-*-* 02:00:00`).
   - Module: `chad/analytics/train_xgb_model.py`.
   - Post-Phase-2 target: writes candidate model + manifest into `runtime/models/xgb_veto/candidates/<UTC ts>/{xgb_veto_model.json, xgb_veto_manifest.json}`. The candidate dir name is `YYYYMMDD_HHMMSS` and becomes the candidate id used by the promotion CLI.
   - **Never overwrites the active model.** Verified by `test_trainer_does_not_overwrite_shared_models`.

2. **List / inspect candidates.**
   ```
   python3 scripts/promote_xgb_veto.py --list
   python3 scripts/promote_xgb_veto.py --status
   ```
   The status command prints the currently active manifest (preferring `runtime/models/xgb_veto/current/` over `shared/models/`). On this host today, status reports `source=baseline`, `model_version=xgb_veto_20260510_020007`, `accuracy=0.7534`, `logloss=0.5364`.

3. **Promote (operator-gated).**
   ```
   # auto-gate path (only if delta_acc >= 0 AND delta_loss <= 0):
   python3 scripts/promote_xgb_veto.py --candidate <YYYYMMDD_HHMMSS>

   # override (operator approval, audited reason recorded in manifest):
   python3 scripts/promote_xgb_veto.py --candidate <YYYYMMDD_HHMMSS> \
       --operator-approve "<reason>"
   ```
   - **Auto-gate criteria:** `accuracy_new >= accuracy_current` AND `logloss_new <= logloss_current`. A regression on either metric blocks promotion (`scripts/promote_xgb_veto.py:202-212`).
   - **Operator override:** mandatory non-empty `--operator-approve "<reason>"`; the reason is persisted to the promoted manifest as `operator_approve_reason`. CLI rejects `--operator-approve` without `--candidate`.
   - **Atomicity:** model is copied with tmp + `os.replace`; manifest is written with tmp + `os.replace`. No partial writes.
   - **Audit fields written into the promoted manifest:** `promoted_at_utc`, `promoted_from_candidate`, `promoted_by` (`auto_gate` or `operator_approve:<reason>`), `promoted_from_source`, `metrics_delta_accuracy`, `metrics_delta_logloss`, `model_path`, `operator_approve_reason` (when override used).
   - The predictor module caches model + manifest on first call; an operator promotion takes effect on the next predictor process restart (the in-process `_load_manifest` / `_load_model` are gated on `_manifest_load_attempted` / `_model_load_attempted`). No service restart was performed by this audit. Effective propagation on the next natural restart cycle of `chad-live-loop.service`, or on demand via the existing operator-controlled restart procedure (not part of this audit).

4. **Rollback.**
   - The previous active model is preserved by the trainer at `<active_manifest>.prior_model_backup` (`shared/models/xgb_veto_model_<prior ts>.json` today).
   - To roll back, the operator promotes the prior-backup candidate by recreating its candidate dir under `runtime/models/xgb_veto/candidates/<prior_ts>/` and running the promote CLI with `--operator-approve "rollback to <prior_ts>"`. The rollback is recorded in the new active manifest via the same audit triple.

5. **Live activation of enforcement (separate from promotion).**
   - Promotion only changes *which* model the predictor loads. It does **not** turn on enforcement.
   - Activation requires both `CHAD_ML_VETO_ENABLED=1` and a non-empty `CHAD_ML_VETO_CANARY_STRATEGIES=<csv>` in the live-loop systemd environment.
   - Per CHAD governance rules §3 / §6 / §7 such changes are **Pending Actions only** — they must not be applied directly by Claude Code or by this audit. The Box-036 closure deliberately does **not** flip enforcement.

## 6. Operator approval requirement (summary)

| Operation | Approval required | Mechanism |
| --- | --- | --- |
| Train candidate | none (timer-driven) | systemd timer |
| Promote candidate (metric-pass) | none (auto-gate) | `promote_xgb_veto.py --candidate <id>` |
| Promote candidate (metric-fail) | **operator** | `--operator-approve "<reason>"` (reason persisted to manifest) |
| Roll back to prior model | **operator** | re-promote prior candidate with `--operator-approve "rollback to <id>"` |
| Enable enforcement | **operator + governance Pending Action** | systemd env: `CHAD_ML_VETO_ENABLED=1` + `CHAD_ML_VETO_CANARY_STRATEGIES=<csv>` |
| Change threshold | **operator + governance Pending Action** | systemd env: `CHAD_ML_VETO_THRESHOLD=<float>` |

## 7. Evidence commands (reproducible)

```bash
# Active model summary (no mutation):
python3 scripts/promote_xgb_veto.py --status

# Active manifest body:
cat shared/models/xgb_veto_manifest.json

# Live env (proves CHAD_ML_VETO_* not set):
systemctl show chad-live-loop.service --property=Environment

# Shadow-soak evidence (last hour):
journalctl -u chad-live-loop.service --since '1 hour ago' \
  | grep -E 'ML_SHADOW|ML_VETO'

# Test suite:
python3 -m pytest chad/tests/test_ml_veto_loop.py \
                  chad/tests/test_xgb_promotion_workflow.py -q
```

## 8. Forbidden by this document

- Do not flip `CHAD_ML_VETO_ENABLED` on without a governance-approved Pending Action.
- Do not populate `CHAD_ML_VETO_CANARY_STRATEGIES` outside a governance-approved Pending Action.
- Do not edit the systemd unit or drop-ins to change ML env vars without an operator-approved Pending Action.
- Do not edit `shared/models/xgb_veto_model.json` or `shared/models/xgb_veto_manifest.json` by hand — use the promote CLI or a controlled rollback per §5.4.
- Do not commit the 2026-05-17 dirty retrain — see `docs/XGB_VETO_MODEL_ARTIFACT_HYGIENE_PLAN_2026-05-17.md` §10.

## 9. Cross-references

- Design plan: `docs/XGB_VETO_MODEL_ARTIFACT_HYGIENE_PLAN_2026-05-17.md`
- Decision doc: `docs/XGB_VETO_WEEKLY_RETRAIN_DIRTY_TREE_DECISION_2026-05-17.md`
- Predictor: `chad/analytics/ml_veto_predictor.py`
- Trainer: `chad/analytics/train_xgb_model.py`
- Promotion CLI: `scripts/promote_xgb_veto.py`
- Live wiring: `chad/core/live_loop.py:1891-1942`
- Tests: `chad/tests/test_ml_veto_loop.py` (22 tests), `chad/tests/test_xgb_promotion_workflow.py` (14 tests)
- Evidence: `runtime/completion_matrix_evidence/BOX-036_GAP-019_XGB_veto_documented.md`
