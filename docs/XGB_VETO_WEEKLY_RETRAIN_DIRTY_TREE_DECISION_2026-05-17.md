# XGB Veto Weekly Retrain Dirty-Tree Decision — 2026-05-17

## 1. Title
XGB Veto Weekly Retrain Dirty-Tree Decision — 2026-05-17

## 2. Status
**DECISION REQUIRED — DO NOT COMMIT BLINDLY**

No action has been taken on the dirty model artifacts. Operator review is
required before the working tree is brought clean.

## 3. What happened

- `chad-xgb-train.timer` (systemd, Sunday 02:00 UTC weekly) fired at
  `2026-05-17 02:00:00 UTC`.
- The timer launched `chad-xgb-train.service`, which executed
  `python3 chad/analytics/train_xgb_model.py`.
- The trainer rewrote both shared model artifacts in place:
  - `shared/models/xgb_veto_manifest.json`
  - `shared/models/xgb_veto_model.json`
- The trainer also wrote a timestamped backup of the prior model alongside
  the active artifact, recorded in the manifest's `prior_model_backup`
  field (the SS08 rollback path).
- File mtimes are `2026-05-17 02:00:01`, which matches the timer's
  last-run timestamp to the second.
- `git status` now reports both files as modified. No other files are dirty.

## 4. Why this matters

- These files are **runtime-generated artifacts** produced by an automated
  weekly retrain, but they are currently **tracked by git**. That is a
  hybrid state introduced under SS08 (see
  `docs/CHAD_UNIFIED_SSOT_v8.9_2026-05-03.md`).
- Every Sunday at 02:00 UTC the working tree will go dirty again
  regardless of operator activity. This will keep happening forever
  unless the tracking model is changed.
- `shared/models/xgb_veto_model.json` is a ~297 KB single-line JSON
  serialized XGBoost booster. Its diffs are not human-reviewable — git
  treats the entire file as one changed line.
- Mid-task model rotations contaminate unrelated commits. They can be
  accidentally swept into a `git add -A` or `git commit -a` during
  unrelated work (e.g. the in-flight Phase D dynamic-universe work),
  violating CHAD governance rule §1 ("one change at a time").
- A blind `git checkout -- shared/models/...` would overwrite the
  live predictor's on-disk weights (loaded by
  `chad/analytics/ml_veto_predictor.py:31-32`) without going through the
  documented rollback procedure.

## 5. Current retrain metrics

Values are taken directly from `git diff -- shared/models/xgb_veto_manifest.json`.

| Metric                    | Previous (2026-05-10) | Current  (2026-05-17) | Direction |
|---------------------------|-----------------------|-----------------------|-----------|
| `validation_accuracy`     | 0.7534                | 0.7112                | regressed (−4.22 pp) |
| `validation_logloss`      | 0.5364                | 0.5953                | worsened (+0.0589)    |
| `metrics.accuracy`        | 0.7534                | 0.7112                | regressed |
| `metrics.logloss`         | 0.5364                | 0.5953                | worsened  |
| `metrics.base_loss_rate`  | 0.6609                | 0.6788                | more losses in window |
| `metrics.n_train`         | 872                   | 926                   | +54 samples |
| `metrics.n_val`           | 219                   | 232                   | +13 samples |
| `training_samples`        | 1,091                 | 1,158                 | +67 samples |
| `val_veto_rate_at_0.65`   | 0.7123                | 0.6379                | less aggressive veto |
| `excluded.quarantined`    | 24                    | 316                   | +292 |
| `excluded.untrusted`      | 8,243                 | 10,396                | +2,153 |
| `model_version`           | `xgb_veto_20260510_020007` | `xgb_veto_20260517_020001` | rotated |
| `trained_at_utc`          | 2026-05-10T02:00:07Z  | 2026-05-17T02:00:01Z  | weekly cadence |
| `schema_version`          | `xgb_manifest.v2`     | `xgb_manifest.v2`     | unchanged |
| `feature_names`           | 10 features           | 10 features (identical order) | unchanged |

**Headline:** accuracy regressed (0.7534 → 0.7112) and logloss worsened
(0.5364 → 0.5953). The retrain is structurally valid (schema and feature
parity intact) but performance is worse than the prior model.

## 6. Immediate decision

**Do not automatically commit.**
**Do not automatically revert.**

The operator must explicitly choose one of:

- **A. Accept retrain and commit model artifacts.** Treat the
  2026-05-17 retrain as the new tracked baseline despite the metrics
  regression. Commits should be isolated from Phase D work.
- **B. Reject retrain and perform controlled rollback.** Restore the
  2026-05-10 model using the documented SS08 rollback
  (`cp <prior_model_backup> shared/models/xgb_veto_model.json`,
  re-emit a manifest for the prior weights, or `git checkout` the prior
  blob), then disable or pause the timer until the regression cause is
  understood.
- **C. Leave uncommitted temporarily while planning artifact
  relocation.** Acknowledge the retrain happened, do not commit it, and
  schedule a controlled follow-up task that removes runtime-generated
  artifacts from the tracked git path.

## 7. Recommended decision

**Recommend C now.**

- Leave the two files uncommitted pending operator review.
- Do **not** mix model artifacts with Phase D commits (in-flight
  dynamic-universe / dashboard work).
- Create a separate future task to either:
  - move generated model artifacts out of the tracked git path, or
  - formalize a model artifact promotion workflow (train → evaluate →
    approve → promote).

This avoids the worst failure modes (accidental commit of a regressed
model under an unrelated PR title; blind rollback that desyncs the
on-disk predictor from the manifest) while deferring the structural
remediation to its own controlled change window, in line with CHAD
governance rule §1 (one change at a time).

## 8. Future remediation options

Non-exhaustive list of structural fixes that should be evaluated in a
dedicated follow-up task (not in this decision record):

- Move generated model artifacts to `runtime/models/` (mirrors the
  existing `runtime/` convention for live-state files).
- Keep a versioned baseline artifact in git but write weekly retrains
  to a separate, untracked path; promotion copies an approved retrain
  into the tracked baseline.
- Add the generated artifact paths to `.gitignore` **after** the
  migration above is in place — not before, since that would silently
  hide the dirty-tree symptom without fixing the underlying tracking
  model.
- Create a model promotion workflow: train → evaluate → approve →
  promote, with explicit operator gating between evaluate and promote.
- Create a rollback command that restores a known-good model from a
  backup and writes an audit trail (timestamp, operator, prior + new
  `model_sha256`, reason).
- Surface retrain metrics regressions to the dashboard / live_readiness
  publisher so a regressed retrain blocks promotion automatically.

## 9. Forbidden actions

The following actions are explicitly forbidden under this decision
record without separate operator approval:

- Do **not** commit `shared/models/xgb_veto_model.json` blindly.
- Do **not** commit `shared/models/xgb_veto_manifest.json` blindly.
- Do **not** revert live predictor weights with `git checkout` without
  going through the documented SS08 rollback procedure (timestamped
  backup → copy → manifest re-emit).
- Do **not** mix model artifacts into unrelated Phase D commits.
- Do **not** disable `chad-xgb-train.timer` without separate operator
  approval (rule §6 protections on systemd unit files apply).
- Do **not** add the model artifact paths to `.gitignore` while they
  remain tracked — that hides the symptom without fixing the cause.

## 10. Verification evidence

### 10.1 Git status at decision time

```
 M shared/models/xgb_veto_manifest.json
 M shared/models/xgb_veto_model.json
```

### 10.2 Diff stat

```
 shared/models/xgb_veto_manifest.json | 32 ++++++++++++++++----------------
 shared/models/xgb_veto_model.json    |  2 +-
 2 files changed, 17 insertions(+), 17 deletions(-)
```

### 10.3 Timer / service evidence

`systemctl list-timers --all` (filtered):

```
Sun 2026-05-24 02:00:00 UTC  6 days  Sun 2026-05-17 02:00:00 UTC  39min ago
  chad-xgb-train.timer  chad-xgb-train.service
```

`systemctl list-units --all` (filtered):

```
chad-xgb-train.service  loaded  inactive  dead     CHAD XGBoost Model Retraining
chad-xgb-train.timer    loaded  active    waiting  CHAD XGBoost Weekly Retraining Timer
```

### 10.4 Writer references

- `chad/analytics/train_xgb_model.py:48-49` defines
  `MODEL_PATH` and `MANIFEST_PATH` pointing at
  `shared/models/xgb_veto_{model,manifest}.json`.
- `chad/analytics/train_xgb_model.py:502` writes the timestamped backup
  (`xgb_veto_model_<UTC ts>.json`) used as `prior_model_backup`.
- `chad/analytics/train_xgb_model.py:532` stamps the
  `model_version = xgb_veto_<UTC ts>` value — matches the value in the
  current dirty manifest exactly (`xgb_veto_20260517_020001`).
- `chad/analytics/ml_veto_predictor.py:31-32` is the live consumer of
  these files; any rollback that bypasses the SS08 procedure will
  desync the predictor from the manifest.

### 10.5 Metrics regression summary

- `validation_accuracy`: 0.7534 → 0.7112 (regressed)
- `validation_logloss`:  0.5364 → 0.5953 (worsened)

Retrain is structurally valid but performance is worse than the prior
model. The regression alone is sufficient justification for not
auto-committing.

### 10.6 Document scope

This record only documents the situation and the decision required.
It does **not** authorize any change to:

- `shared/models/*`
- `chad/*`
- `deploy/*`
- `ops/*`
- `config/*`
- `runtime/*`
- existing docs
- systemd units (start / stop / restart / enable / disable)

A separate, explicitly-authorized task is required to act on any of the
options in §6, §7, or §8.
