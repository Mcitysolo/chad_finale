# XGB Veto Model Artifact Hygiene Plan — 2026-05-17

## 1. Title

XGB Veto Model Artifact Hygiene Plan — 2026-05-17

## 2. Status

**DESIGN ONLY — NO FILE MOVES YET**

This document defines the target architecture for resolving the recurring
"dirty working tree" caused by `chad-xgb-train.timer` rewriting tracked
files under `shared/models/` every Sunday. No code, config, or model
files are modified by this task. No commits are made. The
implementation phases listed in §8 are deliberately deferred to a
follow-up, governance-approved work item.

This plan is the controlled answer to the question raised by
`docs/XGB_VETO_WEEKLY_RETRAIN_DIRTY_TREE_DECISION_2026-05-17.md`
(commit `be0155d`), which intentionally stopped at the "decision
required" step.

## 3. Current problem

- Weekly retrain writes tracked files.
  `chad-xgb-train.service` invokes `chad/analytics/train_xgb_model.py`,
  which overwrites `shared/models/xgb_veto_model.json` and
  `shared/models/xgb_veto_manifest.json` in place.
- Git becomes dirty every Sunday.
  `git status` reports both files modified after each timer firing.
  This contaminates unrelated commits and violates CHAD governance §1
  ("one change at a time").
- Latest retrain metrics regressed.
  `validation_accuracy` 0.7534 → 0.7112 (−4.22 pp).
  `validation_logloss` 0.5364 → 0.5953 (worse).
  The trainer promoted the candidate to active anyway because there
  is no metric gate.
- Current tracked-runtime hybrid is not sustainable.
  Model files are runtime-generated artifacts (serialized XGBoost
  boosters, ~297 KB single-line JSON, not human-reviewable in diffs)
  but git tracks them as if they were source. The contradiction
  surfaces as recurring dirty-tree friction with no clean resolution
  under the current layout.

## 4. Current evidence

### Dirty files

```
 M shared/models/xgb_veto_manifest.json
 M shared/models/xgb_veto_model.json
```

No other files are dirty. Snapshot captured 2026-05-17 ~02:48 UTC, ~48
minutes after the timer fired.

### Tracked artifacts (entire `shared/models/` tracked surface)

```
shared/models/xgb_veto_manifest.json
shared/models/xgb_veto_model.json
```

Only two files. Timestamped backup files
(`xgb_veto_model_<UTC ts>.json`) are **not** tracked and currently live
alongside the active artifact:

```
shared/models/xgb_veto_model_20260503_115106.json   (untracked)
shared/models/xgb_veto_model_20260510_020007.json   (untracked, ~373 KB)
shared/models/xgb_veto_model_20260517_020001.json   (untracked, ~299 KB)
```

The 2026-05-17 backup is a copy of the **prior** (2026-05-10)
active model, taken by the trainer immediately before it overwrote
`xgb_veto_model.json`. See §4 / backup availability below.

### Writer service / timer

- Unit file: `/etc/systemd/system/chad-xgb-train.service`
  `ExecStart=…/venv/bin/python3 …/chad/analytics/train_xgb_model.py`
- Timer:     `/etc/systemd/system/chad-xgb-train.timer`
  `OnCalendar=Sun *-*-* 02:00:00`, `Persistent=true`, `AccuracySec=1h`.
- Next firing: `Sun 2026-05-24 02:00:00 UTC` (per `systemctl list-timers`).
- Last firing: `Sun 2026-05-17 02:00:00 UTC` (the firing that produced
  the current dirty tree).

Per CHAD governance rule §6, these unit files are **not** to be
modified by this plan or by any follow-up implementation phase without
explicit operator instruction.

### Writer code path

`chad/analytics/train_xgb_model.py`:

- Lines 48-49: constants
  `MODEL_PATH = MODELS_DIR / "xgb_veto_model.json"`,
  `MANIFEST_PATH = MODELS_DIR / "xgb_veto_manifest.json"`,
  where `MODELS_DIR = <repo>/shared/models`.
- Lines 499-509: backup step. Before overwriting `MODEL_PATH`, the
  trainer copies the existing file to
  `MODEL_PATH.with_name(f"xgb_veto_model_{ts}.json")` and records that
  path in `manifest.prior_model_backup`.
- Lines 511-515: `bst.save_model(str(MODEL_PATH))` — unconditional
  overwrite of the tracked artifact.
- Lines 540-575: builds and writes `MANIFEST_PATH` with new
  `model_version`, `model_sha256`, `metrics`, `prior_model_backup`.
- No metric gate: training proceeds and promotes regardless of
  whether new metrics are better or worse than the previous run.

### Reader code path

`chad/analytics/ml_veto_predictor.py`:

- Lines 31-32: constants
  `MODEL_PATH = REPO_ROOT / "shared" / "models" / "xgb_veto_model.json"`,
  `MANIFEST_PATH = REPO_ROOT / "shared" / "models" / "xgb_veto_manifest.json"`.
- Lines 154-172: `_load_manifest()` reads `MANIFEST_PATH` once on
  first call, caches result. Fail-open on missing/broken manifest.
- Lines 251-269: `_load_model()` reads `MODEL_PATH` once on first call,
  caches XGBoost `Booster`. Fail-open on missing/broken model.
- Tests already monkeypatch both `MODEL_PATH` and `MANIFEST_PATH`
  (see `chad/tests/test_ml_veto_loop.py:156-157, 254-255`), so the
  predictor surface is already structured to support relocating
  these paths.

### Metrics regression (current dirty state)

| Field                       | HEAD (committed) | Working tree (dirty) | Direction |
|-----------------------------|------------------|----------------------|-----------|
| `model_version`             | `xgb_veto_20260510_020007` | `xgb_veto_20260517_020001` | rotated |
| `metrics.accuracy`          | 0.7534           | 0.7112               | regressed |
| `metrics.logloss`           | 0.5364           | 0.5953               | worsened  |
| `metrics.base_loss_rate`    | 0.6609           | 0.6788               | more losses |
| `metrics.n_train`           | 872              | 926                  | +54 samples |
| `excluded.untrusted`        | (prior value)    | 10396                | quarantine surface large |
| `excluded.quarantined`      | (prior value)    | 316                  | normal |

Source: live `shared/models/xgb_veto_manifest.json` vs
`git show HEAD:shared/models/xgb_veto_manifest.json`.

### Backup availability (rollback feasibility)

- Trainer-emitted backup `shared/models/xgb_veto_model_20260517_020001.json`
  is present on disk and contains the **previous** (2026-05-10) model
  weights. Rollback is therefore possible in principle: copying that
  file over `shared/models/xgb_veto_model.json` and reverting the
  manifest would restore the committed predictor state.
- A second, older backup (`xgb_veto_model_20260510_020007.json`) is also
  present, covering one further regression step.
- These backups are **untracked** under the current layout. Any
  migration must preserve operator access to them.

## 5. Recommended option

**Option C — Add model promotion workflow.**

The recurring dirty-tree symptom is one face of a deeper problem: the
trainer treats every retrain as authoritative and overwrites the live
predictor's weights without any gate. The current regression
(−4.22 pp accuracy, +0.0589 logloss) would have been silently
promoted into the live predictor if an operator had run a routine
`git add -A` during unrelated work. Fixing the git surface (Option B)
without fixing the promotion semantics would still leave that hazard
in place. Option C addresses both.

## 6. Recommended target architecture (Option C)

- `shared/models/` keeps the **reviewed baseline** only. The active
  baseline is the model the operator has explicitly accepted; it is
  intentionally rotated rarely (e.g. when an operator-driven model
  review concludes that a candidate should become the new ground
  truth). It remains tracked by git so that the committed history
  always reflects an auditable, reviewed predictor.
- `runtime/models/xgb_veto/candidates/` stores weekly retrain
  candidates. Filename includes the UTC timestamp and the candidate
  manifest sits alongside its booster. This directory is **not**
  tracked by git (`runtime/` is already gitignored — see §8 Phase 4).
- `runtime/models/xgb_veto/current/` stores the **approved active
  model** (the one the predictor actually loads when present). This
  directory is also not tracked by git. Promotion is the act of
  publishing a candidate into this directory.
- The manifest in `runtime/models/xgb_veto/current/` tracks active
  candidate metadata: `model_version`, `trained_at_utc`,
  `dataset_hash`, `model_sha256`, `metrics`, plus a new
  `promoted_from_candidate` / `promoted_at_utc` / `promoted_by` triple
  for audit.
- `chad/analytics/ml_veto_predictor.py` loads the **current runtime
  model first** (`runtime/models/xgb_veto/current/`); if that
  directory is missing or empty, it falls back to the **baseline** in
  `shared/models/`. Failure to load either falls open as today
  (predictor stays shadow-only / pass). The module-level constants
  `MODEL_PATH` / `MANIFEST_PATH` continue to exist as the baseline
  paths for backwards compatibility with the existing test
  monkeypatch surface.
- `chad/analytics/train_xgb_model.py` writes candidates only into
  `runtime/models/xgb_veto/candidates/<ts>/`. It **does not**
  overwrite the active model or the baseline. The weekly timer can
  therefore continue to fire on Sunday 02:00 UTC without dirtying
  git.
- Promotion requires either (a) a measurable metric improvement
  versus the currently promoted manifest (e.g.
  `accuracy_delta >= 0 AND logloss_delta <= 0`, exact thresholds to
  be set in Phase 3), or (b) explicit operator approval recorded in
  the new manifest fields. Either path is an explicit, auditable
  step — never a silent overwrite.

## 7. Acceptance criteria for future implementation

- Weekly retrain no longer dirties git status.
  `git status --short` is clean immediately after `chad-xgb-train.service`
  completes.
- Existing predictor still loads a valid model. With no runtime
  current/ directory present, the predictor falls back to the
  `shared/models/` baseline and behaves identically to today.
- Full suite passes. `python3 -m pytest chad/tests/ -x -q` is green
  after each phase, and `chad/core/full_cycle_preview.py --dry-run`
  exits clean.
- Model promotion is auditable. Every promotion writes a manifest
  delta into `runtime/models/xgb_veto/current/` that records the
  candidate it came from, the metrics delta vs. the previous
  promoted model, the UTC timestamp, and whether promotion was
  metric-gated or operator-approved.
- Rollback is explicit and logged. Operators can demote the current
  promoted model back to the previous one via a documented command;
  the demotion is recorded in the same manifest stream.
- No strategy behavior changes during migration. Strategy code
  consumes `ml_veto_predictor`'s public surface only; nothing in
  the migration changes that surface, the scoring math, or the
  fail-open behavior.

## 8. Future implementation phases

**Phase 1 — Predictor runtime path support, baseline fallback.**
Add `RUNTIME_MODEL_PATH` / `RUNTIME_MANIFEST_PATH` in
`chad/analytics/ml_veto_predictor.py` pointing at
`runtime/models/xgb_veto/current/`. Load runtime-current first; on
miss, fall back to the existing `shared/models/` baseline. Extend
tests to cover both paths and the fallback transition. No trainer
changes in this phase; the runtime directory will be absent at
first, so predictor behavior is unchanged.

**Phase 2 — Trainer writes candidates, never overwrites active.**
Change `chad/analytics/train_xgb_model.py` so its output target is
`runtime/models/xgb_veto/candidates/<UTC ts>/{model.json,
manifest.json}`. Retire the in-place overwrite of
`shared/models/xgb_veto_model.json` from this code path. The
trainer's existing `prior_model_backup` field continues to record
the previously promoted model for one-step rollback.

**Phase 3 — Promotion command.**
Add a thin promotion CLI (e.g. `scripts/promote_xgb_veto.py`) that
takes a candidate timestamp, compares its metrics to the currently
promoted manifest, and either (a) publishes the candidate into
`runtime/models/xgb_veto/current/` when the gate passes, or (b)
refuses unless given `--operator-approve` with a reason recorded
in the manifest. Promotion clears the predictor's in-process cache
or signals reload as appropriate.

**Phase 4 — `.gitignore` for generated runtime model directories.**
Confirm coverage. `runtime/` is already gitignored at line 7 of
`.gitignore`, so `runtime/models/xgb_veto/**` is implicitly
excluded — verify nothing has been added that re-includes it. Add
an explicit allow/deny line only if required to keep the intent
documented (e.g. `runtime/models/`).

**Phase 5 — Decide whether to commit or revert the current dirty
retrain.**
Once Phase 1-4 are in place and tests are green, make a deliberate
call on the in-tree dirty files. The expected resolution is to
**revert** the current dirty `shared/models/` artifacts back to the
HEAD baseline (using the operator-controlled procedure documented
in §10), because the regressed model would not pass the new
promotion gate. The Sunday 2026-05-24 retrain will then land
cleanly as a runtime candidate.

## 9. Forbidden actions

- Do not commit regressed model blindly.
- Do not `git checkout` model files without controlled rollback.
- Do not disable weekly retrain without operator approval.
- Do not mix model artifact changes into Phase D scanner commits.
- Do not delete backups.
- Do not modify `chad-xgb-train.service` or `chad-xgb-train.timer`
  (CHAD governance §6).
- Do not move or edit any file under `shared/models/` as part of
  this design task.
- Do not edit `.gitignore` as part of this design task.
- Do not stage, revert, or `git add` the currently dirty model
  files as part of this design task.

## 10. Immediate recommendation for current dirty files

**Recommendation: leave uncommitted until migration decision.**

Evidence supporting this choice:

- The candidate model regressed on both reported metrics
  (accuracy −4.22 pp, logloss +0.0589). Committing it would
  propagate worse predictor behavior into the audited history of
  the repository. The decision doc at
  `docs/XGB_VETO_WEEKLY_RETRAIN_DIRTY_TREE_DECISION_2026-05-17.md`
  reaches the same conclusion at §6/§7 — commit is not appropriate
  without operator review.
- Reverting now would discard the regression signal and the
  manifest's recorded `prior_model_backup` pointer, which the
  follow-up phases use to design the promotion gate. The
  prior-model backup blob remains on disk
  (`shared/models/xgb_veto_model_20260517_020001.json` ≡ the
  2026-05-10 weights), so the rollback option is preserved even
  while the working tree stays dirty. Reverting prematurely is
  also a destructive operation under the project's governance
  posture (overwrites uncommitted on-disk state) and should be
  reserved for the deliberate Phase 5 step.
- Disabling the timer is out of scope per §9 and per CHAD governance
  §6/§7. The dirty state is recurring, not catastrophic; the right
  fix is the migration in Phase 1-4, after which the Sunday
  2026-05-24 retrain will land outside the tracked surface.
- The predictor continues to load the dirty model from disk today;
  fail-open behavior protects the system from any pathological
  predictions, and the on-disk regression is recoverable from the
  backup at any point. There is no urgency that justifies an
  irreversible action before the migration design is implemented.

Therefore: do nothing to the dirty files. Hold the decision until
the implementation phases are scheduled, then execute Phase 5 as
the final closure step.
