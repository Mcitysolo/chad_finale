# PENDING ACTION — W6B-6: wire the ML veto baseline (BLOCKED until sufficient)

**Status:** PREPARED, **NOT READY TO APPLY.** Blocked by a gate this lane built on purpose.
**Type:** config (`config/exterminator.json` → `ml`) + one systemd timer.
**Prepared:** 2026-07-24 (W6B-6)

---

## Part 1 — a correction to `config/exterminator.json` that IS ready now

The config asserts:

> "no live rate can be computed or compared" … "veto decisions leave only log lines
> (ML_SHADOW / ML_VETO), no durable artifact, no counter"

**The second clause is true. The first is false, and has been all along.** The ML_SHADOW
lines at `chad/core/live_loop.py:2996-3010` are fully structured and carry every field a
baseline needs. W6B-4/5 computed one.

Replace the `_doc` / `no_baseline_detail` wording with:

> Veto decisions emit structured ML_SHADOW lines. `chad-ml-veto-collector` parses them into
> `runtime/ml_veto_shadow.ndjson` (`ml_veto_shadow.v1`) and publishes
> `runtime/ml_veto_baseline.json` (`ml_veto_baseline.v1`). A production rate IS computable.
> It is still not comparable to the manifest's `val_veto_rate_at_0.65`, which is a
> training-time validation statistic over a different distribution — that comparison is a
> category error regardless of sample size.

This correction is safe to apply immediately; it is a factual fix to a doc string.

## Part 2 — measured production behaviour

First live collection (2026-07-24, `--since -12h`, 526 rows):

| Metric | Value |
|---|---|
| samples | 526 |
| `would_veto=True` | **1** |
| **production veto rate** | **0.0019 (0.19%)** |
| manifest `val_veto_rate_at_0.65` | 0.7123 (**71.2%**) |
| loss_prob mean / p50 / p90 / p99 | 0.179 / 0.165 / 0.269 / 0.569 |
| model versions | 1 (`xgb_veto_20260510_020007`) |

The divergence is roughly **375x**, larger than the ~65x the plan measured on a shorter
window. That gap is itself the headline: in production the model sees a materially
different distribution than in its training validation split.

**And the stratification immediately earned its place:**

| strategy | n | vetoes | rate |
|---|---|---|---|
| gamma | 493 | 0 | 0.0% |
| omega_macro | 30 | 1 | 3.3% |
| beta | 3 | 0 | 0.0% |

The headline 0.19% is really *"gamma never vetoes"* — gamma is 94% of the window. Quoting
the aggregate as a portfolio veto rate would have been wrong in a way no sample-size check
would have caught.

## Part 3 — why this PA is BLOCKED, and by what

```
sufficient = false
insufficient_reasons = ["observed_span_hours=5.26 < min_span_hours=72"]
```

Sufficiency deliberately requires **both** a sample floor (200) and an elapsed-coverage
floor (72h). Sample count alone would have passed this window at 526 rows — and it is
5.26 hours of a single boot, one market session, 94% one strategy. A veto rate wired from
that would be presented as production behaviour and would move the moment the regime
changed.

**Do not apply Part 4 until `runtime/ml_veto_baseline.json` reports `sufficient: true`.**
That is expected roughly 3 days after the collector timer is installed.

Note the store currently holds only what the journal retained: this host keeps the current
boot only, which is precisely why the durable NDJSON exists. Coverage starts accumulating
from the moment the timer is installed, not retroactively.

## Part 4 — the wiring itself (apply only when sufficient)

```json
"ml": {
  "baseline_veto_rate": <veto_rate from runtime/ml_veto_baseline.json>,
  "baseline_veto_rate_source": "runtime/ml_veto_baseline.json (ml_veto_baseline.v1), window_hours=168, min_samples=200, min_span_hours=72",
  "baseline_veto_rate_stratum": "portfolio",
  "veto_rate_drift_band": 0.05
}
```

Two cautions carried from the measurement:

1. **Re-read `by_strategy` before accepting the portfolio number.** If one strategy still
   dominates the window, wire that stratum explicitly rather than pretending the aggregate
   is portfolio-wide.
2. **All observed rows are `intent_class=entry`.** There is no protective-intent coverage
   at all, so the baseline governs entries only and must not be generalised.

## Part 5 — collector timer

`deploy/chad-ml-veto-collector.service` and `.timer` ship with this lane (not installed).
The timer runs `collect` every 15 minutes and `baseline` hourly. Frequent collection is
required, not cosmetic: journald here retains only the current boot, so anything not
collected before a restart is gone.

Installation is an operator action (governance: no systemd edits without instruction).

## Verification

```bash
cd /home/ubuntu/chad_finale && source venv/bin/activate
python3 -m chad.analytics.ml_veto_shadow_collector collect --since="-24h"
python3 -m chad.analytics.ml_veto_shadow_collector baseline --dry-run | head -40
python3 -m pytest chad/tests/test_w6b_ml_veto_baseline.py -q    # 23 tests
```

`collect` is idempotent — overlapping windows are deduplicated by `row_key`, proven by
`test_append_is_idempotent_across_runs`.

## Rollback

Remove `baseline_veto_rate` from `config/exterminator.json` (EXS8 returns to
`no_baseline`), disable the timer, and optionally delete
`runtime/ml_veto_shadow.ndjson`. Nothing in the veto path changes at any point — the
predictor is untouched and stays shadow-only.

## Evidence

| Claim | Source |
|---|---|
| ML_SHADOW lines are structured | `chad/core/live_loop.py:2996-3010` |
| 526 rows / 1 veto / 0.19% | `ml_veto_shadow_collector collect --since=-12h` then `baseline`, 2026-07-24 |
| Training-time 71.2% | `shared/models/xgb_veto_manifest.json` → `metrics.val_veto_rate_at_0.65` |
| gamma 493/0, omega_macro 30/1, beta 3/0 | baseline `by_strategy` |
| sufficient=false, span 5.26h | baseline `insufficient_reasons` |
| Tests | `chad/tests/test_w6b_ml_veto_baseline.py` (23) |
