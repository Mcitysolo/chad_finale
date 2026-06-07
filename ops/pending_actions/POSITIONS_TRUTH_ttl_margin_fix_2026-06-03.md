# Pending Action — positions_truth ttl 60->90s freshness-margin fix
Date: 2026-06-03  •  Author: TEAM CHAD (issued) / SOLO (executes)  •  Status: EXECUTED 2026-06-04 (commit 3bfa924; deploy live-verified; §8(b) observation open)  •  Priority: LOW (hardening; does not block Bug B closure)

## 1. Objective
Raise the published ttl_seconds on positions_truth.json from 60 to 90 to restore positive freshness margin against the lifecycle-truth-publisher cadence (~61-65s), eliminating the recurring marginal-stale window that produces benign FUTURES_POSITION_CAP_UNVERIFIED noise. No behavior change in normal operation.

## 2. Root cause
positions_truth.json embeds ttl_seconds=60, but the publisher fires on OnUnitActiveSec=60 + AccuracySec=5 plus run time -> effective cadence ~61-65s. ttl == cadence (zero slack) means the file is briefly older than its own ttl at the tail of each cycle; fail-closed consumers occasionally reject it -> benign UNVERIFIED. Observed once live at 2026-06-03 20:24:36 (self-healed in 2s).

## 3. The change (single value)
File: chad/ops/lifecycle_truth_publisher.py:693
From: ttl_truth = _env_int("CHAD_POSITIONS_TRUTH_TTL_SECONDS", 60)
To:   ttl_truth = _env_int("CHAD_POSITIONS_TRUTH_TTL_SECONDS", 90)
- Env var CHAD_POSITIONS_TRUTH_TTL_SECONDS is set nowhere (not in unit, drop-in, or repo) -> the literal default is live. Single-value edit; flows to one write (line 720); no other coupling.
- Code-default edit chosen over a systemd env-var override (avoids a Rule-6 systemd change).
- trade_lifecycle_state.json ttl is a separate variable (line 692, written line 719) and is untouched. Scope = positions_truth only.
- No test pins the publisher default to 60 (tests build their own payloads); no test churn expected.

## 4. Consumer-safety table (all 90s-safe)
| Consumer | Honors embedded ttl? | On stale | 90s-safe |
|---|---|---|---|
| live_loop futures cap (live_loop.py:262) | yes | fail-closed -> CAP_UNVERIFIED, entry refused | yes |
| live_gate (live_gate.py:614) | yes | fail-closed -> blocks live_readiness | yes (also removes the same flapping from live_readiness) |
| portfolio_var (portfolio_var.py:130) | no | falls back to snapshot if unparseable | n/a (report-only) |
| profit_lock FileEquityProvider (profit_lock.py:889) | no | returns None (no matching keys in this file) | n/a (inert vs this file) |
| service_failure_alert (service_failure_alert.py:51) | no | forensic snapshot only | n/a |
| advisory_engine (advisory_engine.py:1187) | no | LLM advisory context only | n/a (visibility-only) |

Key: raising ttl does NOT make normal-operation data older (publisher cadence unchanged); it only stops rejecting in the 60-65s band and widens outage detection by 30s. _futures_pending_adds already compensates intra-cycle truth lag. No consumer needs tighter-than-90s freshness.

## 5. Risk profile — LOW
- Normal operation: byte-identical except the marginal-stale rejections stop.
- Genuine publisher outage: fail-closed now triggers at <=90s file age instead of <=60s (+30s blind window). Quantified for the cap: live-loop cadence ~85s -> at most ~1 extra decision cycle on outage-stale truth; against net=217 vs cap=3 the BLOCK verdict cannot flip on 30s of staleness; signal_guard 10-min cooldown throttles repeats. live_gate gap-detection +30s is immaterial in PAPER. The four non-ttl-gated consumers are unchanged.

## 6. Deploy model — zero-restart
Publisher is Type=oneshot (python3 -m chad.ops.lifecycle_truth_publisher --once), fired by chad-lifecycle-truth-publisher.timer (~61-65s). The edit auto-deploys on the next timer fire (<= ~65s after commit) — same pattern as Fix B's harvester. No service restart, no Rule-7 concern. Readers honor whatever ttl the file embeds; no reader change or restart.

## 7. Procedure (Channel 2 — on GO)
1. Edit line 693: 60 -> 90.
2. py_compile lifecycle_truth_publisher.py; CHAD_SKIP_IB_CONNECT=1 python -m pytest -q (expect no churn vs 2799 baseline); full_cycle_preview clean.
3. Commit (Rule #8). No restart.

## 8. Verification (acceptance)
- Within ~2 min of commit: runtime/positions_truth.json embeds "ttl_seconds": 90.
- Over the following hours: ZERO new FUTURES_POSITION_CAP_UNVERIFIED in the chad-live-loop journal, while FUTURES_POSITION_CAP_BLOCK continues at the ~10-min cooldown cadence (cap still binding).
- No new errors in publisher or live-loop journals.

## 9. Rollback
Revert the single literal (90 -> 60); next timer fire restores ttl=60. No restart.

## 10. Status log
- 2026-06-03: authored from read-first audit (single change site + 6-consumer safety table confirmed). PENDING operator GO.
- 2026-06-04T00:21:44Z: EXECUTED per §7 — committed at 3bfa924 (line 693 60->90 only; line 692 untouched; only this file staged). py_compile clean; pytest 2813 passed (suite grew +14 vs the 2799 anticipated at authoring — zero failures, zero churn); full_cycle_preview clean (0 orders, no broker calls).
- 2026-06-04T00:23:20Z: §8(a) deploy live-verified — runtime/positions_truth.json embeds "ttl_seconds": 90 with ts_utc 2026-06-04T00:23:20.784899Z (later than commit time), confirming the oneshot timer published the new value. No service restarted.
- §8(b) OPEN: zero-new-FUTURES_POSITION_CAP_UNVERIFIED-over-hours observation still pending; CAP_BLOCK expected to continue at the ~10-min cooldown cadence. To be closed on a later verification pass.
- 2026-06-06T21:10Z: §8(b) CLOSED — observation window 2026-06-04T00:21:44Z (EXECUTED 3bfa924, deploy live-verified f4d8cea) → 2026-06-06T21:10Z (~69h). FUTURES_POSITION_CAP_UNVERIFIED = 0 in every journal inspection across the window, including the final read (retained journal 2026-06-06T17:55:55Z–21:10Z: 0) and the 19:50Z Stage 0 read (0 over its since-06-04 grep). FUTURES_POSITION_CAP_BLOCK continued at the expected ~10-min cooldown cadence throughout (19 in the final retained slice; 23 logged 16:06:05Z–19:49:33Z at the Stage 0 read) — cap binding, noise zero, exactly the §8 acceptance shape. Caveat: journald retention at close begins 2026-06-06T17:55:55Z, so the early window is evidenced by the dated sampled reads above rather than one continuous grep. Observation window formally CLOSED; PA fully complete.
