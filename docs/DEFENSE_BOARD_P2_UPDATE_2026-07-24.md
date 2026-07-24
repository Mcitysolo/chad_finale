# Defense Board — P2 remnant sweep, 2026-07-24 (W6B)

**Supersedes selectively:** `docs/DEFENSE_BOARD_RECONCILIATION_2026-06-17.md` §P2.
The June record is **not** rewritten. This file records what changed in the 37 days since,
and closes the items W6B acted on. Where June and today agree, that is stated as an
independent re-confirmation, not a citation.

Every item below was re-verified against the live system today before writing.

---

## P2-2 — `operator_intent` default posture → **CLOSED (was: doc clarification only)**

June's disposition — "by-design (allows paper entries; live blocked downstream)" — was
correct but incomplete, because at the time a hold was **not durable**. INCIDENT-0723
exposed two defects that June's "by-design" verdict did not cover. Both are now fixed, and
that is what makes this closable rather than merely explainable:

**D4(a) — the hold no longer gets stomped.** The 10-minute auto-refresh timer used to
rewrite `ALLOW_LIVE` unconditionally, erasing an operator-granted `EXIT_ONLY` within
seconds. `chad/ops/operator_intent_refresher.py::_unexpired_hold` now returns the current
hold when its **own** declared TTL has not elapsed, and the refresher stands down. The
distinction that makes it correct: the hold is judged against the *state's* `ts_utc +
ttl_seconds` (e.g. a 24h hold), not the store's 900s default freshness window — a hold
whose entire purpose is to outlive the refresh cadence cannot be evaluated against that
cadence. Unreadable/absent/malformed state returns `None` and normal refresh proceeds.

**D4(b) — the hold is actually consulted.** `chad/core/live_gate.py::_load_operator_intent`
loads the intent and **fails closed to `DENY_ALL`** when the file is missing or unreadable
(`operator_mode="DENY_ALL"`, `allow_ibkr_live=False`). The live decision trace confirms the
consumption end-to-end: `operator_mode=EXIT_ONLY` → *"Deny all lanes"*.

So the posture is by-design **and** a hold now both persists and binds. Doc-only, no code
changed by W6B.

**Residual gap, deliberately not closed here:** a hold is admission control on *new*
intents. It does not reach orders already working at the broker. That is item (4), built
as W6B-13 — flag-gated and default OFF.

---

## P2-3 — `HALT_BOOST_SUPPRESSED` log noise → **CLOSED (was: open/benign)**

Re-confirmed independently: the clamp is implemented and correct.
`chad/risk/dynamic_risk_allocator.py:388-406` clamps `winner_factor` to `1.0` when a
strategy is in `halted_lookup` with a factor above 1.0, records
`halt_clamp_applied` in `per_strategy_overlay`, and a second independent
contradiction check exists at `chad/ops/exterminator.py:524-560`. A halted strategy
cannot be boosted. Nothing to reconcile.

**W6B-9 closes the residue.** The line fired at `WARNING`, once per halted strategy
**per cycle**. With several strategies halted for days, that was the dominant source of
WARNING volume in the live journal — the failure mode being that operators learn to skim
past warnings generally, which costs them the warnings that matter.

It now narrates on **transition** at `INFO`: one line when a clamp engages, one when it
releases. Note the release line did not exist before, so this is strictly *more*
information at a fraction of the volume. The dedupe cache is module-level rather than
instance state, because `DynamicRiskAllocator` is a frozen dataclass documented as
"deliberately stateless" and log dedupe is not allocation state.

The structural record is untouched — `per_strategy_overlay[...]["halt_clamp_applied"]` is
what `health_monitor_rules.py:591` and the exterminator actually consume, and a named test
(`test_structural_record_survives_the_logging_change`) proves it is still emitted on every
cycle while only the first cycle narrates.

---

## P2-5 — ports 9618/9619/9620 → **CLOSED, double-confirmed. Report-only.**

Re-verified today by two independent means:

1. **All three listeners are loopback-bound.** `ss -tlnp` shows `127.0.0.1:9618`,
   `127.0.0.1:9619`, `127.0.0.1:9620`. Unreachable off-host at any security-group setting.
2. **The security group does not open them at all.** Instance `i-0c0d16702f524d9a6`,
   SG `CHAD-Prod-SG`; ingress is exactly tcp/80 `0.0.0.0/0`, tcp/443 `0.0.0.0/0`, and
   tcp/22 from three `/32` operator IPs. No rule references 9618/9619/9620.

**No Channel 1/3 action required.** Defense in depth is intact in both directions: were
the SG widened, loopback binding would still refuse; were the binding widened, the SG
would still refuse. Recorded as a verified negative.

---

## P2-7 — Telegram / urllib3 churn → **DEFER, with the reason stated (was: open/benign)**

`python-telegram-bot 13.15`, `urllib3 1.26.20`, both pinned. **Zero** urllib3/connectionpool
warnings in the current journal window — the symptom is not currently firing.

June prescribed PTB 20.x. That is an async-first breaking major upgrade requiring a rewrite
of `chad/utils/telegram_bot.py`'s handler surface (`ParseMode`, `telegram.ext` imports at
`:64-66`) — which is the live alerting spine **and** the COACH-VOICE L1/L2 delivery path.

Destabilising alerting to silence a warning that is not firing is a bad trade. **Deferred
to its own lane**, to be revisited only if urllib3 churn actually resurfaces. Recorded as a
decision with a rationale rather than left as a perpetually-open "benign" row.

---

## P2-8 — APScheduler → **CLOSED by W6B-10 (was: open/benign)**

`APScheduler==3.6.3` was pinned at `requirements.txt:7` and is **imported by nothing** —
zero references across `chad/`, `ops/`, `scripts/`. Re-confirmed today.

One correction to the June framing: it is described as a "dead dependency", which reads as
*not installed*. It **is** installed in the venv (3.6.3). That matters, because
`requirements.txt` self-declares as an *"Authoritative full environment freeze generated
from pip freeze"*, so removing the line makes the file diverge from a live `pip freeze` by
one package for the first time.

W6B-10 removes it anyway and **records the divergence in the file header** rather than
letting the manifest silently stop being what it claims to be. The manifest declares what
CHAD depends on; a fresh `pip install -r requirements.txt` should not resurrect a
dependency nothing imports. Removing it from the venv is a separate operator action, not a
repo change.

Verified safe: the only consumer of `requirements.txt` in the test suite is
`test_ib_async_import_parity.py:349`, which asserts on the `ib_async`/`ib-insync` pins only.

---

## P2-9 — options-chain-refresh → **CLOSED-healthy; two new findings, both addressed**

Re-verified: the unit exited `0/SUCCESS` at 12:30:06 today,
`runtime/options_chains_cache.json` is fresh, well-formed, and carries
`schema_version: options_chain_cache.v2`. The loud alert the brief asks for **already
exists** — the live unit declares `OnFailure=chad-service-alert@%N.service`.

**N1 — repo/live unit drift. The June-era framing was wrong and is corrected here.**
The plan reported `deploy/chad-options-chain-refresh.service` missing from the repo. It is
missing *under that name*. The repo carries **`deploy/options_chain_refresh.service`** — a
legacy naming form predating the `chad-` prefix convention. Diffed against the live unit,
the entire delta is one line:

```
4a5
> OnFailure=chad-service-alert@%N.service
```

So the gap is real but small: the alert wiring is unversioned and a redeploy from repo
would silently drop it. W6B-11 adds the line. **A sweep of all 26 repo units against live
found this as the only drift** — the other 25 are byte-identical.

*Wider observation, recorded not fixed:* there are **126** live `chad-*.service` units and
**26** in `deploy/`. Roughly 100 live units are untracked, including
`chad-xgb-train.service`. Closing that wholesale is its own lane; W6B commits repo copies
only for units it modifies (W3B parity pattern).

**N2 — coverage is one symbol.** `chains` contains **SPY only**. Per operator ruling
(D6) that is correct and intended — but "exit 0 with 1 symbol" and "exit 0 with 0 useful
symbols" are indistinguishable to the current check, which is the actual silent-failure
surface. W6B-11 makes the universe **config-declared with an asserted expected count**, so
`fetched < expected` alerts and the two outcomes become distinguishable forever.

---

## P2-11 — dual ledger authority → **CLOSED by W6B-12**

`docs/LEDGER_AUTHORITY_DECLARATION.md` now states which ledger answers which question
(position identity vs realized P&L), that they are not interchangeable in either
direction, and the resolution order when they disagree — broker truth wins on positions,
the hash-chained closed-trade ledger wins on realized P&L, and a contradiction is a
reconciliation finding rather than something to average.

---

## New since June — not on the P2 board

**P3-1 has grown, not stayed stale.** June recorded "telegram dedupe clutter: stale — 29
files (not 1573)". Today there are **65** `runtime/telegram_dedupe_*.json` files, accreted
2026-05-29 → 2026-07-22. The count more than doubled in 37 days. It remains non-gating,
but "optional tidy" understates a set that grows one file per new alert string and whose
filenames are generated from alert text. This is also why W6B-3 classifies unpinned
runtime contracts by **pattern** rather than enumerating paths — see
`audits/W6B_PREMISE_REVERIFY_2026-07-24.md`.

**Four runtime `.json` files are not valid JSON** and are invisible to EXS7 because they
are neither enforced nor listed: `flip_executor_audit.json` and `signal_throttle_audit.json`
are NDJSON under a `.json` name (`json.load()` raises `Extra data: line 2`),
`claude_usage.json` is zero-byte, and `__guard_swallow_probe__.json` is 1 byte of test-probe
leakage. Recorded for disposition; this lane does not mutate runtime files.
