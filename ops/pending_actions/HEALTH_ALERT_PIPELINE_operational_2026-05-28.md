# Health-Monitor Alert Pipeline — Operational Status Confirmation (2026-05-28)

## Status

OPERATIONAL — no action required. This PA records the verified state of
the alert pipeline as of 2026-05-28, after Fix A/B/C audit work surfaced
the need to confirm rather than rebuild.

## Audit trail

- Source audit: see this commit's e2e test file
  (`chad/tests/test_health_alert_pipeline_e2e.py`).
- Forensic audit reference:
  `reports/forensic_audits/CHAD_FORENSIC_FULL_SYSTEM_AUDIT_20260527T142951Z.md`
- Trigger: Fix D scoping (post-Fix-C). Initial design assumed an alert
  dispatcher needed to be built; audit revealed it already existed and
  was firing live for R17b, R19, R20, and ~20 other rules.

## Pipeline shape (verified 2026-05-28)

```
  ┌───────────────────────────────┐
  │ chad-health-monitor.timer     │  every 5 min, OnUnitActiveSec=300,
  │ (Persistent=true)             │  AccuracySec=15
  └───────────────┬───────────────┘
                  │
  ┌───────────────▼───────────────┐
  │ chad-health-monitor.service   │  Type=oneshot, user=ubuntu,
  │ ExecStart=health_monitor.py   │  loads /etc/chad/telegram.env
  └───────────────┬───────────────┘
                  │
  ┌───────────────▼───────────────┐
  │ run_all_rules() emits         │  ~22 rules including R19, R20
  │ List[Finding]                 │
  └───────────────┬───────────────┘
                  │
  ┌───────────────▼───────────────┐
  │ Dispatcher loop:              │  health_monitor.py line ~295
  │ if remedy_action == "notify": │
  │   _notify(...)                │  line ~308
  └───────────────┬───────────────┘
                  │
  ┌───────────────▼───────────────┐
  │ chad.utils.telegram_notify    │  TELEGRAM_NOTIFY_DEDUPE_TTL_SECONDS
  │ .notify(message, severity,    │  default 900 (15 min)
  │  dedupe_key=...)              │  → runtime/telegram_dedupe_<key>.json
  └───────────────┬───────────────┘
                  │
  ┌───────────────▼───────────────┐
  │ Telegram Bot API              │  TELEGRAM_BOT_TOKEN + chat IDs
  │ (operator's chat)             │  from /etc/chad/telegram.env
  └───────────────────────────────┘
```

Implementation note (verified against source, not asserted falsely): the
dispatcher loop calls `_notify()` for **every** finding. The finding's
`remedy_action` selects the message *shape* — a pure "HEALTH MONITOR"
notification when `remedy_action == "notify"`, or a "🔧 AUTO-FIXED" report
otherwise — but the `notify()` call itself is unconditional. R19 and R20
both use `remedy_action="notify"` (NOTIFY_ONLY), so they take the pure
notification branch.

## Live evidence (snapshot at 2026-05-28T21:51Z)

- `chad-health-monitor.timer`: active (every 5 min).
- `runtime/telegram_dedupe_health_R19_IBKRsustainedlatencyaboves.json`:
  `last_sent_unix=1779988612` (2026-05-28T17:16:52Z) — fired during
  today's Gateway hard-wedge.
- `runtime/telegram_dedupe_health_R20_IBKRGatewayversionisstale.json`:
  `last_sent_unix=1780004711` (2026-05-28T21:45:11Z) — first fire ~15 min
  after Fix C commit 3839bfb, confirming new rules wire automatically
  through the existing dispatcher. (Subsequent re-fires update this stamp
  on each TTL boundary; the value above is the first-fire reference cited
  in the audit.)

## Dedup TTL choice

- Default 900s (15 min) via `TELEGRAM_NOTIFY_DEDUPE_TTL_SECONDS`.
- Rationale: long enough to prevent spam during a sustained incident
  (we got one alert per 15 min during the 13h Gateway wedge today),
  short enough that recovery / recurrence within the same trading
  session generates a fresh alert.
- Tunable via env if a specific finding type needs different cadence;
  no current need to override.

## What this PA does NOT cover

- Auto-recovery on a wedge BETWEEN scheduled restarts
  (`IBKR_AUTO_RECOVERY` PA, unchanged) — separate future workstream.
- The actual Gateway upgrade (`IBKR_GATEWAY_VERSION_UPGRADE` PA) —
  separate operator-led Channel 3 step.
- Per-finding alert routing config (currently every notify-action finding
  sends to the one configured chat ID) — could be added if a future need
  for severity-gated routing emerges; no current need.

## Related PAs (cross-reference)

- `ops/pending_actions/IBKR_AUTO_RECOVERY_design_2026-05-27.md`
- `ops/pending_actions/IBKR_RELIABILITY_socket_backpressure_and_gateway_churn_2026-05-27.md`
- `ops/pending_actions/IBKR_GATEWAY_VERSION_UPGRADE_2026-05-28.md`

## Acceptance criteria

- `chad/tests/test_health_alert_pipeline_e2e.py` passes (6 tests).
- R19 and R20 docstrings cross-reference this PA.
- Dedup TTL documented in this PA matches the code default (900s).

## No-live confirmation

This PA does not authorize live trading. `ready_for_live` must remain
false. `allow_ibkr_live` must remain false. `allow_ibkr_paper` must remain
true. No broker orders may be placed or cancelled.
