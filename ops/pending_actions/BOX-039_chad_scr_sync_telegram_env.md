# BOX-039 — chad-scr-sync.service Telegram env-file drop-in (Pending Action)

- **Box:** 039 — GAP-010 / GAP-023 Telegram errors verified
- **Stage:** 3 — Engineering, tests, SSOT, hidden-gap closure
- **Document timestamp (UTC):** 2026-05-20T02:19:21Z
- **Action type:** systemd drop-in (operator-approved; requires `systemctl daemon-reload`; does not authorize live trading)

## Summary

The 7-day journal scan for Box 039 found exactly one recurring Telegram error fingerprint:

```
python3[<pid>]: telegram_notify_failed err=Missing TELEGRAM_BOT_TOKEN
python3[<pid>]: scr_milestone_failed err=Missing TELEGRAM_BOT_TOKEN
```

These two lines fire together every 60 seconds from `chad-scr-sync.service`. The journal contains 248 of each fingerprint within retention (≈4 h 33 m window). The wiring that triggers them was added in commit `e150c0b` ("Upgrade: complete Telegram intelligence layer", 2026-04-23) and has been firing continuously since.

## Root cause

`chad-scr-sync.service` invokes `ops/scr_state_sync.py` every 60 s. That script calls `chad.utils.telegram_notify.check_and_send_scr_milestone`, which calls `load_config()`, which in turn requires `TELEGRAM_BOT_TOKEN` and `TELEGRAM_ALLOWED_CHAT_ID` from the process environment. The unit file does not declare any `EnvironmentFile`, so `os.environ.get("TELEGRAM_BOT_TOKEN")` returns the empty string and `load_config` raises `NotifyError("Missing TELEGRAM_BOT_TOKEN")` (caught and logged as a WARNING).

Comparison:

| Unit | EnvironmentFiles | Telegram errors / 7d |
| --- | --- | --- |
| `chad-live-loop.service` | `/etc/chad/claude.env`, `/etc/chad/chad.env`, `/etc/chad/telegram.env`, `/etc/chad/kraken.env` | **0** |
| `chad-scr-sync.service` | *(none)* | **248** (paired with 248 `scr_milestone_failed`) |

The notifier code in `chad/utils/telegram_notify.py` is correct: it fails loudly, never leaks the token, never propagates the exception, and the SCR sync pipeline continues normally (the JSON output line `{"ok": true, ...}` immediately after the two warnings proves `runtime/scr_state.json` is still written). The bug is purely a unit-file env-loading gap.

## Severity

- **Functional impact:** **None.** SCR sync still completes successfully every cycle (proved by `runtime/scr_state.json` being current — last `ts_utc=2026-05-20T01:15:54Z` at audit time). Trading/risk is unaffected. The only thing that silently degrades is the optional Telegram milestone announcement when SCR transitions (e.g. CAUTIOUS → CONFIDENT). The transition itself is honoured everywhere else.
- **Operational impact:** **Low.** Journal noise + missed milestone push. Not customer-facing.
- **Security impact:** **None.** The error message reveals only the env-var name, not any token value. `telegram_notify.py:171-173` and `:264` use structured logging that never inlines the token.

## Proposed fix (drop-in, operator-approved)

Create `/etc/systemd/system/chad-scr-sync.service.d/10-telegram-env.conf`:

```ini
# Drop-in: load Telegram credentials so check_and_send_scr_milestone
# can dispatch on SCR state transitions. Pattern mirrors
# chad-live-loop.service. `-` prefix tolerates absent file so unit
# still starts on hosts without Telegram configured.
[Service]
EnvironmentFile=-/etc/chad/telegram.env
```

Apply with:

```bash
sudo systemctl daemon-reload   # mandatory for drop-in pickup
# Next firing of chad-scr-sync.timer (≤60 s away) will pick up the new env.
# No service restart required for the .timer itself — daemon-reload
# re-reads unit + drop-in, and the next oneshot invocation gets the
# new EnvironmentFile.
```

Verification after operator approval:

```bash
# Confirm drop-in registered:
systemctl show chad-scr-sync.service --property=EnvironmentFiles
# expect: EnvironmentFiles=/etc/chad/telegram.env (ignore_errors=yes)

# Wait ~70 s, then re-check journal:
journalctl -u chad-scr-sync.service --since "5 minutes ago" --no-pager \
  | grep -E "telegram_notify_failed|scr_milestone_failed" | wc -l
# expect: 0

# Confirm scr_state.json still updates:
jq '.ts_utc' runtime/scr_state.json
```

## Why this Pending Action — not applied by Box 039

Box 039 constraints explicitly forbid:
- `Do not restart services.`
- `Do not start services.`
- `Do not stop services.`
- `Do not run daemon-reload.`

A drop-in without `daemon-reload` would not take effect; with `daemon-reload` the constraint is violated. Therefore the fix is queued as a Pending Action requiring explicit operator approval and execution. CHAD governance §6 ("Never modify systemd service files without explicit instruction") reinforces this.

## Forbidden by this document

- Do not silently apply the drop-in without operator approval recorded in `control/pending_actions/`.
- Do not change `chad/utils/telegram_notify.py` to silently swallow the missing-token error — fail-loud behaviour is correct; the bug is the missing env source.
- Do not change `ops/scr_state_sync.py` to skip the milestone call — the call is correctly wired; the env source is the gap.
- Do not log `TELEGRAM_BOT_TOKEN` values, even masked, in evidence or PR descriptions.

## Cross-references

- Box-039 evidence: `runtime/completion_matrix_evidence/BOX-039_GAP-010_023_Telegram_errors_verified.md`
- Source of the warning: `chad/utils/telegram_notify.py:95-97` (raise), `:264` (log)
- Wiring commit: `e150c0b` "Upgrade: complete Telegram intelligence layer" (2026-04-23)
- Unit file: `/etc/systemd/system/chad-scr-sync.service`
- Reference pattern: `/etc/systemd/system/chad-live-loop.service` (loads `/etc/chad/telegram.env`)
