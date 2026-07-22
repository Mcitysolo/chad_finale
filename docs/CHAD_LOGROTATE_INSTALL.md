# CHAD Log Rotation — W3B-11 Record + Install

## What the 2026-07-22 inventory found

- journald: already bounded (drop-ins `/etc/systemd/journald.conf.d/99-*.conf`,
  SystemMaxUse winner 150M + disk-guard `--vacuum-size=80M`). No action.
- `logs/telegram_bot.log`: bounded by in-process RotatingFileHandler
  (10MB×5) — but DOUBLE-WRITTEN (the systemd unit also `append:`s to it).
  Deliberately untouched; never rename-rotate it (the systemd fd would be
  stranded), and adding logrotate would double-rotate against the handler.
- Genuinely unbounded: `logs/claude/claude_client.log` (12.4MB, active),
  `logs/gpt/gpt_client.log` (docstring claimed "rotating"; code didn't),
  cron `logs/consensus_update.log` (weekly append, tiny).
- `logs/claude/calls_YYYYMMDD.ndjson`: daily files that stop growing after
  their day — rotation is pointless; if the file COUNT ever matters, that is
  an archival policy question, not logrotate. Left alone.
- `runtime/disk_guard_audit.ndjson`: root-owned, appended by the disk-guard
  root cron — its rotation belongs to the disk-guard owner, not CHAD's
  logrotate. Flagged, not touched.

## What W3B-11 changed in code (D7: code-side rotation preferred)

`chad/intel/claude_client.py` and `chad/intel/gpt_client.py` file handlers
became `RotatingFileHandler(maxBytes=10_000_000, backupCount=5)` — the exact
telegram_bot precedent. Self-contained: no operator install needed for the
two main offenders; activates at each consumer's next restart/oneshot fire.

## Operator: installing the cron-log rotation (the one logrotate-only case)

    sudo cp ops/logrotate/chad-logs.conf /etc/logrotate.d/chad-logs
    sudo logrotate --debug /etc/logrotate.d/chad-logs   # dry-run verification

Rollback: `sudo rm /etc/logrotate.d/chad-logs`.
