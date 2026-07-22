# Telegram Dedupe State — runtime/dedupe/ Relocation + Scheduled Cleanup (W3B-10)

Status: repo-only design artifacts. Nothing installed automatically
(CLAUDE.md governance #6/#7).

## What changed in code (active at each service's next restart / oneshot fire)

- `chad/utils/telegram_notify.py::_dedupe_path` now writes per-key dedupe
  state to **`runtime/dedupe/telegram_dedupe_<key>.json`** instead of loose in
  `runtime/`. `_dedupe_mark` creates the directory on first write. Filenames
  are unchanged — only the directory moved.
- `ops/cleanup_telegram_dedupe.py` scans BOTH the legacy loose location and
  `runtime/dedupe/` (migration window), archiving stale files (mtime older
  than ttl×safety) to `_archive/telegram_dedupe/YYYY/MM/`.
- `runtime/telegram_bot_dedupe.json` (a separate single-file mechanism in
  telegram_bot.py) is intentionally NOT moved.

## Split-brain window (accepted, documented)

Oneshot timer services pick the new path up at their next fire; long-running
services (live-loop, telegram-bot, health-monitor host processes) keep
writing the old path until their next gated restart. Until restarts complete,
dedupe suppression is not shared across the two paths — worst case is one
duplicate alert per key per 900s TTL. Old loose files left in `runtime/`
remain valid dedupe state for the un-restarted writers and are swept by the
cleanup tool once they go stale.

## Migration

No manual migration required. Old loose files either stay in use (see above)
or age out and get archived by the tool. To force-consolidate after all
restarts are complete (optional):

    mv runtime/telegram_dedupe_*.json runtime/dedupe/ 2>/dev/null || true

## Operator: installing the scheduled cleanup (optional; BOX-042 left it manual)

    # 1. read-only verification first
    /home/ubuntu/chad_finale/venv/bin/python3 ops/cleanup_telegram_dedupe.py   # dry-run by default
    # 2. install
    sudo cp ops/systemd/chad-dedupe-cleanup.{service,timer} /etc/systemd/system/
    sudo systemctl daemon-reload
    # 3. one supervised run
    sudo systemctl start chad-dedupe-cleanup.service
    journalctl -u chad-dedupe-cleanup -n 20 --no-pager
    # 4. arm the timer
    sudo systemctl enable --now chad-dedupe-cleanup.timer

Rollback: `sudo systemctl disable --now chad-dedupe-cleanup.timer`, remove
the two unit files, `daemon-reload`.

## Trap (do not forget)

`/usr/local/bin/chad_disk_guard.sh` deletes `_archive/` files older than 30
days — the archive is a purgatory, not storage. Archived dedupe files are
disposable state, so that is fine HERE; do not copy this pattern for
anything that must survive.
