# PHASE 5 LOCKED — IBKR Integration (Gateway + Health + Ledger Watcher)

Date (UTC): 2025-12-29

Guarantees:
- IB Gateway is running and API port :4002 is listening.
- Health timer runs and returns ok:true (no ConnectionRefused).
- Paper ledger watcher runs and returns connect_ok.
- Watchdog writes runtime/ibkr_status.json truth (ok:true when logged in).

Proof commands:
- sudo ss -ltnp | egrep ":(4002)\\b"
- journalctl -u chad-ibkr-health.service --since "-6 min" --no-pager -o cat
- journalctl -u chad-ibkr-paper-ledger-watcher.service --since "-6 min" --no-pager -o cat
- sudo cat runtime/ibkr_status.json
