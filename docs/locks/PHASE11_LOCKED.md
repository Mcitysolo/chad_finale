# PHASE 11 LOCKED — Kraken + Legend (Automation) + Portfolio Merge (Manual)

Date (UTC): 2025-12-29

Guarantees:
- Kraken balance collector runs automatically (timer enabled).
- Kraken PnL watcher runs automatically (timer enabled).
- Kraken runtime truth files update continuously:
  - runtime/kraken_balances.json
  - runtime/kraken_pnl_state.json
- Legend refresh runs automatically (weekly timer enabled).
- Portfolio merge is intentionally NOT automated in this locked phase:
  - chad-portfolio-merge.timer is masked
  - merge can be run on-demand via chad-portfolio-merge.service
  - portfolio_snapshot.json is still updated (proof via mtime/logs)

Proof commands:
- systemctl status chad-kraken-collector.timer chad-kraken-pnl-watcher.timer chad-legend.timer chad-portfolio-merge.timer --no-pager -n 12
- stat runtime/kraken_balances.json runtime/kraken_pnl_state.json runtime/portfolio_snapshot.json
- journalctl -u chad-kraken-collector.service --no-pager -n 40 -o cat
- journalctl -u chad-kraken-pnl-watcher.service --no-pager -n 40 -o cat
- journalctl -u chad-legend.service --no-pager -n 40 -o cat
- journalctl -u chad-portfolio-merge.service --no-pager -n 40 -o cat
