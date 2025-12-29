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
  - portfolio_snapshot.json still updates (proof via mtime)

Proof commands:
- systemctl status chad-kraken-collector.timer chad-kraken-pnl-watcher.timer chad-legend.timer chad-portfolio-merge.timer --no-pager -n 12
- stat -c "mtime=%y size=%s path=%n" runtime/kraken_balances.json runtime/kraken_pnl_state.json runtime/portfolio_snapshot.json
- journalctl -u chad-kraken-collector.service -n 30 --no-pager -o cat
- journalctl -u chad-kraken-pnl-watcher.service -n 30 --no-pager -o cat
- systemctl cat chad-kraken-collector.service chad-kraken-pnl-watcher.service chad-legend.service chad-portfolio-merge.service
