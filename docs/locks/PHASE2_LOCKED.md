# PHASE 2 LOCKED — Market Data (Polygon)

Date (UTC): 2025-12-29

Guarantees:
- Polygon streamer runs continuously under systemd.
- Daily NDJSON file rotation exists in data/feeds/.
- Backend price endpoint returns source=polygon.

Proof commands:
- systemctl status chad-polygon-stocks.service --no-pager -n 25
- ls -lah data/feeds/polygon_stocks_*.ndjson | tail -n 5
- curl -sS "http://127.0.0.1:9618/ai/price?symbol=AAPL" | python3 -m json.tool
