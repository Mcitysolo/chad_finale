CHAD — Phase 7 Safety & Operations Runbook
Finalized for: Ubuntu 24.04 (EC2) — User: ubuntu

CHAD is fully online in practice-only mode.
All intelligence is active.
All real trading is disabled by design.

This runbook explains:

What CHAD is doing right now

How to check CHAD’s health

How to prove CHAD cannot trade real money

How to stop or restart CHAD safely

How CHAD stays in DRY_RUN mode no matter what

Everything here uses real commands with no placeholders.

What CHAD Is Doing (Simple)

CHAD is:

Reading real stock market data (Polygon)

Running all strategy brains (Alpha, Beta, Gamma, Omega, etc.)

Simulating trades

Logging everything

Building risk reports and confidence scores

Serving a /status page for monitoring

Never sending a real order

CHAD is alive, but in a cage.
He thinks and plans, but cannot touch real money.

The Safety Guarantee (WHY CHAD CANNOT TRADE REAL MONEY)

These five locks make live trading impossible in Phase 7:

(1) ExecutionConfig Hard-Locked
All IBKR modes (ibkr_paper, ibkr_live) are forced to:
mode = DRY_RUN
ibkr_dry_run = True

(2) Live Gate Logic
Even if CHAD_MODE=LIVE, final decision = NO LIVE TRADING:
allow_ibkr_live = False
allow_ibkr_paper = True

(3) IB Gateway Disabled
Systemd service chad-ibgateway.service is disabled and inactive.

(4) No Web/API Execution Endpoints
/status is read-only. No /trade, /execute, /order endpoints exist.

(5) Directory Permissions
data/, runtime/, logs/, and repo root are set to drwxr-x---
No “other users” can access CHAD internals.

Bottom Line:
There is no code path or service path that can reach a live broker order.

How to Check CHAD’s Current Mode

ExecutionConfig (adapter-level truth):

PYTHONPATH="/home/ubuntu/CHAD FINALE" python -m chad.core.show_execution_config

Expect:
mode = DRY_RUN
ibkr_dry_run = Yes

Live Gate (final yes/no):

PYTHONPATH="/home/ubuntu/CHAD FINALE" python -m chad.core.show_live_gate

Expect:
allow_ibkr_live = False

Risk & Confidence:

PYTHONPATH="/home/ubuntu/CHAD FINALE" python -m chad.core.show_risk_state

Shows CHAD_MODE, caps, SCR, reasons.

How to Check CHAD's Health (System Level)

Backend:
systemctl status chad-backend.service --no-pager -n 20

Orchestrator:
systemctl status chad-orchestrator.service --no-pager -n 20

Polygon Streamer:
systemctl status chad-polygon-stocks.service --no-pager -n 20

Shadow Status (port 9619):
curl -s http://127.0.0.1:9619/status
 | python -m json.tool

How to Start / Stop CHAD Services Safely

Start all:
sudo systemctl start chad-backend chad-orchestrator chad-polygon-stocks chad-shadow-status

Stop all:
sudo systemctl stop chad-backend chad-orchestrator chad-polygon-stocks chad-shadow-status

Restart one:
sudo systemctl restart chad-backend

Log Locations

Polygon feed:
/home/ubuntu/chad_finale/logs/polygon_stocks.log

Backend:
/home/ubuntu/chad_finale/logs/backend-uvicorn.log

Rotated daily via:
/etc/logrotate.d/chad-polygon
/etc/logrotate.d/chad-backend

What Still Cannot Happen

Even if someone:

Sets CHAD_MODE=LIVE

Sets CHAD_EXECUTION_MODE=ibkr_live

Starts IB Gateway

Runs any CLI

Hits /status

Calls internal functions

CHAD still responds internally:
"Live trading is not allowed in Phase 7. I will simulate only."

Zero path to live trading.

What Happens in Phase 8 (Not Active Yet)

Phase 8 will bring:
A real GO-LIVE switch
Multi-step validation (SCR, caps, router)
A STOP command
True live execution

But Phase 8 is not enabled here.
CHAD remains in DRY_RUN only.

Phase 7 Status: COMPLETE
CHAD is fully alive, fully simulated, fully safe.
Zero possibility of live trading until Phase 8 is intentionally activated.
