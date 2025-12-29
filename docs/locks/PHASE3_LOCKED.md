# PHASE 3 LOCKED — Runtime Contracts + Safety Gates

Date (UTC): 2025-12-29

Runtime contract files (SSOT):
- runtime/stop_state.json
- runtime/operator_intent.json
- runtime/live_mode.json
- runtime/dynamic_caps.json
- runtime/portfolio_snapshot.json

Guarantees:
- STOP exists (DENY_ALL) and is enforced by /live-gate.
- Operator intent exists (ALLOW_LIVE / EXIT_ONLY / DENY_ALL) and is enforced by /live-gate.
- live_mode.json exists and remains live=false until Phase 8 approval.
- LiveGate snapshot includes operator_mode/operator_reason and allow_ibkr_paper field.

Proof commands:
- cat runtime/{stop_state.json,operator_intent.json,live_mode.json,dynamic_caps.json,portfolio_snapshot.json}
- curl -sS http://127.0.0.1:9618/live-gate | python3 -m json.tool
- curl -sS http://127.0.0.1:9618/operator-intent | python3 -m json.tool
