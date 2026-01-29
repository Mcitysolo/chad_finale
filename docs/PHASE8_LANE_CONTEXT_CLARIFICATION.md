# CHAD Design Clarification — Segmented Aggression via Risk Lanes (Future Phases Only)

## Timestamp (UTC)
2026-01-28T23:54:32+00:00

## Summary (No Scope Change)
This is an explicit design clarification to prevent future misunderstandings.
It does NOT change SSOT, does NOT add phases, and does NOT modify current Phase 8 behavior.

CHAD is meant to support segmented aggression, not global aggression.

## Phase 8 Status (Unchanged)
- The current Phase 8 work (LiveGate, canary, STOP drills, exactly-once execution semantics) is correct and unchanged.
- Phase 8 assumes a single global execution path and a single LiveGate decision.
- STOP remains globally authoritative and kills everything.

## Future Intent (Do NOT Implement Now)
In later phases (e.g., Phase 5 / Phase 12), SCR and TierManager will be extended to support multiple risk lanes:

- CORE (long-term / income / low volatility)
- TACTICAL (normal Alpha/Gamma activity)
- BEAST (Delta / Futures / Forex / Volatility)

Each lane will have:
- its own SCR confidence
- its own risk budget
- its own kill switch

Execution will still go through the same adapters and LiveGate, but with a lane context (e.g., lane_id).

## Non-Negotiable Constraints (Unchanged)
- No brain ever executes trades.
- No lane bypasses RiskManager, SCR, or LiveGate.
- STOP still kills everything globally.
- Exactly-once semantics remain identical.

## Phase 8 Action (Now)
- Finish Phase 8 as specified.
- Keep execution interfaces clean so a future lane_id can be passed through as an optional parameter (no logic now).
- Do NOT refactor or add complexity now.

