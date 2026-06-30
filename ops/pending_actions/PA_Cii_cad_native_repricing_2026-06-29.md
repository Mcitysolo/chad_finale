# Pending Action — C-ii CAD-native tier/withdrawal re-pricing (Bucket C)

- **Status:** PREPARED — NOT APPLIED. Live config files untouched.
- **Date:** 2026-06-29
- **Basis:** USDCAD = 1.4160 (one-time conversion constant; **never** stored as a live rate in any config).
- **Governance:** Governance rule #3 (non-negotiable) forbids direct mutation of risk/strategy/financial config. `config/tiers.json` and `config/withdrawal_policy.json` are read **live** by active systemd timers (`chad-tier-manager.timer`, `chad-withdrawal-manager.timer`, `chad-business-phase.timer`), so a direct disk edit takes effect on the next tick (~5 min) with **no commit/push/restart**. The two config changes below are therefore staged here as a Pending Action and are applied by the operator **only** at the gated restart described in §Activation.

## Why this is a PA and not part of the code commit

The companion code commit (CAD equity intake + `_usd`→`_cad` field renames + schema bumps) is **inert until a service restart**. The config changes here take effect on the next **timer tick** under whatever code is currently running. If the config went live before the code (i.e. now), the **old USD-intake code** would compare a USD equity against CAD bands — the exact CAD-band-vs-USD-equity mismatch the change exists to remove. Splitting code (committed, inert) from config (this PA, operator-applied at the restart) keeps activation atomic.

The committed code reads band/policy thresholds **dual-key** (`*_equity_cad` preferred, `*_equity_usd` fallback), mirroring the established `chad/risk/drawdown_guard._row_equity` pattern. Consequence: the committed code runs **unchanged** against the current USD config (fallback path) and switches to CAD automatically once this PA is applied.

> ⚠️ **Atomicity requirement:** apply BOTH config files below **and** restart the three services together. Never restart the new code against the old USD config, and never apply this config under the old code.

## A. Proposed `config/tiers.json` (CAD)

Band thresholds renamed `*_equity_usd` → `*_equity_cad` and re-priced ×1.4160 (whole CAD). `risk_profile.*` caps are **excluded** — they measure USD-denominated instrument risk / ledger P&L (`tier_risk_enforcer.py:262` stop-width, `:207-216` realized loss), never equity, so they stay USD. `hysteresis_pct` (dimensionless) unchanged. `schema_version` left at `tiers.v9_1` (per manifest A; no schema bump requested).

| Tier | min_equity_usd → min_equity_cad | max_equity_usd → max_equity_cad | demotion_equity_usd → demotion_equity_cad |
|---|---|---|---|
| MICRO | 0 → **0** | 2500 → **3540** | null → **null** |
| STARTER | 2500 → **3540** | 25000 → **35400** | 2250 → **3186** |
| PRO_GROWTH | 25000 → **35400** | 160000 → **226560** | 22500 → **31860** |
| SCALE | 160000 → **226560** | 10000000 → **14160000** | 144000 → **203904** |

Continuity preserved (STARTER.min==MICRO.max==3540; PRO_GROWTH.min==STARTER.max==35400; SCALE.min==PRO_GROWTH.max==226560). No-flip check: live `ibkr_equity` 161 345.49 CAD ∈ PRO_GROWTH [35 400, 226 560) — same tier as today.

```json
{
  "schema_version": "tiers.v9_1",
  "hysteresis_pct": 5.0,
  "tiers": {
    "MICRO": {
      "description": "Proving account — single instrument, minimum frequency",
      "min_equity_cad": 0,
      "max_equity_cad": 3540,
      "demotion_equity_cad": null,
      "enabled_strategies": ["alpha_intraday_micro"],
      "allowed_instruments": ["MES"],
      "risk_profile": {
        "max_contracts_per_trade": 1,
        "max_risk_per_trade_usd": 10,
        "max_daily_loss_usd": 20,
        "max_weekly_loss_usd": 30,
        "max_trades_per_day": 2,
        "primary_session_only": true,
        "flatten_before_eod": true,
        "flatten_eod_minutes_before_close": 30,
        "stop_width_gate_enabled": true
      }
    },
    "STARTER": {
      "description": "Controlled scaling — two instruments, primary session only",
      "min_equity_cad": 3540,
      "max_equity_cad": 35400,
      "demotion_equity_cad": 3186,
      "enabled_strategies": [
        "alpha_intraday_micro",
        "alpha_intraday",
        "beta_trend",
        "alpha_crypto"
      ],
      "allowed_instruments": ["MES", "MNQ", "SPY", "QQQ", "IWM", "BTC-USD"],
      "risk_profile": {
        "max_contracts_per_trade": 2,
        "max_risk_per_trade_usd": 40,
        "max_daily_loss_usd": 150,
        "max_weekly_loss_usd": 300,
        "max_trades_per_day": 4,
        "primary_session_only": true,
        "flatten_before_eod": true,
        "flatten_eod_minutes_before_close": 30,
        "stop_width_gate_enabled": true
      }
    },
    "PRO_GROWTH": {
      "description": "Full futures + equities — primary and secondary session",
      "min_equity_cad": 35400,
      "max_equity_cad": 226560,
      "demotion_equity_cad": 31860,
      "enabled_strategies": [
        "alpha",
        "alpha_intraday",
        "alpha_intraday_micro",
        "alpha_futures",
        "alpha_crypto",
        "beta",
        "beta_trend",
        "gamma_futures",
        "delta",
        "omega_macro"
      ],
      "allowed_instruments": ["*"],
      "risk_profile": {
        "max_contracts_per_trade": 5,
        "max_risk_per_trade_usd": 200,
        "max_daily_loss_usd": 500,
        "max_weekly_loss_usd": 1500,
        "max_trades_per_day": 10,
        "primary_session_only": false,
        "flatten_before_eod": false,
        "flatten_eod_minutes_before_close": null,
        "stop_width_gate_enabled": true
      }
    },
    "SCALE": {
      "description": "Full stack — all 16 strategies, existing sizing pipeline unchanged",
      "min_equity_cad": 226560,
      "max_equity_cad": 14160000,
      "demotion_equity_cad": 203904,
      "enabled_strategies": ["*"],
      "allowed_instruments": ["*"],
      "risk_profile": {
        "max_contracts_per_trade": null,
        "max_risk_per_trade_usd": null,
        "max_daily_loss_usd": null,
        "max_weekly_loss_usd": null,
        "max_trades_per_day": null,
        "primary_session_only": false,
        "flatten_before_eod": false,
        "flatten_eod_minutes_before_close": null,
        "stop_width_gate_enabled": false
      }
    }
  }
}
```

## B. Proposed `config/withdrawal_policy.json` (CAD)

`seed_capital_usd`→`seed_capital_cad` (50000 → **70800**), `max_monthly_salary_usd`→`max_monthly_salary_cad` (2000 → **2832**). `build_phase_threshold_multiplier` (1.20), `payout_rate_above_hwm`, `drawdown_veto_pct`, `drawdown_lookback_days`, `minimum_history_days` are dimensionless/time and unchanged. Schema bumped `withdrawal_policy.v1` → `withdrawal_policy.v2`.

```json
{
  "schema_version": "withdrawal_policy.v2",
  "_comment": "WithdrawalManager policy. Edit values to tune the salary engine.",
  "_phases": "BUILD: equity < seed*build_mult. GROW: above threshold but SCR not CONFIDENT, OR drawdown > veto, OR below HWM. PAY: above HWM, SCR CONFIDENT, no drawdown veto.",
  "seed_capital_cad": 70800.0,
  "build_phase_threshold_multiplier": 1.20,
  "payout_rate_above_hwm": 0.30,
  "max_monthly_salary_cad": 2832.0,
  "drawdown_veto_pct": 5.0,
  "drawdown_lookback_days": 30,
  "require_scr_confident": true,
  "minimum_history_days": 14
}
```

## Activation (operator, at a gated maintenance window)

1. Confirm the companion code commit is the deployed HEAD.
2. Copy block A over `config/tiers.json` and block B over `config/withdrawal_policy.json`.
3. Restart together: `chad-tier-manager`, `chad-withdrawal-manager`, `chad-business-phase` (and the dashboard/report consumers). Do not leave a tick window between config-apply and restart.
4. Verify: `runtime/tier_state.json` shows `schema_version: tier_state.v3`, `current_equity_cad` populated from `ibkr_equity`, tier still PRO_GROWTH; `runtime/business_phase.json` shows `business_phase.v2`; withdrawal authorization recomputes off CAD.

## Follow-ups not in the code commit (apply when the PA lands)

- `chad/tests/chad_behavioral_audit_harness.py` (manual audit harness — **not** pytest-collected) loads the real config: lines ~690-709 assert the salary cap is `2000.0` and `_select_tier` band numbers off the real USD config. Update its expected cap to `2832.0` and the `_select_tier` band fixtures to the CAD thresholds when this PA is applied.
- `config/tiers.json` `schema_version` may optionally bump to `tiers.v9_2` to mark the key relabel; left at `v9_1` here per the C-ii manifest.

## Residual label notes (documented, intentional)

- The dashboard business block keeps API output keys `high_water_mark_usd` / `authorized_salary_usd` (api.py:680-681, :1300) carrying CAD values post-activation — these are display-only API keys with no non-test consumer; renaming them is deferred to avoid an unscoped API-contract change (manifest §3/§6.1).
- Withdrawal `authorized_withdrawal_usd` and `spendable_surplus_usd` field names retained (value re-prices to CAD); they have no equity-band consumer.
