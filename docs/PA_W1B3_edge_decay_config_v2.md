# Pending Action — edge_decay_config.v2 self-clear keys (W1B-3)

**Status:** PENDING (awaits operator apply). Not applied directly — per
Governance Rule #3 (no direct config mutation; risk/strategy config lands as a
Pending Action only). `config/edge_decay_config.json` is governed strategy
config, so its VALUES are prepared here rather than edited in the W1B-3 commit.

**Context.** W1B-3 ships the edge-decay *self-clear* LOGIC plus conservative
**in-code defaults** in `chad/risk/edge_decay_monitor.py`:

```
DEFAULT_HALT_TTL_DAYS      = 14     # TTL clear of a stale monitor-imposed halt
DEFAULT_CLEAR_ON_RECOVERY  = True   # persist a ledger-resident recovery
```

`_load_config_values()` reads the two new keys **if present** and otherwise
falls back to these defaults, so the current `edge_decay_config.v1` file runs
the shipped behaviour unchanged. This PA only makes the values **explicit and
operator-governed** in the config file.

## Proposed change (apply only on GO)

Replace `config/edge_decay_config.json` with (additive keys + schema bump):

```json
{
  "schema_version": "edge_decay_config.v2",
  "notes": "Calibrated 2026-04-22 (Audit-O): consecutive_threshold lowered 10→5 to match observed max streak of 3 (alpha strategy, 6 streaks total). Threshold 5 is 1.7× the observed maximum, leaving room for normal variance while still halting genuine decay. W1B-3 (2026-07-19): added self-clear keys — halt_ttl_days and clear_on_recovery. Auto-clear only ever releases a MONITOR-imposed halt when the TRUSTED ledger shows no fresh losing streak; operator-imposed/operator-cleared halts are never touched.",
  "consecutive_threshold": 5,
  "min_trades": 20,
  "halt_ttl_days": 14,
  "clear_on_recovery": true
}
```

- `halt_ttl_days` (14): auto-clear a monitor-imposed halt only after this many
  days **and** only when the trusted ledger shows no fresh losing evidence
  (strategy absent from the trusted ledger, or present with streak <
  `consecutive_threshold`). Set `<= 0` to disable TTL clears. Unsticks stale
  orphaned halts (e.g. alpha_crypto, whose fills are all pnl_untrusted/
  validate_only and excluded) without un-halting a genuinely decaying strategy.
- `clear_on_recovery` (true): when a ledger-resident halted strategy's trusted
  streak falls below `consecutive_threshold`, persist the recovery
  (`halted:false`) instead of the historical in-memory-only no-op.

`_load_config_values()` already tolerates a `.v1` file (falls back to the
defaults for the two new keys), so the schema bump is backward-compatible.

## Direction-of-change note

Auto-un-halting moves *toward* trading. Blast radius is low (posture is PAPER),
and the criteria are conservative and evidence-based: only monitor-imposed
halts, only on TTL-elapsed OR recovered, and only when the **trusted** ledger
shows no fresh losing streak (untrusted/validate_only rows can never clear a
halt). Every release emits an `EDGE_DECAY_AUTO_CLEARED` marker + a coach-voiced
operator NOTIFY.

## Verify after apply

```
cd /home/ubuntu/chad_finale && source venv/bin/activate
python3 -c "import chad.risk.edge_decay_monitor as e; print(e._load_config_values())"
# expect (5, 20, 14, True)
python3 -m pytest chad/tests/test_w1b_edge_decay_self_clear.py chad/tests/test_edge_decay_monitor.py -q
```
