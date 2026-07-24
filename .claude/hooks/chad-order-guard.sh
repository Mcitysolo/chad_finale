#!/usr/bin/env bash
# chad-order-guard.sh — blocks live-order entrypoints, even under bypassPermissions.
input=$(cat)
cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // ""')
# W6B-13: hold_cancel_entries added. It cancels working orders at the broker,
# so it belongs in exactly the same Channel-1 posture as the flatten tools —
# operator-invoked in the terminal, never agent-invoked.
if printf '%s' "$cmd" | grep -qiE 'flatten_futures_oneshot|micro_eod_flatten|kraken_roundtrip_runner|reqGlobalCancel|flatten_all|hold_cancel_entries'; then
  printf 'BLOCKED by chad-order-guard: live-order/flatten entrypoint. Run this manually in the terminal (Channel 1).\n' >&2
  exit 2
fi
exit 0
