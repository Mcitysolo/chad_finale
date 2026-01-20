#!/usr/bin/env bash
set -euo pipefail

cd "/home/ubuntu/CHAD FINALE"

# Generate executive report now (also pushes to Telegram via notify()).
OUT="$(./venv/bin/python -m chad.ops.daily_executive_report)"

echo "== GENERATED PATHS =="
echo "$OUT"

echo
MD="$(echo "$OUT" | tail -n 1)"
echo "== LATEST MD =="
echo "$MD"

echo
echo "== PREVIEW (first 120 lines) =="
sed -n '1,120p' "$MD"
