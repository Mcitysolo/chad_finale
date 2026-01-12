#!/usr/bin/env bash
set -euo pipefail

cd "/home/ubuntu/CHAD FINALE"

# Production-grade: avoid massive 'ls' expansions + handle race where JSON is mid-write.
# We use find + sort, then retry JSON parse a few times.

latest_report() {
  find "reports/shadow" -maxdepth 1 -type f -name 'PAPER_SHADOW_EXECUTE_*.json' -print0 \
  | xargs -0 ls -1t 2>/dev/null \
  | head -n 1
}

LATEST="$(latest_report || true)"
if [[ -z "${LATEST}" ]]; then
  echo "ERROR: no reports found under reports/shadow/PAPER_SHADOW_EXECUTE_*.json" >&2
  exit 2
fi

echo "LATEST=${LATEST}"
echo

# Retry JSON load (atomic writes should prevent partial files, but we defend anyway)
for attempt in 1 2 3 4 5; do
  if ./venv/bin/python - "$LATEST" <<'PY'
import json, sys
p=sys.argv[1]
with open(p,"r",encoding="utf-8") as f:
    d=json.load(f)

preview=d.get("paper_intents_preview") or []
print("preview_count:", len(preview))
print("preview_strategies:", sorted({(i.get("strategy") or "").lower() for i in preview}))
print("preview_symbols:", [i.get("symbol") for i in preview])
print("counts:", d.get("counts"))
PY
  then
    exit 0
  fi
  echo "[WARN] JSON parse failed (attempt ${attempt}/5). Retrying..." >&2
  sleep 0.2
done

echo "ERROR: failed to parse latest report after retries: ${LATEST}" >&2
exit 1
