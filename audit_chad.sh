#!/usr/bin/env bash
set -euo pipefail

ts="$(date -u +%Y%m%dT%H%M%SZ)"
outdir="$HOME/audits/chad/$ts"
mkdir -p "$outdir"

log="$outdir/audit_chad.log"
json="$outdir/audit_chad.json"
md="$outdir/audit_chad.md"

exec > >(tee -a "$log") 2>&1

echo "== CHAD AUDIT START (UTC $ts) =="
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "Uptime: $(uptime -p || true)"
echo

# --- Discover CHAD root (no assumptions) ---
candidates=(
  "/home/ubuntu/chad_finale"
  "/home/ubuntu/CHAD FINALE"
  "$HOME/chad_finale"
  "$HOME/CHAD FINALE"
)
CHAD_ROOT=""
for p in "${candidates[@]}"; do
  if [ -d "$p" ] && [ -d "$p/chad" ]; then
    CHAD_ROOT="$p"
    break
  fi
done

if [ -z "$CHAD_ROOT" ]; then
  echo "ERROR: Could not find CHAD root. Searched: ${candidates[*]}"
  echo "Tip: put CHAD at /home/ubuntu/chad_finale or /home/ubuntu/CHAD FINALE per CSB."
  exit 2
fi

echo "CHAD_ROOT=$CHAD_ROOT"
echo

# --- System snapshot ---
echo "== OS / Kernel =="
uname -a || true
cat /etc/os-release 2>/dev/null || true
echo

echo "== Python / Node =="
python3 -V || true
python -V || true
node -v 2>/dev/null || true
npm -v 2>/dev/null || true
echo

echo "== Disk / Memory =="
df -h || true
free -h || true
echo

# --- Repo snapshot ---
echo "== Repo status =="
cd "$CHAD_ROOT"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 && {
  echo "Git HEAD: $(git rev-parse HEAD)"
  echo "Git branch: $(git rev-parse --abbrev-ref HEAD)"
  echo "Git status:"
  git status --porcelain || true
} || {
  echo "NOTE: Not a git repo (or git not initialized)."
}
echo

# --- Secrets posture ---
echo "== Secrets posture (/etc/chad) =="
sudo ls -la /etc/chad 2>/dev/null || true
sudo stat /etc/chad/chad.env 2>/dev/null || true
sudo stat /etc/chad/openai.env 2>/dev/null || true
echo

# --- systemd services status (CSB minimum set) ---
echo "== systemd: chad-* units =="
systemctl list-units --all "chad-*.service" "chad-*.timer" --no-pager || true
echo

echo "== systemd: key unit status (if present) =="
units=(
  chad-orchestrator.service
  chad-backend.service
  chad-polygon-stocks.service
  chad-telegram-bot.service
  chad-reconcile.service
  chad-ibgateway.service
  chad-ibkr-health.service
  chad-ibkr-watchdog.service
)
for u in "${units[@]}"; do
  if systemctl list-unit-files --no-pager | awk '{print $1}' | grep -qx "$u"; then
    echo "--- $u ---"
    systemctl status "$u" --no-pager || true
    echo
  fi
done

# --- Ports / HTTP surfaces (CSB ports) ---
echo "== Ports listening (9618/9619/9620 expected by CSB) =="
ss -ltnp 2>/dev/null | egrep '(:9618|:9619|:9620)\b' || true
echo

echo "== HTTP checks (localhost) =="
curl -fsS "http://127.0.0.1:9618/health" 2>/dev/null || echo "health: FAIL (no response)"
echo
curl -fsS "http://127.0.0.1:9618/live-gate" 2>/dev/null || echo "live-gate: FAIL (no response)"
echo
curl -fsS "http://127.0.0.1:9620/metrics" 2>/dev/null || echo "metrics: FAIL (no response)"
echo

# --- Runtime state freshness (ts_utc + ttl_seconds) ---
echo "== Runtime state freshness =="
runtime_dir="$CHAD_ROOT/runtime"
if [ ! -d "$runtime_dir" ]; then
  echo "NOTE: runtime/ missing at $runtime_dir"
else
  python3 - <<'PY'
import json
import datetime
from pathlib import Path

runtime = Path("runtime")
now = datetime.datetime.now(datetime.timezone.utc)

def parse_ts_any(v):
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.datetime.fromisoformat(s).astimezone(datetime.timezone.utc)
    except Exception:
        return None

print(f"Now UTC: {now.isoformat()}")

CONFIG_FILES = {
    "scr_config.json",
    "ibkr_paper_ledger.json",
    "paper_shadow.json",
    "paper_shadow_execute_test.json",
    "_debug_live_gate.json",
    "_live_gate_tmp.json",
    "full_execution_cycle_last.json",
    "live_gate_20251213T154930Z.json",
}

rows = []
for fp in sorted(runtime.glob("*.json")):
    name = fp.name
    try:
        obj = json.loads(fp.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            rows.append((name, "CORRUPT", "not a JSON object"))
            continue
    except Exception as e:
        rows.append((name, "CORRUPT", f"json_error={type(e).__name__}: {e}"))
        continue

    if name in CONFIG_FILES:
        rows.append((name, "CONFIG", "ttl not required"))
        continue

    ts = parse_ts_any(obj.get("ts_utc"))
    ttl = obj.get("ttl_seconds")

    if ts is None or not isinstance(ttl, (int, float)):
        rows.append((name, "UNKNOWN", "missing ts_utc/ttl_seconds"))
        continue

    age = (now - ts).total_seconds()
    status = "FRESH" if age <= float(ttl) else "STALE"
    rows.append((name, status, f"age={int(age)}s ttl={int(ttl)}s"))

for name, status, detail in rows:
    print(f"{name:28} {status:7} {detail}")
PY
fi
echo

# --- Ledgers / traces presence (append-only expectation) ---
echo "== Ledger / trace files presence =="
for d in data/trades data/traces data/exec_state shared/models reports; do
  if [ -d "$CHAD_ROOT/$d" ]; then
    echo "--- $d ---"
    ls -la "$CHAD_ROOT/$d" | head -n 50 || true
  else
    echo "--- $d: MISSING ---"
  fi
  echo
done

# --- Minimal python import sanity (read-only) ---
echo "== Python import sanity (best-effort) =="
python3 - <<PY
import sys, importlib
sys.path.insert(0, "$CHAD_ROOT")
mods = [
  "chad",
]
for m in mods:
  try:
    importlib.import_module(m)
    print(f"OK import: {m}")
  except Exception as e:
    print(f"FAIL import: {m} -> {e}")
PY
echo

# --- Produce a small machine-readable JSON summary ---
echo "== Writing JSON summary =="
python3 - <<PY
import json, os, subprocess, datetime
from pathlib import Path

ts="$ts"
root=Path("$CHAD_ROOT")
out=Path("$json")
now=datetime.datetime.now(datetime.timezone.utc).isoformat()

def sh(cmd):
  try:
    return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
  except Exception as e:
    return f"ERR: {e}"

summary={
  "ts_utc": now,
  "chad_root": str(root),
  "ports": sh(r"ss -ltnp 2>/dev/null | egrep '(:9618|:9619|:9620)\b' || true"),
  "health": sh(r"curl -fsS http://127.0.0.1:9618/health 2>/dev/null || true"),
  "live_gate": sh(r"curl -fsS http://127.0.0.1:9618/live-gate 2>/dev/null || true"),
  "metrics": sh(r"curl -fsS http://127.0.0.1:9620/metrics 2>/dev/null | head -n 50 || true"),
  "units": sh(r"systemctl list-units --all 'chad-*.service' 'chad-*.timer' --no-pager || true"),
}
out.write_text(json.dumps(summary, indent=2))
print(f"Wrote {out}")
PY

# --- Markdown pointer ---
cat > "$md" <<MD
# CHAD Audit ($ts)

Artifacts:
- \`audit_chad.log\` (full console output)
- \`audit_chad.json\` (summary)

Key checks (CSB-aligned):
- systemd units: \`chad-*\`
- ports: 9618/9619/9620
- \`/health\`, \`/live-gate\`, \`/metrics\`
- runtime JSON freshness: ts_utc + ttl_seconds
- ledgers: data/trades, data/traces, data/exec_state

MD

echo
echo "== DONE =="
echo "Output: $outdir"
