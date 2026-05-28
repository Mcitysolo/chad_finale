#!/usr/bin/env bash
# Installs CHAD pre-commit hooks into .git/hooks/
# Run from repo root: bash scripts/install_precommit_hooks.sh
# Idempotent.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_DIR="${REPO_ROOT}/.git/hooks"
HOOK="${HOOK_DIR}/pre-commit"

mkdir -p "${HOOK_DIR}"

cat > "${HOOK}" <<'EOF'
#!/usr/bin/env bash
# CHAD pre-commit hook
# Enforces:
# 1. Port-binding invariant (no new 0.0.0.0 binds without allowlist)
# 2. No placeholder $100 fill prices in new strategy code

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# 1. Port binding check
if [ -f chad/validators/port_binding.py ]; then
  if ! /home/ubuntu/chad_finale/venv/bin/python3 -m chad.validators.port_binding --check >/dev/null; then
    echo "PRE-COMMIT BLOCKED: port-binding invariant violation."
    echo "Either rebind the service to 127.0.0.1 or add an allowlist entry."
    /home/ubuntu/chad_finale/venv/bin/python3 -m chad.validators.port_binding --check || true
    exit 1
  fi
fi

# 2. Placeholder fill check (light heuristic, not a replacement for the
#    full fill_validation.py runtime check)
if git diff --cached --name-only | grep -qE "^chad/strategies/.*\.py$"; then
  if git diff --cached chad/strategies/ | grep -qE "fill_price.*=.*100\.0|fill_price.*=.*100[^0-9]"; then
    echo "PRE-COMMIT BLOCKED: detected fill_price=100 in chad/strategies/."
    echo "Use the canonical price provider; do not introduce placeholder \$100 fills."
    exit 1
  fi
fi

exit 0
EOF

chmod +x "${HOOK}"
echo "Installed CHAD pre-commit hook at ${HOOK}"
