#!/usr/bin/env bash
# GAP-032 preventive lint guard — thin wrapper.
#
# Delegates to chad.ops.systemd_wants_lint so the .py module is the
# single shared implementation (Phase-25 named entrypoint). Strictly
# read-only at runtime: no sudo, no systemctl enable/disable/start/rm.
#
# Exit code:
#   0  clean (no chad-scoped regular-file corruption)
#   2  chad-scoped regular-file corruption detected
set -euo pipefail
REPO_ROOT="/home/ubuntu/chad_finale"
exec "${REPO_ROOT}/venv/bin/python3" -m chad.ops.systemd_wants_lint "$@"
