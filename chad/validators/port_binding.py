"""Port-binding validator (PORT-BINDING-1).

Scans repo code/config for the default bind hosts of CHAD's internal HTTP
listeners (9618 backend, 9619 status server, 9620 metrics server) and fails
when any of them defaults to a non-localhost address without an entry in the
documented allowlist.

The validator is read-only by default. With ``--live-check`` it additionally
parses ``ss -tlnp`` output to verify the *current* runtime bind, but this is
informational only — the canonical signal is the code/config default.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

EXIT_OK = 0
EXIT_NONLOCAL_DEFAULT = 2
EXIT_LIVE_CHECK_MISMATCH = 3

ALLOWLIST_FILE = REPO_ROOT / "config" / "port_binding_allowlist.json"
ALLOWLIST_SCHEMA_VERSION = "port_binding_allowlist.v1"

ALLOWLIST: dict[int, str] = {
    # port -> documented reason. Empty dict means "no port may default to non-localhost".
    # Add an entry here only with explicit operator approval recorded in
    # ops/pending_actions/.
    # NOTE: This in-code dict is overlaid by load_allowlist_file() at evaluate()
    # time. The on-disk file at config/port_binding_allowlist.json is the
    # canonical operator-domain source.
}


def load_allowlist_file(path: Path | None = None) -> dict[int, dict]:
    """Load the operator-domain allowlist JSON. Returns a {port: entry} map.

    Missing file → empty dict (no allowlist entries).
    Wrong schema_version → raises ValueError.
    """
    fp = path or ALLOWLIST_FILE
    if not fp.is_file():
        return {}
    try:
        doc = json.loads(fp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"port_binding_allowlist.json malformed: {exc}") from exc
    sv = doc.get("schema_version")
    if sv != ALLOWLIST_SCHEMA_VERSION:
        raise ValueError(
            f"port_binding_allowlist.json schema_version mismatch: "
            f"got {sv!r}, expected {ALLOWLIST_SCHEMA_VERSION!r}"
        )
    out: dict[int, dict] = {}
    for service, entry in (doc.get("allowlist") or {}).items():
        if not isinstance(entry, dict):
            continue
        port = entry.get("port")
        if not isinstance(port, int):
            continue
        out[port] = {"service": service, **entry}
    return out


@dataclass
class PortAudit:
    port: int
    owner: str
    source_file: str
    code_default_host: str
    env_var: str | None
    is_localhost: bool
    allowlist_reason: str | None
    live_bind_host: str | None = None


# ---------------------------------------------------------------------------
# Static (code/config) scanning
# ---------------------------------------------------------------------------

# Each entry: (port, owner_label, source_path, regex_to_extract_default_host, env_var_name)
TARGETS: list[tuple[int, str, str, str, str | None]] = [
    (
        9619,
        "chad-shadow-status (chad.web.status_server)",
        "chad/web/status_server.py",
        r'CHAD_STATUS_HOST"\s*,\s*"([^"]+)"',
        "CHAD_STATUS_HOST",
    ),
    (
        9620,
        "chad-metrics (chad.ops.metrics_server)",
        "chad/ops/metrics_server.py",
        r'CHAD_METRICS_HOST"\s*,\s*"([^"]+)"',
        "CHAD_METRICS_HOST",
    ),
    (
        9618,
        "chad-backend (uvicorn backend.app:app — host passed via ExecStart)",
        "(systemd ExecStart in /etc/systemd/system/chad-backend.service)",
        # No Python-side default. We surface this as a known operator-domain
        # gap and rely on the companion systemd-edit Pending Action.
        r"",
        None,
    ),
]


def _scan_code_default(target: tuple[int, str, str, str, str | None]) -> PortAudit:
    port, owner, source, pattern, env_var = target
    audit = PortAudit(
        port=port,
        owner=owner,
        source_file=source,
        code_default_host="unknown",
        env_var=env_var,
        is_localhost=False,
        allowlist_reason=ALLOWLIST.get(port),
    )
    if not pattern:
        audit.code_default_host = "operator_domain_systemd_arg"
        audit.is_localhost = False
        return audit
    fp = REPO_ROOT / source
    if not fp.is_file():
        audit.code_default_host = "source_not_found"
        return audit
    text = fp.read_text(encoding="utf-8")
    m = re.search(pattern, text)
    if not m:
        audit.code_default_host = "pattern_not_found"
        return audit
    host = m.group(1)
    audit.code_default_host = host
    audit.is_localhost = host in {"127.0.0.1", "localhost", "::1"}
    return audit


def _scan_live(port: int) -> str | None:
    try:
        out = subprocess.check_output(
            ["ss", "-tlnp"], stderr=subprocess.DEVNULL, text=True, timeout=5
        )
    except Exception:
        return None
    for line in out.splitlines():
        if f":{port}" not in line:
            continue
        parts = line.split()
        # ss columns: State Recv-Q Send-Q Local-Addr Peer-Addr ...
        for token in parts:
            if token.endswith(f":{port}"):
                host = token.rsplit(":", 1)[0]
                return host
    return None


def run_audit(*, live_check: bool = False) -> list[PortAudit]:
    audits: list[PortAudit] = []
    for target in TARGETS:
        a = _scan_code_default(target)
        if live_check:
            a.live_bind_host = _scan_live(a.port)
        audits.append(a)
    return audits


def evaluate(audits: list[PortAudit], *, file_allowlist: dict[int, dict] | None = None) -> tuple[int, dict]:
    """Classify each audit as pass / warning (allowlisted) / failure.

    ``file_allowlist`` overlays the in-code ``ALLOWLIST`` dict. When omitted,
    the canonical operator-domain file at ``config/port_binding_allowlist.json``
    is loaded.
    """
    if file_allowlist is None:
        try:
            file_allowlist = load_allowlist_file()
        except ValueError as exc:
            # Surface schema problems as a hard failure — operator must fix.
            return EXIT_NONLOCAL_DEFAULT, {
                "validator": "port_binding.v1",
                "error": str(exc),
                "audits": [asdict(a) for a in audits],
                "failures": [{"port": None, "owner": "allowlist_file", "code_default_host": "n/a", "reason": str(exc)}],
                "warnings": [],
            }
    failures: list[dict] = []
    warnings: list[dict] = []
    for a in audits:
        if a.code_default_host == "operator_domain_systemd_arg":
            # 9618 is by-design out of code-scope; the systemd-edit PA tracks it.
            warnings.append({
                "port": a.port,
                "owner": a.owner,
                "note": "bind-host is supplied by systemd ExecStart; see "
                        "ops/pending_actions/PORT_BINDING_systemd_unit_edits_2026-05-27.md",
            })
            continue
        if a.is_localhost:
            continue
        # Resolve allowlist: in-code reason wins if present; else file overlay.
        reason: str | None = a.allowlist_reason
        file_entry = file_allowlist.get(a.port)
        if reason is None and file_entry is not None:
            reason = f"{file_entry.get('service', 'unknown')}: {file_entry.get('reason', 'allowlisted (file)')}"
        if reason:
            warnings.append({
                "port": a.port,
                "owner": a.owner,
                "code_default_host": a.code_default_host,
                "allowlist_reason": reason,
                "allowlist_source": "file" if (a.allowlist_reason is None and file_entry is not None) else "in_code",
            })
            continue
        failures.append({
            "port": a.port,
            "owner": a.owner,
            "code_default_host": a.code_default_host,
        })
    report = {
        "validator": "port_binding.v1",
        "audits": [asdict(a) for a in audits],
        "failures": failures,
        "warnings": warnings,
        "allowlist_file_loaded": file_allowlist is not None and len(file_allowlist) > 0,
        "allowlist_file_entries": sorted(file_allowlist.keys()) if file_allowlist else [],
    }
    code = EXIT_NONLOCAL_DEFAULT if failures else EXIT_OK
    return code, report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="port_binding", description=__doc__)
    p.add_argument("--check", action="store_true", help="Run validator (default action).")
    p.add_argument("--live-check", action="store_true", help="Also parse ss -tlnp output (informational).")
    args = p.parse_args(argv)

    audits = run_audit(live_check=args.live_check)
    code, report = evaluate(audits)
    print(json.dumps(report, indent=2, default=str))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
