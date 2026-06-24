#!/usr/bin/env python3
"""Auto-dismiss the IB Gateway "Paper Account Notice" dialog (IBC 3.23.0 gap).

Context
-------
``chad-ibgateway.service`` (paper, IBC 3.23.0 + IB Gateway 10.37 build 1037) runs
headless on the Xvfb display ``:99``. After a successful "Paper Log In", IBKR
pops a modal dialog titled exactly ``Paper Account Notice``. IBC 3.23.0's
``AcceptNonBrokerageAccountWarning`` title-match does not recognise this renamed
dialog, so it is never auto-clicked. While the modal is up it blocks the API
port (4002) from opening. The dialog has a single bottom-centre ``OK`` button.

What this does
--------------
A bounded, idempotent watcher intended to run as an ``ExecStartPost`` of the
Gateway unit. It polls (``search`` verb) for a window whose name matches the
target title and, when ARMED, clicks the derived ``OK`` button location to
dismiss it. Geometry is read live every tick (nothing about the on-screen
position is hard-coded). Worst case it is a no-op: if no notice is ever seen it
exits 0; if the notice is seen but will not close after the attempt budget it
still exits 0 (status ``STUCK``). It must NEVER block or crash the Gateway
service start, so every code path returns 0.

Safety
------
* Pure stdlib + ``subprocess`` calls to ``xdotool``. No network, no broker, no
  order, no config and no runtime mutation.
* The ONLY ``xdotool`` verbs this module ever invokes are: ``search``,
  ``getwindowgeometry``, ``mousemove``, ``click``, ``windowactivate`` and
  ``key Return``. (``getwindowname`` is intentionally NOT used — it is outside
  the permitted verb set; window-name equality is asserted using ``search``.)
* It is structurally impossible to interact with any window whose name is not
  exactly ``Paper Account Notice``: the only title literal ever passed to
  ``search`` is :data:`TARGET_TITLE`, and a click is gated behind
  :func:`_assert_target_window`, which re-confirms (via ``search`` only) that the
  window id still resolves to that exact title immediately before any pointer or
  key action.
* Clicking is gated behind the ``CHAD_NOTICE_DISMISS_ARMED=1`` environment
  variable. Unset/!= "1" → DRY-RUN: the watcher does the geometry math and logs
  intent but performs NO pointer or key action. This makes the script safe by
  default; production arming is an explicit operator step (see the staged
  drop-in and the companion Pending Action).

Output
------
Writes exactly ONE artifact per run to
``reports/gateway_paper_notice_log/<UTC_BASIC_ISO>.json`` (directory created if
absent). That JSON file is this module's ONLY filesystem write.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Tuple

# --- constants ---------------------------------------------------------------

#: Exact WM_NAME of the dialog. The ONLY window-title literal in this module.
TARGET_TITLE: str = "Paper Account Notice"

#: X display used by chad-ibgateway.service (chad-ibgateway.service:14).
DEFAULT_DISPLAY: str = ":99"

#: Total poll budget and per-tick interval. 120s / 3s -> at most 40 ticks.
POLL_BUDGET_SECONDS: float = 120.0
POLL_INTERVAL_SECONDS: float = 3.0

#: Maximum click attempts before declaring the notice STUCK (non-fatal).
MAX_CLICK_ATTEMPTS: int = 3

#: The single OK button sits ~22px above the dialog's bottom edge, centred.
#: Target = (x + w//2, y + h - BUTTON_OFFSET_FROM_BOTTOM_PX). Derived live.
BUTTON_OFFSET_FROM_BOTTOM_PX: int = 22

#: Env flag that ARMS clicking. Unset / != "1" -> dry-run (no pointer/key action).
ARMED_ENV: str = "CHAD_NOTICE_DISMISS_ARMED"

#: Absolute xdotool path (matches `which xdotool` on the host).
XDOTOOL: str = "/usr/bin/xdotool"

#: Hard timeout on every xdotool subprocess so the watcher is bounded everywhere.
SUBPROC_TIMEOUT_SECONDS: float = 10.0

#: repo_root/reports/gateway_paper_notice_log/ — resolved from this file's path
#: so the (single) write is CWD-independent under systemd.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
ARTIFACT_DIR: Path = REPO_ROOT / "reports" / "gateway_paper_notice_log"

_POSITION_RE = re.compile(r"Position:\s*(-?\d+),(-?\d+)")
_GEOMETRY_RE = re.compile(r"Geometry:\s*(\d+)x(\d+)")


class Geometry(NamedTuple):
    """A window's outer geometry as reported by ``xdotool getwindowgeometry``."""

    x: int
    y: int
    w: int
    h: int


class NoticeWindowMismatch(RuntimeError):
    """Raised when a window id no longer resolves to the target title.

    Fail-closed sentinel: callers must abandon any pointer/key action when this
    is raised, because the window we measured is gone or renamed.
    """


# --- pure parsing / geometry (unit-tested; no subprocess, no X) ---------------


def parse_search_output(stdout: str) -> List[int]:
    """Parse ``xdotool search`` stdout into a list of integer window ids.

    Non-numeric and blank lines are ignored so the parser is robust to xdotool
    diagnostics. Order is preserved.
    """
    wids: List[int] = []
    for line in stdout.splitlines():
        token = line.strip()
        if token.isdigit():
            wids.append(int(token))
    return wids


def parse_window_geometry(stdout: str) -> Geometry:
    """Parse ``xdotool getwindowgeometry <wid>`` (human form) into a Geometry.

    Tolerates the trailing ``(screen: N)`` suffix on the Position line, e.g.::

        Window 12583125
          Position: 638,453 (screen: 0)
          Geometry: 644x175

    Raises ValueError if either the Position or Geometry line is absent.
    """
    pos = _POSITION_RE.search(stdout)
    geo = _GEOMETRY_RE.search(stdout)
    if pos is None or geo is None:
        raise ValueError(f"unparseable getwindowgeometry output: {stdout!r}")
    return Geometry(
        x=int(pos.group(1)),
        y=int(pos.group(2)),
        w=int(geo.group(1)),
        h=int(geo.group(2)),
    )


def compute_click_target(geom: Geometry) -> Tuple[int, int]:
    """Derive the OK-button click point: bottom-centre of the dialog.

    ``(x + w // 2, y + h - BUTTON_OFFSET_FROM_BOTTOM_PX)``. For the live notice
    (x=638, y=453, w=644, h=175) this yields (960, 606).
    """
    return (geom.x + geom.w // 2, geom.y + geom.h - BUTTON_OFFSET_FROM_BOTTOM_PX)


def build_artifact(
    *,
    ts_utc: str,
    display: str,
    notice_seen: bool,
    dismissed: bool,
    attempts: int,
    status: str,
    geometry: Optional[Geometry],
    click_target: Optional[Tuple[int, int]],
) -> dict:
    """Assemble the run artifact dict (shape is part of the public contract)."""
    return {
        "ts_utc": ts_utc,
        "display": display,
        "notice_seen": notice_seen,
        "dismissed": dismissed,
        "attempts": attempts,
        # status in {DISMISSED, NOT_SEEN, STUCK, DRY_RUN}. DRY_RUN = notice seen
        # while un-ARMED (geometry computed, deliberately not clicked).
        "status": status,
        "geometry": (
            None
            if geometry is None
            else {"x": geometry.x, "y": geometry.y, "w": geometry.w, "h": geometry.h}
        ),
        "click_target": None if click_target is None else [click_target[0], click_target[1]],
    }


# --- xdotool wrappers (each uses ONLY a permitted verb) ----------------------


def _log(msg: str) -> None:
    """Emit a journald line (stderr). Not a filesystem write."""
    print(f"[dismiss_paper_account_notice] {msg}", file=sys.stderr, flush=True)


def _run_xdotool(display: str, args: Sequence[str]) -> subprocess.CompletedProcess:
    """Run ``xdotool <args>`` against ``display`` with a hard timeout.

    Never raises on tool failure/timeout/absence: returns a CompletedProcess
    with a non-zero returncode and empty stdout so callers degrade benignly
    (the watcher must never crash the Gateway start).
    """
    env = dict(os.environ)
    env["DISPLAY"] = display
    try:
        return subprocess.run(
            [XDOTOOL, *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=SUBPROC_TIMEOUT_SECONDS,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:  # bounded, benign
        _log(f"xdotool {list(args)} failed: {exc!r}")
        return subprocess.CompletedProcess(args=[XDOTOOL, *args], returncode=127, stdout="", stderr=str(exc))


def search_notice(display: str) -> List[int]:
    """``xdotool search --name "Paper Account Notice"`` -> window ids.

    :data:`TARGET_TITLE` is the only title literal ever passed to ``search``.
    """
    proc = _run_xdotool(display, ["search", "--name", TARGET_TITLE])
    if proc.returncode != 0:
        # xdotool search exits non-zero when nothing matches; that's "no notice".
        return []
    return parse_search_output(proc.stdout)


def get_geometry(display: str, wid: int) -> Geometry:
    """``xdotool getwindowgeometry <wid>`` -> parsed Geometry (may raise ValueError)."""
    proc = _run_xdotool(display, ["getwindowgeometry", str(wid)])
    return parse_window_geometry(proc.stdout)


def _assert_target_window(display: str, wid: int) -> None:
    """Fail-closed: confirm ``wid`` STILL resolves to the exact target title.

    Equality is asserted using only the ``search`` verb: a window id is accepted
    iff it currently appears in ``search --name "Paper Account Notice"``. Raises
    :class:`NoticeWindowMismatch` otherwise. This is the structural guarantee
    that no pointer/key action can ever touch a non-target window.
    """
    if wid not in search_notice(display):
        raise NoticeWindowMismatch(f"window {wid} no longer matches {TARGET_TITLE!r}")


def dismiss_once(display: str, wid: int, target: Tuple[int, int]) -> None:
    """Perform ONE dismissal attempt against an asserted target window.

    Order: assert title -> ``mousemove --sync`` + ``click 1`` at ``target`` ->
    ``windowactivate --sync`` + ``key Return`` (belt-and-suspenders). Caller is
    responsible for re-checking that the window is gone afterwards.

    Only ever called when ARMED. Raises :class:`NoticeWindowMismatch` (before any
    action) if the window vanished/renamed between measurement and click.
    """
    _assert_target_window(display, wid)
    tx, ty = target
    _run_xdotool(display, ["mousemove", "--sync", str(tx), str(ty), "click", "1"])
    _run_xdotool(display, ["windowactivate", "--sync", str(wid), "key", "Return"])


# --- watcher loop ------------------------------------------------------------


def _sleep_until_next_tick(deadline: float) -> None:
    """Sleep one interval without overshooting the overall budget."""
    remaining = deadline - time.monotonic()
    if remaining > 0:
        time.sleep(min(POLL_INTERVAL_SECONDS, remaining))


def run(display: str, armed: bool, ts_utc: str) -> dict:
    """Run the bounded poll loop and return the (unwritten) run artifact dict."""
    deadline = time.monotonic() + POLL_BUDGET_SECONDS
    attempts = 0
    notice_seen = False
    geometry: Optional[Geometry] = None
    click_target: Optional[Tuple[int, int]] = None

    while time.monotonic() < deadline:
        wids = search_notice(display)
        if not wids:
            _sleep_until_next_tick(deadline)
            continue

        wid = wids[0]
        notice_seen = True
        try:
            geometry = get_geometry(display, wid)
        except ValueError as exc:
            _log(f"geometry parse failed for wid {wid}: {exc!r}; retrying")
            _sleep_until_next_tick(deadline)
            continue
        click_target = compute_click_target(geometry)

        if not armed:
            _log(
                f"DRY-RUN ({ARMED_ENV} unset): notice seen wid={wid} "
                f"geometry={geometry} click_target={click_target}; NOT clicking."
            )
            return build_artifact(
                ts_utc=ts_utc,
                display=display,
                notice_seen=True,
                dismissed=False,
                attempts=0,
                status="DRY_RUN",
                geometry=geometry,
                click_target=click_target,
            )

        attempts += 1
        _log(f"ARMED: dismissal attempt {attempts}/{MAX_CLICK_ATTEMPTS} wid={wid} target={click_target}")
        try:
            dismiss_once(display, wid, click_target)
        except NoticeWindowMismatch as exc:
            _log(f"target vanished before click: {exc}; re-polling")
            continue

        if not search_notice(display):
            _log(f"DISMISSED after {attempts} attempt(s).")
            return build_artifact(
                ts_utc=ts_utc,
                display=display,
                notice_seen=True,
                dismissed=True,
                attempts=attempts,
                status="DISMISSED",
                geometry=geometry,
                click_target=click_target,
            )

        if attempts >= MAX_CLICK_ATTEMPTS:
            _log(f"STUCK: notice still open after {attempts} attempt(s); giving up (non-fatal).")
            return build_artifact(
                ts_utc=ts_utc,
                display=display,
                notice_seen=True,
                dismissed=False,
                attempts=attempts,
                status="STUCK",
                geometry=geometry,
                click_target=click_target,
            )
        _sleep_until_next_tick(deadline)

    if notice_seen:
        # Budget elapsed with the notice seen but never confirmed gone.
        _log("STUCK: budget elapsed with notice still present (non-fatal).")
        return build_artifact(
            ts_utc=ts_utc,
            display=display,
            notice_seen=True,
            dismissed=False,
            attempts=attempts,
            status="STUCK",
            geometry=geometry,
            click_target=click_target,
        )

    _log("NOT_SEEN: budget elapsed, no Paper Account Notice ever appeared (benign).")
    return build_artifact(
        ts_utc=ts_utc,
        display=display,
        notice_seen=False,
        dismissed=False,
        attempts=0,
        status="NOT_SEEN",
        geometry=None,
        click_target=None,
    )


def write_artifact(artifact: dict, ts_basic: str) -> Path:
    """Write the single run artifact and return its path (the ONLY write)."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / f"{ts_basic}.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Always returns 0 — must never fail the Gateway start."""
    display = os.environ.get("DISPLAY", DEFAULT_DISPLAY) or DEFAULT_DISPLAY
    armed = os.environ.get(ARMED_ENV) == "1"
    now = datetime.now(timezone.utc)
    ts_utc = now.strftime("%Y-%m-%dT%H:%M:%SZ")  # canonical extended ISO (artifact field)
    ts_basic = now.strftime("%Y%m%dT%H%M%SZ")     # basic ISO (filename; sibling-log convention)

    _log(f"start display={display} armed={armed} budget={POLL_BUDGET_SECONDS:.0f}s interval={POLL_INTERVAL_SECONDS:.0f}s")
    try:
        artifact = run(display=display, armed=armed, ts_utc=ts_utc)
    except Exception as exc:  # absolute backstop: never crash the service start
        _log(f"unexpected error: {exc!r}; recording STUCK and exiting 0")
        artifact = build_artifact(
            ts_utc=ts_utc,
            display=display,
            notice_seen=False,
            dismissed=False,
            attempts=0,
            status="STUCK",
            geometry=None,
            click_target=None,
        )

    try:
        path = write_artifact(artifact, ts_basic)
        _log(f"artifact -> {path} status={artifact['status']}")
    except OSError as exc:
        _log(f"artifact write failed (non-fatal): {exc!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
