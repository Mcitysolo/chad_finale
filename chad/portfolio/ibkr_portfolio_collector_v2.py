from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Runtime file freshness for portfolio snapshot (short on purpose)
PORTFOLIO_SNAPSHOT_TTL_SECONDS = 300  # 5 minutes
POSITIONS_SNAPSHOT_TTL_SECONDS = 300  # 5 minutes

# NEW-GAP-043 wall-clock guard: the collector is a oneshot triggered every
# 120s by chad-ibkr-collector.timer. The IBKR gateway can stall the
# accountSummary()/positions() reply, and the systemd unit historically had
# TimeoutStartSec=infinity, so the service could (and did) sit in
# `activating/start` for hours, blocking subsequent timer fires. This
# code-level alarm is a defense-in-depth bound that ensures the process
# self-terminates even when systemd guardrails are misconfigured. The
# matching unit-level guard is in
# ops/systemd/chad-ibkr-collector.service.d/10-timeout-guards.conf.
DEFAULT_WALL_CLOCK_SECONDS = 60
WALL_CLOCK_ENV = "CHAD_COLLECTOR_WALL_CLOCK_SECONDS"


class CollectorWallClockTimeout(SystemExit):
    """Raised by the SIGALRM handler when the wall-clock guard fires.

    Inherits SystemExit so an uncaught instance produces a non-zero exit
    without a traceback (systemd records exit code 124 — the canonical
    timeout exit code chosen by `timeout(1)` — making the failure visible
    in `systemctl status` / `Result=exit-code` / `ExecMainStatus=124`).
    """

    def __init__(self, seconds: int) -> None:
        super().__init__(124)
        self.seconds = int(seconds)


def _resolve_wall_clock_seconds() -> int:
    raw = os.environ.get(WALL_CLOCK_ENV)
    if raw is None or str(raw).strip() == "":
        return DEFAULT_WALL_CLOCK_SECONDS
    try:
        v = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_WALL_CLOCK_SECONDS
    if v <= 0:
        return DEFAULT_WALL_CLOCK_SECONDS
    return v


def install_wall_clock_guard(seconds: Optional[int] = None) -> int:
    """Install a SIGALRM-based wall-clock guard for the collector process.

    Returns the number of seconds the alarm is set for. Re-arming with a
    new call REPLACES any previous alarm (POSIX semantics). On a non-Linux
    platform (no SIGALRM), this is a no-op that returns 0 — production
    runs only on Linux so this is purely a test-portability guard.

    The handler raises CollectorWallClockTimeout which propagates up and
    forces a non-zero (124) exit. The 124 exit code makes the failure
    visible as `Result=exit-code` in `systemctl status`, distinguishable
    from a normal clean exit (0) or other failures (1).
    """
    s = int(seconds) if seconds is not None else _resolve_wall_clock_seconds()
    if not hasattr(signal, "SIGALRM"):
        return 0

    def _handler(_signum: int, _frame: Any) -> None:  # pragma: no cover (signal-driven)
        print(
            f"[IBKR PORTFOLIO] WALL_CLOCK_TIMEOUT after {s}s — "
            "collector self-terminated to prevent stuck-activating systemd state",
            file=sys.stderr,
            flush=True,
        )
        raise CollectorWallClockTimeout(s)

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(s)
    return s


def disarm_wall_clock_guard() -> None:
    """Disable any installed SIGALRM. Used in tests; production main() does
    not need this — the kernel clears the alarm at process exit."""
    if hasattr(signal, "SIGALRM"):
        signal.alarm(0)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


@dataclass(frozen=True)
class IBKRConnectionConfig:
    """
    IBKR connection settings for paper gateway by default.
    Reads from environment variables used across this repo.
    """
    host: str
    port: int
    client_id: int
    account_id: str

    @staticmethod
    def from_env() -> "IBKRConnectionConfig":
        host = os.getenv("IBKR_HOST", "127.0.0.1")
        port = int(os.getenv("IBKR_PORT", "4002"))
        client_id = int(os.getenv("IBKR_CLIENT_ID", "99"))
        account_id = os.getenv("IBKR_ACCOUNT_ID", "").strip()
        return IBKRConnectionConfig(host=host, port=port, client_id=client_id, account_id=account_id)


class IBKRPortfolioCollector:
    """
    Collector that reads NetLiquidation via ib_insync and writes runtime/portfolio_snapshot.json.

    Contract:
      - Never writes secrets
      - Atomic + fsync
      - Adds ts_utc + ttl_seconds for audit freshness
      - Preserves other venue equities already present in the snapshot
    """

    def __init__(self, cfg: IBKRConnectionConfig) -> None:
        self._cfg = cfg

    def get_net_liquidation(self) -> float:
        """
        Returns NetLiquidation for the configured account, or sum across accounts if not set.

        Backward-compatible thin wrapper around
        :meth:`get_net_liquidation_with_currency` that discards the currency
        tag (used by the ``print`` CLI command).
        """
        value, _currency = self.get_net_liquidation_with_currency()
        return value

    def get_net_liquidation_with_currency(self) -> "tuple[float, str]":
        """
        Returns (NetLiquidation, currency) for the configured account, or the
        sum across accounts if no account is configured.

        The currency tag comes from the NetLiquidation AccountValue row
        (IBKR reports NetLiquidation in the account's base currency). When
        summing across multiple accounts the currency is only returned if
        every contributing account agrees; otherwise an empty string is
        returned so the caller's fail-closed gate engages (BOX-034A §3).
        """
        from ib_async import IB  # type: ignore[import]

        ib = IB()
        try:
            ib.connect(self._cfg.host, self._cfg.port, clientId=self._cfg.client_id, readonly=True, timeout=6.0)

            # accountSummary returns a list of AccountValue(account, tag, value, currency, modelCode)
            rows = ib.accountSummary()
            values_by_account: Dict[str, float] = {}
            currency_by_account: Dict[str, str] = {}
            for r in rows:
                try:
                    if str(getattr(r, "tag", "")) != "NetLiquidation":
                        continue
                    acct = str(getattr(r, "account", "")).strip()
                    val = float(getattr(r, "value", "0") or 0.0)
                    ccy = str(getattr(r, "currency", "") or "").strip().upper()
                    values_by_account[acct] = val
                    currency_by_account[acct] = ccy
                except Exception:
                    continue

            if not values_by_account:
                raise RuntimeError("No NetLiquidation values found in account summary")

            if self._cfg.account_id:
                acct = self._cfg.account_id
                if acct not in values_by_account:
                    raise RuntimeError(f"IBKR_ACCOUNT_ID={acct!r} not found in NetLiquidation summary")
                total = values_by_account[acct]
                currency = currency_by_account.get(acct, "")
            else:
                total = sum(values_by_account.values())
                distinct = {c for c in currency_by_account.values() if c}
                currency = next(iter(distinct)) if len(distinct) == 1 else ""

            return float(total), currency
        finally:
            try:
                ib.disconnect()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # snapshot integration                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _default_snapshot_path() -> Path:
        root = Path(__file__).resolve().parents[2]
        return root / "runtime" / "portfolio_snapshot.json"

    def update_snapshot(self, snapshot_path: Optional[Path] = None) -> Path:
        """
        Read existing portfolio_snapshot.json (if any), overwrite ibkr_equity
        with current NetLiquidation, preserve coinbase_equity and kraken_equity,
        write updated snapshot back atomically, TTL-stamped.

        BOX-034A §3 currency gate (fail-closed): the snapshot's ibkr_equity is
        canonical in the configured base currency (CHAD_BASE_CURRENCY, default
        CAD). The collector reads the NetLiquidation row's currency tag and
        only overwrites ibkr_equity when it matches the base. On a mismatch it
        does NOT write the wrong-currency value — it preserves the prior
        canonical value (read-through), logs a loud error, and records
        ibkr_equity_currency_ok=false so monitoring can see it.
        """
        path = snapshot_path or self._default_snapshot_path()

        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        else:
            data = {}

        base = os.environ.get("CHAD_BASE_CURRENCY", "CAD").strip().upper() or "CAD"
        ibkr_equity, row_currency = self.get_net_liquidation_with_currency()

        # Preserve other venues if present
        def _to_float(x: Any) -> float:
            try:
                return float(x)
            except Exception:
                return 0.0

        coinbase_equity = _to_float(data.get("coinbase_equity", 0.0))
        kraken_equity = _to_float(data.get("kraken_equity", 0.0))

        if row_currency == base:
            equity_to_write = float(ibkr_equity)
            currency_ok = True
        else:
            # Fail-closed: do NOT overwrite the canonical CAD value with a
            # wrong-currency reading. Preserve the prior canonical value
            # (read-through), exactly as the publisher does.
            equity_to_write = _to_float(data.get("ibkr_equity", 0.0))
            currency_ok = False
            print(
                "[IBKR PORTFOLIO] IBKR_NETLIQ_CURRENCY_MISMATCH "
                f"expected={base} got={row_currency!r} — preserving last-known "
                "ibkr_equity, NOT writing wrong-currency value",
                file=sys.stderr,
                flush=True,
            )

        # BOX-034B Step 2 (generalized): the payload is written wholesale, so any
        # prior key this collector does NOT re-list is dropped. Previously a
        # hardcoded 3-key allowlist (kraken currency tags + ibkr USD display) was
        # carried forward — but that still silently erased the publisher-authored
        # authoritative USD block (total_equity_usd_authoritative / usd_ok /
        # usdcad_rate_used) and any future field on every 2-min collector cycle,
        # starving the tier-manager between publisher runs. We now PRESERVE ALL
        # unknown keys generically: start from the existing snapshot (read-through;
        # fail-soft {} on missing/corrupt above) and overlay ONLY the keys this
        # collector authors. No allowlist — a cold/empty snapshot (data == {})
        # yields exactly the owned keys, so nothing is fabricated as null.
        new_payload: Dict[str, Any] = dict(data)
        new_payload.update(
            {
                "ts_utc": _utc_now_iso(),
                "ttl_seconds": int(PORTFOLIO_SNAPSHOT_TTL_SECONDS),
                "ibkr_equity": float(equity_to_write),
                "ibkr_equity_currency": base,
                "ibkr_equity_currency_ok": bool(currency_ok),
                "coinbase_equity": float(coinbase_equity),
                "kraken_equity": float(kraken_equity),
            }
        )

        _atomic_write_json(path, new_payload)
        return path

    # ------------------------------------------------------------------ #
    # positions snapshot                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _default_positions_path() -> Path:
        root = Path(__file__).resolve().parents[2]
        return root / "runtime" / "positions_snapshot.json"

    def collect_positions(self, positions_path: Optional[Path] = None) -> Path:
        """
        Query IBKR for all positions and write runtime/positions_snapshot.json.

        This is the broker-authority source for reconciliation.
        """
        from ib_async import IB  # type: ignore[import]

        path = positions_path or self._default_positions_path()
        ib = IB()
        try:
            ib.connect(
                self._cfg.host, self._cfg.port,
                clientId=self._cfg.client_id, readonly=True, timeout=10.0,
            )
            raw_positions = ib.positions()

            pos_list: List[Dict[str, Any]] = []
            for p in raw_positions:
                c = p.contract
                pos_list.append({
                    "conId": int(c.conId),
                    "symbol": str(c.symbol),
                    "position": float(p.position),
                    "avgCost": float(p.avgCost),
                    "secType": str(c.secType),
                    "currency": str(c.currency),
                })

            payload: Dict[str, Any] = {
                "positions": pos_list,
                "positions_count": len(pos_list),
                "ts_utc": _utc_now_iso(),
                "ttl_seconds": int(POSITIONS_SNAPSHOT_TTL_SECONDS),
                "source": "ibkr_portfolio_collector_v2",
            }

            _atomic_write_json(path, payload)
            return path
        finally:
            try:
                ib.disconnect()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    # NEW-GAP-043: install the wall-clock guard BEFORE arg parsing / network.
    # If anything below (arg parse, IB connect, accountSummary, positions)
    # hangs, the SIGALRM handler exits 124 so the systemd unit transitions
    # cleanly from activating → failed instead of sitting in activating
    # indefinitely. Tests can opt out via CHAD_COLLECTOR_WALL_CLOCK_SECONDS=
    # but the production default (60s) covers all observed healthy runs
    # (typical wall-clock: ~2-5s for connect+accountSummary+positions+write).
    install_wall_clock_guard()

    parser = argparse.ArgumentParser(
        description=(
            "IBKR Portfolio Collector v2\n"
            "Reads NetLiquidation via ib_insync and updates runtime/portfolio_snapshot.json "
            "with ibkr_equity, preserving coinbase_equity and kraken_equity."
        )
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "print",
        help="Print NetLiquidation to stdout (no file writes).",
    )

    collect_parser = subparsers.add_parser(
        "collect",
        help="Update portfolio_snapshot.json with ibkr_equity.",
    )
    collect_parser.add_argument(
        "--snapshot-path",
        type=str,
        default="",
        help="Optional override for portfolio_snapshot.json path.",
    )

    positions_parser = subparsers.add_parser(
        "positions",
        help=(
            "Refresh runtime/positions_snapshot.json only (read-only). "
            "Does not touch portfolio_snapshot.json. Intended for the "
            "chad-positions-snapshot timer."
        ),
    )
    positions_parser.add_argument(
        "--snapshot-path",
        type=str,
        default="",
        help="Optional override for positions_snapshot.json path.",
    )

    args = parser.parse_args(argv)

    if args.command not in ("print", "collect", "positions"):
        parser.print_help()
        return 2

    cfg = IBKRConnectionConfig.from_env()
    collector = IBKRPortfolioCollector(cfg)

    if args.command == "print":
        try:
            val = collector.get_net_liquidation()
        except Exception as exc:
            print(f"[IBKR PORTFOLIO] Error: {exc}")
            return 1
        print(val)
        return 0

    if args.command == "positions":
        try:
            override = (
                Path(args.snapshot_path).expanduser().resolve()
                if args.snapshot_path
                else None
            )
            out_path = collector.collect_positions(positions_path=override)
        except Exception as exc:
            print(f"[IBKR PORTFOLIO] Positions snapshot error: {exc}")
            return 1
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            count = int(payload.get("positions_count", 0))
        except Exception:
            count = -1
        print(
            f"[IBKR PORTFOLIO] Updated positions snapshot: {out_path} "
            f"positions_count={count}"
        )
        return 0

    # collect
    rc = 0
    try:
        snapshot_path = Path(args.snapshot_path).expanduser().resolve() if args.snapshot_path else None
        out_path = collector.update_snapshot(snapshot_path=snapshot_path)
        print(f"[IBKR PORTFOLIO] Updated equity snapshot: {out_path}")
    except Exception as exc:
        print(f"[IBKR PORTFOLIO] Equity snapshot error: {exc}")
        rc = 1

    try:
        pos_path = collector.collect_positions()
        print(f"[IBKR PORTFOLIO] Updated positions snapshot: {pos_path}")
    except Exception as exc:
        print(f"[IBKR PORTFOLIO] Positions snapshot error: {exc}")
        rc = 1

    return rc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
