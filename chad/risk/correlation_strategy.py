from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from chad.risk import correlation_layer

ROOT = Path("/home/ubuntu/chad_finale")
C_PATH = ROOT / "runtime" / "dynamic_caps_correlation.json"
HEALTH_PATH = ROOT / "runtime" / "correlation_overlay_health.json"


def _iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _write_health(
    *,
    health_path: Path,
    ok: bool,
    status: str,
    reason: Optional[str],
    source_file: Path,
    weights_count: int,
) -> None:
    try:
        source_mtime_utc: Optional[str] = None
        if source_file.exists():
            source_mtime_utc = (
                datetime.fromtimestamp(source_file.stat().st_mtime, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
        payload = {
            "ok": bool(ok),
            "status": status,
            "reason": reason,
            "source_file": str(source_file),
            "source_mtime_utc": source_mtime_utc,
            "weights_count": int(weights_count),
            "ts_utc": _iso_z(),
        }
        _atomic_write_json(health_path, payload)
    except Exception:
        # Health writing must never break the strategy.
        pass


class CorrelationOverlayStrategy:
    name = "correlation_overlay"

    def __init__(
        self,
        c_path: Path = C_PATH,
        health_path: Path = HEALTH_PATH,
        refresher=correlation_layer.refresh,
    ) -> None:
        self._c_path = c_path
        self._health_path = health_path
        self._refresher = refresher

    def apply(
        self,
        *,
        repo_root: Path,
        base_weights: Mapping[str, float],
        log,
    ) -> Dict[str, float]:
        producer_reason: Optional[str] = None
        try:
            ok, producer_reason, _ = self._refresher(out_path=self._c_path)
            if not ok:
                # Producer could not refresh (e.g. quarantine_weights missing).
                # Fall through to read whatever is on disk; if also missing,
                # the consumer block below raises with a clear status.
                pass
        except Exception as exc:  # noqa: BLE001
            producer_reason = f"producer_exception: {exc}"

        try:
            if not self._c_path.exists():
                reason = producer_reason or "correlation file missing"
                _write_health(
                    health_path=self._health_path,
                    ok=False,
                    status="missing",
                    reason=reason,
                    source_file=self._c_path,
                    weights_count=0,
                )
                raise RuntimeError(reason)

            data = json.loads(self._c_path.read_text(encoding="utf-8"))
            corr = data.get("correlation_governed_weights")

            if not isinstance(corr, dict) or not corr:
                reason = "invalid correlation_governed_weights"
                _write_health(
                    health_path=self._health_path,
                    ok=False,
                    status="invalid",
                    reason=reason,
                    source_file=self._c_path,
                    weights_count=0,
                )
                raise RuntimeError(reason)

            adjusted = {}
            for k, bw in base_weights.items():
                adjusted[k] = float(corr.get(k, bw))

            status = "ok" if producer_reason is None else "stale_producer_failed"
            _write_health(
                health_path=self._health_path,
                ok=True,
                status=status,
                reason=producer_reason,
                source_file=self._c_path,
                weights_count=len(corr),
            )

            log.info("correlation_overlay_applied", extra={"weights": adjusted})
            return adjusted

        except Exception as e:
            log.warning("correlation_overlay_failed_fallback", extra={"error": str(e)})
            return dict(base_weights)
