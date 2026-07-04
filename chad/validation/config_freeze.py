"""chad/validation/config_freeze.py — Phase 5 config freeze (SSOT §3.2, F2).

Freezes the pass/fail thresholds and the cost configuration *before* a sealed OOS
run, and makes goalpost-moving mathematically expensive (SSOT
``docs/CHAD_EDGE_VALIDATION_HARNESS_DESIGN_v1.1.md`` §3.2 / Part 0 point 5). Two
mechanisms:

  1. **Hash-freeze.** :func:`config_hash` fingerprints ``{thresholds, cost_config}``
     with SHA-256 over canonical JSON. A :class:`FrozenConfig` records that hash and
     the frozen contents; :meth:`FrozenConfig.verify` confirms a candidate
     thresholds/cost pair is byte-identical to what was frozen. The frozen hash is
     printed in every report and bound into the OOS seal (:mod:`chad.validation.oos_lockbox`).

  2. **Post-FAIL change = a new trial.** A :class:`FreezeLedger` persists the current
     frozen config, a running ``trial_count``, and the last verdict. Amending the
     config (:meth:`FreezeLedger.amend`) after a ``FAIL`` **invalidates the prior seal**
     (the old hash is recorded in ``superseded_hashes`` and no longer verifies) **and
     increments ``trial_count``**. That count is fed to the Phase-3 punitive deflation
     N via :func:`deflation_trials` — so nudging the settings after a bad result and
     re-running is penalised as another attempt, not silently allowed. Amending BEFORE
     any result, or after a non-FAIL verdict, is legitimate pre-registration tuning and
     does NOT bump the count; re-amending to the *same* config is a no-op.

Why the deflation coupling matters (SSOT §3.3, S1): only surviving heads/configs are
ever visible, so every abandoned config *was also a trial*. The deflation benchmark
SR*_0 grows with N, so a larger trial count lowers the Deflated Sharpe Ratio — making
the proof strictly harder each time the goalposts move. This module produces the
*count*; :mod:`chad.validation.significance` consumes it.

Isolation (SSOT §1.2 / §2): pure, offline, deterministic, standard-library only —
:mod:`hashlib`, :mod:`json`, :mod:`pathlib`, :mod:`dataclasses`, :mod:`typing`. No
numpy, no broker, no ``runtime/`` reader, no live-loop dependency. It reads/writes ONLY
the single ledger JSON file it is handed; it never writes ``runtime/`` and never
touches ``ready_for_live``.

Sentinel / raise convention: invalid *configuration* raises ``ValueError`` (a non-str
timestamp, a negative ``base_trials``, amending before any freeze). The freeze
contents themselves must be JSON-serialisable mappings of plain scalars — a
non-serialisable value raises at hash time (a caller bug, surfaced fast).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Optional

__all__ = [
    "LEDGER_FILENAME",
    "FrozenConfig",
    "FreezeState",
    "FreezeLedger",
    "config_hash",
    "deflation_trials",
]

LEDGER_FILENAME: Final[str] = "config_freeze_ledger.json"


def _canonical(payload: Mapping[str, Any]) -> str:
    """Canonical JSON of a freeze payload (sorted keys, no NaN, ASCII) for hashing."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def config_hash(thresholds: Mapping[str, Any], cost_config: Mapping[str, Any]) -> str:
    """SHA-256 hex digest of ``{thresholds, cost_config}`` (deterministic, order-free).

    Sorted-key canonical JSON, so key insertion order does not matter but any change to
    a threshold or a cost parameter changes the digest. Both mappings must be
    JSON-serialisable (plain scalars / nested plain containers); a non-serialisable
    value raises ``TypeError``/``ValueError`` at ``json.dumps`` time.
    """
    if not isinstance(thresholds, Mapping):
        raise ValueError(f"thresholds must be a mapping, got {type(thresholds).__name__}")
    if not isinstance(cost_config, Mapping):
        raise ValueError(f"cost_config must be a mapping, got {type(cost_config).__name__}")
    canonical = _canonical({"thresholds": dict(thresholds), "cost_config": dict(cost_config)})
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Frozen config + persisted ledger state (flat, serialisable, report-embedded).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FrozenConfig:
    """The thresholds + cost config frozen before a sealed run, with their hash.

    ``thresholds`` and ``cost_config`` are the exact frozen contents (plain dicts);
    ``config_hash`` is their :func:`config_hash`; ``frozen_at`` is the caller-supplied
    timestamp (deterministic — this module never reads the wall clock).
    """

    thresholds: dict[str, Any]
    cost_config: dict[str, Any]
    config_hash: str
    frozen_at: str

    def verify(self, thresholds: Mapping[str, Any], cost_config: Mapping[str, Any]) -> bool:
        """True iff ``(thresholds, cost_config)`` hash-matches this frozen config.

        The mechanical "has the config changed since freezing?" check — a single bit of
        difference in any threshold or cost parameter makes it ``False`` (invalidating
        the seal in :meth:`FreezeLedger.amend`).
        """
        return config_hash(thresholds, cost_config) == self.config_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "thresholds": dict(self.thresholds),
            "cost_config": dict(self.cost_config),
            "config_hash": self.config_hash,
            "frozen_at": self.frozen_at,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "FrozenConfig":
        return FrozenConfig(
            thresholds=dict(payload["thresholds"]),
            cost_config=dict(payload["cost_config"]),
            config_hash=str(payload["config_hash"]),
            frozen_at=str(payload["frozen_at"]),
        )


@dataclass(frozen=True)
class FreezeState:
    """The persisted freeze ledger: the current frozen config + trial/verdict history.

    ``trial_count`` is the punitive count fed to deflation (base trials + every
    post-FAIL config amendment). ``last_verdict`` is the most recent recorded verdict
    (``None`` until one is recorded). ``superseded_hashes`` is the append-only history
    of config hashes invalidated by post-FAIL amendments — every one of them now fails
    :meth:`FrozenConfig.verify` against the current frozen config.
    """

    frozen: FrozenConfig
    trial_count: int
    last_verdict: Optional[str]
    superseded_hashes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frozen": self.frozen.to_dict(),
            "trial_count": self.trial_count,
            "last_verdict": self.last_verdict,
            "superseded_hashes": list(self.superseded_hashes),
        }

    @staticmethod
    def from_dict(payload: Mapping[str, Any]) -> "FreezeState":
        return FreezeState(
            frozen=FrozenConfig.from_dict(payload["frozen"]),
            trial_count=int(payload["trial_count"]),
            last_verdict=(None if payload.get("last_verdict") is None else str(payload["last_verdict"])),
            superseded_hashes=tuple(str(h) for h in payload.get("superseded_hashes", ())),
        )


def deflation_trials(base_punitive_n: int, state: FreezeState) -> int:
    """Effective deflation trial count = punitive base N + post-FAIL config amendments.

    The Phase-3 :func:`chad.validation.significance.punitive_trial_count` produces the
    survivor-derived base ``N``; this adds the ledger's ``trial_count`` (which includes
    every goalpost-moving amendment, SSOT §3.2) so the Deflated Sharpe benchmark grows
    with each re-run. ``base_punitive_n`` must be a positive int.
    """
    if isinstance(base_punitive_n, bool) or not isinstance(base_punitive_n, int) or base_punitive_n < 1:
        raise ValueError(f"base_punitive_n must be an int >= 1, got {base_punitive_n!r}")
    return base_punitive_n + int(state.trial_count)


# --------------------------------------------------------------------------- #
# The persisted ledger.
# --------------------------------------------------------------------------- #
class FreezeLedger:
    """Persist the frozen config, its trial count, and post-FAIL amendment penalties.

    Bound to a single JSON ledger file. The lifecycle is: :meth:`freeze` once before
    the sealed run → :meth:`record_verdict` after each run → :meth:`amend` if the
    config must change (penalised only after a FAIL). All state is on disk so the
    penalty survives across separate CLI invocations (a re-run in a fresh process still
    sees the accumulated trial count).
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def exists(self) -> bool:
        return self._path.is_file()

    def load(self) -> Optional[FreezeState]:
        """Load the persisted state, or ``None`` if nothing has been frozen yet."""
        if not self._path.is_file():
            return None
        with self._path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise ValueError(f"malformed freeze ledger {self._path}")
        return FreezeState.from_dict(payload)

    def _write(self, state: FreezeState) -> FreezeState:
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(state.to_dict(), fh, sort_keys=True, indent=2)
        tmp.replace(self._path)
        return state

    def freeze(
        self,
        thresholds: Mapping[str, Any],
        cost_config: Mapping[str, Any],
        *,
        timestamp: str,
        base_trials: int = 0,
    ) -> FreezeState:
        """Freeze thresholds + cost config as the initial sealed-run configuration.

        Writes a fresh ledger with ``trial_count = base_trials`` (typically 0; a caller
        may pre-load prior known trials), no recorded verdict, and no superseded hashes.
        Overwrites any existing ledger — freezing is the deliberate "start a clean
        pre-registration" act; post-freeze changes go through :meth:`amend`, which is
        where the penalty lives. ``base_trials`` must be a non-negative int.
        """
        if not isinstance(timestamp, str) or not timestamp:
            raise ValueError("timestamp must be a non-empty str")
        if isinstance(base_trials, bool) or not isinstance(base_trials, int) or base_trials < 0:
            raise ValueError(f"base_trials must be an int >= 0, got {base_trials!r}")
        frozen = FrozenConfig(
            thresholds=dict(thresholds),
            cost_config=dict(cost_config),
            config_hash=config_hash(thresholds, cost_config),
            frozen_at=timestamp,
        )
        return self._write(
            FreezeState(
                frozen=frozen,
                trial_count=base_trials,
                last_verdict=None,
                superseded_hashes=(),
            )
        )

    def record_verdict(self, verdict: str) -> FreezeState:
        """Record the most recent run's verdict (raises if nothing is frozen yet).

        The verdict string (e.g. ``"FAIL"`` / ``"INSUFFICIENT_DATA"`` / ``"PASS"``)
        determines whether the NEXT :meth:`amend` is penalised: only a ``"FAIL"`` makes
        a subsequent config change count as a new trial (SSOT §3.2).
        """
        if not isinstance(verdict, str) or not verdict:
            raise ValueError("verdict must be a non-empty str")
        state = self.load()
        if state is None:
            raise ValueError("cannot record a verdict before freeze() has been called")
        return self._write(
            FreezeState(
                frozen=state.frozen,
                trial_count=state.trial_count,
                last_verdict=verdict,
                superseded_hashes=state.superseded_hashes,
            )
        )

    def amend(
        self,
        thresholds: Mapping[str, Any],
        cost_config: Mapping[str, Any],
        *,
        timestamp: str,
    ) -> FreezeState:
        """Amend the frozen config; penalise the change iff the last verdict was FAIL.

        Behaviour (SSOT §3.2):
          * **Unchanged config** (same hash) → no-op: returns the current state
            unchanged (re-amending to the identical config is not a new trial).
          * **Changed config after a FAIL** → the prior seal is invalidated (its hash is
            appended to ``superseded_hashes`` and no longer :meth:`~FrozenConfig.verify`\\
            s), ``trial_count`` is incremented by 1, and the new config is frozen
            (``last_verdict`` reset to ``None`` for the fresh attempt).
          * **Changed config before any result, or after a non-FAIL verdict** →
            legitimate pre-registration tuning: the new config is frozen with NO trial
            bump (``last_verdict`` reset to ``None``).

        Raises ``ValueError`` if nothing has been frozen yet (call :meth:`freeze` first).
        """
        if not isinstance(timestamp, str) or not timestamp:
            raise ValueError("timestamp must be a non-empty str")
        state = self.load()
        if state is None:
            raise ValueError("cannot amend before freeze() has been called")

        new_hash = config_hash(thresholds, cost_config)
        if new_hash == state.frozen.config_hash:
            return state  # identical config → not a change, not a trial

        penalise = state.last_verdict == "FAIL"
        new_frozen = FrozenConfig(
            thresholds=dict(thresholds),
            cost_config=dict(cost_config),
            config_hash=new_hash,
            frozen_at=timestamp,
        )
        if penalise:
            new_state = FreezeState(
                frozen=new_frozen,
                trial_count=state.trial_count + 1,
                last_verdict=None,
                superseded_hashes=state.superseded_hashes + (state.frozen.config_hash,),
            )
        else:
            new_state = FreezeState(
                frozen=new_frozen,
                trial_count=state.trial_count,
                last_verdict=None,
                superseded_hashes=state.superseded_hashes,
            )
        return self._write(new_state)
