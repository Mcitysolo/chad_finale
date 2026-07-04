"""chad/validation/oos_lockbox.py — Phase 5 OOS lockbox (SSOT §3.1, F1).

The core enforcement mechanism that turns the harness's out-of-sample discipline
from an honour-system *promise* into a *lock* (SSOT
``docs/CHAD_EDGE_VALIDATION_HARNESS_DESIGN_v1.1.md`` §3.1 / Part 0). Four
guarantees, all mechanical:

  1. **Hash-seal at split time.** The OOS partition (the per-head OOS return series)
     is fingerprinted with SHA-256 and the hash is recorded in a seal file, together
     with the frozen-config hash and the code commit. Re-sealing identical content is
     idempotent; sealing *different* content over an existing seal raises rather than
     silently replacing the sealed box.
  2. **Refuse to score OOS without ``final_run``.** :meth:`OOSLockbox.open_oos`
     returns the real sealed series ONLY when called with ``final_run=True``. Any
     attempt without the flag raises :class:`OOSAccessError` *before* returning any
     data — the reviewer's "score OOS without the flag" attempt fails loudly.
  3. **Immutable, append-only access log + high-water anchor.** Every successful open
     appends one record (timestamp + config hash + code commit + OOS hash) to an NDJSON
     run-log opened in append mode only — never truncated. Each record carries a
     hash-chain link (``chain = SHA-256(prev_chain | record)``), and a separate
     high-water anchor file records the running ``(count, terminal chain)``. Together
     they make :meth:`OOSLockbox.verify_log_integrity` detect an edit or reorder of any
     line (broken chain), a deletion of an interior line (broken ``seq`` sequence), AND
     a truncation of the tail or deletion of the whole log (anchor mismatch) — the last
     of which a bare forward chain alone cannot catch. Residual threat model: an
     adversary with write access to BOTH the log and the anchor could rewrite them
     consistently; that is outside this offline harness's scope, which targets honest
     re-runs and accidental corruption, not a filesystem attacker.
  4. **Contamination flag (fail-toward-flagging).** :meth:`OOSLockbox.access_count`
     returns the greater of the logged-record count and the anchored high-water count,
     so a tail-truncation cannot silently *reduce* it below a prior open. A count
     ``> 1`` means the sealed box was opened more than once, which the Phase-5 verdict
     auto-flags ``CONTAMINATED`` (see :mod:`chad.validation.verdict`).

**Dev / debug runs never touch the real OOS.** They call :meth:`OOSLockbox.decoy_oos`
(equivalently the module fn :func:`synthetic_decoy_returns`) for a deterministic,
seeded, synthetic decoy series and drive the whole OOS code path with it. The real
box is sealed (so its hash is on record) but never opened, so ``access_count`` stays
``0`` — exactly the state a non-``--final-run`` CLI run must end in.

Isolation (SSOT §1.2 / §2): pure, offline, deterministic, standard-library only —
:mod:`hashlib`, :mod:`json`, :mod:`random`, :mod:`pathlib`, :mod:`dataclasses`,
:mod:`typing`. No numpy, no broker, no ``runtime/`` reader, no live-loop dependency.
It reads/writes ONLY the two files inside the lockbox ``root`` directory it is handed
(a seal file and an append-only log); it never writes ``runtime/`` state, never
touches ``ready_for_live``, and never imports a live/strategy/broker module.

Sentinel / raise convention (mirrors the rest of the harness): invalid *configuration*
or a *discipline violation* raises (a non-``final_run`` open → :class:`OOSAccessError`;
a not-yet-sealed open, a content/hash mismatch, or a conflicting re-seal →
:class:`OOSSealError`; a malformed argument → ``ValueError``). Degenerate-but-valid
DATA (an empty OOS series) is sealed and opened normally — an empty series is a real,
representable fact the verdict layer will read as "below minimums", not an error here.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Optional, Sequence, Union

__all__ = [
    "OOSAccessError",
    "OOSSealError",
    "OOSSeal",
    "OOSAccessRecord",
    "OOSLockbox",
    "DEFAULT_DECOY_SEED",
    "SEAL_FILENAME",
    "ACCESS_LOG_FILENAME",
    "ACCESS_HEAD_FILENAME",
    "returns_hash",
    "synthetic_decoy_returns",
]

Number = Union[int, float]

# Fixed filenames inside a lockbox ``root`` directory (documented, stable).
SEAL_FILENAME: Final[str] = "oos_seal.json"
ACCESS_LOG_FILENAME: Final[str] = "oos_access_log.ndjson"
# High-water anchor: the running (count, terminal chain) of the access log, so a
# tail-truncation or whole-log deletion is detectable and cannot reduce the count.
ACCESS_HEAD_FILENAME: Final[str] = "oos_access_head.json"

# Default seed for the synthetic decoy OOS series (dev/debug runs). A plain, small
# constant so the decoy is trivially reproducible and documented.
DEFAULT_DECOY_SEED: Final[int] = 20260704

# Standard deviation of the seeded synthetic decoy returns (a plausible daily-ish
# per-trade return scale). Only the shape matters — the decoy exists to exercise the
# OOS *code path*, never to stand in as evidence.
_DECOY_SIGMA: Final[float] = 0.01


class OOSAccessError(Exception):
    """Raised on an attempt to open the sealed OOS without ``final_run=True``.

    The mechanical refusal at the heart of §3.1: the engine "literally refuses to
    open" the test data except on one final, logged run. Raised *before* any OOS
    value is returned, so a leak attempt cannot obtain data and then be logged.
    """


class OOSSealError(Exception):
    """Raised on a seal-integrity violation (unsealed open, hash mismatch, re-seal conflict).

    Distinct from :class:`OOSAccessError` (which is the ``final_run`` gate): this is
    the "the sealed box does not match what you handed me / there is no sealed box"
    class of failure — a tampered or swapped OOS partition, or an attempt to seal
    different content over an existing seal.
    """


# --------------------------------------------------------------------------- #
# Deterministic hashing / decoy helpers (pure module functions).
# --------------------------------------------------------------------------- #
def _coerce_returns(values: Sequence[Number]) -> list[float]:
    """Coerce a return series to ``list[float]``; reject ``bool``/non-numbers/non-finite."""
    out: list[float] = []
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"OOS return[{i}] must be a real number, got {type(v).__name__}: {v!r}")
        f = float(v)
        # NaN/inf would make the seal hash non-reproducible across producers; a real
        # OOS return series is always finite, so a non-finite value is malformed input.
        if f != f or f in (float("inf"), float("-inf")):
            raise ValueError(f"OOS return[{i}] must be finite, got {f}")
        out.append(f)
    return out


def returns_hash(values: Sequence[Number]) -> str:
    """SHA-256 hex digest of an OOS return series (deterministic, order-sensitive).

    Canonicalises to ``{"n": len, "values": [...]}`` via :func:`json.dumps` with sorted
    keys and CPython's shortest-round-trip float ``repr`` (deterministic within the
    interpreter), so the same series always yields the same digest and any change to a
    value, the order, or the count changes it.
    """
    coerced = _coerce_returns(values)
    canonical = json.dumps(
        {"n": len(coerced), "values": coerced},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def synthetic_decoy_returns(n: int, *, seed: int = DEFAULT_DECOY_SEED) -> list[float]:
    """A deterministic synthetic decoy OOS return series of length ``n`` (SSOT §3.1).

    Seeded :class:`random.Random` Gaussian draws — reproducible for a given ``(n, seed)``
    and completely independent of the real (sealed) OOS. Dev/debug runs drive the OOS
    code path with THIS instead of ever opening the real box. ``n`` must be an int
    ``>= 0`` and ``seed`` a plain int (invalid config raises ``ValueError``).
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"n must be an int >= 0, got {n!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {seed!r}")
    rng = random.Random(f"oos-decoy:{seed}")
    return [rng.gauss(0.0, _DECOY_SIGMA) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Records — flat, serialisable, embedded verbatim by the report.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OOSSeal:
    """The recorded seal of an OOS partition (written once at split time).

    ``oos_hash`` fingerprints the sealed return series; ``config_hash`` binds the seal
    to the frozen config that produced it (SSOT §3.2); ``code_commit`` records the code
    that sealed it; ``sealed_at`` is the caller-supplied timestamp (deterministic —
    the lockbox never reads the wall clock). ``n_oos`` is the sealed series length and
    ``decoy_seed`` the seed a dev run's decoy uses.
    """

    oos_hash: str
    n_oos: int
    config_hash: str
    code_commit: str
    sealed_at: str
    decoy_seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "oos_hash": self.oos_hash,
            "n_oos": self.n_oos,
            "config_hash": self.config_hash,
            "code_commit": self.code_commit,
            "sealed_at": self.sealed_at,
            "decoy_seed": self.decoy_seed,
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "OOSSeal":
        return OOSSeal(
            oos_hash=str(payload["oos_hash"]),
            n_oos=int(payload["n_oos"]),
            config_hash=str(payload["config_hash"]),
            code_commit=str(payload["code_commit"]),
            sealed_at=str(payload["sealed_at"]),
            decoy_seed=int(payload["decoy_seed"]),
        )


@dataclass(frozen=True)
class OOSAccessRecord:
    """One immutable append-only log entry: a single logged OOS open (SSOT §3.1).

    ``seq`` is the 0-based position in the log; ``chain`` is the hash-chain link
    ``SHA-256(prev_chain | canonical_core)`` (``prev_chain`` = the seal's ``oos_hash``
    for ``seq == 0``, else the previous record's ``chain``) that makes the log
    tamper-evident. ``final_run`` is always ``True`` for a logged open (a non-final
    open is refused and never logged).
    """

    seq: int
    timestamp: str
    config_hash: str
    code_commit: str
    final_run: bool
    oos_hash: str
    chain: str

    def core_json(self) -> str:
        """The canonical, chain-hashed core (excludes ``seq`` and ``chain`` itself)."""
        return json.dumps(
            {
                "timestamp": self.timestamp,
                "config_hash": self.config_hash,
                "code_commit": self.code_commit,
                "final_run": self.final_run,
                "oos_hash": self.oos_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "config_hash": self.config_hash,
            "code_commit": self.code_commit,
            "final_run": self.final_run,
            "oos_hash": self.oos_hash,
            "chain": self.chain,
        }


def _chain_link(prev_chain: str, core_json: str) -> str:
    """One hash-chain link: ``SHA-256(prev_chain | core_json)`` (hex)."""
    return hashlib.sha256(f"{prev_chain}|{core_json}".encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# The lockbox.
# --------------------------------------------------------------------------- #
class OOSLockbox:
    """Hash-seal + ``final_run`` gate + immutable access log for one OOS partition.

    A lockbox is bound to a ``root`` directory holding exactly two files: the seal
    (:data:`SEAL_FILENAME`) and the append-only access log (:data:`ACCESS_LOG_FILENAME`).
    One lockbox guards one OOS partition (one head's OOS return series, or a portfolio
    track's). Construct one per head with a per-head ``root``.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._seal_path = self._root / SEAL_FILENAME
        self._log_path = self._root / ACCESS_LOG_FILENAME
        self._head_path = self._root / ACCESS_HEAD_FILENAME

    # --- properties / paths ------------------------------------------------- #
    @property
    def root(self) -> Path:
        return self._root

    @property
    def seal_path(self) -> Path:
        return self._seal_path

    @property
    def access_log_path(self) -> Path:
        return self._log_path

    @property
    def access_head_path(self) -> Path:
        return self._head_path

    def is_sealed(self) -> bool:
        """True once :meth:`seal` has written the seal file."""
        return self._seal_path.is_file()

    def load_seal(self) -> OOSSeal:
        """Load the recorded seal (raises :class:`OOSSealError` if not yet sealed)."""
        if not self.is_sealed():
            raise OOSSealError(f"no OOS seal at {self._seal_path}; call seal() first")
        with self._seal_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise OOSSealError(f"malformed seal file {self._seal_path}")
        return OOSSeal.from_dict(payload)

    # --- seal --------------------------------------------------------------- #
    def seal(
        self,
        oos_returns: Sequence[Number],
        *,
        config_hash: str,
        code_commit: str,
        timestamp: str,
        decoy_seed: int = DEFAULT_DECOY_SEED,
    ) -> OOSSeal:
        """Hash-seal the OOS return series at split time (idempotent on identical content).

        Writes the seal file with the series' :func:`returns_hash`, the frozen
        ``config_hash``, the ``code_commit``, and ``timestamp``. If a seal already
        exists: identical ``oos_hash`` returns the existing seal unchanged (a re-run's
        harmless re-seal); a DIFFERENT ``oos_hash`` raises :class:`OOSSealError` rather
        than silently replacing the sealed box (a swapped OOS partition is a discipline
        violation, not a quiet overwrite). Sealing is NOT an access — it never touches
        the run-log and never counts toward contamination.
        """
        if not isinstance(config_hash, str) or not config_hash:
            raise ValueError("config_hash must be a non-empty str")
        if not isinstance(code_commit, str) or not code_commit:
            raise ValueError("code_commit must be a non-empty str")
        if not isinstance(timestamp, str) or not timestamp:
            raise ValueError("timestamp must be a non-empty str")
        if isinstance(decoy_seed, bool) or not isinstance(decoy_seed, int):
            raise ValueError(f"decoy_seed must be an int, got {decoy_seed!r}")

        coerced = _coerce_returns(oos_returns)
        new_hash = returns_hash(coerced)

        if self.is_sealed():
            existing = self.load_seal()
            if existing.oos_hash == new_hash:
                return existing
            raise OOSSealError(
                f"refusing to re-seal {self._seal_path}: existing OOS hash "
                f"{existing.oos_hash[:12]}… != new {new_hash[:12]}… (the sealed OOS "
                "partition changed — this is a swapped test set, not a harmless re-seal)"
            )

        seal = OOSSeal(
            oos_hash=new_hash,
            n_oos=len(coerced),
            config_hash=config_hash,
            code_commit=code_commit,
            sealed_at=timestamp,
            decoy_seed=decoy_seed,
        )
        # Atomic-ish write: dump to a tmp sibling then replace, so a crash mid-write
        # cannot leave a half-written seal that would read as a hash mismatch later.
        tmp = self._seal_path.with_suffix(self._seal_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(seal.to_dict(), fh, sort_keys=True, indent=2)
        tmp.replace(self._seal_path)
        return seal

    # --- decoy (dev/debug) -------------------------------------------------- #
    def decoy_oos(self, n: int, *, seed: Optional[int] = None) -> list[float]:
        """A deterministic synthetic decoy OOS series (dev/debug; the real box untouched).

        ``seed`` defaults to the sealed seal's ``decoy_seed`` when the box is sealed,
        else :data:`DEFAULT_DECOY_SEED`. Never reads the real OOS and never appends to
        the run-log — a run that uses only the decoy ends with ``access_count() == 0``.
        """
        if seed is None:
            seed = self.load_seal().decoy_seed if self.is_sealed() else DEFAULT_DECOY_SEED
        return synthetic_decoy_returns(n, seed=seed)

    # --- open (the gate) ---------------------------------------------------- #
    def open_oos(
        self,
        oos_returns: Sequence[Number],
        *,
        final_run: bool,
        config_hash: str,
        code_commit: str,
        timestamp: str,
    ) -> list[float]:
        """Open the sealed OOS series — ONLY on ``final_run=True`` — and log the access.

        The §3.1 gate. With ``final_run=False`` this raises :class:`OOSAccessError`
        immediately and returns nothing (the reviewer's "score OOS without the flag"
        path). With ``final_run=True`` it verifies ``returns_hash(oos_returns)`` matches
        the seal (mismatch → :class:`OOSSealError`), appends one hash-chained record to
        the immutable run-log, and returns the series. Each call is one logged access;
        a second call makes :meth:`access_count` ``> 1`` → ``CONTAMINATED`` at verdict.
        """
        if not isinstance(final_run, bool):
            raise ValueError(f"final_run must be a bool, got {final_run!r}")
        if not self.is_sealed():
            raise OOSSealError("cannot open OOS before it is sealed; call seal() first")
        if not final_run:
            raise OOSAccessError(
                "refusing to open the sealed OOS partition without final_run=True "
                "(SSOT §3.1: the test data is opened only on one final, logged run). "
                "Use decoy_oos() for development."
            )
        if not isinstance(config_hash, str) or not config_hash:
            raise ValueError("config_hash must be a non-empty str")
        if not isinstance(code_commit, str) or not code_commit:
            raise ValueError("code_commit must be a non-empty str")
        if not isinstance(timestamp, str) or not timestamp:
            raise ValueError("timestamp must be a non-empty str")

        coerced = _coerce_returns(oos_returns)
        seal = self.load_seal()
        got = returns_hash(coerced)
        if got != seal.oos_hash:
            raise OOSSealError(
                f"OOS content does not match the seal (got {got[:12]}…, sealed "
                f"{seal.oos_hash[:12]}…): the OOS partition changed since sealing"
            )

        self._append_access(
            config_hash=config_hash,
            code_commit=code_commit,
            timestamp=timestamp,
            oos_hash=seal.oos_hash,
        )
        return coerced

    def _append_access(
        self, *, config_hash: str, code_commit: str, timestamp: str, oos_hash: str
    ) -> OOSAccessRecord:
        """Append one hash-chained record to the append-only log (never truncates)."""
        existing = self.access_records()
        seq = len(existing)
        prev_chain = existing[-1].chain if existing else oos_hash
        record = OOSAccessRecord(
            seq=seq,
            timestamp=timestamp,
            config_hash=config_hash,
            code_commit=code_commit,
            final_run=True,
            oos_hash=oos_hash,
            chain="",  # placeholder; recomputed below with the true core
        )
        chain = _chain_link(prev_chain, record.core_json())
        record = OOSAccessRecord(
            seq=seq,
            timestamp=timestamp,
            config_hash=config_hash,
            code_commit=code_commit,
            final_run=True,
            oos_hash=oos_hash,
            chain=chain,
        )
        # Append-only: mode "a" never truncates an existing log line.
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")
        # Update the high-water anchor AFTER the append so the recorded count/terminal
        # chain commit to the full log — a later tail-truncation now mismatches it.
        self._write_head(count=seq + 1, last_chain=chain)
        return record

    def _write_head(self, *, count: int, last_chain: str) -> None:
        """Persist the high-water anchor ``(count, last_chain)`` atomically."""
        tmp = self._head_path.with_suffix(self._head_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump({"count": count, "last_chain": last_chain}, fh, sort_keys=True)
        tmp.replace(self._head_path)

    def _load_head(self) -> Optional[dict[str, Any]]:
        """Load the high-water anchor, or ``None`` if no open has been logged yet."""
        if not self._head_path.is_file():
            return None
        with self._head_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise OOSSealError(f"malformed access-head anchor {self._head_path}")
        return payload

    # --- log inspection ----------------------------------------------------- #
    def access_records(self) -> tuple[OOSAccessRecord, ...]:
        """Parse the append-only run-log into records (empty tuple if none yet)."""
        if not self._log_path.is_file():
            return ()
        records: list[OOSAccessRecord] = []
        with self._log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                records.append(
                    OOSAccessRecord(
                        seq=int(payload["seq"]),
                        timestamp=str(payload["timestamp"]),
                        config_hash=str(payload["config_hash"]),
                        code_commit=str(payload["code_commit"]),
                        final_run=bool(payload["final_run"]),
                        oos_hash=str(payload["oos_hash"]),
                        chain=str(payload["chain"]),
                    )
                )
        return tuple(records)

    def access_count(self) -> int:
        """Number of logged OOS opens (fail-toward-flagging). ``> 1`` ⇒ opened more
        than once ⇒ the Phase-5 verdict auto-flags ``CONTAMINATED`` (SSOT §3.1).

        Returns the GREATER of the logged-record count and the high-water anchor count,
        so a tail-truncation of the log cannot silently reduce it below a prior open.
        """
        logged = len(self.access_records())
        head = self._load_head()
        anchored = int(head["count"]) if head is not None else 0
        return max(logged, anchored)

    def verify_log_integrity(self) -> bool:
        """Confirm no log line was edited/reordered/inserted/deleted (chain + anchor).

        Returns ``True`` only when the log is internally consistent AND matches the
        high-water anchor. Detects: a changed field or reorder (broken chain link), an
        interior deletion/insertion (broken ``seq`` sequence), and a tail-truncation or
        whole-log deletion (record count or terminal chain disagrees with the anchor).
        An empty log with no anchor is consistent (``True``); an empty log WITH an anchor
        claiming prior opens is tampered (``False``). Requires the seal (the chain root).
        """
        records = self.access_records()
        head = self._load_head()
        if not records:
            # No records: consistent only if the anchor also records zero opens.
            return head is None or int(head.get("count", 0)) == 0
        if not self.is_sealed():
            return False
        prev_chain = self.load_seal().oos_hash
        for i, rec in enumerate(records):
            if rec.seq != i:
                return False
            expected = _chain_link(prev_chain, rec.core_json())
            if expected != rec.chain:
                return False
            prev_chain = rec.chain
        # Cross-check the high-water anchor: count + terminal chain must agree, so a
        # dropped tail line (which leaves a still-self-consistent prefix) is caught.
        if head is None:
            return False  # records exist but the anchor was removed → tampered
        if int(head.get("count", -1)) != len(records):
            return False
        if str(head.get("last_chain", "")) != records[-1].chain:
            return False
        return True
