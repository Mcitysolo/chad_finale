from typing import Dict
from dataclasses import dataclass
from datetime import datetime, timezone

try:
    from ib_async import IB
except ImportError:
    IB = None


class BrokerTruthUnavailable(RuntimeError):
    """Broker positions could not be read as authoritative truth.

    Signals "unknown", NEVER "flat". Callers must not infer a flat broker from
    this — see XOV-2345 and ``BrokerPositionSync.fetch_positions``.
    """


@dataclass
class BrokerPosition:
    symbol: str
    sec_type: str
    quantity: float
    avg_cost: float
    timestamp: datetime


class BrokerPositionSync:
    """
    Production-grade IBKR position synchronizer.

    - Pulls real broker positions
    - Normalizes into CHAD format
    - Provides single source of truth
    """

    def __init__(self, ib: IB):
        self.ib = ib

    def api_connected(self) -> bool:
        """True iff the IB API connection is up and its caches are trustworthy.

        Test doubles without an ``isConnected`` probe are trusted (legacy
        behaviour) — production always has one.
        """
        probe = getattr(self.ib, "isConnected", None)
        if not callable(probe):
            return True
        try:
            return bool(probe())
        except Exception:  # noqa: BLE001 - an unanswerable probe is not truth
            return False

    def fetch_positions(self) -> Dict[str, BrokerPosition]:
        """
        Pull positions from IBKR.

        Raises ``BrokerTruthUnavailable`` when the API connection is down.

        XOV-2345: ``ib.positions()`` is a pure read of ``wrapper.positions``, a
        LOCAL cache that ib_async empties via ``wrapper.reset()`` on any socket
        drop and only ever repopulates from the ``reqPositions`` subscription
        issued inside ``connectAsync``. A dead connection therefore returns an
        empty list — byte-identical to a genuinely flat broker — with no
        exception. Returning that {} let callers "confirm" the broker was flat
        against a connection that had simply died, so this fetch fails closed:
        no connection, no answer.
        """
        if not self.api_connected():
            raise BrokerTruthUnavailable(
                "IB API connection is down — ib.positions() would return a reset "
                "local cache, not broker truth"
            )

        positions = self.ib.positions()

        out: Dict[str, BrokerPosition] = {}

        for p in positions:
            contract = p.contract

            symbol = contract.localSymbol or contract.symbol
            sec_type = contract.secType

            out[symbol] = BrokerPosition(
                symbol=symbol,
                sec_type=sec_type,
                quantity=float(p.position),
                avg_cost=float(p.avgCost),
                timestamp=datetime.now(timezone.utc),
            )

        return out
