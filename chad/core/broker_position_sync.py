from typing import Dict
from dataclasses import dataclass
from datetime import datetime

try:
    from ib_insync import IB
except ImportError:
    IB = None


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

    def fetch_positions(self) -> Dict[str, BrokerPosition]:
        """
        Pull positions from IBKR
        """
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
                timestamp=datetime.utcnow(),
            )

        return out
