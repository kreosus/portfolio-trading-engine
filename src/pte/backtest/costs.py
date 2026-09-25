"""Transaction cost model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    slippage_bps: float = 2.0
    multiplier: float = 1.0

    @classmethod
    def from_config(cls, c: dict) -> "CostModel":
        return cls(c["maker_fee"], c["taker_fee"], c["slippage_bps"], c.get("multiplier", 1.0))

    def fee(self, notional: float, taker: bool) -> float:
        rate = self.taker_fee if taker else self.maker_fee
        return abs(notional) * rate * self.multiplier

    def slip(self, price: float, side: int) -> float:
        """Adverse price for a taker fill. side=+1 buying, -1 selling."""
        return price * (1 + side * self.slippage_bps * 1e-4 * self.multiplier)
