"""Data objects shared by the live modules (spec §20, §25, §26)."""

from __future__ import annotations

import enum
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class OrderState(str, enum.Enum):
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


# Allowed transitions. Anything else is a bug and raises.
TRANSITIONS = {
    OrderState.CREATED: {OrderState.VALIDATED, OrderState.REJECTED, OrderState.CANCELLED},
    OrderState.VALIDATED: {OrderState.SUBMITTED, OrderState.CANCELLED},
    OrderState.SUBMITTED: {OrderState.ACKNOWLEDGED, OrderState.REJECTED, OrderState.UNKNOWN},
    OrderState.ACKNOWLEDGED: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED, OrderState.UNKNOWN},
    OrderState.PARTIALLY_FILLED: {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELLED, OrderState.UNKNOWN},
    OrderState.UNKNOWN: {OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED, OrderState.FILLED,
                         OrderState.CANCELLED, OrderState.REJECTED},
    OrderState.FILLED: set(),
    OrderState.CANCELLED: set(),
    OrderState.REJECTED: set(),
}


class Side(str, enum.Enum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        return 1 if self is Side.LONG else -1


@dataclass
class Order:
    symbol: str
    side: str                 # BUY / SELL
    type: str                 # MARKET / LIMIT / STOP_MARKET / TAKE_PROFIT_MARKET
    qty: float
    price: float | None = None
    stop_price: float | None = None
    reduce_only: bool = False
    purpose: str = "entry"    # entry / stop / tp1 / tp2 / tp3 / close
    client_id: str = field(default_factory=lambda: "pte-" + uuid.uuid4().hex[:16])
    exchange_id: str | None = None
    state: OrderState = OrderState.CREATED
    filled_qty: float = 0.0
    avg_price: float | None = None
    history: list = field(default_factory=list)

    def transition(self, new: OrderState, note: str = "") -> None:
        if new not in TRANSITIONS[self.state]:
            raise RuntimeError(f"illegal order transition {self.state.value} -> {new.value} ({self.client_id})")
        self.history.append({"t": now_utc().isoformat(), "from": self.state.value, "to": new.value, "note": note})
        self.state = new

    @property
    def open(self) -> bool:
        return self.state in (OrderState.SUBMITTED, OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED,
                              OrderState.UNKNOWN)


@dataclass
class Signal:
    """Spec §20. Produced by the Master Decision Engine. No order is submitted at this stage."""
    symbol: str
    direction: str
    strategy_version: str
    timestamp: str
    htf_bias: str
    structure_state: dict
    liquidity_state: dict
    sweep: dict
    displacement: dict
    bos_choch: dict
    poi: dict
    fvg: dict
    ob: dict
    momentum_state: dict
    btc_context: dict
    funding: float | None
    oi: float | None
    session: str
    regime: str
    news_state: str
    score: float
    score_breakdown: dict
    entry_zone: tuple
    planned_entry: float
    stop: float
    tp1: float
    tp2: float
    tp3: float
    expected_rr: float
    invalidation: str
    setup_bar_time: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    action: str                     # LONG / SHORT / NO_TRADE
    reasons: list                   # every check, pass or fail, in order
    signal: Signal | None = None
    failed: list = field(default_factory=list)

    def explain(self) -> str:
        head = f"{self.action}"
        if self.signal:
            head += f"  score {self.signal.score:.0f}"
        lines = [head]
        for ok, text in self.reasons:
            lines.append(f"  [{'x' if ok else ' '}] {text}")
        return "\n".join(lines)


@dataclass
class Position:
    """Spec §26."""
    symbol: str
    side: str                       # LONG / SHORT
    entry: float
    qty: float
    remaining_qty: float
    stop: float
    initial_stop: float
    tp1: float
    tp2: float
    tp3: float
    leverage: float
    margin: float
    opened_at: str
    signal_id: str
    realized_pnl: float = 0.0
    fees: float = 0.0
    funding: float = 0.0
    tp_done: list = field(default_factory=list)
    mae: float = 0.0                # most adverse excursion, price units
    mfe: float = 0.0                # most favourable excursion, price units
    liquidation_price: float = 0.0

    @property
    def sign(self) -> int:
        return 1 if self.side == "LONG" else -1

    def unrealized(self, price: float) -> float:
        return self.sign * self.remaining_qty * (price - self.entry)
