"""Execution Engine and Position Manager (spec §21, §24-§30, §39).

Only ExecutionEngine talks to venue order endpoints. Every order moves through the state machine
CREATED -> VALIDATED -> SUBMITTED -> ACKNOWLEDGED -> (PARTIALLY_)FILLED / CANCELLED / REJECTED / UNKNOWN.
An UNKNOWN order is reconciled by querying the venue; it is never blindly resubmitted.
"""

from __future__ import annotations

import pandas as pd

from .adapters import round_step, round_tick
from .models import Order, OrderState, Position
from .risk import size_position


class ExecutionEngine:
    def __init__(self, venue, journal, cfg: dict):
        self.venue, self.j, self.cfg = venue, journal, cfg
        self.rules = venue.exchange_rules(cfg["symbol"])

    def _submit(self, o: Order) -> Order:
        o.transition(OrderState.VALIDATED, "passed pre-trade checks")
        o.transition(OrderState.SUBMITTED, f"to {self.venue.name}")
        try:
            resp = self.venue.submit(o)
        except Exception as e:                                          # network / timeout
            o.transition(OrderState.UNKNOWN, f"submit error: {e}")
            self.reconcile(o)
            self.j.order(o)
            return o
        if resp.get("status") == "REJECTED":
            o.transition(OrderState.REJECTED, resp.get("reason", ""))
        else:
            o.exchange_id = str(resp.get("orderId", o.exchange_id))
            o.transition(OrderState.ACKNOWLEDGED, f"venue id {o.exchange_id}")
        self.j.order(o)
        self.j.event("order_submission", purpose=o.purpose, side=o.side, type=o.type, qty=o.qty,
                     price=o.price, stop_price=o.stop_price, state=o.state.value, client_id=o.client_id)
        return o

    def reconcile(self, o: Order) -> None:
        r = self.venue.query(o.client_id)
        st = r.get("status", "UNKNOWN")
        mapping = {"NEW": OrderState.ACKNOWLEDGED, "ACKNOWLEDGED": OrderState.ACKNOWLEDGED,
                   "FILLED": OrderState.FILLED, "PARTIALLY_FILLED": OrderState.PARTIALLY_FILLED,
                   "CANCELED": OrderState.CANCELLED, "CANCELLED": OrderState.CANCELLED,
                   "REJECTED": OrderState.REJECTED}
        if st in mapping and o.state != mapping[st]:
            o.transition(mapping[st], "reconciled with venue")
        self.j.event("reconciliation", client_id=o.client_id, venue_status=st, state=o.state.value)

    # ---- entry (spec §21 entry validation happens in open_from_signal)
    def open_from_signal(self, sig, quote: dict, equity: float, news_blocking: bool) -> tuple[Order | None, object]:
        side = sig.direction
        up = side == "LONG"
        px_now = quote["ask"] if up else quote["bid"]
        spread_bps = (quote["ask"] - quote["bid"]) / px_now * 1e4
        problems = []
        if spread_bps > self.cfg["execution"]["max_spread_bps"]:
            problems.append(f"spread {spread_bps:.1f} bps")
        if news_blocking:
            problems.append("news lock became active")
        if (up and px_now <= sig.stop) or (not up and px_now >= sig.stop):
            problems.append("price already beyond the stop: setup invalidated")
        entry = round_tick(sig.planned_entry, self.rules["tick_size"])
        sizing = size_position(entry, sig.stop, side, equity, self.rules, self.cfg)
        if not sizing.approved:
            problems.append(f"risk: {sizing.reason}")
        if problems:
            self.j.event("risk_rejection" if not sizing.approved else "signal_cancelled", problems=problems)
            return None, sizing
        o = Order(self.cfg["symbol"], "BUY" if up else "SELL", "LIMIT", sizing.qty, price=entry, purpose="entry")
        return self._submit(o), sizing

    def protect(self, pos: Position) -> list[Order]:
        """Place stop + TP1/TP2/TP3 as reduce-only orders immediately after the entry fills."""
        close_side = "SELL" if pos.side == "LONG" else "BUY"
        tick, step = self.rules["tick_size"], self.rules["step_size"]
        out = [self._submit(Order(pos.symbol, close_side, "STOP_MARKET", pos.qty,
                                  stop_price=round_tick(pos.stop, tick), reduce_only=True, purpose="stop"))]
        split = self.cfg["execution"]["tp_split"]
        left = pos.qty
        for i, (tp, frac) in enumerate(zip((pos.tp1, pos.tp2, pos.tp3), split)):
            q = left if i == 2 else max(round_step(pos.qty * frac, step), self.rules["min_qty"])
            q = min(q, left)
            left = round(left - q, 10)
            if q > 0:
                out.append(self._submit(Order(pos.symbol, close_side, "TAKE_PROFIT_MARKET", q,
                                              stop_price=round_tick(tp, tick), reduce_only=True,
                                              purpose=f"tp{i + 1}")))
        if out[0].state != OrderState.ACKNOWLEDGED:
            raise RuntimeError("protective stop could not be placed")     # -> kill switch in the runner
        return out

    def cancel(self, o: Order, why: str) -> None:
        if o.open:
            self.venue.cancel(o.client_id)
            if o.state != OrderState.CANCELLED:
                o.transition(OrderState.CANCELLED, why)
            self.j.order(o)

    def market_close(self, pos: Position, qty: float, quote: dict, ts, why: str) -> dict:
        o = Order(pos.symbol, "SELL" if pos.side == "LONG" else "BUY", "MARKET", qty, reduce_only=True,
                  purpose="close")
        self._submit(o)
        fill = self.venue.market_now(o, quote["bid"], quote["ask"], ts)
        self.j.order(o)
        self.j.event("exit", reason=why, price=fill["price"], qty=qty)
        return fill
