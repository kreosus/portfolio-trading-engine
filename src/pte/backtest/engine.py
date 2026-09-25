"""Event-driven, bar-by-bar portfolio backtester.

Fill rules (conservative by design):
- A signal on bar i's close creates a limit order that is live from bar i+1.
- A limit fills only if price trades THROUGH it (long: low < entry). Fill price
  is min(open, entry) for longs — a gap through the level fills at the open.
- If the bar opens beyond the stop, the order is cancelled (setup invalidated).
- If the target is reached before the order fills, the order is cancelled.
- On the fill bar only the stop is checked (target ignored: order unknown).
- Stop and target touched in the same bar -> stop.
- Stops fill at the worse of stop price and bar open, plus slippage, taker fee.
- Targets fill at the target price, taker fee (conservative), no slippage.
- Time exits and end-of-window exits fill at the close, plus slippage, taker fee.
- Funding is charged at each settlement time on positions held into it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..risk import RiskEngine
from ..strategies.smc_m15 import OrderIntent
from .costs import CostModel


@dataclass
class Position:
    side: int
    qty: float
    entry: float
    stop: float
    target: float
    entry_idx: int
    entry_time: pd.Timestamp
    time_exit_bars: int
    fees: float
    funding: float = 0.0
    risk_frac: float = 0.0


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series           # mark-to-market equity at each bar close
    risk_blocks: dict
    halted_at: object
    initial_equity: float


def run_backtest(features: dict[str, pd.DataFrame], intents: dict[str, list[OrderIntent]],
                 funding: dict[str, pd.DataFrame], risk_cfg: dict, costs: CostModel,
                 start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> BacktestResult:
    symbols = list(features)
    timeline = sorted(set().union(*[set(features[s].index) for s in symbols]))
    timeline = pd.DatetimeIndex(timeline)
    if start is not None:
        timeline = timeline[timeline >= start]
    if end is not None:
        timeline = timeline[timeline < end]

    arr = {}
    for s in symbols:
        f = features[s]
        arr[s] = dict(
            pos=f.index.get_indexer(timeline),
            o=f["open"].to_numpy(), h=f["high"].to_numpy(), l=f["low"].to_numpy(),
            c=f["close"].to_numpy(),
            ev=f["elevated_vol"].to_numpy() if "elevated_vol" in f else np.zeros(len(f), bool),
            funding=set(funding[s].index) if s in funding else set(),
            rates=funding[s]["rate"].to_dict() if s in funding else {},
            intents={it.signal_idx: it for it in intents.get(s, [])},
        )

    risk = RiskEngine(risk_cfg)
    cash = risk_cfg["initial_equity"]
    positions: dict[str, Position] = {}
    pending: dict[str, OrderIntent] = {}
    last_close: dict[str, float] = {}
    trades: list[dict] = []
    eq_times, eq_vals = [], []

    def mtm() -> float:
        return cash + sum(p.side * p.qty * (last_close[s] - p.entry) for s, p in positions.items())

    def close_pos(s: str, price: float, taker: bool, ts, idx: int, reason: str) -> None:
        nonlocal cash
        p = positions.pop(s)
        fee = costs.fee(p.qty * price, taker=taker)
        gross = p.side * p.qty * (price - p.entry)
        cash += gross - fee
        fees = p.fees + fee
        net = gross - fees - p.funding
        risk_usd = p.qty * abs(p.entry - p.stop)
        trades.append(dict(
            symbol=s, side=p.side, entry_time=p.entry_time, exit_time=ts, entry=p.entry, exit=price,
            stop=p.stop, target=p.target, qty=p.qty, gross=gross, fees=fees, funding=p.funding,
            pnl=net, r=net / risk_usd if risk_usd > 0 else np.nan, reason=reason,
            bars_held=idx - p.entry_idx, risk_frac=p.risk_frac,
        ))
        risk.on_trade_closed(net, idx)

    def try_fill(s: str, it: OrderIntent, price: float, ts, idx: int, elevated: bool) -> None:
        nonlocal cash
        equity = mtm() if last_close else cash
        open_risk = sum(p.qty * abs(p.entry - p.stop) for p in positions.values())
        qty = risk.size(equity=equity, entry=price, stop=it.stop, elevated_vol=bool(elevated),
                        open_risk=open_risk, bar_idx=idx, positions_on_symbol=int(s in positions))
        if qty <= 0:
            return
        fee = costs.fee(qty * price, taker=False)
        cash -= fee
        positions[s] = Position(it.side, qty, price, it.stop, it.target, idx, ts,
                                it.time_exit_bars, fee, risk_frac=qty * abs(price - it.stop) / equity)
        a = arr[s]
        # fill bar: stop only (intra-bar order of target vs fill is unknowable)
        if it.side == 1 and a["l"][idx] <= it.stop:
            close_pos(s, costs.slip(it.stop, -1), True, ts, idx, "stop")
        elif it.side == -1 and a["h"][idx] >= it.stop:
            close_pos(s, costs.slip(it.stop, 1), True, ts, idx, "stop")

    for t_i, ts in enumerate(timeline):
        risk.on_bar(ts, mtm() if last_close else cash)
        if risk.s.halted:
            pending.clear()
        for s in symbols:
            a = arr[s]
            i = int(a["pos"][t_i])
            if i < 0:
                continue
            o, h, l, c = a["o"][i], a["h"][i], a["l"][i], a["c"][i]

            # 1) funding on positions held into this settlement
            if s in positions and ts in a["funding"]:
                p = positions[s]
                pay = p.side * p.qty * o * a["rates"][ts]
                cash -= pay
                p.funding += pay

            # 2) exits for positions opened on earlier bars
            if s in positions:
                p = positions[s]
                if p.side == 1:
                    if l <= p.stop:
                        close_pos(s, costs.slip(min(o, p.stop), -1), True, ts, i, "stop")
                    elif h >= p.target:
                        close_pos(s, p.target, True, ts, i, "target")
                else:
                    if h >= p.stop:
                        close_pos(s, costs.slip(max(o, p.stop), 1), True, ts, i, "stop")
                    elif l <= p.target:
                        close_pos(s, p.target, True, ts, i, "target")
                if s in positions and i - p.entry_idx >= p.time_exit_bars:
                    close_pos(s, costs.slip(c, -p.side), True, ts, i, "time")

            # 3) pending limit order
            it = pending.get(s)
            if it is not None and s not in positions:
                if i > it.expires_idx:
                    pending.pop(s)
                elif it.side == 1:
                    if o <= it.stop:
                        pending.pop(s)
                    elif l < it.entry:
                        pending.pop(s)
                        try_fill(s, it, min(o, it.entry), ts, i, a["ev"][i])
                    elif h >= it.target:
                        pending.pop(s)
                else:
                    if o >= it.stop:
                        pending.pop(s)
                    elif h > it.entry:
                        pending.pop(s)
                        try_fill(s, it, max(o, it.entry), ts, i, a["ev"][i])
                    elif l <= it.target:
                        pending.pop(s)

            last_close[s] = c

            # 4) new signal on this bar's close -> live from next bar
            new = a["intents"].get(i)
            if new is not None and not risk.s.halted:
                pending[s] = new

        eq_times.append(ts)
        eq_vals.append(mtm())

    # flatten at end of window
    if len(timeline):
        for s in list(positions):
            a = arr[s]
            idx = int(a["pos"][-1]) if a["pos"][-1] >= 0 else len(a["c"]) - 1
            close_pos(s, costs.slip(last_close[s], -positions[s].side), True, timeline[-1], idx, "end")
        if eq_vals:
            eq_vals[-1] = cash

    return BacktestResult(
        trades=pd.DataFrame(trades), equity=pd.Series(eq_vals, index=pd.DatetimeIndex(eq_times), name="equity"),
        risk_blocks=dict(risk.s.blocks), halted_at=risk.s.halt_time,
        initial_equity=risk_cfg["initial_equity"],
    )

