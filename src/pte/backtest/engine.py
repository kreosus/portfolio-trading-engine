"""Event-driven, bar-by-bar backtester for one or more independent sub-bots.

Each sub-bot (sleeve) runs separately: its own capital (allocation x
initial_equity; the bot uses 1.0 = full account each), cash, positions,
pending orders, risk engine, trades and equity. Sub-bots never share state.
An optional portfolio_halt stops all sub-bots when their summed equity draws
down past it; the bot does not use it.

Fill rules (conservative by design):
- A signal on bar i's close creates an order that is live from bar i+1.
- Market orders fill at bar i+1's open plus slippage, taker fee.
- Limit orders fill only if price trades THROUGH the level (long: low < entry),
  at min(open, entry) for longs; maker fee. Cancelled if the bar opens beyond
  the stop, or if the target trades before the fill.
- On the fill bar only the stop is checked (intra-bar order is unknowable).
- Stop and target touched in the same bar -> stop.
- Stops fill at the worse of stop and open, plus slippage, taker fee.
- Targets fill at the target price, taker fee, no slippage.
- Trailing stops move only on bar closes and only in the trade's favour.
- Time exits and end-of-window exits fill at the close plus slippage, taker fee.
- Funding is charged at each settlement on positions held into it.
- R-multiples use the INITIAL stop distance.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

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
    init_stop: float
    target: float
    entry_idx: int
    entry_time: pd.Timestamp
    time_exit_bars: int
    fees: float
    trail_atr: float | None = None
    funding: float = 0.0
    risk_frac: float = 0.0


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series           # mark-to-market equity at each bar close
    risk_blocks: dict
    halted_at: object
    initial_equity: float


@dataclass
class SleeveSpec:
    name: str
    intents: dict[str, list[OrderIntent]]
    allocation: float = 1.0


@dataclass
class PortfolioResult:
    sleeves: dict[str, BacktestResult]
    equity: pd.Series
    initial_equity: float
    halted_at: object


class _Sleeve:
    def __init__(self, spec: SleeveSpec, risk_cfg: dict, costs: CostModel, arr: dict):
        self.name = spec.name
        self.costs = costs
        self.arr = arr
        cfg = copy.deepcopy(risk_cfg)
        cfg["initial_equity"] = risk_cfg["initial_equity"] * spec.allocation
        self.initial = cfg["initial_equity"]
        self.risk = RiskEngine(cfg)
        self.cash = self.initial
        self.positions: dict[str, Position] = {}
        self.pending: dict[str, OrderIntent] = {}
        self.intents = {s: {it.signal_idx: it for it in lst} for s, lst in spec.intents.items()}
        self.trades: list[dict] = []
        self.eq: list[float] = []

    def mtm(self, last_close: dict) -> float:
        return self.cash + sum(p.side * p.qty * (last_close[s] - p.entry)
                               for s, p in self.positions.items() if s in last_close)

    def close(self, s, price, ts, idx, reason):
        p = self.positions.pop(s)
        fee = self.costs.fee(p.qty * price, taker=True)
        gross = p.side * p.qty * (price - p.entry)
        self.cash += gross - fee
        fees = p.fees + fee
        net = gross - fees - p.funding
        risk_usd = p.qty * abs(p.entry - p.init_stop)
        self.trades.append(dict(
            sleeve=self.name, symbol=s, side=p.side, entry_time=p.entry_time, exit_time=ts,
            entry=p.entry, exit=price, stop=p.init_stop, target=p.target, qty=p.qty, gross=gross,
            fees=fees, funding=p.funding, pnl=net, r=net / risk_usd if risk_usd > 0 else np.nan,
            reason=reason, bars_held=idx - p.entry_idx, risk_frac=p.risk_frac,
        ))
        self.risk.on_trade_closed(net, idx)

    def fill(self, s, it: OrderIntent, price, ts, idx, last_close, taker: bool):
        a = self.arr[s]
        equity = self.mtm(last_close)
        open_risk = sum(p.qty * abs(p.entry - p.stop) for p in self.positions.values())
        stop = it.stop
        if it.entry_type == "market":
            # keep the planned stop DISTANCE when the open differs from the signal close
            stop = price - it.side * abs(it.entry - it.stop)
        qty = self.risk.size(equity=equity, entry=price, stop=stop, elevated_vol=bool(a["ev"][idx]),
                             open_risk=open_risk, bar_idx=idx, positions_on_symbol=int(s in self.positions))
        if qty <= 0:
            return
        fee = self.costs.fee(qty * price, taker=taker)
        self.cash -= fee
        target = it.target
        if it.entry_type == "market" and math.isfinite(it.target):
            target = price + (it.target - it.entry)
        self.positions[s] = Position(it.side, qty, price, stop, stop, target, idx, ts, it.time_exit_bars,
                                     fee, it.trail_atr, risk_frac=qty * abs(price - stop) / equity)
        if it.side == 1 and a["l"][idx] <= stop:
            self.close(s, self.costs.slip(min(a["o"][idx], stop), -1), ts, idx, "stop")
        elif it.side == -1 and a["h"][idx] >= stop:
            self.close(s, self.costs.slip(max(a["o"][idx], stop), 1), ts, idx, "stop")

    def on_symbol_bar(self, s, i, ts, last_close, halted: bool):
        a, costs = self.arr[s], self.costs
        o, h, l, c = a["o"][i], a["h"][i], a["l"][i], a["c"][i]

        if s in self.positions and ts in a["funding"]:
            p = self.positions[s]
            pay = p.side * p.qty * o * a["rates"][ts]
            self.cash -= pay
            p.funding += pay

        if s in self.positions:
            p = self.positions[s]
            if p.side == 1:
                if l <= p.stop:
                    self.close(s, costs.slip(min(o, p.stop), -1), ts, i, "stop")
                elif h >= p.target:
                    self.close(s, p.target, ts, i, "target")
            else:
                if h >= p.stop:
                    self.close(s, costs.slip(max(o, p.stop), 1), ts, i, "stop")
                elif l <= p.target:
                    self.close(s, p.target, ts, i, "target")
            if s in self.positions and i - p.entry_idx >= p.time_exit_bars:
                self.close(s, costs.slip(c, -p.side), ts, i, "time")

        it = self.pending.get(s)
        if it is not None and s not in self.positions:
            if i > it.expires_idx or halted:
                self.pending.pop(s)
            elif it.entry_type == "market":
                self.pending.pop(s)
                self.fill(s, it, costs.slip(o, it.side), ts, i, last_close, taker=True)
            elif it.side == 1:
                if o <= it.stop:
                    self.pending.pop(s)
                elif l < it.entry:
                    self.pending.pop(s)
                    self.fill(s, it, min(o, it.entry), ts, i, last_close, taker=False)
                elif h >= it.target:
                    self.pending.pop(s)
            else:
                if o >= it.stop:
                    self.pending.pop(s)
                elif h > it.entry:
                    self.pending.pop(s)
                    self.fill(s, it, max(o, it.entry), ts, i, last_close, taker=False)
                elif l <= it.target:
                    self.pending.pop(s)

        # trailing stop update on the close (applies from next bar)
        p = self.positions.get(s)
        if p is not None and p.trail_atr and not np.isnan(a["atr"][i]):
            if p.side == 1:
                p.stop = max(p.stop, c - p.trail_atr * a["atr"][i])
            else:
                p.stop = min(p.stop, c + p.trail_atr * a["atr"][i])

    def queue_signal(self, s, i, halted: bool):
        new = self.intents.get(s, {}).get(i)
        if new is not None and not halted and not self.risk.s.halted:
            self.pending[s] = new


def run_portfolio(features: dict[str, pd.DataFrame], sleeves: list[SleeveSpec],
                  funding: dict[str, pd.DataFrame], risk_cfg: dict, costs: CostModel,
                  start=None, end=None, portfolio_halt: float | None = None) -> PortfolioResult:
    symbols = list(features)
    timeline = pd.DatetimeIndex(sorted(set().union(*[set(features[s].index) for s in symbols])))
    if start is not None:
        timeline = timeline[timeline >= start]
    if end is not None:
        timeline = timeline[timeline < end]

    arr = {}
    for s in symbols:
        f = features[s]
        arr[s] = dict(
            pos=f.index.get_indexer(timeline),
            o=f["open"].to_numpy(), h=f["high"].to_numpy(), l=f["low"].to_numpy(), c=f["close"].to_numpy(),
            atr=f["atr"].to_numpy() if "atr" in f else np.full(len(f), np.nan),
            ev=f["elevated_vol"].to_numpy() if "elevated_vol" in f else np.zeros(len(f), bool),
            funding=set(funding[s].index) if s in funding else set(),
            rates=funding[s]["rate"].to_dict() if s in funding else {},
        )
    books = [_Sleeve(sp, risk_cfg, costs, arr) for sp in sleeves]
    total0 = sum(b.initial for b in books)
    last_close: dict[str, float] = {}
    peak = total0
    halted_at = None
    port_eq = []

    for t_i, ts in enumerate(timeline):
        halted = halted_at is not None
        for b in books:
            b.risk.on_bar(ts, b.mtm(last_close))
            if b.risk.s.halted:
                b.pending.clear()
        for s in symbols:
            i = int(arr[s]["pos"][t_i])
            if i < 0:
                continue
            for b in books:
                b.on_symbol_bar(s, i, ts, last_close, halted)
            last_close[s] = arr[s]["c"][i]
            for b in books:
                b.queue_signal(s, i, halted)
        total = 0.0
        for b in books:
            e = b.mtm(last_close)
            b.eq.append(e)
            total += e
        port_eq.append(total)
        peak = max(peak, total)
        if portfolio_halt is not None and halted_at is None and 1 - total / peak >= portfolio_halt:
            halted_at = ts
            for b in books:
                b.pending.clear()

    if len(timeline):
        for b in books:
            for s in list(b.positions):
                idx = int(arr[s]["pos"][-1]) if arr[s]["pos"][-1] >= 0 else len(arr[s]["c"]) - 1
                b.close(s, costs.slip(last_close[s], -b.positions[s].side), timeline[-1], idx, "end")
            if b.eq:
                b.eq[-1] = b.cash
        if port_eq:
            port_eq[-1] = sum(b.cash for b in books)

    idx = pd.DatetimeIndex(timeline)
    results = {b.name: BacktestResult(
        trades=pd.DataFrame(b.trades), equity=pd.Series(b.eq, index=idx, name=b.name),
        risk_blocks=dict(b.risk.s.blocks), halted_at=b.risk.s.halt_time, initial_equity=b.initial,
    ) for b in books}
    return PortfolioResult(results, pd.Series(port_eq, index=idx, name="portfolio"), total0, halted_at)


def run_backtest(features: dict[str, pd.DataFrame], intents: dict[str, list[OrderIntent]],
                 funding: dict[str, pd.DataFrame], risk_cfg: dict, costs: CostModel,
                 start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> BacktestResult:
    """Single-strategy backtest: one sleeve holding all capital."""
    res = run_portfolio(features, [SleeveSpec("main", intents, 1.0)], funding, risk_cfg, costs, start, end)
    r = res.sleeves["main"]
    if not r.trades.empty:
        r.trades = r.trades.drop(columns="sleeve")
    return r
