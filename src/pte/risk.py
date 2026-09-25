"""Risk engine. Sits above every strategy; strategies cannot override it."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class RiskState:
    peak_equity: float
    day_start_equity: float
    week_start_equity: float
    day: object = None
    week: object = None
    consecutive_losses: int = 0
    paused_until_idx: int = -1
    halted: bool = False
    halt_time: object = None
    blocks: dict = field(default_factory=dict)  # reason -> count of rejected entries


class RiskEngine:
    def __init__(self, cfg: dict):
        self.c = cfg
        eq = cfg["initial_equity"]
        self.s = RiskState(peak_equity=eq, day_start_equity=eq, week_start_equity=eq)

    # --- called every bar with mark-to-market equity
    def on_bar(self, ts: pd.Timestamp, equity: float) -> None:
        s = self.s
        day = ts.value // 86_400_000_000_000          # UTC day number
        week = (day + 3) // 7                          # Monday-based week number (epoch was a Thursday)
        if s.day != day:
            s.day, s.day_start_equity = day, equity
        if s.week != week:
            s.week, s.week_start_equity = week, equity
        s.peak_equity = max(s.peak_equity, equity)
        if not s.halted and self.drawdown(equity) >= self.c["drawdown_halt"]:
            s.halted, s.halt_time = True, ts

    def on_trade_closed(self, pnl: float, bar_idx: int) -> None:
        s = self.s
        s.consecutive_losses = s.consecutive_losses + 1 if pnl < 0 else 0
        if s.consecutive_losses >= self.c["max_consecutive_losses"]:
            s.paused_until_idx = bar_idx + self.c["loss_pause_bars"]
            s.consecutive_losses = 0

    def drawdown(self, equity: float) -> float:
        return 1 - equity / self.s.peak_equity

    def risk_fraction(self, equity: float, elevated_vol: bool) -> float:
        if self.drawdown(equity) > self.c["drawdown_reduce"]:
            return self.c["risk_drawdown_5"]
        if elevated_vol:
            return self.c["risk_elevated_vol"]
        return self.c["risk_normal"]

    def _block(self, reason: str) -> float:
        self.s.blocks[reason] = self.s.blocks.get(reason, 0) + 1
        return 0.0

    def size(self, *, equity: float, entry: float, stop: float, elevated_vol: bool,
             open_risk: float, bar_idx: int, positions_on_symbol: int) -> float:
        """Return quantity (units of base asset) or 0 if the trade is not allowed."""
        s, c = self.s, self.c
        if s.halted:
            return self._block("halted_drawdown")
        if bar_idx < s.paused_until_idx:
            return self._block("paused_consecutive_losses")
        if positions_on_symbol >= c["max_positions_per_symbol"]:
            return self._block("max_positions_per_symbol")
        if equity / s.day_start_equity - 1 <= -c["max_daily_loss"]:
            return self._block("daily_loss")
        if equity / s.week_start_equity - 1 <= -c["max_weekly_loss"]:
            return self._block("weekly_loss")
        per_unit = abs(entry - stop)
        if per_unit <= 0:
            return self._block("bad_stop")
        frac = self.risk_fraction(equity, elevated_vol)
        if (open_risk + frac * equity) / equity > c["max_open_risk"] + 1e-12:
            return self._block("max_open_risk")
        qty = frac * equity / per_unit
        max_qty = c["max_leverage"] * equity / entry
        return min(qty, max_qty)
