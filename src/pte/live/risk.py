"""Risk Engine and Position Sizing (spec §22, §23, §29, §32). The strategy cannot override these limits."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .adapters import round_step


@dataclass
class Sizing:
    approved: bool
    reason: str
    qty: float = 0.0
    notional: float = 0.0
    leverage: float = 0.0
    margin: float = 0.0
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    liquidation_price: float = 0.0


def limits_block(state: dict, cfg: dict) -> str | None:
    """Account-level kill switches (spec §39). Returns the reason or None."""
    r = cfg["risk"]
    eq, peak = state["equity"], state["peak_equity"]
    if state.get("halted"):
        return f"halted: {state['halted']}"
    if eq / state["day_start_equity"] - 1 <= -r["max_daily_loss"]:
        return "daily loss limit reached"
    if eq / state["week_start_equity"] - 1 <= -r["max_weekly_loss"]:
        return "weekly loss limit reached"
    if 1 - eq / peak >= r["max_drawdown"]:
        return "max drawdown reached"
    if state["open_positions"] >= r["max_open_positions"]:
        return f"max open positions ({r['max_open_positions']}) reached"
    return None


def size_position(entry: float, stop: float, side: str, equity: float, rules: dict, cfg: dict,
                  maint_margin_rate: float = 0.004) -> Sizing:
    """Worked example in spec §22: $10,000 x 0.5% = $50 risk; stop $500 away -> 0.1 BTC, $10,000 notional."""
    r = cfg["risk"]
    dist = abs(entry - stop)
    if dist <= 0:
        return Sizing(False, "stop distance is zero")
    risk_amt = equity * r["risk_per_trade"]
    raw_qty = risk_amt / dist
    qty = round_step(raw_qty, rules["step_size"])          # never round UP past max risk
    max_qty = round_step(r["max_leverage"] * equity / entry, rules["step_size"])
    qty = min(qty, max_qty)
    if qty < rules["min_qty"]:
        return Sizing(False, f"size {raw_qty:.6f} below exchange min qty {rules['min_qty']}")
    notional = qty * entry
    if notional < rules["min_notional"]:
        return Sizing(False, f"notional {notional:.2f} below exchange minimum {rules['min_notional']}")
    # smallest whole leverage that covers the notional with the account; never above the hard cap
    lev = min(r["max_leverage"], max(1, math.ceil(notional / equity)))
    margin = notional / lev
    # approximate isolated-margin liquidation price
    if side == "LONG":
        liq = entry * (1 - 1 / lev + maint_margin_rate)
    else:
        liq = entry * (1 + 1 / lev - maint_margin_rate)
    liq_dist = abs(entry - liq)
    if lev > 1 and liq_dist < r["min_liq_distance_x_stop"] * dist:
        return Sizing(False, f"liquidation {liq:.2f} too close ({liq_dist:.0f} < {r['min_liq_distance_x_stop']}x stop)")
    return Sizing(True, "approved", qty, notional, float(lev), margin, qty * dist, qty * dist / equity, liq)
