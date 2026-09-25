import copy

import numpy as np
import pandas as pd
import pytest

from pte.backtest.costs import CostModel
from pte.backtest.engine import run_backtest
from pte.config import load_config
from pte.strategies.smc_m15 import OrderIntent

RISK = load_config()["risk"]
ZERO = CostModel(0, 0, 0)


def _feats(rows):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="15min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


def _run(rows, intent, costs=ZERO, risk=RISK, funding=None):
    f = _feats(rows)
    fu = {"X": funding} if funding is not None else {}
    return run_backtest({"X": f}, {"X": [intent]}, fu, risk, costs)


FLAT = [100, 101, 99, 100]


def test_limit_needs_trade_through():
    # entry 99: bar low touches exactly 99 -> no fill
    it = OrderIntent(0, 1, 99, 95, 107, 5, 96)
    res = _run([FLAT, [100, 101, 99, 100], [100, 101, 99, 100]], it)
    assert res.trades.empty


def test_long_hits_target_and_sizes_by_risk():
    it = OrderIntent(0, 1, 99, 95, 107, 5, 96)
    rows = [FLAT, [100, 101, 98.5, 100], [100, 108, 99, 107]]
    res = _run(rows, it)
    t = res.trades.iloc[0]
    assert t.reason == "target" and t.exit == 107
    risk_usd = RISK["initial_equity"] * RISK["risk_normal"]
    assert t.pnl == pytest.approx(2 * risk_usd)
    assert t.r == pytest.approx(2.0)


def test_same_bar_stop_and_target_counts_as_stop():
    it = OrderIntent(0, 1, 99, 95, 107, 5, 96)
    rows = [FLAT, [100, 101, 98.5, 100], [100, 110, 90, 100]]
    t = _run(rows, it).trades.iloc[0]
    assert t.reason == "stop"


def test_gap_through_stop_fills_at_open():
    it = OrderIntent(0, 1, 99, 95, 107, 5, 96)
    rows = [FLAT, [100, 101, 98.5, 100], [90, 92, 89, 91]]
    t = _run(rows, it).trades.iloc[0]
    assert t.exit == 90 and t.r < -1


def test_target_before_fill_cancels_order():
    it = OrderIntent(0, 1, 99, 95, 107, 5, 96)
    rows = [FLAT, [100, 108, 100, 107], [100, 101, 98, 100]]
    assert _run(rows, it).trades.empty


def test_order_expires():
    it = OrderIntent(0, 1, 99, 95, 107, 1, 96)
    rows = [FLAT, [100, 101, 100, 100], [100, 101, 98, 100]]
    assert _run(rows, it).trades.empty


def test_costs_reduce_pnl():
    it = OrderIntent(0, 1, 99, 95, 107, 5, 96)
    rows = [FLAT, [100, 101, 98.5, 100], [100, 108, 99, 107]]
    free = _run(rows, it).trades.iloc[0].pnl
    paid = _run(rows, it, CostModel(0.0002, 0.0005, 2)).trades.iloc[0]
    assert paid.pnl < free and paid.fees > 0


def test_funding_charged_to_long():
    it = OrderIntent(0, 1, 99, 95, 200, 5, 96)
    rows = [FLAT, [100, 101, 98.5, 100], [100, 101, 99, 100], [100, 101, 99, 100]]
    idx = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
    fu = pd.DataFrame({"rate": [0.01]}, index=pd.DatetimeIndex([idx[2]], name="time"))
    t = _run(rows, it, funding=fu).trades.iloc[0]
    assert t.funding > 0 and t.reason == "end"


def test_short_mirror():
    it = OrderIntent(0, -1, 101, 105, 93, 5, 96)
    rows = [FLAT, [100, 101.5, 99, 100], [100, 101, 92, 93]]
    t = _run(rows, it).trades.iloc[0]
    assert t.reason == "target" and t.r == pytest.approx(2.0)


def test_drawdown_halt_blocks_new_trades():
    risk = copy.deepcopy(RISK); risk["drawdown_halt"] = 0.004
    its = [OrderIntent(0, 1, 99, 95, 107, 5, 96), OrderIntent(3, 1, 99, 95, 107, 8, 96)]
    rows = [FLAT, [100, 101, 98.5, 100], [96, 96, 94, 94.5], [99, 100, 98, 99.5],
            [99.5, 100, 98, 99], [99, 100, 98, 99]]
    f = _feats(rows)
    res = run_backtest({"X": f}, {"X": its}, {}, risk, ZERO)
    assert len(res.trades) == 1 and res.halted_at is not None


def test_random_walk_has_no_edge_before_costs():
    """Sanity: on a driftless random walk, the full strategy should not show a big edge."""
    from pte.config import load_config
    from pte.data.synthetic import make_bars
    from pte.pipeline import prepare
    cfg = load_config()
    bars = {"X": make_bars(30_000, seed=11)}
    feats, intents = prepare(bars, cfg)
    res = run_backtest(feats, intents, {}, cfg["risk"], ZERO)
    assert len(res.trades) > 20
    assert abs(res.trades.r.mean()) < 0.35
