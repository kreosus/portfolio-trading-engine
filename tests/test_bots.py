import math

import numpy as np
import pandas as pd
import pytest

from pte.backtest.costs import CostModel
from pte.backtest.engine import SleeveSpec, run_backtest, run_portfolio
from pte.bots import build_features_all, build_sleeves
from pte.config import load_config
from pte.data.synthetic import make_bars, make_funding
from pte.features.common import build_common, funding_on_bars
from pte.strategies.library import REGISTRY
from pte.strategies.smc_m15 import OrderIntent

CFG = load_config()
RISK = CFG["risk"]
ZERO = CostModel(0, 0, 0)


@pytest.fixture(scope="module")
def bars():
    return make_bars(6000, seed=5)


@pytest.mark.parametrize("cut", [3100, 4500])
def test_common_features_are_causal(bars, cut):
    fu = make_funding(bars)
    full = build_common(bars, fu)
    part = build_common(bars.iloc[:cut], fu[fu.index < bars.index[cut]])
    for col in part.columns:
        assert np.allclose(full[col].iloc[:cut].to_numpy(), part[col].to_numpy(), equal_nan=True), col


@pytest.mark.parametrize("name", list(REGISTRY))
def test_library_signals_are_causal(bars, name):
    fu = {"X": make_funding(bars)}
    cut = 4200
    f_full = build_features_all({"X": bars}, fu, CFG)["X"]
    fu_cut = {"X": fu["X"][fu["X"].index < bars.index[cut]]}
    f_part = build_features_all({"X": bars.iloc[:cut]}, fu_cut, CFG)["X"]
    p = CFG["bots"]["sleeves"][name]
    full = [s for s in REGISTRY[name](f_full, p) if s.signal_idx < cut]
    part = REGISTRY[name](f_part, p)
    assert full == part


def test_funding_known_only_after_settlement():
    idx = pd.date_range("2024-01-01 07:00", periods=8, freq="15min", tz="UTC")
    df = pd.DataFrame({"close": 1.0}, index=idx)
    fu = pd.DataFrame({"rate": [0.0005]}, index=pd.DatetimeIndex([pd.Timestamp("2024-01-01 08:00", tz="UTC")], name="time"))
    s = funding_on_bars(df, fu)
    assert s[pd.Timestamp("2024-01-01 07:30", tz="UTC")] == 0.0     # closes 07:45: not yet settled
    assert s[pd.Timestamp("2024-01-01 07:45", tz="UTC")] == 0.0005  # closes 08:00: settled


def _f(rows, atr=1.0):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="15min", tz="UTC")
    f = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)
    f["atr"] = atr
    return f


def test_market_entry_fills_next_open_with_slippage_and_taker_fee():
    it = OrderIntent(0, 1, 100, 98, math.inf, 1, 96, entry_type="market")
    f = _f([[100, 100, 100, 100], [101, 101.5, 100.5, 101], [101, 101, 100.5, 101]])
    costs = CostModel(0.0002, 0.0005, 2.0)
    t = run_backtest({"X": f}, {"X": [it]}, {}, RISK, costs).trades.iloc[0]
    assert t.entry == pytest.approx(101 * 1.0002)
    assert t.stop == pytest.approx(t.entry - 2)          # stop distance preserved from signal
    assert t.reason == "end"


def test_trailing_stop_ratchets_and_exits():
    it = OrderIntent(0, 1, 100, 98, math.inf, 1, 96, entry_type="market", trail_atr=1.0)
    rows = [[100, 100, 100, 100], [100, 101, 99.5, 101], [101, 105, 101, 105],
            [105, 105, 103.5, 104], [104, 104, 103, 103.5]]
    t = run_backtest({"X": _f(rows)}, {"X": [it]}, {}, RISK, ZERO).trades.iloc[0]
    # after bar 2 close 105 -> stop 104; bar 3 low 103.5 hits it
    assert t.reason == "stop" and t.exit == pytest.approx(104)
    assert t.r == pytest.approx((104 - 100) / 2)          # R uses the initial stop


def test_sub_bots_are_independent(bars):
    """Running sub-bots together must give exactly what each gives alone."""
    fu = {"X": make_funding(bars)}
    feats = build_features_all({"X": bars}, fu, CFG)
    specs = build_sleeves(feats, CFG)
    together = run_portfolio(feats, specs, fu, RISK, CostModel.from_config(CFG["costs"]))
    for sp in specs:
        alone = run_portfolio(feats, [sp], fu, RISK, CostModel.from_config(CFG["costs"]))
        a, b = together.sleeves[sp.name], alone.sleeves[sp.name]
        assert a.initial_equity == RISK["initial_equity"]            # full account each
        assert np.allclose(a.equity.to_numpy(), b.equity.to_numpy())
        assert len(a.trades) == len(b.trades)
