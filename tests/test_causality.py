"""Look-ahead guard: every feature and signal at bar i must be identical whether
or not bars after i exist. If any future bar changes a past value, the feature
peeks ahead."""

import numpy as np
import pandas as pd
import pytest

from pte.config import load_config
from pte.data.synthetic import make_bars
from pte.features.indicators import elevated_vol
from pte.features.smc import build_features
from pte.strategies.smc_m15 import generate_signals

CFG = load_config()


@pytest.fixture(scope="module")
def bars():
    return make_bars(4000, seed=7)


@pytest.mark.parametrize("cut", [1500, 2333, 3001])
def test_features_are_causal(bars, cut):
    full = build_features(bars, CFG["smc"])
    part = build_features(bars.iloc[:cut], CFG["smc"])
    a, b = full.iloc[:cut], part
    for col in part.columns:
        x, y = a[col].to_numpy(), b[col].to_numpy()
        if x.dtype.kind == "f":
            assert np.allclose(x, y, equal_nan=True), f"{col} changes when future bars are added"
        else:
            assert (x == y).all(), f"{col} changes when future bars are added"


def test_elevated_vol_is_causal(bars):
    f = build_features(bars, CFG["smc"])
    full = elevated_vol(f["atr"], 800, 0.8)
    part = elevated_vol(f["atr"].iloc[:2000], 800, 0.8)
    assert (full.iloc[:2000] == part).all()


@pytest.mark.parametrize("cut", [1800, 3200])
def test_signals_are_causal(bars, cut):
    full = generate_signals(build_features(bars, CFG["smc"]), CFG["strategy"])
    part = generate_signals(build_features(bars.iloc[:cut], CFG["smc"]), CFG["strategy"])
    assert [s for s in full if s.signal_idx < cut] == part


def test_htf_bias_waits_for_bar_close():
    # A 1h bar starting 10:00 is only known at 11:00, i.e. from the M15 bar opening 10:45.
    idx = pd.date_range("2024-01-01", periods=400, freq="15min", tz="UTC")
    df = make_bars(400, seed=3).set_axis(idx)
    f = build_features(df, CFG["smc"])
    from pte.features.smc import htf_bias
    full = htf_bias(df, "1h", 3)
    # truncate so the last 1h bar is incomplete: its trend must not appear
    part = htf_bias(df.iloc[:-2], "1h", 3)
    assert (full.iloc[:-2].to_numpy() == part.to_numpy()).all()
    assert len(f) == 400
