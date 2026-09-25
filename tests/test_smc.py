import numpy as np
import pandas as pd

from pte.features.smc import fvg, structure, swings


def _df(highs, lows, closes=None, opens=None):
    n = len(highs)
    closes = closes if closes is not None else [(h + l) / 2 for h, l in zip(highs, lows)]
    opens = opens if opens is not None else closes
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}, index=idx, dtype=float)


def test_swing_high_recorded_at_confirmation_bar():
    h = np.array([1, 2, 3, 9, 3, 2, 1, 1], float)
    l = h - 0.5
    sh, sl = swings(h, l, 3)
    assert np.isnan(sh[3])          # not known on the swing bar itself
    assert sh[6] == 9               # known after 3 right-hand bars close


def test_equal_highs_are_not_swings():
    h = np.array([1, 2, 5, 5, 2, 1, 1, 1], float)
    sh, _ = swings(h, h - 1, 2)
    assert np.isnan(sh).all()


def test_choch_then_bos():
    # swing high 10 at bar 3 (confirmed bar 5); close 11 at bar 7 -> CHoCH up (from unknown).
    highs = [5, 6, 7, 10, 7, 6, 8, 11.5, 9, 8, 7, 13, 12, 11, 14.5]
    lows = [h - 2 for h in highs]
    closes = list(highs); closes[7] = 11; closes[14] = 14
    df = _df(highs, lows, closes)
    st = structure(df, 2)
    assert st["choch_up"].iloc[7] and st["trend"].iloc[7] == 1
    # swing high 13 at bar 11 confirmed at bar 13; close 14 on bar 14 -> BOS up
    assert st["bos_up"].iloc[14]


def test_sweep_requires_close_back_inside():
    highs = [10, 9, 8, 9, 10, 10, 10.5]
    lows = [8, 7, 5, 7, 8, 8, 4.5]          # swing low 5 at bar 2 confirmed bar 4
    closes = [9, 8, 6, 8, 9, 9, 6]          # bar 6 wicks to 4.5, closes 6 > 5 -> sweep
    st = structure(_df(highs, lows, closes), 2)
    assert st["sweep_up"].iloc[6] and st["sweep_level"].iloc[6] == 5
    assert not st["choch_dn"].iloc[6]


def test_close_below_is_break_not_sweep():
    highs = [10, 9, 8, 9, 10, 10, 10]
    lows = [8, 7, 5, 7, 8, 8, 4]
    closes = [9, 8, 6, 8, 9, 9, 4.5]
    st = structure(_df(highs, lows, closes), 2)
    assert st["choch_dn"].iloc[6] and not st["sweep_up"].iloc[6]


def test_bullish_fvg_zone():
    df = _df([10, 12, 15], [9, 10, 11])
    atr = pd.Series(1.0, index=df.index)
    g = fvg(df, atr, 0.1)
    assert g["fvg_up"].iloc[2]
    assert g["fvg_up_lo"].iloc[2] == 10 and g["fvg_up_hi"].iloc[2] == 11
    assert not g["fvg_dn"].any()


def test_fvg_min_size_filter():
    df = _df([10, 12, 15], [9, 10, 10.05])
    g = fvg(df, pd.Series(1.0, index=df.index), 0.1)
    assert not g["fvg_up"].iloc[2]
