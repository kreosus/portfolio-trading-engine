import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from pte.config import load_config
from pte.data import forward
from pte.data.synthetic import make_bars
from pte.paper import entry_hash, run_entries, write_report


def test_funding_estimate_formula():
    idx = pd.date_range("2026-09-01", periods=480, freq="1min", tz="UTC")
    t = pd.DatetimeIndex([pd.Timestamp("2026-09-01 08:00", tz="UTC")])
    # small premium: clamp absorbs it -> exactly the 0.01% baseline
    assert forward.estimate_funding(pd.Series(0.0002, index=idx), t).iloc[0] == 0.0001
    # large premium: P + clamp(I - P) = P - 0.0005 + ... -> 0.001 + (-0.0005)
    assert forward.estimate_funding(pd.Series(0.001, index=idx), t).iloc[0] == pytest.approx(0.0005)
    # later samples weigh more
    s = pd.Series(np.r_[np.zeros(240), np.full(240, 0.002)], index=idx)
    # linear weights put 75% of the weight on the second half: P = 0.0015 -> F = 0.0010 (unweighted would give 0.0005)
    assert forward.estimate_funding(s, t).iloc[0] == pytest.approx(0.0010, abs=2e-5)


def _zip(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("x.csv", "\n".join(",".join(map(str, r)) for r in rows) + "\n")


def _kline_rows(df):
    ms = df.index.as_unit("ms").asi8
    return [[t, o, h, l, c, v, t + 899_999, v * c, 10, v / 2, v * c / 2, 0]
            for t, o, h, l, c, v in zip(ms, df.open, df.high, df.low, df.close, df.volume)]


def test_load_funding_actual_then_estimated(tmp_path):
    raw = tmp_path
    _zip(raw / "fundingRate" / "X" / "X-fundingRate-2026-08.zip",
         [["calc_time", "funding_interval_hours", "last_funding_rate"],
          [int(pd.Timestamp("2026-08-31 16:00", tz="UTC").value // 10**6), 8, 0.0001]])
    idx = pd.date_range("2026-08-31 16:00", "2026-09-01 23:59", freq="1min", tz="UTC")
    rows = [[int(t.value // 10**6), 0, 0, 0, 0.0003, 0, 0, 0, 0, 0, 0, 0] for t in idx]
    _zip(raw / "premium_daily" / "X" / "X-1m-2026-09-01.zip", rows)
    f = forward.load_funding("X", raw)
    assert not f.loc[pd.Timestamp("2026-08-31 16:00", tz="UTC"), "estimated"]
    est = f[f.estimated]
    assert est.index[0] == pd.Timestamp("2026-09-01 00:00", tz="UTC")
    assert (est.rate == 0.0001).all() and len(est) == 4   # 00:00, 08:00, 16:00 and 2026-09-02 00:00


def test_paper_end_to_end_on_synthetic_archives(tmp_path):
    cfg = load_config()
    raw = tmp_path / "raw"
    b = make_bars(4 * 24 * 400, start="2025-08-01", seed=4)
    for sym in ("BTCUSDT", "ETHUSDT"):
        for m, g in b.groupby(b.index.strftime("%Y-%m")):
            _zip(raw / "klines" / sym / f"{sym}-15m-{m}.zip", _kline_rows(g))
    bars15 = {s: forward.load_klines(s, raw) for s in ("BTCUSDT", "ETHUSDT")}
    assert len(bars15["BTCUSDT"]) == len(b)
    reg = {"id": "t", "symbols": ["BTCUSDT", "ETHUSDT"], "history_start": "2025-08",
           "review": {"date": "2027-03-01"},
           "entries": [{"id": "vb4h", "sub_bot": "vol_breakout", "timeframe": "4h", "scoring_start": "2026-06-01",
                        "entry_hash": entry_hash(cfg, "vol_breakout", "4h")}]}
    res = run_entries(reg, cfg, bars15, {})
    st = write_report(reg, res, bars15, {}, {}, tmp_path / "out")
    assert (tmp_path / "out" / "README.md").exists() and (tmp_path / "out" / "vb4h" / "trades.csv").exists()
    assert st["entries"][0]["id"] == "vb4h"
    # rules changed -> refuse
    reg["entries"][0]["entry_hash"] = "deadbeef0000"
    with pytest.raises(SystemExit):
        run_entries(reg, cfg, bars15, {})
