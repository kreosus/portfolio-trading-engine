import zipfile

import pandas as pd

from pte.data import binance
from pte.data.store import split_holdout
from pte.data.synthetic import make_bars

ROW = "{t},100,101,99,100.5,10,{t2},1000,5,4,400,0\n"


def _zip(path, name, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, body)


def test_process_klines_with_and_without_header(tmp_path):
    t0 = 1704067200000  # 2024-01-01 00:00 UTC
    no_hdr = "".join(ROW.format(t=t0 + k * 900_000, t2=t0 + k * 900_000 + 899_999) for k in range(3))
    hdr = ",".join(binance.KLINE_COLS) + "\n" + "".join(
        ROW.format(t=t0 + k * 900_000, t2=0) for k in range(3, 5))
    raw = tmp_path / "raw" / "klines" / "BTCUSDT"
    _zip(raw / "BTCUSDT-15m-2024-01.zip", "a.csv", no_hdr)
    _zip(raw / "BTCUSDT-15m-2024-02.zip", "b.csv", hdr)
    df = binance.process_klines("BTCUSDT", tmp_path / "raw", tmp_path / "proc")
    assert len(df) == 5
    assert df.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert df.index[-1] == pd.Timestamp("2024-01-01 01:00", tz="UTC")
    assert (tmp_path / "proc" / "BTCUSDT_15m.parquet").exists()


def test_process_funding_rounds_calc_time(tmp_path):
    body = "calc_time,funding_interval_hours,last_funding_rate\n1704067200005,8,0.0001\n1704096000003,8,-0.0002\n"
    _zip(tmp_path / "raw" / "fundingRate" / "BTCUSDT" / "f.zip", "f.csv", body)
    fu = binance.process_funding("BTCUSDT", tmp_path / "raw", tmp_path / "proc")
    assert list(fu.index) == [pd.Timestamp("2024-01-01 00:00", tz="UTC"), pd.Timestamp("2024-01-01 08:00", tz="UTC")]
    assert fu.rate.iloc[1] == -0.0002


def test_holdout_is_whole_recent_months():
    df = make_bars(3000 * 14, start="2024-01-01")
    dev, hold = split_holdout(df, 12)
    assert hold.index[0].day == 1 and hold.index[0].hour == 0
    assert dev.index[-1] < hold.index[0]
    last = df.index[-1].tz_convert(None)
    assert hold.index[0].tz_convert(None).to_period("M") == (last - pd.DateOffset(months=11)).to_period("M")


def test_month_range():
    assert binance.month_range("2024-11", "2025-02") == ["2024-11", "2024-12", "2025-01", "2025-02"]
