"""Forward (paper-trading) data from data.binance.vision.

Binance's live futures API is geo-blocked in some regions (including the US);
the public archive is not. The archive publishes:
  - daily 15m klines with about a one-day lag,
  - daily 1m premium-index klines,
  - funding rates only as a MONTHLY file, after the month ends.

So for months without a published funding file, funding is ESTIMATED from the
1m premium index using Binance's formula:
    P = time-weighted average premium index over the 8h interval
        (weights 1..n, later samples weigh more)
    F = round(P + clamp(0.01% - P, -0.05%, +0.05%), 8 decimals)
Checked against real rates for Jul-Aug 2026: same side of the 0.01% baseline in
98-100% of settlements. Estimated rows are flagged, and results that depend on
them are provisional until the monthly file replaces them.
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .binance import KLINE_COLS, _read_zip_csv, _to_utc

log = logging.getLogger(__name__)
BASE = "https://data.binance.vision/data/futures/um"
INTEREST = 0.0001
CLAMP = 0.0005


def _get(session: requests.Session, url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    r = session.get(url, timeout=60)
    if r.status_code == 404:
        return False
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.replace(dest)
    return True


def sync(symbols: list[str], start_month: str, raw_dir: str | Path, today: date | None = None) -> dict:
    """Download everything needed from start_month through yesterday. Returns coverage info."""
    today = today or date.today()
    raw = Path(raw_dir)
    s = requests.Session()
    cur = pd.Period(today, "M")
    info = {}
    for sym in symbols:
        months_daily, funding_missing = [], []
        for m in pd.period_range(start_month, cur, freq="M"):
            ms = m.strftime("%Y-%m")
            got_k = m < cur and _get(s, f"{BASE}/monthly/klines/{sym}/15m/{sym}-15m-{ms}.zip",
                                     raw / "klines" / sym / f"{sym}-15m-{ms}.zip")
            got_f = m < cur and _get(s, f"{BASE}/monthly/fundingRate/{sym}/{sym}-fundingRate-{ms}.zip",
                                     raw / "fundingRate" / sym / f"{sym}-fundingRate-{ms}.zip")
            if got_k and got_f:
                continue
            days = pd.date_range(m.start_time, min(m.end_time, pd.Timestamp(today) - pd.Timedelta(days=1)), freq="D")
            for d in days:
                ds = d.strftime("%Y-%m-%d")
                if not got_k:
                    _get(s, f"{BASE}/daily/klines/{sym}/15m/{sym}-15m-{ds}.zip",
                         raw / "klines_daily" / sym / f"{sym}-15m-{ds}.zip")
                if not got_f:
                    _get(s, f"{BASE}/daily/premiumIndexKlines/{sym}/1m/{sym}-1m-{ds}.zip",
                         raw / "premium_daily" / sym / f"{sym}-1m-{ds}.zip")
            if not got_k:
                months_daily.append(ms)
            if not got_f:
                funding_missing.append(ms)
        info[sym] = {"klines_from_daily": months_daily, "funding_estimated_months": funding_missing}
    return info


def load_klines(sym: str, raw_dir: str | Path) -> pd.DataFrame:
    raw = Path(raw_dir)
    files = sorted((raw / "klines" / sym).glob("*.zip")) + sorted((raw / "klines_daily" / sym).glob("*.zip"))
    if not files:
        raise FileNotFoundError(f"no klines for {sym}")
    df = pd.concat([_read_zip_csv(f, KLINE_COLS) for f in files], ignore_index=True)
    df["open_time"] = _to_utc(df["open_time"])
    df = df[["open_time", "open", "high", "low", "close", "volume", "quote_volume", "count"]]
    df = df.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")
    return df.astype({"open": float, "high": float, "low": float, "close": float,
                      "volume": float, "quote_volume": float, "count": int})


def estimate_funding(premium_1m: pd.Series, settlements: pd.DatetimeIndex) -> pd.Series:
    """Binance funding estimate at each settlement from 1m premium-index closes (indexed by open time)."""
    out = {}
    for t in settlements:
        w = premium_1m[(premium_1m.index >= t - pd.Timedelta("8h")) & (premium_1m.index < t)]
        if len(w) < 470:
            continue
        weights = np.arange(1, len(w) + 1)
        p = float(np.dot(weights, w.to_numpy()) / weights.sum())
        out[t] = round(p + min(max(INTEREST - p, -CLAMP), CLAMP), 8)
    return pd.Series(out, name="rate", dtype=float)


def load_funding(sym: str, raw_dir: str | Path) -> pd.DataFrame:
    """Actual funding where published, estimated after that. Column `estimated` flags estimates."""
    raw = Path(raw_dir)
    parts = []
    for f in sorted((raw / "fundingRate" / sym).glob("*.zip")):
        d = _read_zip_csv(f, None)
        d.columns = [c.strip().lower() for c in d.columns]
        tcol = "calc_time" if "calc_time" in d.columns else d.columns[0]
        rcol = "last_funding_rate" if "last_funding_rate" in d.columns else d.columns[-1]
        parts.append(pd.DataFrame({"time": pd.DatetimeIndex(_to_utc(d[tcol])).round("1min"), "rate": d[rcol].astype(float).to_numpy()}))
    actual = (pd.concat(parts).drop_duplicates("time").set_index("time").sort_index()
              if parts else pd.DataFrame({"rate": []}, index=pd.DatetimeIndex([], tz="UTC", name="time")))
    actual["estimated"] = False
    pfiles = sorted((raw / "premium_daily" / sym).glob("*.zip"))
    if not pfiles:
        return actual
    p = pd.concat([_read_zip_csv(f, KLINE_COLS) for f in pfiles], ignore_index=True)
    p["open_time"] = _to_utc(p["open_time"])
    prem = p.drop_duplicates("open_time").set_index("open_time").sort_index()["close"].astype(float)
    first = actual.index.max() + pd.Timedelta("8h") if len(actual) else prem.index.min().ceil("8h")
    last = (prem.index.max() + pd.Timedelta("1min")).floor("8h")
    settles = pd.date_range(first, last, freq="8h") if last >= first else pd.DatetimeIndex([], tz="UTC")
    est = estimate_funding(prem, settles)
    if len(est):
        e = pd.DataFrame({"rate": est, "estimated": True})
        e.index.name = "time"
        actual = pd.concat([actual, e]).sort_index()
    return actual
