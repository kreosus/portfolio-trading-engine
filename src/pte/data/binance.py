"""Download Binance USD-M perpetual archives (data.binance.vision).

Raw zips are kept exactly as downloaded (immutable). Processing to Parquet is a
separate, repeatable step so every research result can be rebuilt from raw.
"""

from __future__ import annotations

import hashlib
import io
import logging
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE = "https://data.binance.vision/data/futures/um/monthly"
KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]


def month_range(start: str, end: str | None = None) -> list[str]:
    """Months from start (YYYY-MM) through the last fully completed month."""
    if end is None:
        today = date.today()
        last = (pd.Timestamp(today.year, today.month, 1) - pd.offsets.MonthBegin(1))
        end = last.strftime("%Y-%m")
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def _url(kind: str, symbol: str, interval: str, month: str) -> str:
    if kind == "klines":
        return f"{BASE}/klines/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    if kind == "fundingRate":
        return f"{BASE}/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"
    raise ValueError(kind)


def _download(url: str, dest: Path, session: requests.Session) -> bool:
    if dest.exists():
        return True
    r = session.get(url, timeout=60)
    if r.status_code == 404:
        log.warning("missing: %s", url)
        return False
    r.raise_for_status()
    body = r.content
    c = session.get(url + ".CHECKSUM", timeout=30)
    if c.ok:
        expected = c.text.split()[0].strip()
        actual = hashlib.sha256(body).hexdigest()
        if expected != actual:
            raise IOError(f"checksum mismatch for {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(body)
    tmp.rename(dest)
    return True


def download(symbols: list[str], interval: str, start: str, raw_dir: str | Path,
             end: str | None = None) -> None:
    raw = Path(raw_dir)
    s = requests.Session()
    for sym in symbols:
        for m in month_range(start, end):
            for kind in ("klines", "fundingRate"):
                url = _url(kind, sym, interval, m)
                dest = raw / kind / sym / Path(url).name
                ok = _download(url, dest, s)
                log.info("%s %s %s %s", sym, kind, m, "ok" if ok else "missing")


def _read_zip_csv(path: Path, columns: list[str] | None) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        raw = z.read(name)
    first = raw.split(b"\n", 1)[0]
    has_header = not first[:1].isdigit()
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    if columns is not None:
        df = df.iloc[:, : len(columns)]
        df.columns = columns
    return df


def _to_utc(ts: pd.Series) -> pd.Series:
    ts = ts.astype("int64")
    unit = np.where(ts > 10**14, "us", "ms")  # guard: some newer archives use microseconds
    ms = np.where(unit == "us", ts // 1000, ts)
    return pd.to_datetime(ms, unit="ms", utc=True)


def process_klines(symbol: str, raw_dir: str | Path, processed_dir: str | Path) -> pd.DataFrame:
    files = sorted((Path(raw_dir) / "klines" / symbol).glob("*.zip"))
    if not files:
        raise FileNotFoundError(f"No raw klines for {symbol}. Run `pte download` first.")
    df = pd.concat([_read_zip_csv(f, KLINE_COLS) for f in files], ignore_index=True)
    df["open_time"] = _to_utc(df["open_time"])
    df = df[["open_time", "open", "high", "low", "close", "volume", "quote_volume", "count"]]
    df = df.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")
    df = df.astype({"open": float, "high": float, "low": float, "close": float,
                    "volume": float, "quote_volume": float, "count": int})
    _report_gaps(df, symbol)
    out = Path(processed_dir) / f"{symbol}_15m.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return df


def process_funding(symbol: str, raw_dir: str | Path, processed_dir: str | Path) -> pd.DataFrame:
    files = sorted((Path(raw_dir) / "fundingRate" / symbol).glob("*.zip"))
    if not files:
        raise FileNotFoundError(f"No raw funding for {symbol}.")
    parts = []
    for f in files:
        d = _read_zip_csv(f, None)
        d.columns = [c.strip().lower() for c in d.columns]
        time_col = "calc_time" if "calc_time" in d.columns else d.columns[0]
        rate_col = "last_funding_rate" if "last_funding_rate" in d.columns else d.columns[-1]
        parts.append(pd.DataFrame({"time": _to_utc(d[time_col]), "rate": d[rate_col].astype(float)}))
    df = pd.concat(parts).drop_duplicates("time").sort_values("time")
    # Settlements are on the hour; calc_time can carry a few ms of jitter.
    df["time"] = df["time"].dt.round("1min")
    df = df.set_index("time")
    out = Path(processed_dir) / f"{symbol}_funding.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return df


def _report_gaps(df: pd.DataFrame, symbol: str) -> None:
    diffs = df.index.to_series().diff().dropna()
    gaps = diffs[diffs > pd.Timedelta("15min")]
    if len(gaps):
        log.warning("%s: %d gaps in 15m data (largest %s)", symbol, len(gaps), gaps.max())
