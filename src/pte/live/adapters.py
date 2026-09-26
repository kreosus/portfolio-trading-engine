"""Exchange and market-data adapters (spec §3, §24, §40).

Strategy code never imports this module's HTTP details; it gets DataFrames and quotes.
Only the ExecutionEngine calls the order methods.

Market data:
  CoinbaseMarketData  public BTC-USD candles + ticker (US-accessible; no funding/OI: spot)
  BinanceMarketData   public USD-M futures data (geo-restricted in some regions, incl. the US)
  SnapshotMarketData  a JSON snapshot captured elsewhere (used for demos and tests)

Order venues:
  SimulatedExchange     shadow mode: real prices, simulated fills, no real order (spec §38)
  BinanceFuturesAdapter Binance Futures DEMO environment (fake funds). Mainnet is refused.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import time
import urllib.parse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .models import Order, OrderState

GRAN = {"15m": 900, "1h": 3600}


# ----------------------------------------------------------------------------- market data

def _cb_rows_to_df(rows: list, gran_s: int, now: pd.Timestamp) -> pd.DataFrame:
    """Coinbase rows [time, low, high, open, close, volume] -> closed bars only."""
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close", "volume"]).drop_duplicates("t")
    df.index = pd.to_datetime(df.pop("t"), unit="s", utc=True)
    df.index.name = "open_time"
    df = df.sort_index()[["open", "high", "low", "close", "volume"]].astype(float)
    closed = df.index + pd.Timedelta(seconds=gran_s) <= now      # drop the bar still forming
    return df[closed]


class SnapshotMarketData:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.d = json.loads(self.path.read_text())
        self.now = pd.Timestamp(self.d["fetched_at"])
        self.source = f"snapshot:{self.d.get('source', '?')}"

    def candles(self, tf: str) -> pd.DataFrame:
        return _cb_rows_to_df(self.d[{"15m": "m15", "1h": "h1"}[tf]], GRAN[tf], self.now)

    def quote(self) -> dict:
        t = self.d["ticker"]
        return {"bid": float(t["bid"]), "ask": float(t["ask"]), "time": pd.Timestamp(t["time"])}

    def derivatives(self) -> dict | None:
        return self.d.get("derivatives")

    def calendar(self) -> list | None:
        return self.d.get("calendar")

    def clock(self) -> pd.Timestamp:
        return self.now


class CoinbaseMarketData:
    BASE = "https://api.exchange.coinbase.com/products/BTC-USD"

    def __init__(self, session=None):
        import requests
        self.s = session or requests.Session()
        self.source = "coinbase BTC-USD (spot, public)"

    def _get(self, path: str, **params):
        r = self.s.get(self.BASE + path, params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def candles(self, tf: str, n: int = 1000) -> pd.DataFrame:
        gran = GRAN[tf]
        end = pd.Timestamp.now(tz="UTC").floor(f"{gran}s")
        rows = []
        while len(rows) < n:
            start = end - pd.Timedelta(seconds=gran * 300)
            chunk = self._get("/candles", granularity=gran, start=start.isoformat(), end=end.isoformat())
            if not chunk:
                break
            rows += chunk
            end = start
        return _cb_rows_to_df(rows, gran, pd.Timestamp.now(tz="UTC"))

    def quote(self) -> dict:
        t = self._get("/ticker")
        return {"bid": float(t["bid"]), "ask": float(t["ask"]), "time": pd.Timestamp(t["time"])}

    def derivatives(self) -> dict | None:
        return None        # spot venue: no funding / open interest

    def calendar(self) -> list | None:
        return None

    def clock(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="UTC")


class BinanceMarketData:
    """Public USD-M futures data. Binance refuses restricted locations (HTTP 451 / code 0)."""

    def __init__(self, base_url: str = "https://fapi.binance.com", symbol: str = "BTCUSDT", session=None):
        import requests
        self.s = session or requests.Session()
        self.base, self.symbol = base_url, symbol
        self.source = f"binance {base_url}"

    def _get(self, path, **params):
        r = self.s.get(self.base + path, params=params, timeout=15)
        if r.status_code == 451 or (r.ok and isinstance(r.json(), dict) and r.json().get("code") == 0):
            raise PermissionError("Binance: service unavailable from a restricted location")
        r.raise_for_status()
        return r.json()

    def candles(self, tf: str, n: int = 1000) -> pd.DataFrame:
        rows = self._get("/fapi/v1/klines", symbol=self.symbol, interval=tf, limit=min(n, 1500))
        df = pd.DataFrame([r[:6] for r in rows], columns=["t", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df.pop("t"), unit="ms", utc=True)
        df = df.astype(float)
        now = pd.Timestamp.now(tz="UTC")
        return df[df.index + pd.Timedelta(seconds=GRAN[tf]) <= now]

    def quote(self) -> dict:
        t = self._get("/fapi/v1/ticker/bookTicker", symbol=self.symbol)
        return {"bid": float(t["bidPrice"]), "ask": float(t["askPrice"]),
                "time": pd.Timestamp(int(t["time"]), unit="ms", tz="UTC")}

    def derivatives(self) -> dict | None:
        p = self._get("/fapi/v1/premiumIndex", symbol=self.symbol)
        oi = self._get("/fapi/v1/openInterest", symbol=self.symbol)
        return {"funding": float(p["lastFundingRate"]), "mark": float(p["markPrice"]),
                "index": float(p["indexPrice"]), "oi": float(oi["openInterest"])}

    def calendar(self) -> list | None:
        return None

    def clock(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="UTC")


# ----------------------------------------------------------------------------- order venues

def round_step(x: float, step: float) -> float:
    """Round DOWN to the exchange step (never round size up past max risk, spec §22)."""
    return round(math.floor(x / step + 1e-9) * step, 8)


def round_tick(price: float, tick: float) -> float:
    return round(round(price / tick) * tick, 10)


class SimulatedExchange:
    """Shadow-mode venue. Keeps orders and one net position, fills against real prices.

    Fills: MARKET at ask/bid plus slippage (taker). LIMIT when the bar trades THROUGH the price
    (maker). STOP_MARKET / TAKE_PROFIT_MARKET when touched; stop and TP in the same bar -> stop.
    State persists to a JSON file so consecutive shadow cycles continue the same account.
    """

    name = "simulated"

    def __init__(self, rules: dict, state_path: str | Path | None = None, slippage_bps: float = 2.0):
        self.rules = rules
        self.slip = slippage_bps * 1e-4
        self.path = Path(state_path) if state_path else None
        self.orders: dict[str, Order] = {}
        self.position = {"qty": 0.0, "entry": 0.0}      # signed qty
        self.fills: list[dict] = []
        if self.path and self.path.exists():
            self._load()

    # -- persistence
    def _load(self):
        d = json.loads(self.path.read_text())
        for o in d["orders"]:
            o["state"] = OrderState(o["state"])
            self.orders[o["client_id"]] = Order(**o)
        self.position, self.fills = d["position"], d["fills"]

    def save(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        d = {"orders": [{**asdict(o), "state": o.state.value} for o in self.orders.values()],
             "position": self.position, "fills": self.fills}
        self.path.write_text(json.dumps(d, indent=1, default=str))

    # -- exchange API surface (what the ExecutionEngine calls)
    def exchange_rules(self, symbol: str) -> dict:
        return dict(self.rules)

    def submit(self, order: Order) -> dict:
        if order.qty < self.rules["min_qty"] - 1e-12:
            return {"status": "REJECTED", "reason": "qty below min_qty"}
        order.exchange_id = "sim-" + order.client_id[-8:]
        self.orders[order.client_id] = order
        return {"status": "NEW", "orderId": order.exchange_id}

    def cancel(self, client_id: str) -> dict:
        o = self.orders.get(client_id)
        if o and o.open:
            o.transition(OrderState.CANCELLED, "cancel requested")
        return {"status": o.state.value if o else "UNKNOWN"}

    def query(self, client_id: str) -> dict:
        o = self.orders.get(client_id)
        return {"status": o.state.value, "executedQty": o.filled_qty, "avgPrice": o.avg_price} if o else \
            {"status": "UNKNOWN"}

    def position_risk(self, symbol: str) -> dict:
        return dict(self.position)

    # -- market simulation
    def _fill(self, o: Order, price: float, ts, taker: bool):
        fee_rate = self.rules["taker_fee"] if taker else self.rules["maker_fee"]
        signed = o.qty if o.side == "BUY" else -o.qty
        q0, e0 = self.position["qty"], self.position["entry"]
        realized = 0.0
        if q0 == 0 or (q0 > 0) == (signed > 0):
            q1 = q0 + signed
            self.position = {"qty": q1, "entry": (q0 * e0 + signed * price) / q1 if q1 else 0.0}
        else:
            closing = min(abs(signed), abs(q0))
            realized = closing * (price - e0) * (1 if q0 > 0 else -1)
            q1 = q0 + signed
            self.position = {"qty": q1, "entry": e0 if q1 and (q1 > 0) == (q0 > 0) else (price if q1 else 0.0)}
        o.filled_qty, o.avg_price = o.qty, price
        if o.state == OrderState.ACKNOWLEDGED:
            o.transition(OrderState.FILLED, f"filled @ {price:.2f}")
        fee = abs(o.qty * price) * fee_rate
        self.fills.append({"t": str(ts), "client_id": o.client_id, "purpose": o.purpose, "side": o.side,
                           "qty": o.qty, "price": price, "fee": fee, "realized": realized})

    def acknowledge_all(self):
        for o in self.orders.values():
            if o.state == OrderState.SUBMITTED:
                o.transition(OrderState.ACKNOWLEDGED, "venue ack")

    def process_bar(self, ts, o_: float, h: float, l: float, c: float) -> list[dict]:
        """Run resting orders against one closed bar. Returns fills that happened."""
        n0 = len(self.fills)
        self.acknowledge_all()
        live = [o for o in self.orders.values() if o.state == OrderState.ACKNOWLEDGED]
        # protective orders first (conservative: stop before take-profit inside one bar)
        live.sort(key=lambda o: {"stop": 0, "close": 0, "tp1": 1, "tp2": 1, "tp3": 1, "entry": 2}.get(o.purpose, 3))
        for o in live:
            if o.state != OrderState.ACKNOWLEDGED:
                continue
            if o.reduce_only:
                if abs(self.position["qty"]) < 1e-12:
                    o.transition(OrderState.CANCELLED, "reduce-only with no position")
                    continue
                o.qty = min(o.qty, abs(self.position["qty"]))    # reduce-only never flips a position
            buy = o.side == "BUY"
            if o.type == "MARKET":
                self._fill(o, o_ * (1 + self.slip) if buy else o_ * (1 - self.slip), ts, True)
            elif o.type == "LIMIT":
                if (buy and l < o.price) or (not buy and h > o.price):
                    self._fill(o, min(o_, o.price) if buy else max(o_, o.price), ts, False)
            elif o.type == "STOP_MARKET":
                if (buy and h >= o.stop_price) or (not buy and l <= o.stop_price):
                    px = max(o_, o.stop_price) if buy else min(o_, o.stop_price)
                    self._fill(o, px * (1 + self.slip) if buy else px * (1 - self.slip), ts, True)
            elif o.type == "TAKE_PROFIT_MARKET":
                if (buy and l <= o.stop_price) or (not buy and h >= o.stop_price):
                    self._fill(o, o.stop_price, ts, True)
            if o.reduce_only and o.state == OrderState.FILLED:
                self._clip_reduce_only()
        return self.fills[n0:]

    def market_now(self, o: Order, bid: float, ask: float, ts) -> dict:
        """Immediate market fill at the live quote (used for closes)."""
        self.acknowledge_all()
        px = ask * (1 + self.slip) if o.side == "BUY" else bid * (1 - self.slip)
        self._fill(o, px, ts, True)
        return self.fills[-1]

    def _clip_reduce_only(self):
        # a reduce-only order may not flip the position
        if abs(self.position["qty"]) < 1e-12:
            self.position = {"qty": 0.0, "entry": 0.0}


class BinanceFuturesAdapter:
    """Binance USD-M Futures REST, DEMO environment only (fake funds).

    Keys are read from environment variables named in config (never from files in the repo, never
    typed into anything by the bot). Binance restricts some regions, including the US, for both the
    live and demo environments; calls from a restricted location raise PermissionError.
    """

    name = "binance-demo"

    def __init__(self, cfg: dict, session=None):
        import requests
        b = cfg["binance"]
        self.base = b["demo_base_url"]
        if "demo" not in self.base and "testnet" not in self.base:
            raise PermissionError("This build only talks to the Binance demo/test environment.")
        self.key = os.environ.get(b["api_key_env"], "")
        self.secret = os.environ.get(b["api_secret_env"], "")
        if not self.key or not self.secret:
            raise RuntimeError(f"Set {b['api_key_env']} and {b['api_secret_env']} for demo mode.")
        self.s = session or requests.Session()
        self.s.headers["X-MBX-APIKEY"] = self.key

    @staticmethod
    def sign(query: str, secret: str) -> str:
        return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()

    def _req(self, method: str, path: str, signed: bool = False, **params):
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params.setdefault("recvWindow", 5000)
            q = urllib.parse.urlencode(params)
            q += "&signature=" + self.sign(q, self.secret)
        else:
            q = urllib.parse.urlencode(params)
        r = self.s.request(method, f"{self.base}{path}?{q}", timeout=15)
        if r.status_code == 451:
            raise PermissionError("Binance: restricted location")
        data = r.json()
        if isinstance(data, dict) and data.get("code") not in (None, 200) and "orderId" not in data:
            if data.get("code") == 0:
                raise PermissionError(data.get("msg", "restricted"))
            raise RuntimeError(f"Binance error {data.get('code')}: {data.get('msg')}")
        return data

    def exchange_rules(self, symbol: str) -> dict:
        info = self._req("GET", "/fapi/v1/exchangeInfo")
        s = next(x for x in info["symbols"] if x["symbol"] == symbol)
        f = {x["filterType"]: x for x in s["filters"]}
        return {"tick_size": float(f["PRICE_FILTER"]["tickSize"]), "step_size": float(f["LOT_SIZE"]["stepSize"]),
                "min_qty": float(f["LOT_SIZE"]["minQty"]), "min_notional": float(f["MIN_NOTIONAL"]["notional"]),
                "maker_fee": 0.0002, "taker_fee": 0.0005}

    def set_leverage(self, symbol: str, leverage: int):
        return self._req("POST", "/fapi/v1/leverage", True, symbol=symbol, leverage=int(leverage))

    def submit(self, o: Order) -> dict:
        p = {"symbol": o.symbol, "side": o.side, "type": o.type, "newClientOrderId": o.client_id}
        if o.type in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
            p.update(stopPrice=o.stop_price, closePosition="false", quantity=o.qty, reduceOnly="true",
                     workingType="MARK_PRICE")
        else:
            p["quantity"] = o.qty
            if o.type == "LIMIT":
                p.update(price=o.price, timeInForce="GTC")
            if o.reduce_only:
                p["reduceOnly"] = "true"
        return self._req("POST", "/fapi/v1/order", True, **p)

    def cancel(self, client_id: str) -> dict:
        return self._req("DELETE", "/fapi/v1/order", True, symbol="BTCUSDT", origClientOrderId=client_id)

    def query(self, client_id: str) -> dict:
        return self._req("GET", "/fapi/v1/order", True, symbol="BTCUSDT", origClientOrderId=client_id)

    def position_risk(self, symbol: str) -> dict:
        r = self._req("GET", "/fapi/v2/positionRisk", True, symbol=symbol)
        p = r[0] if r else {"positionAmt": 0, "entryPrice": 0}
        return {"qty": float(p["positionAmt"]), "entry": float(p["entryPrice"])}
