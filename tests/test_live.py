import copy

import numpy as np
import pandas as pd
import pytest

from pte.live import engines as E
from pte.live.adapters import BinanceFuturesAdapter, SimulatedExchange, round_step
from pte.live.bot import load_live_config
from pte.live.models import Order, OrderState
from pte.live.risk import limits_block, size_position

CFG = load_live_config()
RULES = CFG["exchange_rules"]


def test_order_state_machine_rejects_illegal_transitions():
    o = Order("BTCUSDT", "BUY", "LIMIT", 0.1, price=100)
    o.transition(OrderState.VALIDATED)
    o.transition(OrderState.SUBMITTED)
    with pytest.raises(RuntimeError):
        o.transition(OrderState.FILLED)          # must be acknowledged first
    o.transition(OrderState.UNKNOWN, "timeout")
    o.transition(OrderState.ACKNOWLEDGED, "reconciled")
    o.transition(OrderState.FILLED)
    with pytest.raises(RuntimeError):
        o.transition(OrderState.CANCELLED)       # terminal


def test_sizing_matches_spec_worked_example():
    # spec §22: $10,000 equity, 0.5% risk, entry 100,000, stop 99,500 -> 0.1 BTC, $10,000 notional
    s = size_position(100_000, 99_500, "LONG", 10_000, RULES, CFG)
    assert s.approved and s.qty == pytest.approx(0.1) and s.notional == pytest.approx(10_000)
    assert s.risk_amount == pytest.approx(50) and s.leverage == 1


def test_sizing_caps_leverage_at_5x_and_rounds_down():
    s = size_position(100_000, 99_950, "LONG", 10_000, RULES, CFG)   # tiny stop wants 1 BTC
    assert s.leverage <= 5 and s.notional <= 5 * 10_000 + 1e-6
    assert s.risk_amount <= 50 + 1e-9                                # never above max risk
    assert round_step(0.1239, 0.001) == 0.123


def test_liquidation_distance_check_blocks_tight_margin():
    cfg = copy.deepcopy(CFG); cfg["risk"]["min_liq_distance_x_stop"] = 500   # 500 x $50 stop > ~$19.6k liq distance at 5x
    s = size_position(100_000, 99_950, "LONG", 10_000, RULES, cfg)
    assert not s.approved and "liquidation" in s.reason


def test_kill_switch_limits():
    st = {"equity": 9790, "peak_equity": 10000, "day_start_equity": 10000, "week_start_equity": 10000,
          "open_positions": 0}
    assert limits_block(st, CFG) == "daily loss limit reached"
    st.update(equity=10000, open_positions=1)
    assert "max open positions" in limits_block(st, CFG)


def test_binance_signature_matches_documented_example():
    q = "symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1&recvWindow=5000&timestamp=1499827319559"
    secret = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
    assert BinanceFuturesAdapter.sign(q, secret) == "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71"


def test_binance_adapter_refuses_mainnet():
    cfg = copy.deepcopy(CFG); cfg["binance"]["demo_base_url"] = "https://fapi.binance.com"
    with pytest.raises(PermissionError):
        BinanceFuturesAdapter(cfg)


def _ack(ex, o):
    o.transition(OrderState.VALIDATED); o.transition(OrderState.SUBMITTED)
    ex.submit(o); o.transition(OrderState.ACKNOWLEDGED)


def test_simulated_exchange_stop_beats_tp_in_same_bar_and_reduce_only_never_flips():
    ex = SimulatedExchange(RULES, slippage_bps=0)
    e = Order("BTCUSDT", "BUY", "LIMIT", 0.1, price=100); _ack(ex, e)
    ex.process_bar(pd.Timestamp("2026-01-01", tz="UTC"), 101, 101, 99, 100)
    assert e.state == OrderState.FILLED and ex.position["qty"] == pytest.approx(0.1)
    stop = Order("BTCUSDT", "SELL", "STOP_MARKET", 0.1, stop_price=95, reduce_only=True, purpose="stop"); _ack(ex, stop)
    tp = Order("BTCUSDT", "SELL", "TAKE_PROFIT_MARKET", 0.1, stop_price=110, reduce_only=True, purpose="tp1"); _ack(ex, tp)
    ex.process_bar(pd.Timestamp("2026-01-01 00:15", tz="UTC"), 100, 111, 94, 100)
    assert stop.state == OrderState.FILLED and tp.state == OrderState.CANCELLED
    assert ex.position["qty"] == 0


def test_touch_is_not_a_sweep():
    idx = pd.date_range("2026-01-01", periods=60, freq="15min", tz="UTC")
    base = np.full(60, 100.0)
    df = pd.DataFrame({"open": base, "high": base + 1, "low": base - 1, "close": base, "volume": 1.0}, index=idx)
    levels = [{"type": "PDL", "side": "SELL", "level": 99.0, "weight": 1.0}]
    assert E.detect_sweeps(df, levels, CFG) == []                           # lows only touch 99
    df.iloc[-2, df.columns.get_loc("low")] = 98.0                            # trades through, closes back above
    sw = E.detect_sweeps(df, levels, CFG)
    assert len(sw) == 1 and sw[0]["bias"] == "LONG"
    df.iloc[-2, df.columns.get_loc("close")] = 98.5                          # closes below: a break, not a sweep
    assert E.detect_sweeps(df, levels, CFG) == []


def test_news_states():
    now = pd.Timestamp("2026-09-30 12:00", tz="UTC")
    ev = lambda m: [{"title": "CPI m/m", "country": "USD", "impact": "High",
                     "date": (now + pd.Timedelta(minutes=m)).isoformat()}]
    assert E.news_state(ev(3), now, CFG)["state"] == "EVENT_LOCK"
    assert E.news_state(ev(10), now, CFG)["blocking"]
    s = E.news_state(ev(45), now, CFG)
    assert s["state"] == "PRE_EVENT" and not s["blocking"] and s["penalty"] > 0
    assert E.news_state(ev(-10), now, CFG)["blocking"]
    assert E.news_state(ev(300), now, CFG)["state"] == "NORMAL"
    un = E.news_state(None, now, CFG)
    assert un["state"] == "NEWS_DATA_UNAVAILABLE" and un["blocking"]     # never assume "no news"
