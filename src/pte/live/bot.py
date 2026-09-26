"""The confluence bot: one cycle per closed M15 bar (spec §48 order of operations)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from . import engines as E
from .adapters import SimulatedExchange
from .data_quality import check_candles, check_quote
from .decision import decide
from .execution import ExecutionEngine
from .journal import Journal
from .models import OrderState, Position
from .risk import limits_block

LIVE_CFG = Path(__file__).resolve().parents[3] / "config" / "live.yaml"


def load_live_config(path=None) -> dict:
    return yaml.safe_load(Path(path or LIVE_CFG).read_text())


class Bot:
    def __init__(self, cfg: dict, venue=None, root: str | Path = "."):
        if cfg["mode"] == "live":
            raise PermissionError("live-money mode is disabled in this build")
        self.cfg = cfg
        root = Path(root)
        self.j = Journal(root / cfg["journal"]["path"], root / cfg["journal"]["log_path"])
        self.venue = venue or SimulatedExchange(cfg["exchange_rules"], root / "live" / "sim_exchange.json")
        self.ex = ExecutionEngine(self.venue, self.j, cfg)
        self.st = self.j.get("state") or {
            "equity": cfg["risk"]["initial_equity"], "peak_equity": cfg["risk"]["initial_equity"],
            "realized": 0.0, "day": None, "week": None,
            "day_start_equity": cfg["risk"]["initial_equity"], "week_start_equity": cfg["risk"]["initial_equity"],
            "pending": None, "position": None, "last_bar": None, "halted": None, "fills_seen": 0}
        self.last_ctx = None

    # ------------------------------------------------------------------ analysis
    def analyse(self, m15: pd.DataFrame, h1: pd.DataFrame, quote: dict, now: pd.Timestamp,
                derivatives: dict | None, calendar: list | None) -> dict:
        c = self.cfg
        dq = check_candles(m15, 900, now, c, "M15") + check_candles(h1, 3600, now, c, "H1")
        qp, spread = check_quote(quote, now, c)
        dq += qp
        h4 = E.resample(h1, "4h")
        s4 = E.structure_engine(h4, c["structure"]["htf_swing_n"])
        s1 = E.structure_engine(h1, c["structure"]["htf_swing_n"])
        s15 = E.structure_engine(m15, c["structure"]["swing_n"])
        levels = E.liquidity_levels(m15, h1, c["structure"]["swing_n"], c)
        ctx = {
            "now": now, "m15": m15, "h1": h1, "h4": h4, "quote": quote, "spread_bps": spread,
            "dq_problems": dq, "h4_struct": s4, "h1_struct": s1, "m15_struct": s15,
            "htf_bias": E.htf_bias(s4, s1), "levels": levels, "sweeps": E.detect_sweeps(m15, levels, c),
            "momentum": E.momentum_engine(m15, c), "session": E.session_of(now, c),
            "regime": E.regime_engine(h4, s4), "derivatives": derivatives,
            "news": E.news_state(calendar, now, c),
        }
        self._roll_account(now)
        self.st["open_positions"] = 1 if self.st["position"] else 0
        ctx["risk_block"] = limits_block(self.st, c)
        self.last_ctx = ctx
        return ctx

    def _roll_account(self, now):
        day, week = str(now.floor("D")), str((now - pd.Timedelta(days=now.weekday())).floor("D"))
        if self.st["day"] != day:
            self.st["day"], self.st["day_start_equity"] = day, self.st["equity"]
        if self.st["week"] != week:
            self.st["week"], self.st["week_start_equity"] = week, self.st["equity"]

    # ------------------------------------------------------------------ one cycle
    def cycle(self, m15, h1, quote, now, derivatives=None, calendar=None) -> dict:
        self.j.clock = now
        report = {"time": str(now), "events": []}
        # 1) venue processes every closed bar since the last cycle (shadow fills)
        bars = m15 if self.st["last_bar"] is None else m15[m15.index > pd.Timestamp(self.st["last_bar"])]
        if self.st["last_bar"] is None:
            bars = m15.iloc[-1:]
        for ts, b in bars.iterrows():
            if hasattr(self.venue, "process_bar"):
                self.venue.process_bar(ts, b.open, b.high, b.low, b.close)
            self._manage(ts, b, quote, report)
        if len(m15):
            self.st["last_bar"] = str(m15.index[-1])
        # 2) analysis + decision
        ctx = self.analyse(m15, h1, quote, now, derivatives, calendar)
        self._mark_to_market(float(m15.close.iloc[-1]))
        if self.st["position"] or self.st["pending"]:
            report["decision"] = "IN_TRADE" if self.st["position"] else "ENTRY_PENDING"
            self._structure_failure(ctx, quote, now, report)
        else:
            d = decide(ctx, self.cfg)
            self.j.decision(str(m15.index[-1]), d)
            report["decision"] = d.action
            report["explain"] = d.explain()
            if d.signal:
                o, sizing = self.ex.open_from_signal(d.signal, quote, self.st["equity"], ctx["news"]["blocking"])
                if o is not None and o.state == OrderState.ACKNOWLEDGED:
                    self.st["pending"] = {"cid": o.client_id, "signal": d.signal.to_dict(),
                                          "sizing": sizing.__dict__, "placed_bar": str(m15.index[-1]),
                                          "bars_live": 0}
                    report["events"].append(f"ENTRY ORDER placed: {d.signal.direction} LIMIT {o.qty} @ {o.price}"
                                            f" (risk ${sizing.risk_amount:.2f}, {sizing.leverage:.0f}x)")
                else:
                    report["events"].append(f"signal cancelled at entry validation: {sizing.reason}")
        self.st["equity_curve_last"] = self.st["equity"]
        self.j.put("state", self.st)
        if hasattr(self.venue, "save"):
            self.venue.save()
        report["equity"] = round(self.st["equity"], 2)
        return report

    # ------------------------------------------------------------------ position management
    def _manage(self, ts, bar, quote, report):
        p = self.st["pending"]
        if p:
            o = self.venue.orders.get(p["cid"])
            if o and o.state == OrderState.FILLED:
                self._on_entry_filled(o, ts, report)
            else:
                p["bars_live"] += 1
                sig = p["signal"]
                up = sig["direction"] == "LONG"
                invalid = (bar.low <= sig["stop"]) if up else (bar.high >= sig["stop"])
                if p["bars_live"] > self.cfg["poi"]["entry_valid_bars"] or invalid:
                    self.ex.cancel(o, "setup invalidated" if invalid else "entry window expired")
                    report["events"].append(f"ENTRY CANCELLED ({'invalidated' if invalid else 'expired'})")
                    self.j.event("signal_cancelled", reason="invalidated" if invalid else "expired")
                    self.st["pending"] = None
        pos = self.st["position"]
        if pos:
            P = Position(**pos["p"])
            up = P.side == "LONG"
            P.mae = max(P.mae, (P.entry - bar.low) if up else (bar.high - P.entry))
            P.mfe = max(P.mfe, (bar.high - P.entry) if up else (P.entry - bar.low))
            for purpose, cid in list(pos["orders"].items()):
                if purpose == "entry":
                    continue
                o = self.venue.orders.get(cid)
                if o and o.state == OrderState.FILLED and purpose not in P.tp_done:
                    P.tp_done.append(purpose)
                    report["events"].append(f"{purpose.upper()} FILLED {o.qty} @ {o.avg_price:.2f}")
                    self.j.event("order_fill", purpose=purpose, qty=o.qty, price=o.avg_price)
                    self.j.order(o)
                    if purpose == "tp1" and self.cfg["execution"]["breakeven_after_tp1"] and "stop" not in P.tp_done:
                        self._move_stop(P, pos, P.entry, report)
            pos["p"] = P.__dict__
            vq = self.venue.position_risk(P.symbol)["qty"]
            if abs(vq) < 1e-12:
                self._close_trade(P, pos, ts, report)

    def _move_stop(self, P, pos, new_stop, report):
        old = self.venue.orders.get(pos["orders"]["stop"])
        self.ex.cancel(old, "stop moved")
        remaining = abs(self.venue.position_risk(P.symbol)["qty"])
        from .models import Order
        o = self.ex._submit(Order(P.symbol, "SELL" if P.side == "LONG" else "BUY", "STOP_MARKET", remaining,
                                  stop_price=round(new_stop, 1), reduce_only=True, purpose="stop"))
        pos["orders"]["stop"] = o.client_id
        P.stop = new_stop
        report["events"].append(f"STOP moved to breakeven {new_stop:.2f}")
        self.j.event("position_update", stop=new_stop)

    def _on_entry_filled(self, o, ts, report):
        p = self.st["pending"]
        sig = p["signal"]
        sz = p["sizing"]
        P = Position(symbol=sig["symbol"], side=sig["direction"], entry=o.avg_price, qty=o.qty, remaining_qty=o.qty,
                     stop=sig["stop"], initial_stop=sig["stop"], tp1=sig["tp1"], tp2=sig["tp2"], tp3=sig["tp3"],
                     leverage=sz["leverage"], margin=o.qty * o.avg_price / sz["leverage"], opened_at=str(ts),
                     signal_id=sig["timestamp"], liquidation_price=sz["liquidation_price"])
        try:
            prot = self.ex.protect(P)
        except RuntimeError as e:        # kill switch: protective stop missing -> emergency exit
            self.st["halted"] = f"protective stop missing: {e}"
            report["events"].append("KILL SWITCH: protective stop missing -> emergency close")
            return
        orders = {"entry": o.client_id, **{x.purpose: x.client_id for x in prot}}
        tid = self.j.open_trade(opened_at=str(ts), symbol=P.symbol, direction=P.side,
                                strategy_version=sig["strategy_version"], entry=P.entry, stop=P.stop, tp1=P.tp1,
                                tp2=P.tp2, tp3=P.tp3, qty=P.qty, leverage=P.leverage, margin=P.margin,
                                risk_amount=sz["risk_amount"], risk_pct=sz["risk_pct"], notional=sz["notional"],
                                score=sig["score"], signal=sig, order_ids=orders)
        self.st["position"] = {"p": P.__dict__, "orders": orders, "trade_id": tid, "fills_from": o.client_id}
        self.st["pending"] = None
        report["events"].append(f"ENTRY FILLED: {P.side} {P.qty} BTC @ {P.entry:.2f}; stop {P.stop:.2f}, "
                                f"TP1 {P.tp1:.2f} / TP2 {P.tp2:.2f} / TP3 {P.tp3:.2f}")
        self.j.event("order_fill", purpose="entry", qty=P.qty, price=P.entry)

    def _close_trade(self, P, pos, ts, report, reason=None):
        for purpose, cid in pos["orders"].items():
            o = self.venue.orders.get(cid)
            if o and o.open:
                self.ex.cancel(o, "position closed")
        cids = set(pos["orders"].values())
        fills = [f for f in self.venue.fills if f["client_id"] in cids or f.get("purpose") == "close"
                 and pd.Timestamp(f["t"]) >= pd.Timestamp(P.opened_at)]
        gross = sum(f["realized"] for f in fills)
        fees = sum(f["fee"] for f in fills)
        net = gross - fees
        risk_amt = abs(P.entry - P.initial_stop) * P.qty
        exits = [f["purpose"] for f in fills if f["purpose"] != "entry"]
        reason = reason or (exits[-1] if exits else "flat")
        self.st["equity"] += 0  # realized already counted in mark-to-market below
        self.st["realized"] += net
        self.j.close_trade(pos["trade_id"], closed_at=str(ts), exit_reason=reason, gross_pnl=gross, fees=fees,
                           funding=0.0, net_pnl=net, r_multiple=net / risk_amt if risk_amt else None,
                           mae=P.mae, mfe=P.mfe)
        report["events"].append(f"POSITION CLOSED ({reason}): net {net:+.2f} USD ({net / risk_amt:+.2f}R)")
        self.j.event("exit", reason=reason, net=net)
        self.st["position"] = None
        self._mark_to_market(None)

    def _structure_failure(self, ctx, quote, now, report):
        pos = self.st["position"]
        if not pos:
            return
        P = Position(**pos["p"])
        s = ctx["m15_struct"]["last_choch"]
        opp = "DOWN" if P.side == "LONG" else "UP"
        if s and s["dir"] == opp and pd.Timestamp(s["time"]) > pd.Timestamp(P.opened_at):
            qty = abs(self.venue.position_risk(P.symbol)["qty"])
            if qty > 0:
                self.ex.market_close(P, qty, quote, now, "opposite M15 CHoCH (structure failure)")
                self._close_trade(P, pos, now, report, reason="structure failure")

    def close_now(self, quote, now, why="manual close") -> dict:
        """Operator close (and the emergency path)."""
        report = {"time": str(now), "events": []}
        pos = self.st["position"]
        if not pos:
            report["events"].append("no open position")
            return report
        P = Position(**pos["p"])
        qty = abs(self.venue.position_risk(P.symbol)["qty"])
        self.ex.market_close(P, qty, quote, now, why)
        self._close_trade(P, pos, now, report, reason=why)
        self.j.put("state", self.st)
        if hasattr(self.venue, "save"):
            self.venue.save()
        return report

    def _mark_to_market(self, price):
        pos_q = self.venue.position_risk(self.cfg["symbol"])
        unreal = pos_q["qty"] * (price - pos_q["entry"]) if (price and pos_q["qty"]) else 0.0
        self.st["equity"] = self.cfg["risk"]["initial_equity"] + self.st["realized"] + unreal
        self.st["peak_equity"] = max(self.st["peak_equity"], self.st["equity"])
