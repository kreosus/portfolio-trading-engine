"""Master Decision Engine (spec §19-§21, §31, §45-§46).

Hard requirements (all must hold) -> soft confirmations (weighted score) -> Signal or NO TRADE.
The core edge stays Liquidity + Structure + Displacement + POI. Everything else only scores.
"""

from __future__ import annotations

import pandas as pd

from ..features.indicators import atr as atr_fn
from . import engines as E
from .models import Decision, Signal


def _targets(levels: list[dict], entry: float, bias: str, h4: dict, risk: float) -> tuple[list, list]:
    """TP1 nearest opposing liquidity, TP2 next significant, TP3 extended HTF (spec §27-§28).
    Falls back to R multiples where no liquidity exists, and says so."""
    up = bias == "LONG"
    side = "BUY" if up else "SELL"
    opp = sorted({round(L["level"], 2): L for L in levels if L["side"] == side and
                  ((L["level"] > entry + 0.5 * risk) if up else (L["level"] < entry - 0.5 * risk))}.values(),
                 key=lambda L: L["level"] if up else -L["level"])
    tps, src = [], []
    for L in opp[:2]:
        tps.append(L["level"]); src.append(L["type"])
    htf = h4["major_swing_high"] if up else h4["major_swing_low"]
    if (up and htf > (tps[-1] if tps else entry)) or (not up and htf < (tps[-1] if tps else entry)):
        tps.append(htf); src.append("H4_SWING")
    while len(tps) < 3:          # no further liquidity: extend by 1R beyond the last target, and say so
        base = tps[-1] if tps else entry
        tps.append(base + (1 if up else -1) * risk)
        src.append(f"{src[-1] if src else 'entry'}+1R (no liquidity)")
    return tps[:3], src[:3]


def decide(ctx: dict, cfg: dict) -> Decision:
    """ctx: output of the runner's analysis step (all engines run, data-quality + quote checked)."""
    reasons, failed = [], []

    def check(ok: bool, text: str):
        reasons.append((bool(ok), text))
        if not ok:
            failed.append(text)
        return ok

    m15, now = ctx["m15"], ctx["now"]
    # ---- execution-condition gates first: they can veto everything (spec §31)
    check(not ctx["dq_problems"], "data quality ok" if not ctx["dq_problems"] else
          "data quality: " + "; ".join(ctx["dq_problems"]))
    news = ctx["news"]
    check(not news["blocking"], f"news state {news['state']}" + (f" ({news['event']['title']}, "
          f"{news['event']['minutes_until']} min)" if news.get("event") else ""))
    check(not ctx["risk_block"], "risk limits ok" if not ctx["risk_block"] else f"risk block: {ctx['risk_block']}")

    # ---- hard requirements, per direction; keep the best candidate
    best = None
    for sw in sorted(ctx["sweeps"], key=lambda s: (s["bar"], s["strength"]), reverse=True):
        bias = sw["bias"]
        up = bias == "LONG"
        cand = [(True, f"{sw['direction'].replace('_', '-').lower()} liquidity swept: {sw['level_type']} "
                       f"{sw['level']:.2f} at {sw['time'][11:16]} (strength {sw['strength']})")]
        htf_ok = ctx["htf_bias"] == ("BULLISH" if up else "BEARISH")
        disp = E.displacement_after(m15, sw["bar"], bias, ctx["m15_struct"]["_frame"], cfg)
        transition = bool(disp and disp["structure_break"] and disp["structure_break"]["type"] == "CHoCH")
        cand.append((htf_ok or transition, f"structural context: HTF {ctx['htf_bias']}" +
                     (", M15 CHoCH transition" if transition else "")))
        cand.append((disp is not None, "displacement confirmed" + (
            f" ({disp['body_atr']}x ATR body, {disp['consecutive']} candles)" if disp else " - none after sweep")))
        brk = disp["structure_break"] if disp else None
        cand.append((brk is not None, f"{brk['type']} confirmed at {brk['time'][11:16]}" if brk else
                     "no BOS/CHoCH after displacement"))
        poi = E.poi_engine(m15, disp, bias, cfg, (ctx["m15_struct"]["major_swing_low"],
                                                  ctx["m15_struct"]["major_swing_high"])) if brk else None
        cand.append((poi is not None, f"valid POI: {poi['zone']['type']} {poi['zone']['low']:.2f}-"
                     f"{poi['zone']['high']:.2f}" if poi else "no fresh, unmitigated FVG/OB"))
        n_ok = sum(ok for ok, _ in cand)
        if best is None or n_ok > best[0]:
            best = (n_ok, sw, bias, disp, brk, poi, cand)
        if n_ok == len(cand):
            break

    if best is None:
        check(False, "no liquidity sweep in the last "
              f"{cfg['liquidity']['sweep_lookback_bars']} M15 bars (a touch is not a sweep)")
        return Decision("NO_TRADE", reasons, None, failed)
    n_ok, sw, bias, disp, brk, poi, cand = best
    for ok, text in cand:
        check(ok, text)
    if failed:
        return Decision("NO_TRADE", reasons, None, failed)

    # ---- plan the trade
    up = bias == "LONG"
    a = float(atr_fn(m15, 14).iloc[-1])
    q = ctx["quote"]
    zone = poi["zone"]
    entry = zone["mid"]
    if up and q["ask"] < entry:           # price already inside/through the POI: no better than the ask
        entry = q["ask"]
    if not up and q["bid"] > entry:
        entry = q["bid"]
    stop = sw["extreme"] - 0.1 * a if up else sw["extreme"] + 0.1 * a
    risk = abs(entry - stop)
    check(risk > 0 and ((stop < entry) if up else (stop > entry)), f"stop {stop:.2f} beyond sweep extreme")
    tps, tp_src = _targets(ctx["levels"], entry, bias, ctx["h4_struct"], risk)
    rr = abs(tps[1] - entry) / risk if risk else 0
    check(rr >= cfg["decision"]["min_rr"], f"R:R to TP2 {rr:.2f} (min {cfg['decision']['min_rr']})")
    drift = abs((q["ask"] if up else q["bid"]) - entry) / a
    check(drift <= cfg["execution"]["max_entry_drift_atr"] or ((q["ask"] < zone["high"]) if up else (q["bid"] > zone["low"])),
          f"price {drift:.2f} ATR from planned entry")

    # ---- soft confirmations -> score
    w = cfg["decision"]["weights"]
    mo, ctxd = ctx["momentum"], ctx["derivatives"]
    parts = {
        "htf_alignment": w["htf_alignment"] * (1.0 if ctx["htf_bias"] == ("BULLISH" if up else "BEARISH") else
                                              0.5 if ctx["htf_bias"] in ("NEUTRAL", "MIXED") else 0.0),
        "fvg_ob_overlap": w["fvg_ob_overlap"] * (1.0 if poi["fvg_ob_overlap"] else 0.0),
        "premium_discount": w["premium_discount"] * (1.0 if poi["favourable_location"] else 0.0),
        "volume": w["volume"] * (1.0 if disp["volume_confirm"] else 0.5 if disp["range_expansion"] else 0.0),
        "vwap": w["vwap"] * (1.0 if mo["above_vwap"] == up else 0.3),
        "ema": w["ema"] * (1.0 if mo["ema_state"] == ("BULLISH" if up else "BEARISH") else 0.0),
        "rsi": w["rsi"] * (1.0 if (40 <= mo["rsi"] <= 70 if up else 30 <= mo["rsi"] <= 60) else 0.3),
        "session": w["session"] * (1.0 if any(s in ctx["session"] for s in ("LONDON", "NEW_YORK")) else 0.5),
        "derivatives": w["derivatives"] * (0.5 if not ctxd else
                                           1.0 if (ctxd["funding"] <= 0.0003 if up else ctxd["funding"] >= -0.0003)
                                           else 0.2),
    }
    score = max(0.0, sum(parts.values()) - news.get("penalty", 0))
    thr = cfg["decision"]["score_threshold"]
    check(score >= thr, f"confluence score {score:.0f} (threshold {thr}, UNVALIDATED placeholder)")
    if failed:
        return Decision("NO_TRADE", reasons, None, failed)

    sig = Signal(
        symbol=cfg["symbol"], direction=bias, strategy_version=cfg["strategy_version"], timestamp=str(now),
        htf_bias=ctx["htf_bias"],
        structure_state={k: {x: v for x, v in ctx[k].items() if not x.startswith("_")}
                         for k in ("h4_struct", "h1_struct", "m15_struct")},
        liquidity_state={"swept": sw["level_type"], "level": sw["level"]},
        sweep={k: v for k, v in sw.items() if k != "bar"},
        displacement={k: v for k, v in disp.items() if k != "bar"}, bos_choch=brk,
        poi=poi["zone"], fvg=poi["fvg"] or {}, ob=poi["ob"] or {}, momentum_state=mo,
        btc_context={"h4": ctx["h4_struct"]["direction"], "regime": ctx["regime"]["regime"]},
        funding=ctxd["funding"] if ctxd else None, oi=ctxd["oi"] if ctxd else None,
        session=ctx["session"], regime=ctx["regime"]["regime"], news_state=news["state"],
        score=round(score, 1), score_breakdown={k: round(v, 1) for k, v in parts.items()},
        entry_zone=(zone["low"], zone["high"]), planned_entry=round(entry, 2), stop=round(stop, 2),
        tp1=round(tps[0], 2), tp2=round(tps[1], 2), tp3=round(tps[2], 2), expected_rr=round(rr, 2),
        invalidation=f"close {'below' if up else 'above'} {stop:.2f} or opposite M15 CHoCH",
        setup_bar_time=str(m15.index[-1]),
    )
    sig.score_breakdown["targets"] = " / ".join(tp_src)
    return Decision(bias, reasons, sig, failed)
