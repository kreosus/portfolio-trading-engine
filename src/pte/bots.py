"""One bot with five sub-bots. Each sub-bot trades the full account independently
(own capital, risk engine, positions and trades) and is reported on its own.
Sub-bots are simulated in a single pass over the data for speed; nothing is shared."""

from __future__ import annotations

import pandas as pd

from .backtest.costs import CostModel
from .backtest.engine import PortfolioResult, SleeveSpec, run_portfolio
from .backtest.metrics import daily_returns, summarize
from .config import with_overrides
from .features.common import build_common
from .features.indicators import elevated_vol
from .features.smc import build_features
from .strategies.library import REGISTRY
from .strategies.smc_m15 import generate_signals


def build_features_all(bars: dict, funding: dict, cfg: dict) -> dict[str, pd.DataFrame]:
    feats = {}
    for s, df in bars.items():
        f = build_features(df, cfg["smc"])
        extra = build_common(df, funding.get(s), cfg["smc"]["atr_period"]).drop(columns="atr")
        f = pd.concat([f, extra], axis=1)
        f["elevated_vol"] = elevated_vol(f["atr"], cfg["risk"]["vol_lookback_bars"], cfg["risk"]["vol_percentile"])
        feats[s] = f
    return feats


def build_sleeves(feats: dict, cfg: dict) -> list[SleeveSpec]:
    specs = []
    for name, sc in cfg["bots"]["sleeves"].items():
        if not sc.get("enabled", True):
            continue
        if name == "smc":
            intents = {s: generate_signals(f, cfg["strategy"]) for s, f in feats.items()}
        else:
            intents = {s: REGISTRY[name](f, sc) for s, f in feats.items()}
        specs.append(SleeveSpec(name, intents, 1.0))   # full capital each
    return specs


def run_bot(bars: dict, funding: dict, cfg: dict, start=None, end=None,
            feats: dict | None = None, specs: list | None = None) -> PortfolioResult:
    feats = feats if feats is not None else build_features_all(bars, funding, cfg)
    specs = specs if specs is not None else build_sleeves(feats, cfg)
    return run_portfolio(feats, specs, funding, cfg["risk"], CostModel.from_config(cfg["costs"]), start, end)


def bot_report(bars: dict, funding: dict, cfg: dict, start=None, end=None) -> dict:
    feats = build_features_all(bars, funding, cfg)
    specs = build_sleeves(feats, cfg)
    res = run_bot(bars, funding, cfg, start, end, feats, specs)
    res2 = run_bot(bars, funding, with_overrides(cfg, {"costs.multiplier": 2.0}), start, end, feats, specs)
    rows = {}
    rets = {}
    kc = cfg["kill_criteria"]
    for name, r in res.sleeves.items():
        m = summarize(r.trades, r.equity, r.initial_equity)
        m2 = summarize(res2.sleeves[name].trades, res2.sleeves[name].equity, r.initial_equity)
        hy = _period_returns(r.equity)
        status = cfg["bots"]["sleeves"][name].get("status", "research")
        gross_r = float((r.trades.gross / (r.trades.qty * (r.trades.entry - r.trades.stop).abs())).mean()) \
            if len(r.trades) else 0.0
        checks = {
            "trades": m["trades"] >= kc["min_oos_trades"],
            "pf": m["profit_factor"] > kc["min_profit_factor"],
            "sharpe": m["sharpe"] > kc["min_sharpe"],
            "pf_2x": m2["profit_factor"] > kc["min_profit_factor_2x_costs"],
            "periods": (hy > 0).mean() >= kc["min_profitable_fold_frac"] if len(hy) else False,
        }
        rows[name] = {
            "status": status,
            "trades": m["trades"], "return": m["total_return"], "sharpe": m["sharpe"],
            "pf": m["profit_factor"], "pf_2x_costs": m2["profit_factor"], "max_dd": m["max_drawdown"],
            "win_rate": m["win_rate"], "avg_r": m["avg_r"], "gross_avg_r": gross_r,
            "profitable_halves": float((hy > 0).mean()) if len(hy) else 0.0,
            "halted": r.halted_at is not None,
            "passes": sum(checks.values()), "verdict": "PASS" if all(checks.values()) else "FAIL",
        }
        rets[name] = daily_returns(r.equity)
    corr = pd.DataFrame(rets).corr()   # informational: how alike the sub-bots' daily returns are
    return {"sleeves": pd.DataFrame(rows).T, "correlation": corr, "result": res}


def _period_returns(equity: pd.Series) -> pd.Series:
    e = equity.resample("1D").last().dropna()
    if e.empty:
        return pd.Series(dtype=float)
    key = e.index.year * 2 + (e.index.month - 1) // 6
    first = e.groupby(key).first()
    last = e.groupby(key).last()
    prev_last = last.shift(1).fillna(first)
    return last / prev_last - 1
