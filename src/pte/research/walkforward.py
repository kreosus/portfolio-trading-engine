"""Walk-forward validation, experiment logging, holdout lock and kill criteria."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..backtest.costs import CostModel
from ..backtest.engine import run_backtest
from ..backtest.metrics import daily_returns, max_drawdown, sharpe, summarize
from ..config import config_hash, with_overrides
from ..features.indicators import elevated_vol
from ..features.smc import build_features
from ..strategies.smc_m15 import generate_signals

GRID_KEYS = {"swing_n": "smc.swing_n", "target_r": "strategy.target_r"}


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(first: pd.Timestamp, last: pd.Timestamp, train_m: int, test_m: int, step_m: int) -> list[Fold]:
    start = first.tz_convert(None).to_period("M").to_timestamp().tz_localize("UTC")
    if first > start:
        start += pd.DateOffset(months=1)
    folds = []
    while True:
        tr_end = start + pd.DateOffset(months=train_m)
        te_end = tr_end + pd.DateOffset(months=test_m)
        if te_end > last + pd.Timedelta("15min"):
            break
        folds.append(Fold(start, tr_end, tr_end, te_end))
        start += pd.DateOffset(months=step_m)
    return folds


class ExperimentLog:
    """Append-only log of every configuration evaluated (multiple-testing record)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **rec) -> None:
        rec["logged_at"] = datetime.now(timezone.utc).isoformat()
        with self.path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def count(self) -> int:
        return sum(1 for _ in self.path.open()) if self.path.exists() else 0


class Prepared:
    """Caches features per swing_n and signals per (swing_n, strategy params)."""

    def __init__(self, bars: dict[str, pd.DataFrame], cfg: dict):
        self.bars, self.cfg = bars, cfg
        self._feats: dict = {}
        self._sigs: dict = {}

    def get(self, cfg: dict):
        fkey = json.dumps(cfg["smc"], sort_keys=True)
        if fkey not in self._feats:
            feats = {}
            for s, df in self.bars.items():
                f = build_features(df, cfg["smc"])
                f["elevated_vol"] = elevated_vol(f["atr"], cfg["risk"]["vol_lookback_bars"],
                                                 cfg["risk"]["vol_percentile"])
                feats[s] = f
            self._feats[fkey] = feats
        feats = self._feats[fkey]
        skey = fkey + json.dumps(cfg["strategy"], sort_keys=True)
        if skey not in self._sigs:
            self._sigs[skey] = {s: generate_signals(f, cfg["strategy"]) for s, f in feats.items()}
        return feats, self._sigs[skey]


def grid_configs(cfg: dict) -> list[dict]:
    grid = cfg["walkforward"]["grid"]
    keys = list(grid)
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        out.append(with_overrides(cfg, {GRID_KEYS[k]: v for k, v in zip(keys, combo)}))
    return out


def params_of(cfg: dict) -> dict:
    return {"swing_n": cfg["smc"]["swing_n"], "target_r": cfg["strategy"]["target_r"]}


def walk_forward(bars: dict[str, pd.DataFrame], funding: dict[str, pd.DataFrame], cfg: dict,
                 log: ExperimentLog | None = None, verbose: bool = True) -> dict:
    wf = cfg["walkforward"]
    first = max(df.index[0] for df in bars.values())
    last = min(df.index[-1] for df in bars.values())
    folds = make_folds(first, last, wf["train_months"], wf["test_months"], wf["step_months"])
    if not folds:
        raise ValueError("Not enough data for one walk-forward fold.")
    prep = Prepared(bars, cfg)
    candidates = grid_configs(cfg)

    fold_rows, oos_trades, oos_trades_2x, oos_rets, oos_rets_2x = [], [], [], [], []
    for k, fold in enumerate(folds):
        best, best_score = None, -float("inf")
        for c in candidates:
            feats, sigs = prep.get(c)
            res = run_backtest(feats, sigs, funding, c["risk"], CostModel.from_config(c["costs"]),
                               fold.train_start, fold.train_end)
            m = summarize(res.trades, res.equity, res.initial_equity)
            if log:
                log.write(kind="train", fold=k, params=params_of(c), cfg_hash=config_hash(c),
                          start=fold.train_start, end=fold.train_end,
                          **{x: m[x] for x in ("trades", "sharpe", "profit_factor", "max_drawdown")})
            score = m["sharpe"] if m["trades"] >= wf["min_train_trades"] else -1e9 + m["trades"]
            if score > best_score:
                best, best_score = c, score

        feats, sigs = prep.get(best)
        res = run_backtest(feats, sigs, funding, best["risk"], CostModel.from_config(best["costs"]),
                           fold.test_start, fold.test_end)
        stress = with_overrides(best, {"costs.multiplier": 2.0})
        res2 = run_backtest(feats, sigs, funding, stress["risk"], CostModel.from_config(stress["costs"]),
                            fold.test_start, fold.test_end)
        m = summarize(res.trades, res.equity, res.initial_equity)
        m2 = summarize(res2.trades, res2.equity, res2.initial_equity)
        if log:
            log.write(kind="test", fold=k, params=params_of(best), cfg_hash=config_hash(best),
                      start=fold.test_start, end=fold.test_end,
                      **{x: m[x] for x in ("trades", "sharpe", "profit_factor", "max_drawdown")})
        row = {"fold": k, "test_start": fold.test_start.date(), "test_end": fold.test_end.date(),
               **params_of(best), "trades": m["trades"], "return": m["total_return"],
               "sharpe": m["sharpe"], "pf": m["profit_factor"], "max_dd": m["max_drawdown"],
               "return_2x_costs": m2["total_return"], "halted": res.halted_at is not None}
        fold_rows.append(row)
        if verbose:
            print(f"fold {k}: test {row['test_start']}..{row['test_end']} params={params_of(best)} "
                  f"trades={m['trades']} ret={m['total_return']:+.2%} sharpe={m['sharpe']:.2f}")
        if len(res.trades):
            oos_trades.append(res.trades.assign(fold=k))
        if len(res2.trades):
            oos_trades_2x.append(res2.trades.assign(fold=k))
        oos_rets.append(daily_returns(res.equity))
        oos_rets_2x.append(daily_returns(res2.equity))

    trades = pd.concat(oos_trades, ignore_index=True) if oos_trades else pd.DataFrame()
    trades2 = pd.concat(oos_trades_2x, ignore_index=True) if oos_trades_2x else pd.DataFrame()
    rets = pd.concat(oos_rets) if oos_rets else pd.Series(dtype=float)
    rets2 = pd.concat(oos_rets_2x) if oos_rets_2x else pd.Series(dtype=float)
    stitched = (1 + rets).cumprod()
    return {
        "folds": pd.DataFrame(fold_rows),
        "oos_trades": trades,
        "oos_equity_index": stitched,
        "summary": _oos_summary(trades, trades2, rets, rets2, fold_rows),
        "configs_tried": len(candidates) * len(folds),
        "latest_params": params_of(best),
    }


def _pf(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    w = trades.loc[trades.pnl > 0, "pnl"].sum()
    l_ = -trades.loc[trades.pnl < 0, "pnl"].sum()
    return float(w / l_) if l_ > 0 else float("inf")


def _oos_summary(trades, trades2, rets, rets2, fold_rows) -> dict:
    eq = (1 + rets).cumprod()
    return {
        "oos_trades": int(len(trades)),
        "oos_sharpe": sharpe(rets),
        "oos_profit_factor": _pf(trades),
        "oos_profit_factor_2x_costs": _pf(trades2),
        "oos_sharpe_2x_costs": sharpe(rets2),
        "oos_max_drawdown": max_drawdown(pd.concat([pd.Series([1.0]), eq.reset_index(drop=True)])),
        "oos_total_return": float(eq.iloc[-1] - 1) if len(eq) else 0.0,
        "oos_avg_r": float(trades.r.mean()) if len(trades) else 0.0,
        "oos_win_rate": float((trades.pnl > 0).mean()) if len(trades) else 0.0,
        "profitable_fold_frac": float(sum(r["return"] > 0 for r in fold_rows) / len(fold_rows)),
    }


def kill_check(summary: dict, kc: dict, holdout_return: float | None = None) -> list[tuple[str, bool, str]]:
    checks = [
        ("OOS trades >= %d" % kc["min_oos_trades"], summary["oos_trades"] >= kc["min_oos_trades"],
         str(summary["oos_trades"])),
        ("Profit factor > %.2f" % kc["min_profit_factor"], summary["oos_profit_factor"] > kc["min_profit_factor"],
         f"{summary['oos_profit_factor']:.2f}"),
        ("Sharpe > %.2f" % kc["min_sharpe"], summary["oos_sharpe"] > kc["min_sharpe"], f"{summary['oos_sharpe']:.2f}"),
        ("PF at 2x costs > %.2f" % kc["min_profit_factor_2x_costs"],
         summary["oos_profit_factor_2x_costs"] > kc["min_profit_factor_2x_costs"],
         f"{summary['oos_profit_factor_2x_costs']:.2f}"),
        ("Profitable folds >= %.0f%%" % (100 * kc["min_profitable_fold_frac"]),
         summary["profitable_fold_frac"] >= kc["min_profitable_fold_frac"], f"{summary['profitable_fold_frac']:.0%}"),
    ]
    if holdout_return is not None:
        checks.append(("Holdout year positive", holdout_return > 0, f"{holdout_return:+.2%}"))
    return checks
