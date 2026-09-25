"""Performance metrics computed from trades and daily equity."""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return pd.Series(dtype=float)
    d = equity.resample("1D").last().dropna()
    return d.pct_change().dropna()


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float((1 - equity / peak).max())


def sharpe(rets: pd.Series, periods: int = 365) -> float:
    if len(rets) < 2 or rets.std() == 0:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(periods))


def sortino(rets: pd.Series, periods: int = 365) -> float:
    down = rets[rets < 0]
    if len(down) < 2 or down.std() == 0:
        return 0.0
    return float(rets.mean() / down.std() * np.sqrt(periods))


def summarize(trades: pd.DataFrame, equity: pd.Series, initial: float) -> dict:
    rets = daily_returns(equity)
    n = len(trades)
    out = {
        "trades": n,
        "total_return": float(equity.iloc[-1] / initial - 1) if len(equity) else 0.0,
        "sharpe": sharpe(rets),
        "sortino": sortino(rets),
        "max_drawdown": max_drawdown(pd.concat([pd.Series([initial]), equity.reset_index(drop=True)])),
    }
    if n:
        wins = trades.loc[trades.pnl > 0, "pnl"].sum()
        losses = -trades.loc[trades.pnl < 0, "pnl"].sum()
        out.update({
            "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
            "win_rate": float((trades.pnl > 0).mean()),
            "avg_r": float(trades.r.mean()),
            "fees": float(trades.fees.sum()),
            "funding": float(trades.funding.sum()),
            "exit_reasons": trades.reason.value_counts().to_dict(),
        })
    else:
        out.update({"profit_factor": 0.0, "win_rate": 0.0, "avg_r": 0.0, "fees": 0.0, "funding": 0.0,
                    "exit_reasons": {}})
    days = (equity.index[-1] - equity.index[0]).days if len(equity) > 1 else 0
    if days > 0 and out["total_return"] > -1:
        out["cagr"] = float((1 + out["total_return"]) ** (365 / days) - 1)
        out["avg_monthly"] = float((1 + out["total_return"]) ** (30.44 / days) - 1)
    return out
