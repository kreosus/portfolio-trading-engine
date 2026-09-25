"""Forward paper trading from a pre-registered registry.

Each run downloads new archive data, then RE-RUNS every registered sub-bot from
the fixed history start with the same engine as the backtests. Recomputing from
scratch keeps results deterministic and self-correcting: when real funding
replaces estimates, or a late file arrives, the numbers update. Nothing is
stored as mutable state, so results cannot drift from the code.

Positions still open at the last bar are marked to market and reported as open.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from .backtest.metrics import summarize
from .bots import build_features_all, run_bot
from .config import with_overrides
from .data import forward
from .data.store import HTF_FOR, resample_bars


def entry_hash(cfg: dict, sub_bot: str, timeframe: str) -> str:
    """Hash of everything that affects one sub-bot's trades (not the rest of the config)."""
    part = {"sub_bot": sub_bot, "timeframe": timeframe, "smc": cfg["smc"], "risk": cfg["risk"],
            "costs": cfg["costs"], "params": cfg["bots"]["sleeves"][sub_bot],
            "strategy": cfg["strategy"] if sub_bot == "smc" else None}
    return hashlib.sha256(json.dumps(part, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _entry_cfg(cfg: dict, sub_bot: str, timeframe: str) -> dict:
    c = with_overrides(cfg, {"smc.htf_rule": HTF_FOR[timeframe]})
    sleeves = {k: {**v, "enabled": k == sub_bot} for k, v in c["bots"]["sleeves"].items()}
    return {**c, "bots": {**c["bots"], "sleeves": sleeves}}


def run_entries(reg: dict, cfg: dict, bars15: dict, funding: dict) -> list[dict]:
    out = []
    for e in reg["entries"]:
        name, tf = e["sub_bot"], e["timeframe"]
        h = entry_hash(cfg, name, tf)
        if h != e["entry_hash"]:
            raise SystemExit(f"{e['id']}: rules changed since registration ({h} != {e['entry_hash']}). "
                             "Register a new entry instead of editing a live one.")
        ec = _entry_cfg(cfg, name, tf)
        bars = {s: resample_bars(df, tf) for s, df in bars15.items()}
        start = pd.Timestamp(e["scoring_start"], tz="UTC")
        res = run_bot(bars, funding, ec, start=start).sleeves[name]
        m = summarize(res.trades, res.equity, res.initial_equity)
        t = res.trades
        closed = t[t.reason != "end"] if len(t) else t
        open_ = t[t.reason == "end"] if len(t) else t
        out.append({"entry": e, "metrics": m, "trades": t, "equity": res.equity,
                    "closed": len(closed), "open": open_, "halted_at": res.halted_at})
    return out


def update(reg_path: str | Path, cfg: dict, raw_dir: str | Path, out_dir: str | Path,
           today: date | None = None) -> dict:
    reg = json.loads(Path(reg_path).read_text())
    info = forward.sync(reg["symbols"], reg["history_start"], raw_dir, today)
    bars15 = {s: forward.load_klines(s, raw_dir) for s in reg["symbols"]}
    fund = {s: forward.load_funding(s, raw_dir) for s in reg["symbols"]}
    results = run_entries(reg, cfg, bars15, {s: f[["rate"]] for s, f in fund.items()})
    return write_report(reg, results, bars15, fund, info, out_dir)


def write_report(reg, results, bars15, fund, info, out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    last_bar = min(df.index[-1] for df in bars15.values())
    est_since = [f[f.estimated].index.min() for f in fund.values() if "estimated" in f and f.estimated.any()]
    est_since = min(est_since) if est_since else None
    now = datetime.now(timezone.utc).isoformat(timespec="minutes")
    rows, status = [], {"updated_at": now, "data_through": str(last_bar + pd.Timedelta("15min")),
                        "funding_estimated_since": str(est_since) if est_since is not None else None,
                        "coverage": info, "entries": []}
    for r in results:
        e, m = r["entry"], r["metrics"]
        eid = e["id"]
        d = out / eid
        d.mkdir(exist_ok=True)
        r["trades"].to_csv(d / "trades.csv", index=False)
        r["equity"].resample("1D").last().to_csv(d / "equity_daily.csv")
        rec = {"id": eid, "sub_bot": e["sub_bot"], "timeframe": e["timeframe"], "scoring_start": e["scoring_start"],
               "closed_trades": r["closed"], "open_positions": len(r["open"]),
               "return": m["total_return"], "profit_factor": m["profit_factor"], "max_drawdown": m["max_drawdown"],
               "avg_r": m["avg_r"], "halted_at": str(r["halted_at"]) if r["halted_at"] is not None else None}
        status["entries"].append(rec)
        rows.append(rec)
    (out / "status.json").write_text(json.dumps(status, indent=2, default=str))
    hist = out / "history.csv"
    df = pd.DataFrame([{"run_at": now, "data_through": status["data_through"], **{k: r[k] for k in (
        "id", "closed_trades", "open_positions", "return", "profit_factor", "max_drawdown")}} for r in rows])
    df.to_csv(hist, mode="a", header=not hist.exists(), index=False)
    lines = [f"# Paper trading ({reg['id']})", "",
             f"Updated {now} UTC. Market data through {status['data_through']} UTC.",
             "Live risk rules, $10,000 per sub-bot, BTCUSDT + ETHUSDT perpetuals, same engine as the backtests.",
             ""]
    if est_since is not None:
        lines += [f"**Provisional:** funding since {est_since} is estimated from the premium index until Binance "
                  "publishes the monthly file. Numbers after that date can still change.", ""]
    lines += ["| Entry | Scoring since | Closed trades | Open | Return | Profit factor | Max DD | Avg R | Halted |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        pf = "n/a" if r["closed_trades"] == 0 else f"{r['profit_factor']:.2f}"
        lines.append(f"| {r['id']} | {r['scoring_start']} | {r['closed_trades']} | {r['open_positions']} | "
                     f"{r['return']:+.2%} | {pf} | {r['max_drawdown']:.1%} | {r['avg_r']:+.2f} | "
                     f"{r['halted_at'] or 'no'} |")
    lines += ["", f"Review date: {reg['review']['date']}. Criteria: see `paper/PAPER.json` in the main branch."]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    return status
