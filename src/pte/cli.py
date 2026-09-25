"""Command line: pte download | process | backtest | walkforward | holdout | demo"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .backtest.metrics import summarize
from .config import load_config, with_overrides
from .data import binance
from .data.store import load_bars, load_funding, split_holdout
from .data.synthetic import make_bars, make_funding
from .pipeline import backtest
from .research.walkforward import ExperimentLog, kill_check, walk_forward

REPORTS = Path("reports")
HOLDOUT_LOCK = REPORTS / "HOLDOUT_OPENED.json"


def _load(cfg, include_holdout=False):
    d = cfg["data"]
    bars, funding, holdout = {}, {}, {}
    for s in d["symbols"]:
        df = load_bars(s, d["processed_dir"])
        dev, hold = split_holdout(df, d["holdout_months"])
        bars[s], holdout[s] = dev, hold
        funding[s] = load_funding(s, d["processed_dir"])
    return (bars, funding, holdout) if include_holdout else (bars, funding)


def _synthetic(cfg, n):
    bars = {s: make_bars(n, seed=i, price=30000 / (i * 15 + 1)) for i, s in enumerate(cfg["data"]["symbols"])}
    funding = {s: make_funding(b, seed=i) for i, (s, b) in enumerate(bars.items())}
    return bars, funding


def _print_summary(m: dict) -> None:
    for k, v in m.items():
        print(f"  {k:28s} {v:.4f}" if isinstance(v, float) else f"  {k:28s} {v}")


def _print_kill(checks) -> bool:
    print("\nKill criteria (fixed before testing):")
    for name, ok, val in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:32s} {val}")
    passed = all(ok for _, ok, _ in checks)
    print("\nVERDICT:", "advance to Phase 2" if passed else "shelve strategy (do not re-tune)")
    return passed


def cmd_download(cfg, a):
    d = cfg["data"]
    binance.download(d["symbols"], d["interval"], a.start or d["start"], d["raw_dir"], a.end)


def cmd_process(cfg, a):
    d = cfg["data"]
    for s in d["symbols"]:
        df = binance.process_klines(s, d["raw_dir"], d["processed_dir"])
        fu = binance.process_funding(s, d["raw_dir"], d["processed_dir"])
        print(f"{s}: {len(df):,} bars {df.index[0]} .. {df.index[-1]}; {len(fu):,} funding prints")


def cmd_backtest(cfg, a):
    bars, funding = _synthetic(cfg, a.synthetic) if a.synthetic else _load(cfg)
    res = backtest(bars, funding, cfg)
    m = summarize(res.trades, res.equity, res.initial_equity)
    print(f"Backtest on {'SYNTHETIC' if a.synthetic else 'development'} data (holdout excluded):")
    _print_summary(m)
    print("  risk blocks:", res.risk_blocks, "| halted at:", res.halted_at)
    REPORTS.mkdir(exist_ok=True)
    res.trades.to_csv(REPORTS / "backtest_trades.csv", index=False)
    res.equity.to_csv(REPORTS / "backtest_equity.csv")
    ExperimentLog(REPORTS / "experiments.jsonl").write(kind="full_backtest", synthetic=bool(a.synthetic),
                                                       **{k: m[k] for k in ("trades", "sharpe", "profit_factor")})


def cmd_walkforward(cfg, a):
    bars, funding = _synthetic(cfg, a.synthetic) if a.synthetic else _load(cfg)
    REPORTS.mkdir(exist_ok=True)
    log = ExperimentLog(REPORTS / ("experiments_synthetic.jsonl" if a.synthetic else "experiments.jsonl"))
    out = walk_forward(bars, funding, cfg, log=log)
    out["folds"].to_csv(REPORTS / "walkforward_folds.csv", index=False)
    out["oos_trades"].to_csv(REPORTS / "walkforward_oos_trades.csv", index=False)
    out["oos_equity_index"].to_csv(REPORTS / "walkforward_oos_equity.csv")
    print(f"\nOut-of-sample summary ({'SYNTHETIC' if a.synthetic else 'real'} data):")
    _print_summary(out["summary"])
    print(f"  configurations evaluated this run: {out['configs_tried']}; "
          f"all-time log entries: {log.count()}")
    (REPORTS / "walkforward_summary.json").write_text(json.dumps(
        {**out["summary"], "latest_params": out["latest_params"], "configs_tried": out["configs_tried"]},
        indent=2, default=str))
    _print_kill(kill_check(out["summary"], cfg["kill_criteria"]))


def cmd_bots(cfg, a):
    from .bots import bot_report
    bars, funding = _synthetic(cfg, a.synthetic) if a.synthetic else _load(cfg)
    start = None if a.synthetic else pd.Timestamp(cfg["bots"]["report_start"], tz="UTC")
    rep = bot_report(bars, funding, cfg, start=start)
    t = rep["sleeves"]
    first = min(df.index[0] for df in bars.values()) if start is None else start
    last = min(df.index[-1] for df in bars.values())
    print(f"Sub-bots, each trading the full ${cfg['risk']['initial_equity']:,.0f} independently")
    print(f"Window: {first.date()} .. {last.date()} ({'SYNTHETIC' if a.synthetic else 'real'} data, holdout excluded)\n")
    fmt = t.copy()
    for col in ("return", "max_dd", "win_rate", "profitable_halves"):
        fmt[col] = fmt[col].map(lambda v: f"{v:+.1%}" if col == "return" else f"{v:.0%}")
    for col in ("sharpe", "pf", "pf_2x_costs", "avg_r", "gross_avg_r"):
        fmt[col] = fmt[col].map(lambda v: f"{v:.2f}")
    fmt["passes"] = fmt["passes"].map(lambda v: f"{v}/5")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(fmt.to_string())
        print("\nDaily-return correlation between sub-bots:")
        print(rep["correlation"].round(2).to_string())
    REPORTS.mkdir(exist_ok=True)
    t.to_csv(REPORTS / "bots_summary.csv")
    rep["correlation"].to_csv(REPORTS / "bots_correlation.csv")
    for name, r in rep["result"].sleeves.items():
        r.trades.to_csv(REPORTS / f"bots_{name}_trades.csv", index=False)
        r.equity.resample("1D").last().to_csv(REPORTS / f"bots_{name}_equity_daily.csv")
    log = ExperimentLog(REPORTS / ("experiments_synthetic.jsonl" if a.synthetic else "experiments.jsonl"))
    for name, row in t.iterrows():
        log.write(kind="bot_sleeve", sleeve=name, trades=int(row.trades), sharpe=float(row.sharpe),
                  profit_factor=float(row.pf), start=first, end=last)


def cmd_holdout(cfg, a):
    if HOLDOUT_LOCK.exists() and not a.force:
        print("Holdout already opened:", HOLDOUT_LOCK.read_text())
        print("Re-running it turns the holdout into training data. Use --force only if you accept that.")
        return
    if not a.confirm:
        print("This opens the locked holdout ONCE. Re-run with --confirm when Phase 1 research is final.")
        return
    summ = json.loads((REPORTS / "walkforward_summary.json").read_text())
    params = summ["latest_params"]
    run_cfg = with_overrides(cfg, {"smc.swing_n": params["swing_n"], "strategy.target_r": params["target_r"]})
    dev, funding, hold = _load(cfg, include_holdout=True)
    full = {s: pd.concat([dev[s], hold[s]]) for s in dev}
    start = min(h.index[0] for h in hold.values())
    res = backtest(full, funding, run_cfg, start=start)
    m = summarize(res.trades, res.equity, res.initial_equity)
    print(f"Holdout from {start.date()} with params {params}:")
    _print_summary(m)
    HOLDOUT_LOCK.write_text(json.dumps({"opened_at": datetime.now(timezone.utc).isoformat(),
                                        "params": params, "return": m["total_return"]}, indent=2))
    ExperimentLog(REPORTS / "experiments.jsonl").write(kind="holdout", params=params,
                                                       **{k: m[k] for k in ("trades", "sharpe", "profit_factor")})
    _print_kill(kill_check(summ, cfg["kill_criteria"], holdout_return=m["total_return"]))


def main(argv=None):
    p = argparse.ArgumentParser(prog="pte")
    p.add_argument("--config", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="download Binance USD-M archives")
    d.add_argument("--start"); d.add_argument("--end")
    sub.add_parser("process", help="raw zips -> parquet")
    for name in ("backtest", "walkforward"):
        s = sub.add_parser(name)
        s.add_argument("--synthetic", type=int, default=0, metavar="N_BARS",
                       help="run on N synthetic bars instead of real data (pipeline check only)")
    b = sub.add_parser("bots", help="run all five sub-bots, each on the full account, and report separately")
    b.add_argument("--synthetic", type=int, default=0, metavar="N_BARS")
    h = sub.add_parser("holdout", help="open the locked holdout once")
    h.add_argument("--confirm", action="store_true"); h.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    cfg = load_config(a.config)
    {"download": cmd_download, "process": cmd_process, "backtest": cmd_backtest,
     "walkforward": cmd_walkforward, "bots": cmd_bots, "holdout": cmd_holdout}[a.cmd](cfg, a)


if __name__ == "__main__":
    main()
