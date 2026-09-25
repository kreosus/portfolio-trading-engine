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
from .data.store import HTF_FOR, load_bars, load_funding, resample_bars, split_holdout
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
    tf = a.timeframe
    bars = {s: resample_bars(df, tf) for s, df in bars.items()}
    cfg = with_overrides(cfg, {"smc.htf_rule": HTF_FOR[tf]})
    if a.research_mode:
        # measurement only: let each sub-bot trade the whole window instead of stopping at its drawdown halt
        cfg = with_overrides(cfg, {"risk.drawdown_halt": 1.0, "risk.max_consecutive_losses": 10**9})
    tag = f"{tf}{'_research' if a.research_mode else ''}"
    start = None if a.synthetic else pd.Timestamp(cfg["bots"]["report_start"], tz="UTC")
    rep = bot_report(bars, funding, cfg, start=start)
    t = rep["sleeves"]
    first = min(df.index[0] for df in bars.values()) if start is None else start
    last = min(df.index[-1] for df in bars.values())
    print(f"Sub-bots on {tf} bars, each trading the full ${cfg['risk']['initial_equity']:,.0f} independently"
          + (" [research mode: drawdown halt and loss-streak pause off]" if a.research_mode else ""))
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
    t.to_csv(REPORTS / f"bots_{tag}_summary.csv")
    rep["correlation"].to_csv(REPORTS / f"bots_{tag}_correlation.csv")
    for name, r in rep["result"].sleeves.items():
        r.trades.to_csv(REPORTS / f"bots_{tag}_{name}_trades.csv", index=False)
        r.equity.resample("1D").last().to_csv(REPORTS / f"bots_{tag}_{name}_equity_daily.csv")
    log = ExperimentLog(REPORTS / ("experiments_synthetic.jsonl" if a.synthetic else "experiments.jsonl"))
    for name, row in t.iterrows():
        log.write(kind="bot_sleeve", sleeve=name, timeframe=tf, research_mode=bool(a.research_mode), trades=int(row.trades), sharpe=float(row.sharpe),
                  profit_factor=float(row.pf), start=first, end=last)


def cmd_holdout_bots(cfg, a):
    """One-shot holdout test of pre-registered sub-bots. Reads everything from the JSON."""
    from .backtest.metrics import summarize
    from .bots import run_bot
    from .config import config_hash
    reg = json.loads(Path(a.prereg).read_text())
    lock = REPORTS / f"HOLDOUT_OPENED_{reg['id']}.json"
    if lock.exists():
        print("This holdout test was already run:", lock.read_text())
        return
    if not a.confirm:
        print(f"Pre-registration {reg['id']} will use the locked holdout ONCE. Re-run with --confirm.")
        return
    dev, funding, hold = _load(cfg, include_holdout=True)
    start = pd.Timestamp(reg["holdout_window"]["start"], tz="UTC")
    assert min(h.index[0] for h in hold.values()) == start, "holdout window does not match pre-registration"
    full15 = {s: pd.concat([dev[s], hold[s]]) for s in dev}
    crit = reg["criteria"]
    results, all_pass = [], True
    for c in reg["candidates"]:
        name, tf = c["sub_bot"], c["timeframe"]
        run_cfg = with_overrides(cfg, {"smc.htf_rule": HTF_FOR[tf]})
        if config_hash(run_cfg) != c["config_hash"]:
            raise SystemExit(f"{name}: config changed since pre-registration ({config_hash(run_cfg)} != {c['config_hash']})")
        only = {**run_cfg["bots"]["sleeves"]}
        for k in only:
            only[k] = {**only[k], "enabled": k == name}
        run_cfg = {**run_cfg, "bots": {**run_cfg["bots"], "sleeves": only}}
        bars = {s: resample_bars(df, tf) for s, df in full15.items()}
        out = {}
        for mode, rc in (("live", run_cfg),
                         ("research", with_overrides(run_cfg, {"risk.drawdown_halt": 1.0,
                                                               "risk.max_consecutive_losses": 10**9}))):
            r1 = run_bot(bars, funding, rc, start=start).sleeves[name]
            r2 = run_bot(bars, funding, with_overrides(rc, {"costs.multiplier": 2.0}), start=start).sleeves[name]
            m = summarize(r1.trades, r1.equity, r1.initial_equity)
            m2 = summarize(r2.trades, r2.equity, r2.initial_equity)
            out[mode] = {**{k: m[k] for k in ("trades", "total_return", "sharpe", "profit_factor",
                                                "max_drawdown", "win_rate", "avg_r")},
                         "profit_factor_2x_costs": m2["profit_factor"], "halted_at": str(r1.halted_at)}
            if mode == "live":
                r1.trades.to_csv(REPORTS / f"holdout_{name}_{tf}_trades.csv", index=False)
        L = out["live"]
        checks = {
            f"trades >= {c['min_trades']}": L["trades"] >= c["min_trades"],
            "net return > 0": L["total_return"] > crit["net_return_gt"],
            f"profit factor >= {crit['profit_factor_gte']}": L["profit_factor"] >= crit["profit_factor_gte"],
            f"PF at 2x costs >= {crit['profit_factor_2x_costs_gte']}":
                L["profit_factor_2x_costs"] >= crit["profit_factor_2x_costs_gte"],
            "not halted": L["halted_at"] == "None",
        }
        passed = all(checks.values())
        all_pass &= passed
        results.append({"sub_bot": name, "timeframe": tf, "verdict": "PASS" if passed else "FAIL",
                        "checks": checks, **out})
        print(f"\n{name} {tf}: {'PASS' if passed else 'FAIL'}")
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        print(f"  live:     trades {L['trades']}, return {L['total_return']:+.2%}, PF {L['profit_factor']:.2f}, "
              f"PF 2x {L['profit_factor_2x_costs']:.2f}, Sharpe {L['sharpe']:.2f}, max DD {L['max_drawdown']:.1%}, "
              f"avg R {L['avg_r']:+.2f}, halted {L['halted_at']}")
        R = out["research"]
        print(f"  no halt:  trades {R['trades']}, return {R['total_return']:+.2%}, PF {R['profit_factor']:.2f}, "
              f"Sharpe {R['sharpe']:.2f}, max DD {R['max_drawdown']:.1%}  (reported, not scored)")
    rec = {"prereg": reg["id"], "opened_at": datetime.now(timezone.utc).isoformat(), "results": results}
    REPORTS.mkdir(exist_ok=True)
    lock.write_text(json.dumps(rec, indent=2, default=str))
    ExperimentLog(REPORTS / "experiments.jsonl").write(kind="holdout_bots", prereg=reg["id"],
                                                       verdicts={r["sub_bot"]: r["verdict"] for r in results})


def cmd_robustness(cfg, a):
    from .research.robustness import random_entry_baseline, robustness
    bars, funding = _load(cfg)
    start = pd.Timestamp(cfg["bots"]["report_start"], tz="UTC")
    r = robustness(bars, funding, cfg, a.sub_bot, a.timeframe, start)
    print(f"{a.sub_bot} on {a.timeframe}, 2020-2025 development data, research mode\n")
    print(r["neighbourhood"].to_string(float_format=lambda v: f"{v:.2f}"))
    print("\nby year:", "  ".join(f"{y}: {v:+.1%}" for y, v in r["yearly"].items()))
    print(r["splits"].round(2).to_string())
    s = r["stats"]
    print(f"\nmean R {s['mean_r']:+.3f}, bootstrap 95% CI {s['mean_r_ci95'][0]:+.3f}..{s['mean_r_ci95'][1]:+.3f}, "
          f"P(mean<=0) {s['p_mean_le_0']:.3f} (uncorrected for configurations tried)")
    b = random_entry_baseline(bars, funding, cfg, a.sub_bot, a.timeframe, start, n_seeds=a.seeds)
    print(f"random entries, same exits: median mean R {b['random_mean_r_median']:+.3f} "
          f"(5-95%: {b['random_mean_r_p5_p95'][0]:+.3f}..{b['random_mean_r_p5_p95'][1]:+.3f}); "
          f"random >= real in {b['share_random_ge_real']:.0%} of {b['n_seeds']} runs")


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
    b.add_argument("--timeframe", choices=["15m", "1h", "4h"], default="15m",
                   help="bar size; parameters are in bars, so 1h/4h span 4x/16x the time")
    b.add_argument("--research-mode", action="store_true",
                   help="turn off drawdown halt and loss-streak pause to measure the full-window edge")
    rb = sub.add_parser("robustness", help="settings neighbourhood, splits, bootstrap and random-entry baseline")
    rb.add_argument("--sub-bot", required=True, choices=["trend_pullback", "vol_breakout", "vwap_reversion", "funding_momentum"])
    rb.add_argument("--timeframe", choices=["15m", "1h", "4h"], default="4h")
    rb.add_argument("--seeds", type=int, default=40)
    hb = sub.add_parser("holdout-bots", help="one-shot holdout test of pre-registered sub-bots")
    hb.add_argument("--prereg", default="research/2026-09-25-holdout-bots.json")
    hb.add_argument("--confirm", action="store_true")
    h = sub.add_parser("holdout", help="open the locked holdout once")
    h.add_argument("--confirm", action="store_true"); h.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    cfg = load_config(a.config)
    {"download": cmd_download, "process": cmd_process, "backtest": cmd_backtest,
     "walkforward": cmd_walkforward, "bots": cmd_bots, "holdout-bots": cmd_holdout_bots, "robustness": cmd_robustness, "holdout": cmd_holdout}[a.cmd](cfg, a)


if __name__ == "__main__":
    main()
