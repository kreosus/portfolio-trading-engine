"""Trade database, journal and structured event log (spec §33, §34, §44)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .models import now_utc

SCHEMA = """
create table if not exists decisions (id integer primary key, ts text, bar text, action text, score real,
  reasons text, signal text);
create table if not exists orders (client_id text primary key, ts text, purpose text, side text, type text,
  qty real, price real, stop_price real, state text, exchange_id text, avg_price real, history text);
create table if not exists trades (id integer primary key, opened_at text, closed_at text, symbol text,
  direction text, strategy_version text, entry real, stop real, tp1 real, tp2 real, tp3 real, qty real,
  leverage real, margin real, risk_amount real, risk_pct real, notional real, exit_reason text,
  gross_pnl real, fees real, funding real, net_pnl real, r_multiple real, mae real, mfe real,
  score real, signal text, order_ids text, status text);
create table if not exists account (k text primary key, v text);
"""


class Journal:
    def __init__(self, path: str | Path, log_path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(SCHEMA)
        self.log_path = Path(log_path)
        self.clock = None                      # replay sets a simulated clock

    def ts(self) -> str:
        return str(self.clock) if self.clock is not None else now_utc().isoformat()

    def event(self, kind: str, **data):
        rec = {"ts": self.ts(), "event": kind, **data}
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def decision(self, bar: str, d) -> None:
        self.db.execute("insert into decisions (ts, bar, action, score, reasons, signal) values (?,?,?,?,?,?)",
                        (self.ts(), bar, d.action, d.signal.score if d.signal else None,
                         json.dumps([[ok, t] for ok, t in d.reasons]),
                         json.dumps(d.signal.to_dict(), default=str) if d.signal else None))
        self.db.commit()
        self.event("approved_signal" if d.signal else "rejected_signal", bar=bar, action=d.action,
                   failed=d.failed)

    def order(self, o) -> None:
        self.db.execute("""insert or replace into orders values (?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (o.client_id, self.ts(), o.purpose, o.side, o.type, o.qty, o.price, o.stop_price,
                         o.state.value, o.exchange_id, o.avg_price, json.dumps(o.history)))
        self.db.commit()

    def open_trade(self, **t) -> int:
        cols = ",".join(t)
        cur = self.db.execute(f"insert into trades ({cols}, status) values ({','.join('?' * len(t))}, 'OPEN')",
                              tuple(json.dumps(v, default=str) if isinstance(v, (dict, list)) else v
                                    for v in t.values()))
        self.db.commit()
        return cur.lastrowid

    def close_trade(self, trade_id: int, **t) -> None:
        sets = ",".join(f"{k}=?" for k in t)
        self.db.execute(f"update trades set {sets}, status='CLOSED' where id=?", (*t.values(), trade_id))
        self.db.commit()

    def open_trades(self) -> list[dict]:
        cur = self.db.execute("select * from trades where status='OPEN'")
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def trades(self) -> list[dict]:
        cur = self.db.execute("select * from trades order by id")
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def get(self, k: str, default=None):
        r = self.db.execute("select v from account where k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else default

    def put(self, k: str, v) -> None:
        self.db.execute("insert or replace into account values (?,?)", (k, json.dumps(v, default=str)))
        self.db.commit()
