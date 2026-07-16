"""
Persistent memory layer. Every completed audit is stored so future runs on
the same domain can reason about trends ("score dropped from 82 to 71 since
last week") instead of treating each audit as a stateless one-off. This is
what gives the agent long-term memory across separate invocations.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from urllib.parse import urlparse

from .config import DB_PATH


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                url TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                overall_score REAL NOT NULL,
                grade TEXT,
                report_json TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_audits_domain ON audits(domain)")


def domain_of(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return urlparse(url).netloc


def save_audit(url: str, report: dict) -> int:
    init_db()
    domain = domain_of(url)
    timestamp = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO audits (domain, url, timestamp, overall_score, grade, report_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (domain, url, timestamp, report.get("overall_score", 0), report.get("grade", ""), json.dumps(report)),
        )
        return cur.lastrowid


def get_last_audit(url: str) -> dict | None:
    init_db()
    domain = domain_of(url)
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM audits WHERE domain = ? ORDER BY id DESC LIMIT 1", (domain,)
        ).fetchone()
    if not row:
        return None
    report = json.loads(row["report_json"])
    report["_timestamp"] = row["timestamp"]
    return report


def get_history(url: str, limit: int = 10) -> list[dict]:
    init_db()
    domain = domain_of(url)
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, url, timestamp, overall_score, grade FROM audits "
            "WHERE domain = ? ORDER BY id DESC LIMIT ?",
            (domain, limit),
        ).fetchall()
    return [dict(r) for r in rows]
