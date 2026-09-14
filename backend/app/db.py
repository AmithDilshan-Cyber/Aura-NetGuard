"""Lightweight SQLite persistence.

Kept deliberately simple (stdlib sqlite3, no ORM) since this is a research
prototype: it exists so alerts/feedback survive a server restart and so the
metric history / feedback log can be queried later for evaluation, not to
serve as a production data-store design.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

DB_PATH = Path(__file__).parent.parent / "data" / "aura_netguard.db"
METRIC_RETENTION_HOURS = 6

SCHEMA = """
CREATE TABLE IF NOT EXISTS metrics (
    device_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    latency_ms REAL, packet_loss_pct REAL, jitter_ms REAL, bandwidth_util_pct REAL,
    cpu_pct REAL, memory_pct REAL, interface_errors_per_min REAL, temperature_c REAL,
    retransmits_per_min REAL, link_up INTEGER, device_state TEXT, active_fault TEXT,
    failure_event INTEGER
);
CREATE INDEX IF NOT EXISTS idx_metrics_device_ts ON metrics(device_id, ts);

CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    device_id TEXT, device_name TEXT, device_type TEXT, site TEXT,
    category TEXT, severity TEXT, probability REAL, eta_minutes REAL,
    summary TEXT, root_cause TEXT, causes_json TEXT, actions_json TEXT,
    status TEXT, created_at TEXT, updated_at TEXT, last_seen_at TEXT,
    detections_count INTEGER, peak_probability REAL,
    feedback TEXT, feedback_at TEXT, resolved_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
"""


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_metric(snapshot: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO metrics (device_id, ts, latency_ms, packet_loss_pct, jitter_ms,
               bandwidth_util_pct, cpu_pct, memory_pct, interface_errors_per_min, temperature_c,
               retransmits_per_min, link_up, device_state, active_fault, failure_event)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                snapshot["device_id"], snapshot["timestamp"], snapshot["latency_ms"],
                snapshot["packet_loss_pct"], snapshot["jitter_ms"], snapshot["bandwidth_util_pct"],
                snapshot["cpu_pct"], snapshot["memory_pct"], snapshot["interface_errors_per_min"],
                snapshot["temperature_c"], snapshot["retransmits_per_min"],
                int(snapshot["link_up"]), snapshot["device_state"], snapshot["active_fault"],
                int(snapshot["failure_event"]),
            ),
        )


def fetch_recent_metrics(device_id: str, limit: int = 120) -> list[dict]:
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM metrics WHERE device_id=? ORDER BY ts DESC LIMIT ?",
            (device_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def prune_old_metrics() -> None:
    cutoff = (datetime.utcnow() - timedelta(hours=METRIC_RETENTION_HOURS)).isoformat()
    with get_conn() as conn:
        conn.execute("DELETE FROM metrics WHERE ts < ?", (cutoff,))


def upsert_alert(alert_dict: dict) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO alerts (id, device_id, device_name, device_type, site, category,
               severity, probability, eta_minutes, summary, root_cause, causes_json, actions_json,
               status, created_at, updated_at, last_seen_at, detections_count, peak_probability,
               feedback, feedback_at, resolved_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 severity=excluded.severity, probability=excluded.probability,
                 eta_minutes=excluded.eta_minutes, summary=excluded.summary,
                 root_cause=excluded.root_cause, causes_json=excluded.causes_json,
                 actions_json=excluded.actions_json, status=excluded.status,
                 updated_at=excluded.updated_at, last_seen_at=excluded.last_seen_at,
                 detections_count=excluded.detections_count, peak_probability=excluded.peak_probability,
                 feedback=excluded.feedback, feedback_at=excluded.feedback_at,
                 resolved_at=excluded.resolved_at
            """,
            (
                alert_dict["id"], alert_dict["device_id"], alert_dict["device_name"],
                alert_dict["device_type"], alert_dict["site"], alert_dict["category"],
                alert_dict["severity"], alert_dict["probability"], alert_dict["eta_minutes"],
                alert_dict["summary"], alert_dict["root_cause"],
                json.dumps(alert_dict["causes"]), json.dumps(alert_dict["recommended_actions"]),
                alert_dict["status"], alert_dict["created_at"], alert_dict["updated_at"],
                alert_dict["last_seen_at"], alert_dict["detections_count"], alert_dict["peak_probability"],
                alert_dict["feedback"], alert_dict["feedback_at"], alert_dict["resolved_at"],
            ),
        )


def fetch_all_alerts() -> list[dict]:
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM alerts ORDER BY updated_at DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["causes"] = json.loads(d.pop("causes_json"))
        d["recommended_actions"] = json.loads(d.pop("actions_json"))
        out.append(d)
    return out


def feedback_stats() -> dict:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT feedback, COUNT(*) FROM alerts WHERE feedback IS NOT NULL GROUP BY feedback"
        ).fetchall()
    stats = {"true_positive": 0, "false_positive": 0}
    for fb, count in rows:
        stats[fb] = count
    total = stats["true_positive"] + stats["false_positive"]
    stats["precision"] = round(stats["true_positive"] / total, 3) if total else None
    stats["total_rated"] = total
    return stats
