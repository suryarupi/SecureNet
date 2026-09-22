"""
SecureNet - Live Data Store
============================

SQLite bridge between live_capture/packet_capture.py (writer) and
dashboard/app.py (reader).

Design:
- WAL mode so the capture process can write while the dashboard reads
  concurrently without locking errors.
- `flows` table holds the compact decision summary shown in the UI
  (matches the exact columns requested for the dashboard).
- `flow_features` is a SEPARATE table holding the 52 raw features,
  keyed by flow id. It is NOT joined into the main flow table or
  shown in the live table - it exists only so the dashboard can
  re-run SHAP on demand via the "Explain Prediction" button.
- `heartbeat` is a single-row table the capture process updates every
  sniff cycle (~2s) so the dashboard can show LIVE / OFFLINE status
  without needing a second process check.

All read functions open/close their own short-lived connection -
this is the simplest reliable pattern for a Streamlit app that
reruns the whole script on every refresh.
"""

import os
import sqlite3
import time
import json

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DB_PATH = os.path.join(PROJECT_ROOT, "data", "securenet_live.db")


# ============================================================
# Connection helpers
# ============================================================

def _ensure_data_dir():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def get_connection():
    """
    Open a new connection with WAL mode enabled.
    Safe to call from either the capture process or the
    dashboard process - each call is independent.
    """

    _ensure_data_dir()

    conn = sqlite3.connect(
        DB_PATH,
        timeout=10,
        check_same_thread=False
    )

    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.row_factory = sqlite3.Row

    return conn


def _json_default(value):
    # Handles numpy scalar types (np.float64, np.int64, np.bool_)
    # that json.dumps can't serialize directly.
    if hasattr(value, "item"):
        return value.item()
    return str(value)


# ============================================================
# Schema
# ============================================================

def init_db(conn=None):
    """
    Create tables if they don't exist yet. Safe to call every
    time the capture process or dashboard starts up.
    """

    owns_conn = conn is None

    if owns_conn:
        conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS flows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            src_ip TEXT,
            src_port INTEGER,
            dst_ip TEXT,
            dst_port INTEGER,
            protocol TEXT,
            status TEXT,
            severity TEXT,
            attack_type TEXT,
            attack_probability REAL,
            attack_confidence REAL,
            anomaly INTEGER,
            anomaly_score REAL,
            reason TEXT
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_flows_timestamp
        ON flows (timestamp DESC)
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS flow_features (
            flow_id INTEGER PRIMARY KEY,
            features_json TEXT NOT NULL,
            FOREIGN KEY (flow_id) REFERENCES flows(id)
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS heartbeat (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            interface TEXT,
            last_seen REAL
        )
    """)

    conn.commit()

    if owns_conn:
        conn.close()


# ============================================================
# Writer (used by live_capture/packet_capture.py)
# ============================================================

class LiveDataStore:
    """
    Long-lived writer used by the capture process. Keeping one
    connection open for the life of the capture process avoids
    reopening SQLite on every single flow.
    """

    def __init__(self):
        self.conn = get_connection()
        init_db(self.conn)

    def insert_flow(self, row, features=None):
        """
        row: dict with the flows table columns (timestamp, src_ip,
             src_port, dst_ip, dst_port, protocol, status, severity,
             attack_type, attack_probability, attack_confidence,
             anomaly, anomaly_score, reason)
        features: optional dict of the 52 raw features for this flow,
                  stored separately for on-demand SHAP explanation.

        Never raises - a storage failure must not take down the
        live capture pipeline.
        """

        try:
            cur = self.conn.execute(
                """
                INSERT INTO flows (
                    timestamp, src_ip, src_port, dst_ip, dst_port,
                    protocol, status, severity, attack_type,
                    attack_probability, attack_confidence,
                    anomaly, anomaly_score, reason
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    row.get("timestamp"),
                    row.get("src_ip"),
                    row.get("src_port"),
                    row.get("dst_ip"),
                    row.get("dst_port"),
                    row.get("protocol"),
                    row.get("status"),
                    row.get("severity"),
                    row.get("attack_type"),
                    row.get("attack_probability"),
                    row.get("attack_confidence"),
                    int(bool(row.get("anomaly"))),
                    row.get("anomaly_score"),
                    row.get("reason"),
                )
            )

            flow_id = cur.lastrowid

            if features is not None:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO flow_features
                        (flow_id, features_json)
                    VALUES (?, ?)
                    """,
                    (flow_id, json.dumps(features, default=_json_default))
                )

            self.conn.commit()
            return flow_id

        except Exception as e:
            print(f"[!] Failed to store flow in database: {e}")
            return None

    def heartbeat(self, interface):
        """
        Upsert the single heartbeat row. Never raises.
        """

        try:
            self.conn.execute(
                """
                INSERT INTO heartbeat (id, interface, last_seen)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    interface = excluded.interface,
                    last_seen = excluded.last_seen
                """,
                (interface, time.time())
            )
            self.conn.commit()

        except Exception as e:
            print(f"[!] Failed to update heartbeat: {e}")

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


# ============================================================
# Readers (used by dashboard/app.py)
# ============================================================

def fetch_recent_flows(limit=500):
    """
    Returns a pandas DataFrame of the most recent flows,
    newest first. Returns an empty DataFrame (not an error)
    if the database or table doesn't exist yet.
    """

    import pandas as pd

    try:
        conn = get_connection()
        init_db(conn)

        df = pd.read_sql_query(
            "SELECT * FROM flows ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,)
        )

        conn.close()
        return df

    except Exception as e:
        print(f"[!] Failed to fetch flows: {e}")
        return pd.DataFrame()


def fetch_summary_counts():
    """
    Returns a dict of live counters. Always returns valid
    zeroed counts even if the table is empty or missing.
    """

    keys = ("total", "normal", "suspicious", "attack",
            "high", "medium", "low")
    counts = {k: 0 for k in keys}

    try:
        conn = get_connection()
        init_db(conn)

        def count(where=None):
            q = "SELECT COUNT(*) FROM flows"
            if where:
                q += f" WHERE {where}"
            return conn.execute(q).fetchone()[0]

        counts["total"] = count()
        counts["normal"] = count("status = 'NORMAL'")
        counts["suspicious"] = count("status = 'SUSPICIOUS'")
        counts["attack"] = count("status = 'ATTACK'")
        counts["high"] = count("severity = 'HIGH'")
        counts["medium"] = count("severity = 'MEDIUM'")
        counts["low"] = count("severity = 'LOW'")

        conn.close()

    except Exception as e:
        print(f"[!] Failed to fetch summary counts: {e}")

    return counts


def get_heartbeat():
    """
    Returns {"interface": str, "last_seen": float} or None
    if capture has never run.
    """

    try:
        conn = get_connection()
        init_db(conn)

        row = conn.execute(
            "SELECT interface, last_seen FROM heartbeat WHERE id = 1"
        ).fetchone()

        conn.close()

        if row is None:
            return None

        return {"interface": row["interface"], "last_seen": row["last_seen"]}

    except Exception as e:
        print(f"[!] Failed to read heartbeat: {e}")
        return None


def fetch_flow_features(flow_id):
    """
    Returns the stored 52-feature dict for a flow, or None if
    unavailable (e.g. flow was ignored, or predates this feature).
    """

    try:
        conn = get_connection()
        init_db(conn)

        row = conn.execute(
            "SELECT features_json FROM flow_features WHERE flow_id = ?",
            (flow_id,)
        ).fetchone()

        conn.close()

        if row is None:
            return None

        return json.loads(row["features_json"])

    except Exception as e:
        print(f"[!] Failed to fetch flow features: {e}")
        return None


def clear_all_flows(keep_heartbeat=True):
    """
    Deletes all rows from `flows` and `flow_features` so the dashboard
    starts fresh. Safe to call at any time - if the capture process is
    running concurrently and inserts a new flow right after this runs,
    that new flow is unaffected (it just becomes the new first row).

    keep_heartbeat: if True (default), the LIVE/OFFLINE status is left
        alone - clearing history shouldn't make a running capture
        process look offline. Set False to also reset heartbeat.

    Returns True on success, False on failure (never raises).
    """

    try:
        conn = get_connection()
        init_db(conn)

        conn.execute("DELETE FROM flow_features")
        conn.execute("DELETE FROM flows")

        if not keep_heartbeat:
            conn.execute("DELETE FROM heartbeat")

        # Reclaim the autoincrement counter so new flow ids start
        # back at 1 (purely cosmetic, but nicer for a fresh demo run).
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'flows'")

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"[!] Failed to clear flow history: {e}")
        return False