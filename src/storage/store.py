"""
Storage Module — SQLite-backed receipt store with full-text search support.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List, Optional
import os

DB_PATH = os.environ.get("DB_PATH", "outputs/receipts.db")


class ReceiptStore:
    def __init__(self, db_path: str = DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS receipts (
                id          TEXT PRIMARY KEY,
                vendor      TEXT,
                date        TEXT,
                total       TEXT,
                receipt_id  TEXT,
                raw_json    TEXT,
                confidence  TEXT,
                created_at  TEXT
            )
        """)
        # FTS5 table for natural language search
        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS receipts_fts
            USING fts5(id, vendor, date, total, receipt_id, content=receipts)
        """)
        self.conn.commit()

    def save(self, extraction: Dict) -> str:
        row_id = str(uuid.uuid4())
        self.conn.execute("""
            INSERT INTO receipts (id, vendor, date, total, receipt_id, raw_json, confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row_id,
            extraction.get("vendor", ""),
            extraction.get("date", ""),
            extraction.get("total", ""),
            extraction.get("receipt_id", ""),
            json.dumps(extraction),
            json.dumps(extraction.get("confidence", {})),
            datetime.utcnow().isoformat(),
        ))
        # Update FTS
        self.conn.execute("""
            INSERT INTO receipts_fts (id, vendor, date, total, receipt_id)
            VALUES (?, ?, ?, ?, ?)
        """, (row_id,
              extraction.get("vendor", ""),
              extraction.get("date", ""),
              extraction.get("total", ""),
              extraction.get("receipt_id", "")))
        self.conn.commit()
        return row_id

    def get(self, row_id: str) -> Optional[Dict]:
        cur = self.conn.execute("SELECT raw_json FROM receipts WHERE id=?", (row_id,))
        row = cur.fetchone()
        return json.loads(row[0]) if row else None

    def get_all(self) -> List[Dict]:
        cur = self.conn.execute(
            "SELECT id, vendor, date, total, receipt_id, created_at FROM receipts"
            " ORDER BY created_at DESC"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def search(self, query: str) -> List[Dict]:
        """FTS5 search across all text fields."""
        cur = self.conn.execute("""
            SELECT r.id, r.vendor, r.date, r.total, r.receipt_id, r.created_at
            FROM receipts r
            JOIN receipts_fts f ON r.id = f.id
            WHERE receipts_fts MATCH ?
            ORDER BY rank
        """, (query,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self):
        self.conn.close()
