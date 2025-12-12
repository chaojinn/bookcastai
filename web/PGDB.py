"""PostgreSQL database helper for application tables."""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import connection as PGConnection


class PGDB:
    """Thin wrapper around a PostgreSQL connection with table bootstrap helpers."""

    def __init__(self) -> None:
        self.conn: Optional[PGConnection] = None

    def _cursor(self):
        if self.conn is None:
            raise RuntimeError("Database not connected")
        return self.conn.cursor()

    def connect(self) -> PGConnection:
        """Connect using environment variables and return a live connection."""
        if self.conn is not None:
            return self.conn

        load_dotenv(override=False)
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASS")
        host = os.getenv("DB_HOST")
        db_name = os.getenv("DB_NAME")
        port = int(os.getenv("DB_PORT", "5432"))

        if not all([user, password, host, db_name]):
            raise ValueError("Database credentials are missing; check DB_USER/DB_PASS/DB_HOST/DB_NAME.")

        self.conn = psycopg2.connect(
            dbname=db_name,
            user=user,
            password=password,
            host=host,
            port=port,
        )
        self.conn.autocommit = True
        return self.conn

    def disconnect(self) -> None:
        """Close the database connection if open."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def init(self) -> None:
        """Create required tables if they do not already exist."""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS player_queue (
                    user_id CHAR(36),
                    idx INT,
                    pod_id VARCHAR(128),
                    episode_idx INT,
                    start_pos INT DEFAULT 0,
                    PRIMARY KEY (user_id, idx)
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS last_location (
                    user_id CHAR(36) PRIMARY KEY,
                    uri VARCHAR(128)
                );
                """
            )

    def update_last_location(self, user_id: str, uri: str) -> None:
        """Insert or update the last_location for a user."""
        if not user_id or not uri:
            raise ValueError("user_id and uri are required")
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO last_location (user_id, uri)
                VALUES (%s, %s)
                ON CONFLICT (user_id) DO UPDATE SET uri = EXCLUDED.uri;
                """,
                (user_id, uri),
            )

    def get_last_location(self, user_id: str) -> Optional[str]:
        """Fetch the last recorded URI for a user; returns None if not found."""
        if not user_id:
            raise ValueError("user_id is required")
        with self._cursor() as cur:
            cur.execute(
                "SELECT uri FROM last_location WHERE user_id = %s LIMIT 1;",
                (user_id,),
            )
            row = cur.fetchone()
            return row[0] if row else None
