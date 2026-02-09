"""PostgreSQL database helper for application tables."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

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
                    is_current BOOLEAN DEFAULT FALSE,
                    PRIMARY KEY (user_id, idx)
                );
                """
            )
            cur.execute(
                """
                ALTER TABLE player_queue
                ADD COLUMN IF NOT EXISTS is_current BOOLEAN DEFAULT FALSE;
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
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS books (
                    id SERIAL PRIMARY KEY,
                    user_id CHAR(36) NOT NULL,
                    book_title VARCHAR(255) NOT NULL,
                    folder_path VARCHAR(512) NOT NULL,
                    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    modified_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    visibility SMALLINT DEFAULT 1,
                    UNIQUE(user_id, book_title)
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_books_user_id ON books(user_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_books_visibility ON books(visibility);"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS clips (
                    id SERIAL PRIMARY KEY,
                    user_id CHAR(36) NOT NULL,
                    pod_title VARCHAR(255) NOT NULL,
                    start_timestamp DOUBLE PRECISION NOT NULL,
                    end_timestamp DOUBLE PRECISION NOT NULL,
                    transcript JSONB NOT NULL,
                    video_file_path VARCHAR(512),
                    video_url VARCHAR(512),
                    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_clips_user_id ON clips(user_id);"
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

    def add_to_queue(
        self,
        user_id: str,
        pod_id: str,
        episode_idx: int,
        start_pos: int = 0,
        is_current: bool = False,
    ) -> bool:
        """
        Add an episode to the player's queue or update its start position if it already exists.

        Returns False when an existing record was updated; otherwise inserts with idx = current max + 1
        and returns True.
        """
        if not user_id or not pod_id:
            raise ValueError("user_id and pod_id are required")
        if episode_idx is None:
            raise ValueError("episode_idx is required")

        with self._cursor() as cur:
            cur.execute(
                """
                SELECT start_pos FROM player_queue
                WHERE user_id = %s AND pod_id = %s AND episode_idx = %s
                LIMIT 1;
                """,
                (user_id, pod_id, episode_idx),
            )
            existing = cur.fetchone()
            if existing:
                # Update the start position for the existing queue entry.
                cur.execute(
                    """
                    UPDATE player_queue
                    SET start_pos = %s,
                        is_current = %s
                    WHERE user_id = %s AND pod_id = %s AND episode_idx = %s;
                    """,
                    (start_pos, is_current, user_id, pod_id, episode_idx),
                )
                return False

            cur.execute(
                "SELECT COALESCE(MAX(idx), 0) FROM player_queue WHERE user_id = %s;",
                (user_id,),
            )
            max_idx_row = cur.fetchone()
            next_idx = (max_idx_row[0] if max_idx_row else 0) + 1

            cur.execute(
                """
                INSERT INTO player_queue (user_id, idx, pod_id, episode_idx, start_pos, is_current)
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                (user_id, next_idx, pod_id, episode_idx, start_pos, is_current),
            )
            return True

    def get_player_queue(self, user_id: str) -> list[dict[str, object]]:
        """Return the player's queue for a user ordered by idx."""
        if not user_id:
            raise ValueError("user_id is required")
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT idx, pod_id, episode_idx, start_pos, is_current
                FROM player_queue
                WHERE user_id = %s
                ORDER BY idx ASC;
                """,
                (user_id,),
            )
            rows = cur.fetchall()
            return [
                {
                    "idx": row[0],
                    "pod_id": row[1],
                    "episode_idx": row[2],
                    "start_pos": row[3],
                    "is_current": row[4],
                }
                for row in rows
            ]

    def move_queue_item(self, user_id: str, idx: int, direction: str) -> bool:
        """Move a queue item up or down; returns False when move is not possible."""
        if not user_id:
            raise ValueError("user_id is required")
        if direction not in ("up", "down"):
            raise ValueError("direction must be 'up' or 'down'")
        delta = -1 if direction == "up" else 1
        target_idx = idx + delta

        with self._cursor() as cur:
            # Ensure both source and target rows exist for the swap.
            cur.execute(
                """
                SELECT idx FROM player_queue
                WHERE user_id = %s AND idx IN (%s, %s)
                """,
                (user_id, idx, target_idx),
            )
            rows = {row[0] for row in cur.fetchall()}
            if idx not in rows or target_idx not in rows:
                return False

            # Use a temporary placeholder to avoid PK conflicts during swap.
            cur.execute(
                "UPDATE player_queue SET idx = -1 WHERE user_id = %s AND idx = %s;",
                (user_id, target_idx),
            )
            cur.execute(
                "UPDATE player_queue SET idx = %s WHERE user_id = %s AND idx = %s;",
                (target_idx, user_id, idx),
            )
            cur.execute(
                "UPDATE player_queue SET idx = %s WHERE user_id = %s AND idx = -1;",
                (idx, user_id),
            )
            return True

    def delete_queue_item(self, user_id: str, idx: int) -> bool:
        """Remove a queue item and compact indices; returns False when idx missing."""
        if not user_id:
            raise ValueError("user_id is required")
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM player_queue WHERE user_id = %s AND idx = %s;",
                (user_id, idx),
            )
            if cur.rowcount == 0:
                return False

            # Shift down any items that were after the removed item one by one to
            # avoid primary key collisions.
            cur.execute(
                """
                SELECT idx
                FROM player_queue
                WHERE user_id = %s AND idx > %s
                ORDER BY idx ASC;
                """,
                (user_id, idx),
            )
            for (current_idx,) in cur.fetchall():
                cur.execute(
                    """
                    UPDATE player_queue
                    SET idx = %s
                    WHERE user_id = %s AND idx = %s;
                    """,
                    (current_idx - 1, user_id, current_idx),
                )
            return True

    def insert_book(
        self,
        user_id: str,
        book_title: str,
        folder_path: str,
        visibility: int = 1,
    ) -> int:
        """Insert a new book record or update if exists; returns the book id."""
        if not user_id or not book_title or not folder_path:
            raise ValueError("user_id, book_title, and folder_path are required")
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO books (user_id, book_title, folder_path, visibility)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (user_id, book_title) DO UPDATE SET
                    modified_time = CURRENT_TIMESTAMP,
                    visibility = EXCLUDED.visibility
                RETURNING id;
                """,
                (user_id, book_title, folder_path, visibility),
            )
            row = cur.fetchone()
            return row[0] if row else 0

    def get_user_books(self, user_id: str) -> list[dict[str, object]]:
        """Return all books owned by user OR marked as public."""
        if not user_id:
            raise ValueError("user_id is required")
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, book_title, folder_path, visibility
                FROM books
                WHERE user_id = %s OR visibility = 1
                ORDER BY modified_time DESC;
                """,
                (user_id,),
            )
            rows = cur.fetchall()
            return [
                {
                    "id": row[0],
                    "user_id": row[1],
                    "book_title": row[2],
                    "folder_path": row[3],
                    "visibility": row[4],
                }
                for row in rows
            ]

    def get_book_by_folder(self, folder_path: str) -> Optional[dict[str, object]]:
        """Return a book by its folder path."""
        if not folder_path:
            raise ValueError("folder_path is required")
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, book_title, folder_path, visibility
                FROM books
                WHERE folder_path = %s
                LIMIT 1;
                """,
                (folder_path,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "id": row[0],
                "user_id": row[1],
                "book_title": row[2],
                "folder_path": row[3],
                "visibility": row[4],
            }

    def get_book_for_user(
        self, book_title: str, user_id: str
    ) -> Optional[dict[str, object]]:
        """Return a book by title if user owns it or it's public. Prefers user's own book."""
        if not book_title or not user_id:
            raise ValueError("book_title and user_id are required")
        with self._cursor() as cur:
            # First check if user owns a book with this title
            cur.execute(
                """
                SELECT id, user_id, book_title, folder_path, visibility
                FROM books
                WHERE book_title = %s AND user_id = %s
                LIMIT 1;
                """,
                (book_title, user_id),
            )
            row = cur.fetchone()
            if row:
                return {
                    "id": row[0],
                    "user_id": row[1],
                    "book_title": row[2],
                    "folder_path": row[3],
                    "visibility": row[4],
                }
            # Then check for public book with this title
            cur.execute(
                """
                SELECT id, user_id, book_title, folder_path, visibility
                FROM books
                WHERE book_title = %s AND visibility = 1
                LIMIT 1;
                """,
                (book_title,),
            )
            row = cur.fetchone()
            if row:
                return {
                    "id": row[0],
                    "user_id": row[1],
                    "book_title": row[2],
                    "folder_path": row[3],
                    "visibility": row[4],
                }
            return None

    def insert_clip(
        self,
        user_id: str,
        pod_title: str,
        start_timestamp: float,
        end_timestamp: float,
        transcript: list[dict[str, Any]],
        video_file_path: str,
        video_url: str,
    ) -> int:
        """Insert a new clip record and return its id."""
        if not user_id or not pod_title:
            raise ValueError("user_id and pod_title are required")
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO clips (user_id, pod_title, start_timestamp, end_timestamp,
                                   transcript, video_file_path, video_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    user_id,
                    pod_title,
                    start_timestamp,
                    end_timestamp,
                    json.dumps(transcript),
                    video_file_path,
                    video_url,
                ),
            )
            row = cur.fetchone()
            return row[0] if row else 0
