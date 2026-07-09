"""`ProfileStore`의 SQLite 기반 구현체.

세션이 끝나거나 프로세스가 재시작돼도 사용자 프로필이 유지되도록 파일 DB에 저장한다.
`ProfileStore` 프로토콜만 지키므로 `app/agent.py`는 이 구현체로 바꿔도 그래프 로직을
전혀 건드릴 필요가 없다 (`InMemoryProfileStore`에서 이 구현체로 교체만 하면 됨).
"""

import sqlite3
from pathlib import Path

from app.schemas import UserProfile

DEFAULT_DB_PATH = Path(__file__).parent / "profiles.db"


class SQLiteProfileStore:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS profiles (user_id TEXT PRIMARY KEY, profile_json TEXT NOT NULL)"
            )

    def get(self, user_id: str) -> UserProfile | None:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT profile_json FROM profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row is None:
            return None
        return UserProfile.model_validate_json(row[0])

    def save(self, user_id: str, profile: UserProfile) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO profiles (user_id, profile_json) VALUES (?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET profile_json = excluded.profile_json",
                (user_id, profile.model_dump_json()),
            )
