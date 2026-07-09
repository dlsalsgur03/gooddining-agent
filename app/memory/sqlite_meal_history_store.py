"""`MealHistoryStore`의 SQLite 기반 구현체.

세션이 끝나거나 프로세스가 재시작돼도 날짜별 식단 기록이 유지되도록 파일 DB에 저장한다.
"""

import sqlite3
from pathlib import Path

from app.schemas import MealPlan

DEFAULT_DB_PATH = Path(__file__).parent / "meal_history.db"


class SQLiteMealHistoryStore:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS meal_history ("
                "user_id TEXT NOT NULL, date TEXT NOT NULL, meal_plan_json TEXT NOT NULL, "
                "PRIMARY KEY (user_id, date))"
            )

    def get(self, user_id: str, date: str) -> MealPlan | None:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT meal_plan_json FROM meal_history WHERE user_id = ? AND date = ?",
                (user_id, date),
            ).fetchone()
        if row is None:
            return None
        return MealPlan.model_validate_json(row[0])

    def save(self, user_id: str, date: str, meal_plan: MealPlan) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO meal_history (user_id, date, meal_plan_json) VALUES (?, ?, ?) "
                "ON CONFLICT(user_id, date) DO UPDATE SET meal_plan_json = excluded.meal_plan_json",
                (user_id, date, meal_plan.model_dump_json()),
            )

    def list_dates(self, user_id: str) -> list[str]:
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT date FROM meal_history WHERE user_id = ? ORDER BY date", (user_id,)
            ).fetchall()
        return [row[0] for row in rows]
