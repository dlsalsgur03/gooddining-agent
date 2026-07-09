"""날짜별 식단 기록(달력/사이드 패널 표시용) 장기 저장소.

`user_id` + `date`(YYYY-MM-DD) 기준으로 그 날 마지막에 생성된 `MealPlan`을 저장한다.
하루에 여러 번 식단을 추천받아도(예: 아침엔 일반 식단 → 점심에 배달로 변경) 가장 최근
생성된 것으로 덮어쓴다 — "오늘의 최종 식단"이라는 단순한 모델을 따른다.
"""

from typing import Protocol

from app.schemas import MealPlan


class MealHistoryStore(Protocol):
    def get(self, user_id: str, date: str) -> MealPlan | None: ...

    def save(self, user_id: str, date: str, meal_plan: MealPlan) -> None: ...

    def list_dates(self, user_id: str) -> list[str]: ...


class InMemoryMealHistoryStore:
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], MealPlan] = {}

    def get(self, user_id: str, date: str) -> MealPlan | None:
        return self._data.get((user_id, date))

    def save(self, user_id: str, date: str, meal_plan: MealPlan) -> None:
        self._data[(user_id, date)] = meal_plan

    def list_dates(self, user_id: str) -> list[str]:
        return sorted(date for (uid, date) in self._data if uid == user_id)
