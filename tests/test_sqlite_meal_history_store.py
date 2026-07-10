from app.memory.sqlite_meal_history_store import SQLiteMealHistoryStore
from app.schemas import Dish, MacroTargets, Meal, MealPlan


def make_meal_plan(calorie_target=2000.0):
    return MealPlan(
        summary="테스트용 식단 요약",
        daily_calorie_target=calorie_target,
        daily_macros=MacroTargets(protein_g=150, carbs_g=200, fat_g=60),
        meals=[
            Meal(
                meal_type="아침",
                dishes=[
                    Dish(
                        name="참치 샐러드",
                        brand=None,
                        calories=250.0,
                        macros=MacroTargets(protein_g=30, carbs_g=15, fat_g=6),
                        recipe=None,
                        tags=[],
                    )
                ],
            )
        ],
    )


class TestSQLiteMealHistoryStore:
    def test_returns_none_for_unknown_date(self, tmp_path):
        store = SQLiteMealHistoryStore(db_path=tmp_path / "meal_history.db")
        assert store.get("user-1", "2026-07-09") is None

    def test_save_then_get_round_trips(self, tmp_path):
        store = SQLiteMealHistoryStore(db_path=tmp_path / "meal_history.db")
        meal_plan = make_meal_plan()

        store.save("user-1", "2026-07-09", meal_plan)

        assert store.get("user-1", "2026-07-09") == meal_plan

    def test_save_overwrites_same_day(self, tmp_path):
        store = SQLiteMealHistoryStore(db_path=tmp_path / "meal_history.db")
        store.save("user-1", "2026-07-09", make_meal_plan(calorie_target=2000.0))

        store.save("user-1", "2026-07-09", make_meal_plan(calorie_target=1800.0))

        assert store.get("user-1", "2026-07-09").daily_calorie_target == 1800.0

    def test_list_dates_returns_sorted_dates_for_user_only(self, tmp_path):
        store = SQLiteMealHistoryStore(db_path=tmp_path / "meal_history.db")
        store.save("user-1", "2026-07-09", make_meal_plan())
        store.save("user-1", "2026-07-07", make_meal_plan())
        store.save("user-2", "2026-07-08", make_meal_plan())

        assert store.list_dates("user-1") == ["2026-07-07", "2026-07-09"]

    def test_persists_across_separate_store_instances(self, tmp_path):
        db_path = tmp_path / "meal_history.db"
        meal_plan = make_meal_plan()
        SQLiteMealHistoryStore(db_path=db_path).save("user-1", "2026-07-09", meal_plan)

        reloaded_store = SQLiteMealHistoryStore(db_path=db_path)
        assert reloaded_store.get("user-1", "2026-07-09") == meal_plan
