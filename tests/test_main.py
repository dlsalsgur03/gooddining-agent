from fastapi.testclient import TestClient

from app import main
from app.main import app
from app.memory.meal_history_store import InMemoryMealHistoryStore
from app.memory.profile_store import InMemoryProfileStore
from app.schemas import ActivityLevel, Dish, Gender, Goal, MacroTargets, Meal, MealPlan, UserProfile

client = TestClient(app)


def make_meal_plan():
    return MealPlan(
        summary="테스트용 식단 요약",
        daily_calorie_target=2000.0,
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


class TestHealth:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestMealHistoryEndpoints:
    def test_list_dates_empty_for_unknown_user(self, monkeypatch):
        monkeypatch.setattr(main, "_meal_history_store", InMemoryMealHistoryStore())

        response = client.get("/meals/unknown-user/dates")

        assert response.status_code == 200
        assert response.json() == {"dates": []}

    def test_list_dates_returns_saved_dates(self, monkeypatch):
        store = InMemoryMealHistoryStore()
        store.save("user-1", "2026-07-09", make_meal_plan())
        monkeypatch.setattr(main, "_meal_history_store", store)

        response = client.get("/meals/user-1/dates")

        assert response.json() == {"dates": ["2026-07-09"]}

    def test_get_meal_by_date_returns_404_when_missing(self, monkeypatch):
        monkeypatch.setattr(main, "_meal_history_store", InMemoryMealHistoryStore())

        response = client.get("/meals/user-1/2026-07-09")

        assert response.status_code == 404

    def test_get_meal_by_date_returns_saved_plan(self, monkeypatch):
        store = InMemoryMealHistoryStore()
        meal_plan = make_meal_plan()
        store.save("user-1", "2026-07-09", meal_plan)
        monkeypatch.setattr(main, "_meal_history_store", store)

        response = client.get("/meals/user-1/2026-07-09")

        assert response.status_code == 200
        assert response.json()["daily_calorie_target"] == 2000.0


def make_profile(**overrides):
    defaults = dict(
        gender=Gender.MALE,
        age=27,
        height_cm=175,
        weight_kg=78,
        activity_level=ActivityLevel.MODERATE,
        goal=Goal.LOSE,
        allergies=[],
        disliked_ingredients=[],
        custom_bmr_kcal=None,
        custom_tdee_kcal=None,
    )
    defaults.update(overrides)
    return UserProfile(**defaults)


class TestProfileEndpoints:
    def test_get_profile_returns_404_when_missing(self, monkeypatch):
        monkeypatch.setattr(main, "_profile_store", InMemoryProfileStore())

        response = client.get("/profile/unknown-user")

        assert response.status_code == 404

    def test_get_profile_returns_saved_profile(self, monkeypatch):
        store = InMemoryProfileStore()
        store.save("user-1", make_profile(age=30))
        monkeypatch.setattr(main, "_profile_store", store)

        response = client.get("/profile/user-1")

        assert response.status_code == 200
        assert response.json()["age"] == 30
