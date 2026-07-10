from app.memory.sqlite_profile_store import SQLiteProfileStore
from app.schemas import ActivityLevel, Gender, Goal, UserProfile


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


class TestSQLiteProfileStore:
    def test_returns_none_for_unknown_user(self, tmp_path):
        store = SQLiteProfileStore(db_path=tmp_path / "profiles.db")
        assert store.get("unknown") is None

    def test_save_then_get_round_trips(self, tmp_path):
        store = SQLiteProfileStore(db_path=tmp_path / "profiles.db")
        profile = make_profile()

        store.save("user-1", profile)

        assert store.get("user-1") == profile

    def test_save_overwrites_existing_profile(self, tmp_path):
        store = SQLiteProfileStore(db_path=tmp_path / "profiles.db")
        store.save("user-1", make_profile(weight_kg=78))

        store.save("user-1", make_profile(weight_kg=80))

        assert store.get("user-1").weight_kg == 80

    def test_persists_across_separate_store_instances(self, tmp_path):
        db_path = tmp_path / "profiles.db"
        profile = make_profile(age=30)

        SQLiteProfileStore(db_path=db_path).save("user-1", profile)

        # 새 인스턴스(다른 프로세스 재시작을 흉내) 로도 같은 파일에서 읽힌다.
        reloaded_store = SQLiteProfileStore(db_path=db_path)
        assert reloaded_store.get("user-1") == profile

    def test_different_users_are_isolated(self, tmp_path):
        store = SQLiteProfileStore(db_path=tmp_path / "profiles.db")
        store.save("user-1", make_profile(age=27))
        store.save("user-2", make_profile(age=40))

        assert store.get("user-1").age == 27
        assert store.get("user-2").age == 40
