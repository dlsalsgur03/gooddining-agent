from langchain_core.messages import AIMessage, HumanMessage

from app import agent
from app.agent import (
    _ProfileExtraction,
    _RequestClassification,
    ask_for_more_info,
    calculate_targets,
    check_completeness,
    check_profile,
    classify_request,
    route_after_check_completeness,
    route_after_check_profile,
)
from app.memory.profile_store import InMemoryProfileStore
from app.schemas import ActivityLevel, Gender, Goal, UserProfile


class FakeStructuredLLM:
    def __init__(self, result):
        self._result = result

    def invoke(self, prompt):
        return self._result


def make_state(**overrides):
    base = {
        "messages": [],
        "user_id": "user-1",
        "profile": None,
        "calorie_target": None,
        "request_type": None,
        "needs_more_info": False,
        "structured_response": None,
    }
    base.update(overrides)
    return base


class TestProfileStoreRoutingIntegration:
    def test_check_profile_finds_saved_profile(self, monkeypatch):
        store = InMemoryProfileStore()
        profile = UserProfile(
            gender=Gender.MALE,
            age=27,
            height_cm=175,
            weight_kg=78,
            activity_level=ActivityLevel.MODERATE,
            goal=Goal.LOSE,
            allergies=[],
            disliked_ingredients=[],
        )
        store.save("user-1", profile)
        monkeypatch.setattr(agent, "_profile_store", store)

        result = check_profile(make_state(user_id="user-1"))
        assert result["profile"] == profile

    def test_check_profile_returns_none_when_missing(self, monkeypatch):
        monkeypatch.setattr(agent, "_profile_store", InMemoryProfileStore())

        result = check_profile(make_state(user_id="unknown"))
        assert result["profile"] is None

    def test_route_after_check_profile(self):
        assert route_after_check_profile(make_state(profile=object())) == "calculate_targets"
        assert route_after_check_profile(make_state(profile=None)) == "check_completeness"


class TestCheckCompleteness:
    def test_complete_extraction_saves_profile(self, monkeypatch):
        store = InMemoryProfileStore()
        monkeypatch.setattr(agent, "_profile_store", store)
        fake = FakeStructuredLLM(
            _ProfileExtraction(
                gender=Gender.MALE,
                age=27,
                height_cm=175,
                weight_kg=78,
                activity_level=ActivityLevel.MODERATE,
                goal=Goal.LOSE,
            )
        )
        monkeypatch.setattr(agent, "_get_profile_extraction_llm", lambda: fake)

        state = make_state(messages=[HumanMessage(content="27살 남자, 175cm, 78kg, 보통, 감량")])
        result = check_completeness(state)

        assert result["needs_more_info"] is False
        assert result["profile"].age == 27
        assert store.get("user-1") == result["profile"]

    def test_incomplete_extraction_flags_needs_more_info(self, monkeypatch):
        monkeypatch.setattr(agent, "_profile_store", InMemoryProfileStore())
        fake = FakeStructuredLLM(_ProfileExtraction(gender=Gender.MALE, age=27))
        monkeypatch.setattr(agent, "_get_profile_extraction_llm", lambda: fake)

        state = make_state(messages=[HumanMessage(content="나 27살 남자야")])
        result = check_completeness(state)

        assert result["needs_more_info"] is True
        assert "profile" not in result

    def test_route_after_check_completeness(self):
        assert route_after_check_completeness(make_state(needs_more_info=True)) == "ask_for_more_info"
        assert route_after_check_completeness(make_state(needs_more_info=False)) == "calculate_targets"


class TestAskForMoreInfo:
    def test_produces_ai_message_listing_required_fields(self):
        result = ask_for_more_info(make_state())
        message = result["messages"][0]
        assert isinstance(message, AIMessage)
        for keyword in ("성별", "나이", "몸무게", "목표"):
            assert keyword in message.content


class TestCalculateTargets:
    def test_matches_pure_nutrition_calc_functions(self):
        profile = UserProfile(
            gender=Gender.MALE,
            age=27,
            height_cm=175,
            weight_kg=78,
            activity_level=ActivityLevel.MODERATE,
            goal=Goal.LOSE,
            allergies=[],
            disliked_ingredients=[],
        )
        result = calculate_targets(make_state(profile=profile))

        from app.tools import nutrition_calc

        expected_bmr_tdee = nutrition_calc.calculate_bmr_tdee(profile)
        expected_target = nutrition_calc.calculate_calorie_target(
            expected_bmr_tdee, profile.goal, profile.weight_kg
        )
        assert result["calorie_target"] == expected_target


class TestClassifyRequest:
    def test_sets_request_type_and_appends_hint_message(self, monkeypatch):
        fake = FakeStructuredLLM(_RequestClassification(request_type="delivery"))
        monkeypatch.setattr(agent, "_get_request_classifier_llm", lambda: fake)

        state = make_state(messages=[HumanMessage(content="저녁은 배달로 버거킹 먹고 싶어")])
        result = classify_request(state)

        assert result["request_type"] == "delivery"
        assert "search_delivery_menu" in result["messages"][0].content
