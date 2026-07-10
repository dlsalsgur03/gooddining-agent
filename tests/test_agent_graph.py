from langchain_core.messages import AIMessage, HumanMessage

from app import agent
from app.agent import (
    _RequestClassification,
    ask_for_more_info,
    calculate_targets,
    check_completeness,
    check_profile,
    classify_request,
    route_after_check_completeness,
    route_after_check_profile,
    route_after_verify_meal_plan,
    verify_meal_plan,
)
from app.memory.profile_store import InMemoryProfileStore
from app.schemas import (
    ActivityLevel,
    CalorieTarget,
    Dish,
    Gender,
    Goal,
    MacroTargets,
    Meal,
    MealPlan,
    ProfileExtraction,
    UserProfile,
)


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
        "partial_profile": ProfileExtraction(),
        "calorie_target": None,
        "request_type": None,
        "needs_more_info": False,
        "structured_response": None,
        "meal_plan_retries": 0,
        "needs_meal_plan_retry": False,
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
            custom_bmr_kcal=None,
            custom_tdee_kcal=None,
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
            ProfileExtraction(
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
        assert result["partial_profile"] == ProfileExtraction()

    def test_incomplete_extraction_flags_needs_more_info_and_keeps_partial_profile(self, monkeypatch):
        monkeypatch.setattr(agent, "_profile_store", InMemoryProfileStore())
        fake = FakeStructuredLLM(ProfileExtraction(gender=Gender.MALE, age=27))
        monkeypatch.setattr(agent, "_get_profile_extraction_llm", lambda: fake)

        state = make_state(messages=[HumanMessage(content="나 27살 남자야")])
        result = check_completeness(state)

        assert result["needs_more_info"] is True
        assert "profile" not in result
        assert result["partial_profile"].gender == Gender.MALE
        assert result["partial_profile"].age == 27

    def test_accumulates_partial_profile_across_turns(self, monkeypatch):
        store = InMemoryProfileStore()
        monkeypatch.setattr(agent, "_profile_store", store)

        fake_turn_1 = FakeStructuredLLM(ProfileExtraction(gender=Gender.MALE, age=24))
        monkeypatch.setattr(agent, "_get_profile_extraction_llm", lambda: fake_turn_1)
        turn_1_state = make_state(messages=[HumanMessage(content="24살 남자")])
        turn_1_result = check_completeness(turn_1_state)

        assert turn_1_result["needs_more_info"] is True
        assert turn_1_result["partial_profile"].age == 24
        assert turn_1_result["partial_profile"].gender == Gender.MALE

        fake_turn_2 = FakeStructuredLLM(
            ProfileExtraction(height_cm=175, weight_kg=70, activity_level=ActivityLevel.MODERATE, goal=Goal.LOSE)
        )
        monkeypatch.setattr(agent, "_get_profile_extraction_llm", lambda: fake_turn_2)
        turn_2_state = make_state(
            messages=[HumanMessage(content="175cm 70kg 활동량 보통 감량 목표")],
            partial_profile=turn_1_result["partial_profile"],
        )
        turn_2_result = check_completeness(turn_2_state)

        assert turn_2_result["needs_more_info"] is False
        assert turn_2_result["profile"].age == 24
        assert turn_2_result["profile"].gender == Gender.MALE
        assert turn_2_result["profile"].height_cm == 175

    def test_route_after_check_completeness(self):
        assert route_after_check_completeness(make_state(needs_more_info=True)) == "ask_for_more_info"
        assert route_after_check_completeness(make_state(needs_more_info=False)) == "calculate_targets"


class TestAskForMoreInfo:
    def test_produces_ai_message_listing_required_fields_when_nothing_known(self):
        result = ask_for_more_info(make_state())
        message = result["messages"][0]
        assert isinstance(message, AIMessage)
        for keyword in ("성별", "나이", "몸무게", "목표"):
            assert keyword in message.content

    def test_acknowledges_known_fields_and_asks_only_for_missing_ones(self):
        partial = ProfileExtraction(gender=Gender.MALE, age=24)
        result = ask_for_more_info(make_state(partial_profile=partial))
        content = result["messages"][0].content
        acknowledgement, _, remaining_ask = content.partition("확인했어요!")

        assert "24세" in acknowledgement
        assert "남성" in acknowledgement
        assert "키(cm)" in remaining_ask
        assert "몸무게(kg)" in remaining_ask
        assert "나이" not in remaining_ask
        assert "성별" not in remaining_ask


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
            custom_bmr_kcal=None,
            custom_tdee_kcal=None,
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


def make_dish(name="참치 샐러드", calories=250.0):
    return Dish(
        name=name,
        brand=None,
        calories=calories,
        macros=MacroTargets(protein_g=30, carbs_g=15, fat_g=6),
        recipe=None,
        tags=[],
    )


def make_meal_plan(meal_types_and_calories):
    return MealPlan(
        summary="테스트용 식단 요약",
        daily_calorie_target=2000.0,
        daily_macros=MacroTargets(protein_g=150, carbs_g=200, fat_g=60),
        meals=[
            Meal(meal_type=meal_type, dishes=[make_dish(calories=calories)])
            for meal_type, calories in meal_types_and_calories
        ],
    )


CALORIE_TARGET = CalorieTarget(
    tdee_kcal=2500.0,
    target_kcal=2000.0,
    macros=MacroTargets(protein_g=150, carbs_g=200, fat_g=60),
    is_below_safe_minimum=False,
)


class TestVerifyMealPlan:
    def test_skips_non_general_requests(self):
        meal_plan = make_meal_plan([("아침", 100)])
        state = make_state(
            structured_response=meal_plan, calorie_target=CALORIE_TARGET, request_type="delivery"
        )

        result = verify_meal_plan(state)

        assert result == {"needs_meal_plan_retry": False}

    def test_accepts_plan_within_tolerance_and_all_meal_types(self):
        meal_plan = make_meal_plan([("아침", 660), ("점심", 670), ("저녁", 670)])
        state = make_state(
            structured_response=meal_plan, calorie_target=CALORIE_TARGET, request_type="general"
        )

        result = verify_meal_plan(state)

        assert result == {"needs_meal_plan_retry": False}

    def test_flags_missing_meal_type_and_requests_retry(self):
        meal_plan = make_meal_plan([("점심", 660), ("간식", 670), ("저녁", 670)])
        state = make_state(
            structured_response=meal_plan, calorie_target=CALORIE_TARGET, request_type="general"
        )

        result = verify_meal_plan(state)

        assert result["needs_meal_plan_retry"] is True
        assert result["meal_plan_retries"] == 1
        assert "아침" in result["messages"][0].content

    def test_flags_calorie_mismatch_beyond_tolerance(self):
        meal_plan = make_meal_plan([("아침", 100), ("점심", 100), ("저녁", 100)])
        state = make_state(
            structured_response=meal_plan, calorie_target=CALORIE_TARGET, request_type="general"
        )

        result = verify_meal_plan(state)

        assert result["needs_meal_plan_retry"] is True
        assert "300" in result["messages"][0].content
        assert "2000" in result["messages"][0].content

    def test_gives_up_after_max_retries(self):
        meal_plan = make_meal_plan([("아침", 100)])
        state = make_state(
            structured_response=meal_plan,
            calorie_target=CALORIE_TARGET,
            request_type="general",
            meal_plan_retries=2,
        )

        result = verify_meal_plan(state)

        assert result == {"needs_meal_plan_retry": False}


class TestRouteAfterVerifyMealPlan:
    def test_routes_to_agent_when_retry_needed(self):
        assert route_after_verify_meal_plan(make_state(needs_meal_plan_retry=True)) == "agent"

    def test_routes_to_end_when_valid(self):
        assert route_after_verify_meal_plan(make_state(needs_meal_plan_retry=False)) == "end"
