from langchain_core.messages import AIMessage

from app.middleware.guardrail_middleware import (
    LOW_CALORIE_WARNING,
    SKIP_MEAL_APPENDIX,
    GuardrailMiddleware,
)
from app.schemas import CalorieTarget, MacroTargets

middleware = GuardrailMiddleware()

SAFE_CALORIE_TARGET = CalorieTarget(
    tdee_kcal=2000.0,
    target_kcal=1800.0,
    macros=MacroTargets(protein_g=120.0, carbs_g=180.0, fat_g=50.0),
    is_below_safe_minimum=False,
)

UNSAFE_CALORIE_TARGET = CalorieTarget(
    tdee_kcal=1300.0,
    target_kcal=900.0,
    macros=MacroTargets(protein_g=80.0, carbs_g=60.0, fat_g=20.0),
    is_below_safe_minimum=True,
)


class TestGuardrailMiddleware:
    def test_ignores_intermediate_tool_call_message(self):
        message = AIMessage(content="", tool_calls=[{"name": "search_recipes", "args": {}, "id": "1"}])
        state = {"messages": [message], "calorie_target": SAFE_CALORIE_TARGET}

        result = middleware.after_model(state, runtime=None)

        assert result is None

    def test_appends_skip_meal_notice_when_response_suggests_skipping(self):
        message = AIMessage(content="오늘은 저녁 끼니를 거르는 게 좋겠어요.", id="msg-1")
        state = {"messages": [message], "calorie_target": SAFE_CALORIE_TARGET}

        result = middleware.after_model(state, runtime=None)

        assert result is not None
        updated = result["messages"][0]
        assert updated.id == "msg-1"
        assert SKIP_MEAL_APPENDIX in updated.content

    def test_appends_low_calorie_warning_when_target_unsafe(self):
        message = AIMessage(content="오늘의 식단을 추천해드릴게요.", id="msg-2")
        state = {"messages": [message], "calorie_target": UNSAFE_CALORIE_TARGET}

        result = middleware.after_model(state, runtime=None)

        assert result is not None
        updated = result["messages"][0]
        assert updated.id == "msg-2"
        assert LOW_CALORIE_WARNING in updated.content

    def test_appends_both_when_both_conditions_met(self):
        message = AIMessage(content="끼니를 거르는 걸 추천해요.", id="msg-3")
        state = {"messages": [message], "calorie_target": UNSAFE_CALORIE_TARGET}

        result = middleware.after_model(state, runtime=None)

        updated = result["messages"][0]
        assert SKIP_MEAL_APPENDIX in updated.content
        assert LOW_CALORIE_WARNING in updated.content

    def test_returns_none_when_nothing_to_flag(self):
        message = AIMessage(content="오늘의 식단을 추천해드릴게요.", id="msg-4")
        state = {"messages": [message], "calorie_target": SAFE_CALORIE_TARGET}

        result = middleware.after_model(state, runtime=None)

        assert result is None

    def test_returns_none_when_calorie_target_missing(self):
        message = AIMessage(content="오늘의 식단을 추천해드릴게요.", id="msg-5")
        state = {"messages": [message], "calorie_target": None}

        result = middleware.after_model(state, runtime=None)

        assert result is None
