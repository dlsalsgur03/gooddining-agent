from app.schemas import UserProfile
from app.tools import nutrition_calc
from app.tools.agent_tools import (
    calc_remaining_budget,
    calculate_bmr_tdee,
    calculate_calorie_target,
)

PROFILE_INPUT = {
    "gender": "male",
    "age": 27,
    "height_cm": 175,
    "weight_kg": 78,
    "activity_level": "moderate",
    "goal": "lose",
}


class TestToolMetadata:
    def test_tools_expose_name_and_description(self):
        for wrapped_tool in (calculate_bmr_tdee, calculate_calorie_target, calc_remaining_budget):
            assert wrapped_tool.name
            assert wrapped_tool.description


class TestCalculateBmrTdeeTool:
    def test_invoke_matches_pure_function(self):
        result = calculate_bmr_tdee.invoke({"profile": PROFILE_INPUT})
        expected = nutrition_calc.calculate_bmr_tdee(UserProfile(**PROFILE_INPUT))

        assert result.bmr_kcal == expected.bmr_kcal
        assert result.tdee_kcal == expected.tdee_kcal


class TestCalculateCalorieTargetTool:
    def test_invoke_matches_pure_function(self):
        bmr_tdee_input = {"bmr_kcal": 1500.0, "tdee_kcal": 2000.0}

        result = calculate_calorie_target.invoke(
            {"bmr_tdee": bmr_tdee_input, "goal": "lose", "weight_kg": 70}
        )

        assert result.target_kcal == 1500.0
        assert result.macros.protein_g == 70 * 1.8


class TestCalcRemainingBudgetTool:
    def test_invoke_matches_pure_function(self):
        daily_target_input = {
            "tdee_kcal": 2000.0,
            "target_kcal": 1800.0,
            "macros": {"protein_g": 120.0, "carbs_g": 180.0, "fat_g": 50.0},
            "is_below_safe_minimum": False,
        }
        consumed_macros_input = {"protein_g": 40.0, "carbs_g": 60.0, "fat_g": 15.0}

        result = calc_remaining_budget.invoke(
            {
                "daily_target": daily_target_input,
                "consumed_kcal": 600.0,
                "consumed_macros": consumed_macros_input,
                "meals_remaining": 2,
            }
        )

        assert result.remaining_kcal == 1200.0
        assert result.per_meal_kcal == 600.0
        assert result.is_over_budget is False
