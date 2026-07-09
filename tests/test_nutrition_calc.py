import pytest

from app.schemas import (
    ActivityLevel,
    BmrTdeeResult,
    CalorieTarget,
    Gender,
    Goal,
    MacroTargets,
    UserProfile,
)
from app.tools.nutrition_calc import (
    ACTIVITY_MULTIPLIERS,
    MIN_PER_MEAL_KCAL,
    calc_remaining_budget,
    calculate_bmr_tdee,
    calculate_calorie_target,
)


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
    )
    defaults.update(overrides)
    return UserProfile(**defaults)


class TestCalculateBmrTdee:
    def test_male_reference_value(self):
        profile = make_profile(activity_level=ActivityLevel.MODERATE)
        result = calculate_bmr_tdee(profile)
        assert result.bmr_kcal == pytest.approx(1743.75)
        assert result.tdee_kcal == pytest.approx(1743.75 * 1.55)

    def test_female_reference_value(self):
        profile = make_profile(
            gender=Gender.FEMALE,
            age=30,
            height_cm=160,
            weight_kg=55,
            activity_level=ActivityLevel.SEDENTARY,
        )
        result = calculate_bmr_tdee(profile)
        assert result.bmr_kcal == pytest.approx(1239.0)
        assert result.tdee_kcal == pytest.approx(1239.0 * 1.2)

    @pytest.mark.parametrize("activity_level", list(ActivityLevel))
    def test_activity_multiplier_applied(self, activity_level):
        profile = make_profile(activity_level=activity_level)
        result = calculate_bmr_tdee(profile)
        assert result.tdee_kcal == pytest.approx(
            result.bmr_kcal * ACTIVITY_MULTIPLIERS[activity_level]
        )


class TestCalculateCalorieTarget:
    @pytest.mark.parametrize(
        "goal,expected_offset",
        [(Goal.LOSE, -500.0), (Goal.MAINTAIN, 0.0), (Goal.GAIN, 400.0)],
    )
    def test_goal_offset_and_macro_sum(self, goal, expected_offset):
        bmr_tdee = BmrTdeeResult(bmr_kcal=1500.0, tdee_kcal=2000.0)
        result = calculate_calorie_target(bmr_tdee, goal, weight_kg=70)

        assert result.target_kcal == pytest.approx(2000.0 + expected_offset)

        macro_kcal = (
            result.macros.protein_g * 4
            + result.macros.carbs_g * 4
            + result.macros.fat_g * 9
        )
        assert macro_kcal == pytest.approx(result.target_kcal)

    def test_is_below_safe_minimum_flag(self):
        low = calculate_calorie_target(
            BmrTdeeResult(bmr_kcal=1000.0, tdee_kcal=1000.0), Goal.LOSE, weight_kg=50
        )
        assert low.is_below_safe_minimum is True

        normal = calculate_calorie_target(
            BmrTdeeResult(bmr_kcal=1300.0, tdee_kcal=1300.0), Goal.MAINTAIN, weight_kg=70
        )
        assert normal.is_below_safe_minimum is False


class TestCalcRemainingBudget:
    def make_daily_target(self):
        return CalorieTarget(
            tdee_kcal=2000.0,
            target_kcal=1800.0,
            macros=MacroTargets(protein_g=120.0, carbs_g=180.0, fat_g=50.0),
            is_below_safe_minimum=False,
        )

    def test_normal_case_splits_evenly(self):
        daily_target = self.make_daily_target()
        consumed_macros = MacroTargets(protein_g=40.0, carbs_g=60.0, fat_g=15.0)

        result = calc_remaining_budget(
            daily_target, consumed_kcal=600.0, consumed_macros=consumed_macros, meals_remaining=2
        )

        assert result.remaining_kcal == pytest.approx(1200.0)
        assert result.per_meal_kcal == pytest.approx(600.0)
        assert result.per_meal_macros.protein_g == pytest.approx(40.0)
        assert result.per_meal_macros.carbs_g == pytest.approx(60.0)
        assert result.per_meal_macros.fat_g == pytest.approx(17.5)
        assert result.is_over_budget is False

    def test_over_budget_floors_per_meal_kcal_instead_of_skipping(self):
        daily_target = self.make_daily_target()
        consumed_macros = MacroTargets(protein_g=150.0, carbs_g=200.0, fat_g=60.0)

        result = calc_remaining_budget(
            daily_target, consumed_kcal=2200.0, consumed_macros=consumed_macros, meals_remaining=1
        )

        assert result.remaining_kcal == pytest.approx(-400.0)
        assert result.is_over_budget is True
        # 끼니를 거르라고 하지 않기 위해 최소 칼로리는 보장되어야 한다.
        assert result.per_meal_kcal == pytest.approx(MIN_PER_MEAL_KCAL)
        assert result.remaining_macros.protein_g == 0.0
        assert result.remaining_macros.carbs_g == 0.0
        assert result.remaining_macros.fat_g == 0.0

    def test_zero_meals_remaining_raises(self):
        daily_target = self.make_daily_target()
        consumed_macros = MacroTargets(protein_g=0.0, carbs_g=0.0, fat_g=0.0)

        with pytest.raises(ValueError):
            calc_remaining_budget(
                daily_target, consumed_kcal=0.0, consumed_macros=consumed_macros, meals_remaining=0
            )
