"""BMR/TDEE, 칼로리 목표, 잔여 예산 계산 순수 함수.

LangChain 의존성 없이 독립적으로 테스트 가능하도록 작성한다.
`@tool` 래핑은 다음 단계(Tool 등록)에서 이 함수들을 얇게 감싸서 처리한다.
"""

from app.schemas import (
    ActivityLevel,
    BmrTdeeResult,
    CalorieTarget,
    Goal,
    MacroTargets,
    RemainingBudget,
    Gender,
    UserProfile,
)

ACTIVITY_MULTIPLIERS: dict[ActivityLevel, float] = {
    ActivityLevel.SEDENTARY: 1.2,
    ActivityLevel.LIGHT: 1.375,
    ActivityLevel.MODERATE: 1.55,
    ActivityLevel.ACTIVE: 1.725,
    ActivityLevel.VERY_ACTIVE: 1.9,
}

GOAL_CALORIE_OFFSET: dict[Goal, float] = {
    Goal.LOSE: -500.0,
    Goal.MAINTAIN: 0.0,
    Goal.GAIN: 400.0,
}

PROTEIN_G_PER_KG: dict[Goal, float] = {
    Goal.LOSE: 1.8,
    Goal.MAINTAIN: 1.2,
    Goal.GAIN: 1.8,
}

FAT_RATIO = 0.25
SAFE_MIN_CALORIES = 1200.0
MIN_PER_MEAL_KCAL = 150.0


def calculate_bmr_tdee(profile: UserProfile) -> BmrTdeeResult:
    """Mifflin-St Jeor 공식으로 BMR을 구하고 활동계수를 곱해 TDEE를 산출한다."""
    if profile.gender == Gender.MALE:
        bmr = 10 * profile.weight_kg + 6.25 * profile.height_cm - 5 * profile.age + 5
    else:
        bmr = 10 * profile.weight_kg + 6.25 * profile.height_cm - 5 * profile.age - 161

    tdee = bmr * ACTIVITY_MULTIPLIERS[profile.activity_level]
    return BmrTdeeResult(bmr_kcal=bmr, tdee_kcal=tdee)


def calculate_calorie_target(
    bmr_tdee: BmrTdeeResult, goal: Goal, weight_kg: float
) -> CalorieTarget:
    """목표(감량/유지/증량)에 따라 TDEE를 조정하고 매크로(단백질/탄수화물/지방)를 산출한다."""
    target_kcal = bmr_tdee.tdee_kcal + GOAL_CALORIE_OFFSET[goal]

    protein_g = weight_kg * PROTEIN_G_PER_KG[goal]
    fat_g = (target_kcal * FAT_RATIO) / 9
    carbs_kcal = target_kcal - protein_g * 4 - fat_g * 9
    carbs_g = max(carbs_kcal, 0.0) / 4

    return CalorieTarget(
        tdee_kcal=bmr_tdee.tdee_kcal,
        target_kcal=target_kcal,
        macros=MacroTargets(protein_g=protein_g, carbs_g=carbs_g, fat_g=fat_g),
        is_below_safe_minimum=target_kcal < SAFE_MIN_CALORIES,
    )


def calc_remaining_budget(
    daily_target: CalorieTarget,
    consumed_kcal: float,
    consumed_macros: MacroTargets,
    meals_remaining: int,
) -> RemainingBudget:
    """하루 목표에서 이미 섭취한 양을 제외하고 남은 끼니 수에 맞게 예산을 재분배한다."""
    if meals_remaining <= 0:
        raise ValueError("meals_remaining must be a positive integer")

    remaining_kcal = daily_target.target_kcal - consumed_kcal
    remaining_macros = MacroTargets(
        protein_g=max(daily_target.macros.protein_g - consumed_macros.protein_g, 0.0),
        carbs_g=max(daily_target.macros.carbs_g - consumed_macros.carbs_g, 0.0),
        fat_g=max(daily_target.macros.fat_g - consumed_macros.fat_g, 0.0),
    )

    # 잔여 예산이 부족해도 끼니를 거르라고 하지 않기 위해, 끼니당 최소 칼로리를 보장한다.
    per_meal_kcal = max(remaining_kcal / meals_remaining, MIN_PER_MEAL_KCAL)
    per_meal_macros = MacroTargets(
        protein_g=remaining_macros.protein_g / meals_remaining,
        carbs_g=remaining_macros.carbs_g / meals_remaining,
        fat_g=remaining_macros.fat_g / meals_remaining,
    )

    return RemainingBudget(
        remaining_kcal=remaining_kcal,
        remaining_macros=remaining_macros,
        per_meal_kcal=per_meal_kcal,
        per_meal_macros=per_meal_macros,
        is_over_budget=remaining_kcal <= 0,
    )
