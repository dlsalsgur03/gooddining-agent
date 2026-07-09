"""`nutrition_calc.py`의 순수 함수를 LangChain `@tool`로 노출하는 인터페이스 레이어.

여기서는 계산 로직을 다시 구현하지 않고 얇게 위임만 한다. 순수 함수와 이름이
겹치므로 `nutrition_calc` 모듈을 통째로 import해 접두사로 구분한다.
"""

from langchain_core.tools import tool

from app.schemas import (
    BmrTdeeResult,
    CalorieTarget,
    Goal,
    MacroTargets,
    RemainingBudget,
    UserProfile,
)
from app.tools import nutrition_calc


@tool
def calculate_bmr_tdee(profile: UserProfile) -> BmrTdeeResult:
    """사용자의 성별/나이/키/몸무게/활동수준으로 기초대사량(BMR)과 활동대사량(TDEE)을 계산한다."""
    return nutrition_calc.calculate_bmr_tdee(profile)


@tool
def calculate_calorie_target(
    bmr_tdee: BmrTdeeResult, goal: Goal, weight_kg: float
) -> CalorieTarget:
    """TDEE와 목표(감량/유지/증량)에 따라 하루 목표 칼로리와 매크로(단백질/탄수화물/지방)를 산출한다."""
    return nutrition_calc.calculate_calorie_target(bmr_tdee, goal, weight_kg)


@tool
def calc_remaining_budget(
    daily_target: CalorieTarget,
    consumed_kcal: float,
    consumed_macros: MacroTargets,
    meals_remaining: int,
) -> RemainingBudget:
    """하루 목표에서 이미 섭취한 만큼을 제외하고, 남은 끼니 수에 맞춰 잔여 칼로리·매크로 예산을 재분배한다."""
    return nutrition_calc.calc_remaining_budget(
        daily_target, consumed_kcal, consumed_macros, meals_remaining
    )
