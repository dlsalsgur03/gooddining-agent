from enum import Enum

from pydantic import BaseModel, Field


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"


class ActivityLevel(str, Enum):
    SEDENTARY = "sedentary"
    LIGHT = "light"
    MODERATE = "moderate"
    ACTIVE = "active"
    VERY_ACTIVE = "very_active"


class Goal(str, Enum):
    LOSE = "lose"
    MAINTAIN = "maintain"
    GAIN = "gain"


class ProfileExtraction(BaseModel):
    """대화에서 추출한 프로필 정보. 여러 턴에 걸쳐 조금씩 모일 수 있어 전부 Optional."""

    gender: Gender | None = None
    age: int | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    activity_level: ActivityLevel | None = None
    goal: Goal | None = None
    # 사용자가 자신의 정확한 기초대사량/활동대사량을 직접 알려준 경우에만 채워짐.
    custom_bmr_kcal: float | None = None
    custom_tdee_kcal: float | None = None


class UserProfile(BaseModel):
    gender: Gender
    age: int = Field(gt=0, lt=120)
    height_cm: float = Field(gt=0)
    weight_kg: float = Field(gt=0)
    activity_level: ActivityLevel
    goal: Goal
    # 기본값을 두지 않는다: OpenAI 구조화 출력/strict Tool 스키마는 모든 필드가
    # required이길 요구한다(선택 항목은 default가 아니라 값 자체를 빈 리스트로 채워야 함).
    allergies: list[str]
    disliked_ingredients: list[str]
    # 사용자가 직접 알려준 값. 둘 다 채워졌을 때만 계산식 대신 이 값을 사용한다.
    custom_bmr_kcal: float | None
    custom_tdee_kcal: float | None


class ProfileView(UserProfile):
    """`/profile` 응답 전용. 저장된 프로필에 계산/적용된 BMR·TDEE를 덧붙여 보여준다."""

    bmr_kcal: float
    tdee_kcal: float
    is_custom_metabolism: bool


class MacroTargets(BaseModel):
    protein_g: float
    carbs_g: float
    fat_g: float


class BmrTdeeResult(BaseModel):
    bmr_kcal: float
    tdee_kcal: float


class CalorieTarget(BaseModel):
    tdee_kcal: float
    target_kcal: float
    macros: MacroTargets
    is_below_safe_minimum: bool


class RemainingBudget(BaseModel):
    remaining_kcal: float
    remaining_macros: MacroTargets
    per_meal_kcal: float
    per_meal_macros: MacroTargets
    is_over_budget: bool


class Dish(BaseModel):
    name: str
    # 배달 메뉴(search_delivery_menu) 결과인 경우 브랜드명(예: "서브웨이"), 그 외(레시피/웹검색)는 None.
    brand: str | None
    calories: float
    macros: MacroTargets
    recipe: str | None
    tags: list[str]


class Meal(BaseModel):
    meal_type: str
    dishes: list[Dish]


class MealPlan(BaseModel):
    daily_calorie_target: float
    daily_macros: MacroTargets
    meals: list[Meal]
    summary: str


class MealNutritionEstimate(BaseModel):
    calories: float
    macros: MacroTargets
    disclaimer: str = (
        "이 수치는 LLM이 일반 지식으로 추정한 값으로, 실제 영양성분과 다를 수 있습니다."
    )


class DeliveryMenuItem(BaseModel):
    brand: str
    category: str
    name: str
    serving_size_g: float | None = None
    calories: float | None = None
    protein_g: float | None = None
    sugar_g: float | None = None
    saturated_fat_g: float | None = None
    calories_min: float | None = None
    calories_max: float | None = None
