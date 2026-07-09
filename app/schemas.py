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


class UserProfile(BaseModel):
    gender: Gender
    age: int = Field(gt=0, lt=120)
    height_cm: float = Field(gt=0)
    weight_kg: float = Field(gt=0)
    activity_level: ActivityLevel
    goal: Goal
    allergies: list[str] = Field(default_factory=list)
    disliked_ingredients: list[str] = Field(default_factory=list)


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
    calories: float
    macros: MacroTargets
    recipe: str | None = None
    tags: list[str] = Field(default_factory=list)


class Meal(BaseModel):
    meal_type: str
    dishes: list[Dish]


class MealPlan(BaseModel):
    daily_calorie_target: float
    daily_macros: MacroTargets
    meals: list[Meal]


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
