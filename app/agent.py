"""LangGraph StateGraph 조립.

바깥쪽 StateGraph는 결정적(deterministic) 라우팅(조건부 분기 2개)을 담당하고,
안쪽 `agent` 노드는 `create_agent`로 만든 자율적 Tool-calling 서브그래프다.
"""

from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.memory.profile_store import ProfileStore
from app.memory.sqlite_profile_store import SQLiteProfileStore
from app.middleware.duplicate_guard_middleware import DuplicateToolCallGuardMiddleware
from app.middleware.guardrail_middleware import GuardrailMiddleware
from app.middleware.logging_middleware import LoggingMiddleware
from app.middleware.search_budget_middleware import SearchCallBudgetMiddleware
from app.middleware.tool_call_budget_middleware import ToolCallBudgetMiddleware
from app.schemas import (
    ActivityLevel,
    BmrTdeeResult,
    Gender,
    Goal,
    MealPlan,
    ProfileExtraction,
    UserProfile,
)
from app.state import AgentState, InnerAgentState
from app.tools import nutrition_calc
from app.tools.agent_tools import (
    calc_remaining_budget,
    calculate_bmr_tdee as calculate_bmr_tdee_tool,
    calculate_calorie_target as calculate_calorie_target_tool,
)
from app.tools.delivery_menu_search import search_delivery_menu
from app.tools.meal_estimation import estimate_meal_nutrition
from app.tools.recipe_search import search_recipes
from app.tools.web_food_search import search_web_food

SYSTEM_PROMPT = (
    "당신은 GoodDining Agent입니다. 사용자의 신체 정보와 목표에 맞는 하루 식단과 레시피를 추천합니다. "
    "각 요리(Dish)의 칼로리와 영양성분(단백질/탄수화물/지방)은 반드시 search_recipes, search_delivery_menu, "
    "estimate_meal_nutrition, search_web_food Tool이 실제로 반환한 값을 그대로 사용하세요 — Tool 검색 결과에 "
    "없는 요리를 임의로 만들어내거나 수치를 추측해서 채우지 마세요. "
    "search_recipes/search_delivery_menu에서 적절한 요리를 찾지 못하면(결과가 없거나 목표와 너무 안 맞으면) "
    "같은 검색을 반복하는 대신 search_web_food로 웹에서 그 음식의 칼로리·영양성분을 찾아 보완하세요. "
    "search_web_food도 결과를 못 찾으면(None) 다른 요리로 대체하세요. "
    "search_delivery_menu 결과로 요리를 구성할 때는 Dish의 brand 필드에 그 메뉴의 브랜드명(예: '서브웨이', '버거킹')을 "
    "반드시 채우세요 — 배달 메뉴는 브랜드 표시 없이 추천하면 사용자가 어디서 주문해야 할지 알 수 없습니다. "
    "search_recipes/search_web_food 결과로 구성한 요리는 brand를 null로 두세요. "
    "Dish는 name, brand, calories, macros, recipe, tags 6개 필드를 항상 전부 채워야 합니다 — "
    "해당 사항이 없는 필드도 생략하지 말고 명시적으로 null 또는 빈 리스트([])로 채우세요. "
    "하루 식단을 구성할 때는 그날 추천하는 모든 요리의 칼로리 합계가 사전 계산된 목표 칼로리(또는 잔여 예산)에 "
    "최대한 근접하도록(±10% 이내) 요리 개수와 구성을 조정하세요 — 지나치게 적거나 초과하는 합계는 피하세요. "
    "끼니 하나에 요리 1개만으로는 목표에 크게 못 미치는 경우가 많으니, 필요하면 한 끼에 요리를 2개 이상 "
    "조합하거나 search_recipes를 여러 번 호출해 끼니별 예산(대략 하루 목표를 끼니 수로 나눈 값)에 맞추세요. "
    "잔여 칼로리 예산이 부족하더라도 끼니를 거르라고 답하지 마세요 — 가벼운 저칼로리 메뉴로 조절하도록 안내하세요. "
    "estimate_meal_nutrition, search_delivery_menu, search_web_food의 결과는 추정치이며 정밀한 수치가 아닐 수 있음을 응답에 자연스럽게 언급하세요. "
    "이미 시도했던 것과 동일하거나 거의 같은 검색어로 search_recipes/search_delivery_menu를 반복 호출하지 마세요 — "
    "검색 결과가 완벽하게 이상적이지 않더라도, 이미 얻은 결과 중 가장 적합한 것을 선택해 식단을 완성하세요. "
    "search_recipes와 search_delivery_menu는 대화 전체에서 합쳐서 최대 5회까지만 호출하세요. "
    "5회를 다 쓰지 않았더라도 끼니마다 적당한 요리를 이미 찾았다면 더 검색하지 말고 바로 최종 식단을 완성하세요. "
    "MealPlan의 summary 필드에는 새 수치를 지어내지 말고 이 대화에서 실제로 계산된 값만 언급하세요. "
    "이 서비스는 전문 의료·영양 상담을 대체하지 않습니다."
)

REQUEST_TYPE_HINTS: dict[str, str] = {
    "unplanned_meal": (
        "사용자가 계획에 없던 식사를 이미 했다고 언급했습니다. estimate_meal_nutrition으로 섭취량을 먼저 "
        "추정한 뒤 calc_remaining_budget으로 잔여 예산을 재계산하고, search_recipes로 검색한 요리 중 "
        "그 잔여 예산 합계에 맞는 메뉴를 추천하세요. "
        "최종 MealPlan은 하루 전체를 나타내야 합니다 — 이미 드신 계획 외 식사도 하나의 끼니로 포함하세요 "
        "(meal_type은 '계획 외 식사'로 하고, estimate_meal_nutrition이 반환한 값을 그대로 사용). "
        "그 뒤에 잔여 예산에 맞게 추천하는 나머지 끼니들을 이어서 구성하세요. "
        "daily_calorie_target/daily_macros는 잔여 예산이 아니라 원래 하루 전체 목표값을 그대로 유지하세요. "
        "MealPlan.summary에는 estimate_meal_nutrition이 추정한 계획 외 식사의 칼로리(예: '삼겹살+소주로 "
        "대략 1,400kcal를 드신 것 같아요')와 calc_remaining_budget이 계산한 잔여 예산(예: '오늘 남은 예산은 "
        "650kcal 정도예요')을 반드시 언급하고, '무리하게 굶기보다는 가벼운 저칼로리 메뉴로 조절하자'는 안내로 "
        "마무리하세요. 두 Tool이 실제로 반환한 값만 쓰고 새로운 숫자를 만들어내지 마세요."
    ),
    "delivery": (
        "사용자가 배달 음식을 원합니다. search_delivery_menu로 원하는 브랜드/메뉴를 검색해 "
        "잔여 예산 합계에 맞는 메뉴 조합을 추천하세요. 각 Dish의 brand 필드에 브랜드명을 반드시 채우세요. "
        "MealPlan.summary에는 calc_remaining_budget으로 계산한 잔여 예산과, search_delivery_menu로 찾은 "
        "메뉴(브랜드/메뉴명)가 왜 그 예산에 맞는지를 1~2문장으로 언급하세요. 실제로 계산·검색된 값만 "
        "사용하고 지어내지 마세요."
    ),
    "general": (
        "사용자가 일반적인 하루 식단을 원합니다. meal_type은 반드시 '아침', '점심', '저녁' 3개를 모두 사용하세요. "
        "search_recipes를 끼니마다 따로 호출해 각 끼니에 요리를 1~2개씩 배정하고, 하루 목표 칼로리에 "
        "합계가 최대한 맞도록(±10% 이내) 요리 개수를 조정하세요. "
        "MealPlan.summary에는 앞선 [사전 계산됨] 시스템 메시지에 있는 BMR/TDEE·하루 목표 칼로리·매크로 "
        "수치를 근거로, 목표(감량/유지/증량)에 따라 왜 이렇게 설정했는지 1~2문장으로 설명하세요. 그 "
        "메시지에 있는 수치만 그대로 인용하고 새로 계산하거나 지어내지 마세요."
    ),
}


_profile_store: ProfileStore = SQLiteProfileStore()


_PROFILE_EXTRACTION_SYSTEM_PROMPT = (
    "사용자 메시지에서 명시적으로 언급된 신체 정보/목표만 추출하세요. "
    "메시지에 직접 언급되지 않은 필드는 절대 추측하거나 임의의 값으로 채우지 말고 "
    "반드시 null(비워둠)로 남기세요. "
    "사용자가 자신의 정확한 기초대사량(BMR)이나 활동대사량(TDEE) 수치를 직접 알려준 경우 "
    "custom_bmr_kcal/custom_tdee_kcal에 그 값을 추출하세요."
)

PROFILE_FIELDS = ("gender", "age", "height_cm", "weight_kg", "activity_level", "goal")

PROFILE_FIELD_LABELS: dict[str, str] = {
    "gender": "성별",
    "age": "나이",
    "height_cm": "키(cm)",
    "weight_kg": "몸무게(kg)",
    "activity_level": "활동량(적음·보통·많음)",
    "goal": "목표(감량·유지·증량)",
}

_GENDER_DISPLAY = {Gender.MALE: "남성", Gender.FEMALE: "여성"}
_ACTIVITY_LEVEL_DISPLAY = {
    ActivityLevel.SEDENTARY: "매우 적음",
    ActivityLevel.LIGHT: "적음",
    ActivityLevel.MODERATE: "보통",
    ActivityLevel.ACTIVE: "많음",
    ActivityLevel.VERY_ACTIVE: "매우 많음",
}
_GOAL_DISPLAY = {Goal.LOSE: "감량", Goal.MAINTAIN: "유지", Goal.GAIN: "증량"}

_PROFILE_FIELD_DISPLAY: dict[str, dict] = {
    "gender": _GENDER_DISPLAY,
    "activity_level": _ACTIVITY_LEVEL_DISPLAY,
    "goal": _GOAL_DISPLAY,
}


def _display_profile_value(field: str, value) -> str:
    if field in _PROFILE_FIELD_DISPLAY:
        return _PROFILE_FIELD_DISPLAY[field][value]
    if field == "age":
        return f"{value}세"
    if field in ("height_cm", "weight_kg"):
        return f"{value:.0f}{'cm' if field == 'height_cm' else 'kg'}"
    return str(value)


def _is_profile_complete(extraction: ProfileExtraction) -> bool:
    return all(getattr(extraction, field) is not None for field in PROFILE_FIELDS)


def _merge_profile_extraction(
    existing: ProfileExtraction, new: ProfileExtraction
) -> ProfileExtraction:
    merged = existing.model_dump()
    for field, value in new.model_dump().items():
        if value is not None:
            merged[field] = value
    return ProfileExtraction(**merged)


class _RequestClassification(BaseModel):
    request_type: Literal["unplanned_meal", "delivery", "general", "profile_update"] = Field(
        description=(
            "unplanned_meal: 사용자가 이미 무언가를 먹었다고 명시적으로 언급한 경우만. "
            "delivery: 배달 음식/특정 프랜차이즈 메뉴를 원하는 경우. "
            "profile_update: 메시지에 '기초대사량'/'BMR' 또는 '활동대사량'/'TDEE'라는 단어가 "
            "명시적으로 등장하며 그 수치를 알려주는 경우만. "
            "general: 그 외 하루 식단 전체를 추천해달라는 일반적인 요청."
        )
    )


_REQUEST_CLASSIFICATION_SYSTEM_PROMPT = (
    "사용자의 최신 메시지를 아래 네 유형 중 하나로 분류하세요.\n"
    "- unplanned_meal: 사용자가 이미 어떤 음식/식사를 했다고 명시적으로 언급했을 때만 해당합니다.\n"
    "- delivery: 사용자가 배달 음식이나 특정 프랜차이즈 메뉴를 원할 때 해당합니다.\n"
    "- profile_update: 메시지에 '기초대사량'/'BMR' 또는 '활동대사량'/'TDEE'라는 단어가 명시적으로 "
    "등장하며 그 수치를 알려줄 때만 해당합니다(예: '내 기초대사량은 1650이고 활동대사량은 2400이야').\n"
    "- general: 그 외 하루 식단 전체(아침/점심/저녁)를 추천해달라는 요청은 모두 general입니다.\n"
    "먹은 음식을 명시적으로 언급하지 않았다면 절대 unplanned_meal로 분류하지 마세요. "
    "'기초대사량'/'BMR'/'활동대사량'/'TDEE'라는 단어가 메시지에 실제로 등장하지 않으면, 나이·키·몸무게 "
    "같은 다른 신체 수치가 언급되었더라도 절대 profile_update로 분류하지 마세요."
)


def _latest_human_text(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


_profile_extraction_llm: Runnable | None = None


def _get_profile_extraction_llm() -> Runnable:
    global _profile_extraction_llm
    if _profile_extraction_llm is None:
        from langchain_openai import ChatOpenAI

        _profile_extraction_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(
            ProfileExtraction
        )
    return _profile_extraction_llm


_request_classifier_llm: Runnable | None = None


def _get_request_classifier_llm() -> Runnable:
    global _request_classifier_llm
    if _request_classifier_llm is None:
        from langchain_openai import ChatOpenAI

        _request_classifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(
            _RequestClassification
        )
    return _request_classifier_llm


_inner_agent = None


def _get_inner_agent():
    global _inner_agent
    if _inner_agent is None:
        from langchain.agents import create_agent
        from langchain_openai import ChatOpenAI

        _inner_agent = create_agent(
            model=ChatOpenAI(model="gpt-4o-mini", temperature=0),
            tools=[
                calculate_bmr_tdee_tool,
                calculate_calorie_target_tool,
                calc_remaining_budget,
                search_recipes,
                search_delivery_menu,
                estimate_meal_nutrition,
                search_web_food,
            ],
            system_prompt=SYSTEM_PROMPT,
            response_format=MealPlan,
            middleware=[
                ToolCallBudgetMiddleware(),
                SearchCallBudgetMiddleware(),
                DuplicateToolCallGuardMiddleware(),
                LoggingMiddleware(),
                GuardrailMiddleware(),
            ],
            state_schema=InnerAgentState,
        )
    return _inner_agent


def check_profile(state: AgentState) -> dict:
    profile = _profile_store.get(state["user_id"])
    return {"profile": profile, "profile_just_created": False}


def route_after_check_profile(state: AgentState) -> str:
    return "calculate_targets" if state.get("profile") is not None else "check_completeness"


def check_completeness(state: AgentState) -> dict:
    latest_message = _latest_human_text(state["messages"])
    new_extraction = _get_profile_extraction_llm().invoke(
        [
            SystemMessage(content=_PROFILE_EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=latest_message),
        ]
    )
    merged = _merge_profile_extraction(state.get("partial_profile") or ProfileExtraction(), new_extraction)

    if not _is_profile_complete(merged):
        return {"needs_more_info": True, "partial_profile": merged}

    profile = UserProfile(
        gender=merged.gender,
        age=merged.age,
        height_cm=merged.height_cm,
        weight_kg=merged.weight_kg,
        activity_level=merged.activity_level,
        goal=merged.goal,
        allergies=[],
        disliked_ingredients=[],
        custom_bmr_kcal=merged.custom_bmr_kcal,
        custom_tdee_kcal=merged.custom_tdee_kcal,
    )
    _profile_store.save(state["user_id"], profile)
    return {
        "profile": profile,
        "profile_just_created": True,
        "needs_more_info": False,
        "partial_profile": ProfileExtraction(),
    }


def route_after_check_completeness(state: AgentState) -> str:
    return "ask_for_more_info" if state["needs_more_info"] else "calculate_targets"


def ask_for_more_info(state: AgentState) -> dict:
    partial = state.get("partial_profile") or ProfileExtraction()
    known_parts = [
        f"{PROFILE_FIELD_LABELS[field]} {_display_profile_value(field, getattr(partial, field))}"
        for field in PROFILE_FIELDS
        if getattr(partial, field) is not None
    ]
    missing_labels = [
        PROFILE_FIELD_LABELS[field] for field in PROFILE_FIELDS if getattr(partial, field) is None
    ]

    content = ""
    if known_parts:
        content += f"{', '.join(known_parts)} 확인했어요! "
    content += f"{', '.join(missing_labels)} 정보도 알려주세요!"

    return {"messages": [AIMessage(content=content)]}


def calculate_targets(state: AgentState) -> dict:
    profile = state["profile"]
    if profile.custom_bmr_kcal is not None and profile.custom_tdee_kcal is not None:
        bmr_tdee = BmrTdeeResult(bmr_kcal=profile.custom_bmr_kcal, tdee_kcal=profile.custom_tdee_kcal)
    else:
        bmr_tdee = nutrition_calc.calculate_bmr_tdee(profile)
    calorie_target = nutrition_calc.calculate_calorie_target(bmr_tdee, profile.goal, profile.weight_kg)
    macros = calorie_target.macros
    bmr_source = "사용자가 직접 알려준" if profile.custom_bmr_kcal is not None else "계산된"
    summary = (
        f"[사전 계산됨] {bmr_source} BMR {bmr_tdee.bmr_kcal:.0f}kcal, TDEE {bmr_tdee.tdee_kcal:.0f}kcal, "
        f"하루 목표 칼로리 {calorie_target.target_kcal:.0f}kcal, "
        f"단백질 {macros.protein_g:.0f}g / 탄수화물 {macros.carbs_g:.0f}g / 지방 {macros.fat_g:.0f}g. "
        "이미 계산된 값이니 calculate_bmr_tdee/calculate_calorie_target을 다시 호출하지 말고 이 값을 사용하세요."
    )
    return {
        "calorie_target": calorie_target,
        "messages": [SystemMessage(content=summary)],
    }


def classify_request(state: AgentState) -> dict:
    latest_message = _latest_human_text(state["messages"])
    classification = _get_request_classifier_llm().invoke(
        [
            SystemMessage(content=_REQUEST_CLASSIFICATION_SYSTEM_PROMPT),
            HumanMessage(content=latest_message),
        ]
    )
    updates: dict = {"request_type": classification.request_type}
    # profile_update는 식단 추천 agent로 가지 않으므로 REQUEST_TYPE_HINTS에 없다.
    hint_text = REQUEST_TYPE_HINTS.get(classification.request_type)
    if hint_text is not None:
        updates["messages"] = [SystemMessage(content=hint_text)]
    return updates


def route_after_classify_request(state: AgentState) -> str:
    return "update_profile" if state["request_type"] == "profile_update" else "agent"


class _CustomMetabolismExtraction(BaseModel):
    custom_bmr_kcal: float | None = Field(
        default=None, description="사용자가 직접 알려준 기초대사량(BMR, kcal). 언급 없으면 null."
    )
    custom_tdee_kcal: float | None = Field(
        default=None, description="사용자가 직접 알려준 활동대사량(TDEE, kcal). 언급 없으면 null."
    )


_CUSTOM_METABOLISM_SYSTEM_PROMPT = (
    "사용자 메시지에서 기초대사량(BMR)과 활동대사량(TDEE) 수치(kcal)를 추출하세요. "
    "메시지에 언급되지 않은 값은 절대 추측하지 말고 null로 남기세요."
)

_custom_metabolism_llm: Runnable | None = None


def _get_custom_metabolism_llm() -> Runnable:
    global _custom_metabolism_llm
    if _custom_metabolism_llm is None:
        from langchain_openai import ChatOpenAI

        _custom_metabolism_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(
            _CustomMetabolismExtraction
        )
    return _custom_metabolism_llm


def update_profile(state: AgentState) -> dict:
    latest_message = _latest_human_text(state["messages"])
    extraction = _get_custom_metabolism_llm().invoke(
        [
            SystemMessage(content=_CUSTOM_METABOLISM_SYSTEM_PROMPT),
            HumanMessage(content=latest_message),
        ]
    )

    if extraction.custom_bmr_kcal is None or extraction.custom_tdee_kcal is None:
        content = (
            "기초대사량과 활동대사량을 정확히 반영하려면 두 값이 모두 필요해요. "
            "예: '내 기초대사량은 1650kcal, 활동대사량은 2400kcal야'처럼 두 값을 함께 알려주세요."
        )
        return {"messages": [AIMessage(content=content)]}

    profile = state["profile"]
    updated_profile = profile.model_copy(
        update={
            "custom_bmr_kcal": extraction.custom_bmr_kcal,
            "custom_tdee_kcal": extraction.custom_tdee_kcal,
        }
    )
    _profile_store.save(state["user_id"], updated_profile)

    calorie_target = nutrition_calc.calculate_calorie_target(
        BmrTdeeResult(bmr_kcal=extraction.custom_bmr_kcal, tdee_kcal=extraction.custom_tdee_kcal),
        updated_profile.goal,
        updated_profile.weight_kg,
    )
    macros = calorie_target.macros
    content = (
        f"기초대사량 {extraction.custom_bmr_kcal:.0f}kcal, 활동대사량 {extraction.custom_tdee_kcal:.0f}kcal로 "
        "직접 알려주신 값을 저장했어요. 앞으로는 이 값을 기준으로 계산할게요. "
        f"목표를 반영한 하루 목표 칼로리는 약 {calorie_target.target_kcal:.0f}kcal, "
        f"단백질 {macros.protein_g:.0f}g / 탄수화물 {macros.carbs_g:.0f}g / 지방 {macros.fat_g:.0f}g이에요."
    )
    return {"profile": updated_profile, "messages": [AIMessage(content=content)]}


MEAL_PLAN_CALORIE_TOLERANCE = 0.15
MAX_MEAL_PLAN_RETRIES = 2
REQUIRED_MEAL_TYPES = {"아침", "점심", "저녁"}


def verify_meal_plan(state: AgentState) -> dict:
    """하루 전체 식단(`request_type == "general"`)에 한해 칼로리 합계·끼니 구성을 검증.

    `calc_remaining_budget`이 계산하는 잔여 예산은 outer state에 노출되지 않아 unplanned_meal/
    delivery 요청은 검증 대상에서 제외한다(하루 목표 전체가 아니라 부분 예산이 기준이라 비교 불가).
    """
    meal_plan = state.get("structured_response")
    if meal_plan is None or state.get("request_type") != "general":
        return {"needs_meal_plan_retry": False}

    total_calories = sum(dish.calories for meal in meal_plan.meals for dish in meal.dishes)
    target_calories = state["calorie_target"].target_kcal
    diff_ratio = abs(total_calories - target_calories) / target_calories
    missing_meal_types = REQUIRED_MEAL_TYPES - {meal.meal_type for meal in meal_plan.meals}
    retries = state.get("meal_plan_retries", 0)

    is_valid = diff_ratio <= MEAL_PLAN_CALORIE_TOLERANCE and not missing_meal_types
    if is_valid or retries >= MAX_MEAL_PLAN_RETRIES:
        return {"needs_meal_plan_retry": False}

    feedback_parts = []
    if missing_meal_types:
        feedback_parts.append(
            f"{', '.join(sorted(missing_meal_types))} 끼니가 빠졌습니다. 아침/점심/저녁을 모두 포함하세요."
        )
    if diff_ratio > MEAL_PLAN_CALORIE_TOLERANCE:
        direction = "부족" if total_calories < target_calories else "초과"
        feedback_parts.append(
            f"현재 요리 칼로리 합계는 {total_calories:.0f}kcal로 목표 {target_calories:.0f}kcal 대비 "
            f"{direction}합니다. 요리 개수나 구성을 조정해 목표에 더 가깝게 다시 만드세요."
        )
    feedback = SystemMessage(content="[검증 실패] " + " ".join(feedback_parts))

    return {
        "messages": [feedback],
        "meal_plan_retries": retries + 1,
        "needs_meal_plan_retry": True,
    }


def route_after_verify_meal_plan(state: AgentState) -> str:
    return "agent" if state["needs_meal_plan_retry"] else "end"


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("check_profile", check_profile)
    builder.add_node("check_completeness", check_completeness)
    builder.add_node("ask_for_more_info", ask_for_more_info)
    builder.add_node("calculate_targets", calculate_targets)
    builder.add_node("classify_request", classify_request)
    builder.add_node("update_profile", update_profile)
    builder.add_node("agent", _get_inner_agent())
    builder.add_node("verify_meal_plan", verify_meal_plan)

    builder.add_edge(START, "check_profile")
    builder.add_conditional_edges(
        "check_profile",
        route_after_check_profile,
        {"calculate_targets": "calculate_targets", "check_completeness": "check_completeness"},
    )
    builder.add_conditional_edges(
        "check_completeness",
        route_after_check_completeness,
        {"ask_for_more_info": "ask_for_more_info", "calculate_targets": "calculate_targets"},
    )
    builder.add_edge("ask_for_more_info", END)
    builder.add_edge("calculate_targets", "classify_request")
    builder.add_conditional_edges(
        "classify_request",
        route_after_classify_request,
        {"agent": "agent", "update_profile": "update_profile"},
    )
    builder.add_edge("update_profile", END)
    builder.add_edge("agent", "verify_meal_plan")
    builder.add_conditional_edges(
        "verify_meal_plan",
        route_after_verify_meal_plan,
        {"agent": "agent", "end": END},
    )

    return builder.compile(checkpointer=MemorySaver())
