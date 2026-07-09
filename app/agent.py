"""LangGraph StateGraph 조립.

바깥쪽 StateGraph는 결정적(deterministic) 라우팅(조건부 분기 2개)을 담당하고,
안쪽 `agent` 노드는 `create_agent`로 만든 자율적 Tool-calling 서브그래프다.
"""

from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.memory.profile_store import ProfileStore
from app.memory.sqlite_profile_store import SQLiteProfileStore
from app.middleware.guardrail_middleware import GuardrailMiddleware
from app.middleware.logging_middleware import LoggingMiddleware
from app.schemas import ActivityLevel, Gender, Goal, MealPlan, UserProfile
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

SYSTEM_PROMPT = (
    "당신은 GoodDining Agent입니다. 사용자의 신체 정보와 목표에 맞는 하루 식단과 레시피를 추천합니다. "
    "잔여 칼로리 예산이 부족하더라도 끼니를 거르라고 답하지 마세요 — 가벼운 저칼로리 메뉴로 조절하도록 안내하세요. "
    "estimate_meal_nutrition, search_delivery_menu의 결과는 추정치이며 정밀한 수치가 아닐 수 있음을 응답에 자연스럽게 언급하세요. "
    "이 서비스는 전문 의료·영양 상담을 대체하지 않습니다."
)

REQUEST_TYPE_HINTS: dict[str, str] = {
    "unplanned_meal": (
        "사용자가 계획에 없던 식사를 이미 했다고 언급했습니다. estimate_meal_nutrition으로 섭취량을 먼저 "
        "추정한 뒤 calc_remaining_budget으로 잔여 예산을 재계산하고, 그에 맞는 메뉴를 추천하세요."
    ),
    "delivery": (
        "사용자가 배달 음식을 원합니다. search_delivery_menu로 원하는 브랜드/메뉴를 검색해 "
        "잔여 예산에 맞는 메뉴를 추천하세요."
    ),
    "general": (
        "사용자가 일반적인 하루 식단을 원합니다. search_recipes를 활용해 하루 전체 예산 기준으로 "
        "아침·점심·저녁 식단을 구성하세요."
    ),
}

_profile_store: ProfileStore = SQLiteProfileStore()


class _ProfileExtraction(BaseModel):
    """대화에서 추출한 프로필 정보. 일부만 언급됐을 수 있어 전부 Optional."""

    gender: Gender | None = None
    age: int | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    activity_level: ActivityLevel | None = None
    goal: Goal | None = None


_PROFILE_EXTRACTION_SYSTEM_PROMPT = (
    "사용자 메시지에서 명시적으로 언급된 신체 정보/목표만 추출하세요. "
    "메시지에 직접 언급되지 않은 필드는 절대 추측하거나 임의의 값으로 채우지 말고 "
    "반드시 null(비워둠)로 남기세요."
)


def _is_profile_complete(extraction: _ProfileExtraction) -> bool:
    return all(
        value is not None
        for value in (
            extraction.gender,
            extraction.age,
            extraction.height_cm,
            extraction.weight_kg,
            extraction.activity_level,
            extraction.goal,
        )
    )


class _RequestClassification(BaseModel):
    request_type: Literal["unplanned_meal", "delivery", "general"]


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
            _ProfileExtraction
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

        _inner_agent = create_agent(
            model="openai:gpt-4o-mini",
            tools=[
                calculate_bmr_tdee_tool,
                calculate_calorie_target_tool,
                calc_remaining_budget,
                search_recipes,
                search_delivery_menu,
                estimate_meal_nutrition,
            ],
            system_prompt=SYSTEM_PROMPT,
            response_format=MealPlan,
            middleware=[LoggingMiddleware(), GuardrailMiddleware()],
            state_schema=InnerAgentState,
        )
    return _inner_agent


def check_profile(state: AgentState) -> dict:
    profile = _profile_store.get(state["user_id"])
    return {"profile": profile}


def route_after_check_profile(state: AgentState) -> str:
    return "calculate_targets" if state.get("profile") is not None else "check_completeness"


def check_completeness(state: AgentState) -> dict:
    latest_message = _latest_human_text(state["messages"])
    extraction = _get_profile_extraction_llm().invoke(
        [
            SystemMessage(content=_PROFILE_EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=latest_message),
        ]
    )

    if not _is_profile_complete(extraction):
        return {"needs_more_info": True}

    profile = UserProfile(
        gender=extraction.gender,
        age=extraction.age,
        height_cm=extraction.height_cm,
        weight_kg=extraction.weight_kg,
        activity_level=extraction.activity_level,
        goal=extraction.goal,
        allergies=[],
        disliked_ingredients=[],
    )
    _profile_store.save(state["user_id"], profile)
    return {"profile": profile, "needs_more_info": False}


def route_after_check_completeness(state: AgentState) -> str:
    return "ask_for_more_info" if state["needs_more_info"] else "calculate_targets"


def ask_for_more_info(state: AgentState) -> dict:
    message = AIMessage(
        content=(
            "식단을 추천해드리려면 성별/나이/키(cm)/몸무게(kg)/활동량(적음·보통·많음)/목표"
            "(감량·유지·증량)를 알려주세요!"
        )
    )
    return {"messages": [message]}


def calculate_targets(state: AgentState) -> dict:
    profile = state["profile"]
    bmr_tdee = nutrition_calc.calculate_bmr_tdee(profile)
    calorie_target = nutrition_calc.calculate_calorie_target(bmr_tdee, profile.goal, profile.weight_kg)
    macros = calorie_target.macros
    summary = (
        f"[사전 계산됨] 하루 목표 칼로리 {calorie_target.target_kcal:.0f}kcal, "
        f"단백질 {macros.protein_g:.0f}g / 탄수화물 {macros.carbs_g:.0f}g / 지방 {macros.fat_g:.0f}g. "
        "이미 계산된 값이니 calculate_bmr_tdee/calculate_calorie_target을 다시 호출하지 말고 이 값을 사용하세요."
    )
    return {
        "calorie_target": calorie_target,
        "messages": [SystemMessage(content=summary)],
    }


def classify_request(state: AgentState) -> dict:
    latest_message = _latest_human_text(state["messages"])
    classification = _get_request_classifier_llm().invoke(latest_message)
    hint = SystemMessage(content=REQUEST_TYPE_HINTS[classification.request_type])
    return {"request_type": classification.request_type, "messages": [hint]}


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("check_profile", check_profile)
    builder.add_node("check_completeness", check_completeness)
    builder.add_node("ask_for_more_info", ask_for_more_info)
    builder.add_node("calculate_targets", calculate_targets)
    builder.add_node("classify_request", classify_request)
    builder.add_node("agent", _get_inner_agent())

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
    builder.add_edge("classify_request", "agent")
    builder.add_edge("agent", END)

    return builder.compile(checkpointer=MemorySaver())
