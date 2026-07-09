"""FastAPI 래퍼: /chat, /health, / (정적 채팅 UI)."""

from datetime import date
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel

from app.agent import build_graph
from app.memory.sqlite_meal_history_store import SQLiteMealHistoryStore
from app.memory.sqlite_profile_store import SQLiteProfileStore
from app.middleware.guardrail_middleware import LOW_CALORIE_WARNING, SKIP_MEAL_APPENDIX
from app.schemas import MealPlan, UserProfile

# 데이터셋에 없는 메뉴를 계속 검색하는 등 agent의 Tool 호출 루프가 수렴하지 못할 때
# 무한 반복(및 불필요한 API 비용)을 막기 위한 상한. 정상 시나리오(검색 여러 번 + 검증 재시도 2회)는
# 이 안에서 충분히 끝나는 것을 실측으로 확인함.
GRAPH_RECURSION_LIMIT = 60

app = FastAPI(title="GoodDining Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_graph = None
_meal_history_store = SQLiteMealHistoryStore()
_profile_store = SQLiteProfileStore()


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    needs_more_info: bool
    meal_plan: MealPlan | None = None


def _format_meal_plan(meal_plan: MealPlan) -> str:
    macros = meal_plan.daily_macros
    lines = [
        f"오늘의 목표 칼로리: {meal_plan.daily_calorie_target:.0f}kcal "
        f"(단백질 {macros.protein_g:.0f}g / 탄수화물 {macros.carbs_g:.0f}g / 지방 {macros.fat_g:.0f}g)"
    ]
    for meal in meal_plan.meals:
        lines.append(f"\n[{meal.meal_type}]")
        for dish in meal.dishes:
            dish_macros = dish.macros
            dish_label = f"{dish.name} ({dish.brand})" if dish.brand else dish.name
            lines.append(
                f"- {dish_label} ({dish.calories:.0f}kcal, "
                f"단백질 {dish_macros.protein_g:.0f}g / 탄수화물 {dish_macros.carbs_g:.0f}g / "
                f"지방 {dish_macros.fat_g:.0f}g)"
            )
    return "\n".join(lines)


def _extract_warnings(content: str) -> list[str]:
    warnings = []
    if SKIP_MEAL_APPENDIX in content:
        warnings.append(SKIP_MEAL_APPENDIX.strip())
    if LOW_CALORIE_WARNING in content:
        warnings.append(LOW_CALORIE_WARNING.strip())
    return warnings


def _build_reply(result: dict) -> tuple[str, MealPlan | None]:
    meal_plan = result.get("structured_response")
    last_message_content = result["messages"][-1].content

    if meal_plan is None:
        return last_message_content, None

    warnings = _extract_warnings(last_message_content)
    reply = _format_meal_plan(meal_plan)
    if warnings:
        reply += "\n\n" + "\n".join(warnings)
    return reply, meal_plan


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    graph = _get_graph()
    config = {
        "configurable": {"thread_id": request.session_id},
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }
    try:
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=request.message)],
                "user_id": request.user_id,
            },
            config=config,
        )
    except (GraphRecursionError, StructuredOutputValidationError):
        return ChatResponse(
            reply=(
                "죄송해요, 지금 조건에 맞는 메뉴를 찾는 데 어려움을 겪고 있어요. "
                "요청을 조금 더 구체적으로 다시 말씀해주시겠어요?"
            ),
            needs_more_info=False,
            meal_plan=None,
        )
    reply, meal_plan = _build_reply(result)
    if meal_plan is not None:
        _meal_history_store.save(request.user_id, date.today().isoformat(), meal_plan)
    return ChatResponse(
        reply=reply,
        needs_more_info=result.get("needs_more_info", False),
        meal_plan=meal_plan,
    )


@app.get("/profile/{user_id}", response_model=UserProfile)
def get_profile(user_id: str):
    profile = _profile_store.get(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="저장된 프로필이 없습니다.")
    return profile


@app.get("/meals/{user_id}/dates")
def list_meal_dates(user_id: str):
    return {"dates": _meal_history_store.list_dates(user_id)}


@app.get("/meals/{user_id}/{meal_date}", response_model=MealPlan)
def get_meal_by_date(user_id: str, meal_date: str):
    meal_plan = _meal_history_store.get(user_id, meal_date)
    if meal_plan is None:
        raise HTTPException(status_code=404, detail="해당 날짜의 식단 기록이 없습니다.")
    return meal_plan
