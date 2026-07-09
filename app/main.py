"""FastAPI 래퍼: /chat, /health, / (정적 채팅 UI)."""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from app.agent import build_graph
from app.middleware.guardrail_middleware import LOW_CALORIE_WARNING, SKIP_MEAL_APPENDIX
from app.schemas import MealPlan

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
            lines.append(
                f"- {dish.name} ({dish.calories:.0f}kcal, "
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
    return {"message": "GoodDining Agent. 채팅 UI는 /static/index.html 을 확인하세요."}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    graph = _get_graph()
    config = {"configurable": {"thread_id": request.session_id}}
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=request.message)],
            "user_id": request.user_id,
        },
        config=config,
    )
    reply, meal_plan = _build_reply(result)
    return ChatResponse(
        reply=reply,
        needs_more_info=result.get("needs_more_info", False),
        meal_plan=meal_plan,
    )
