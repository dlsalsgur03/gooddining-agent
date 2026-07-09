"""무리한 절식 유도를 막고, 극단적으로 낮은 칼로리 목표에 경고를 붙이는 Middleware."""

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

from app.state import InnerAgentState

SKIP_MEAL_KEYWORDS = ("끼니를 거르", "굶는 게", "굶으세요", "식사를 거르", "한 끼 정도 거르")

SKIP_MEAL_APPENDIX = "\n\n(참고: 끼니를 거르기보다 가벼운 저칼로리 메뉴로 조절하는 걸 권장합니다.)"

LOW_CALORIE_WARNING = (
    "\n\n⚠️ 설정하신 목표 칼로리가 일일 권장 최소치(1200kcal)보다 낮습니다. "
    "무리한 절식은 건강에 좋지 않으니 실제 적용 전 전문가와 상담을 권장드립니다."
)


class GuardrailMiddleware(AgentMiddleware):
    state_schema = InnerAgentState

    def after_model(self, state, runtime) -> dict | None:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return None  # 아직 Tool 호출 중인 중간 메시지 — 최종 응답이 아니므로 건드리지 않음

        content = last_message.content
        appended = False

        if any(keyword in content for keyword in SKIP_MEAL_KEYWORDS):
            content += SKIP_MEAL_APPENDIX
            appended = True

        calorie_target = state.get("calorie_target")
        if calorie_target is not None and calorie_target.is_below_safe_minimum:
            content += LOW_CALORIE_WARNING
            appended = True

        if not appended:
            return None
        return {"messages": [AIMessage(content=content, id=last_message.id)]}
