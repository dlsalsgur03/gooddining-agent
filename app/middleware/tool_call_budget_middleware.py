"""전체 Tool 호출 총량을 강제로 제한해 무한 루프를 끊는 Middleware.

`SearchCallBudgetMiddleware`(검색류 한도)와 `DuplicateToolCallGuardMiddleware`(중복 호출 차단)는
한도를 넘으면 Tool을 재실행하지 않고 "그만하고 최종 응답을 완성하라"는 안내 메시지로 대체한다.
하지만 gpt-4o-mini가 그 안내를 무시하고 (검색 Tool이 아닌) estimate_meal_nutrition 같은 다른
Tool로 완전히 동일한 무한 반복에 빠지는 경우가 실측으로 확인됐다. 안내 메시지만으로는 모델을
멈출 강제력이 없으므로, 대화 전체에서 실행된 Tool 호출 수(차단된 것 포함)가 한도를 넘으면
다음 모델 호출에서 Tool 자체를 목록에서 제거해 모델이 반드시 최종 구조화 응답(MealPlan)을
내도록 강제한다.
"""

from langchain.agents.middleware import AgentMiddleware

MAX_TOTAL_TOOL_CALLS = 20


def _count_tool_calls(state) -> int:
    messages = state.get("messages", []) if hasattr(state, "get") else []
    return sum(1 for message in messages for _ in (getattr(message, "tool_calls", None) or []))


class ToolCallBudgetMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        if _count_tool_calls(request.state) >= MAX_TOTAL_TOOL_CALLS:
            request.tools = []
        return handler(request)
