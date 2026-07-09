"""검색류 Tool(search_recipes/search_delivery_menu/search_web_food) 호출 총량을 강제로 제한하는 Middleware.

시스템 프롬프트로 "최대 N회까지만 검색하라"고 지시해도 모델이 이를 안정적으로 지키지 않아
(실측으로 확인됨) 계속 새 조합을 탐색하다 끝내 응답을 완성하지 못하는 경우가 있었다.
이 Middleware는 대화당 검색 호출 횟수를 실제로 세어, 한도를 넘으면 Tool을 재실행하지 않고
"더 이상 검색하지 말고 지금 있는 결과로 응답을 완성하라"는 지시로 대체한다.
"""

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

SEARCH_TOOL_NAMES = {"search_recipes", "search_delivery_menu", "search_web_food"}
MAX_SEARCH_TOOL_CALLS = 8


class SearchCallBudgetMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        name = request.tool_call["name"]
        if name not in SEARCH_TOOL_NAMES:
            return handler(request)

        current_id = request.tool_call["id"]
        messages = request.state.get("messages", []) if hasattr(request.state, "get") else []
        prior_search_calls = sum(
            1
            for message in messages
            for tool_call in (getattr(message, "tool_calls", None) or [])
            if tool_call["name"] in SEARCH_TOOL_NAMES and tool_call["id"] != current_id
        )

        if prior_search_calls >= MAX_SEARCH_TOOL_CALLS:
            return ToolMessage(
                content=(
                    "[검색 한도 초과] 검색 Tool 호출 한도에 도달했습니다. "
                    "더 이상 검색하지 말고 지금까지 얻은 결과만으로 반드시 지금 최종 응답(MealPlan)을 완성하세요."
                ),
                tool_call_id=current_id,
            )
        return handler(request)
