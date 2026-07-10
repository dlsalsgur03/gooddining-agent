"""검색류 Tool(search_recipes/search_delivery_menu/search_web_food) 호출 총량을 강제로 제한하는 Middleware.

시스템 프롬프트로 "최대 N회까지만 검색하라"고 지시해도, 심지어 한도 초과를 알리는
ToolMessage로 재차 요청해도 모델이 이를 안정적으로 지키지 않아(실측으로 확인됨) 같은 검색을
무한 반복하며 끝내 응답을 완성하지 못하고 GraphRecursionError로 죽는 경우가 있었다.
"부탁"만으로는 강제력이 없으므로, 한도를 넘으면 `wrap_model_call`에서 검색 Tool 자체를
모델의 tool 목록에서 제거해 애초에 호출할 수 없게 만든다. `wrap_tool_call`은 한 턴에서
여러 검색 Tool이 한꺼번에 호출돼 그 배치 안에서 한도를 넘는 경우를 막는 보조 안전장치다.
"""

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

SEARCH_TOOL_NAMES = {"search_recipes", "search_delivery_menu", "search_web_food"}
MAX_SEARCH_TOOL_CALLS = 8


def _tool_name(tool) -> str | None:
    if isinstance(tool, dict):
        return tool.get("name") or tool.get("function", {}).get("name")
    return getattr(tool, "name", None)


def _count_search_calls(state) -> int:
    messages = state.get("messages", []) if hasattr(state, "get") else []
    return sum(
        1
        for message in messages
        for tool_call in (getattr(message, "tool_calls", None) or [])
        if tool_call["name"] in SEARCH_TOOL_NAMES
    )


class SearchCallBudgetMiddleware(AgentMiddleware):
    def wrap_model_call(self, request, handler):
        if _count_search_calls(request.state) >= MAX_SEARCH_TOOL_CALLS:
            request.tools = [t for t in request.tools if _tool_name(t) not in SEARCH_TOOL_NAMES]
        return handler(request)

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
