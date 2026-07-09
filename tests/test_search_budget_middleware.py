from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

from app.middleware.search_budget_middleware import MAX_SEARCH_TOOL_CALLS, SearchCallBudgetMiddleware

middleware = SearchCallBudgetMiddleware()


def make_request(tool_call, messages):
    return SimpleNamespace(tool_call=tool_call, state={"messages": messages})


def make_prior_search_messages(count):
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "search_recipes", "args": {"query": f"q{i}"}, "id": f"call-{i}"}],
        )
        for i in range(count)
    ]


class TestSearchCallBudgetMiddleware:
    def test_allows_non_search_tools_unconditionally(self):
        tool_call = {"name": "calc_remaining_budget", "args": {}, "id": "call-x"}
        request = make_request(tool_call, make_prior_search_messages(MAX_SEARCH_TOOL_CALLS + 5))

        result = middleware.wrap_tool_call(request, lambda req: "handler-called")

        assert result == "handler-called"

    def test_allows_search_calls_under_budget(self):
        tool_call = {"name": "search_recipes", "args": {"query": "new"}, "id": "call-new"}
        request = make_request(tool_call, make_prior_search_messages(MAX_SEARCH_TOOL_CALLS - 1))

        result = middleware.wrap_tool_call(request, lambda req: "handler-called")

        assert result == "handler-called"

    def test_blocks_search_calls_over_budget(self):
        tool_call = {"name": "search_web_food", "args": {"query": "new"}, "id": "call-new"}
        request = make_request(tool_call, make_prior_search_messages(MAX_SEARCH_TOOL_CALLS))

        result = middleware.wrap_tool_call(request, lambda req: "handler-called")

        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "call-new"
        assert "검색 한도" in result.content

    def test_does_not_count_the_current_calls_own_entry(self):
        tool_call = {"name": "search_recipes", "args": {"query": "new"}, "id": "call-current"}
        current_message = AIMessage(content="", tool_calls=[tool_call])
        prior = make_prior_search_messages(MAX_SEARCH_TOOL_CALLS - 1)
        request = make_request(tool_call, [*prior, current_message])

        result = middleware.wrap_tool_call(request, lambda req: "handler-called")

        assert result == "handler-called"
