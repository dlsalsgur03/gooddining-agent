from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

from app.middleware.duplicate_guard_middleware import DuplicateToolCallGuardMiddleware

middleware = DuplicateToolCallGuardMiddleware()


def make_request(tool_call, messages):
    return SimpleNamespace(tool_call=tool_call, state={"messages": messages})


class TestDuplicateToolCallGuardMiddleware:
    def test_calls_handler_when_no_prior_identical_call(self):
        tool_call = {"name": "search_recipes", "args": {"query": "닭가슴살"}, "id": "call-1"}
        prior_message = AIMessage(
            content="",
            tool_calls=[{"name": "search_recipes", "args": {"query": "샐러드"}, "id": "call-0"}],
        )
        request = make_request(tool_call, [prior_message])

        result = middleware.wrap_tool_call(request, lambda req: "handler-called")

        assert result == "handler-called"

    def test_blocks_identical_repeated_call(self):
        tool_call = {"name": "search_recipes", "args": {"query": "닭가슴살"}, "id": "call-2"}
        prior_message = AIMessage(
            content="",
            tool_calls=[{"name": "search_recipes", "args": {"query": "닭가슴살"}, "id": "call-1"}],
        )
        request = make_request(tool_call, [prior_message])

        result = middleware.wrap_tool_call(request, lambda req: "handler-called")

        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "call-2"
        assert "중복 호출 차단" in result.content

    def test_ignores_the_current_calls_own_entry(self):
        tool_call = {"name": "search_recipes", "args": {"query": "닭가슴살"}, "id": "call-1"}
        current_message = AIMessage(content="", tool_calls=[tool_call])
        request = make_request(tool_call, [current_message])

        result = middleware.wrap_tool_call(request, lambda req: "handler-called")

        assert result == "handler-called"

    def test_different_args_are_not_treated_as_duplicate(self):
        tool_call = {"name": "search_recipes", "args": {"query": "닭가슴살"}, "id": "call-2"}
        prior_message = AIMessage(
            content="",
            tool_calls=[{"name": "search_recipes", "args": {"query": "두부"}, "id": "call-1"}],
        )
        request = make_request(tool_call, [prior_message])

        result = middleware.wrap_tool_call(request, lambda req: "handler-called")

        assert result == "handler-called"
