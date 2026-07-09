import logging
from types import SimpleNamespace

from langchain_core.messages import AIMessage

from app.middleware.logging_middleware import LoggingMiddleware

middleware = LoggingMiddleware()


class TestLoggingMiddleware:
    def test_before_model_logs_message_count(self, caplog):
        state = {"messages": [AIMessage(content="hi")]}

        with caplog.at_level(logging.INFO, logger="gooddining.agent"):
            middleware.before_model(state, runtime=None)

        assert "model_call_start" in caplog.text
        assert "message_count=1" in caplog.text

    def test_after_model_logs_whether_tool_calls_present(self, caplog):
        message = AIMessage(content="", tool_calls=[{"name": "search_recipes", "args": {}, "id": "1"}])
        state = {"messages": [message]}

        with caplog.at_level(logging.INFO, logger="gooddining.agent"):
            middleware.after_model(state, runtime=None)

        assert "model_call_end" in caplog.text
        assert "has_tool_calls=True" in caplog.text

    def test_wrap_tool_call_logs_and_returns_handler_result(self, caplog):
        request = SimpleNamespace(tool_call={"name": "search_recipes", "args": {"query": "닭가슴살"}})

        def fake_handler(req):
            return "fake-result"

        with caplog.at_level(logging.INFO, logger="gooddining.agent"):
            result = middleware.wrap_tool_call(request, fake_handler)

        assert result == "fake-result"
        assert "tool_call_start" in caplog.text
        assert "search_recipes" in caplog.text
        assert "tool_call_end" in caplog.text
