"""모델/Tool 호출 전후로 요청·응답을 로깅하는 Middleware."""

import logging

from langchain.agents.middleware import AgentMiddleware

logger = logging.getLogger("gooddining.agent")


class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state, runtime) -> None:
        logger.info("model_call_start message_count=%d", len(state["messages"]))
        return None

    def after_model(self, state, runtime) -> None:
        last_message = state["messages"][-1]
        logger.info(
            "model_call_end has_tool_calls=%s",
            bool(getattr(last_message, "tool_calls", None)),
        )
        return None

    def wrap_tool_call(self, request, handler):
        name = request.tool_call["name"]
        logger.info("tool_call_start tool=%s args=%s", name, request.tool_call["args"])
        result = handler(request)
        logger.info("tool_call_end tool=%s", name)
        return result
