"""동일한 (Tool, 인자) 조합의 반복 호출을 차단하는 Middleware.

`temperature=0`인 모델은 직전과 동일한 대화 맥락에서는 항상 동일한 다음 행동을 선택한다.
검색 결과가 마음에 들지 않아 같은 쿼리로 재검색을 시도하면, 아무 것도 달라지지 않았으니
다시 똑같은 재검색을 선택하게 되어 결정적 무한 루프에 빠질 수 있다(실측으로 확인됨).
이 Middleware는 실제 Tool을 다시 실행하지 않고 "이미 시도했다"는 안내로 대체해 그 루프를 끊는다.
"""

import json

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage


def _call_signature(name: str, args: dict) -> tuple[str, str]:
    return name, json.dumps(args, sort_keys=True, ensure_ascii=False)


class DuplicateToolCallGuardMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        current_id = request.tool_call["id"]
        current_signature = _call_signature(request.tool_call["name"], request.tool_call["args"])

        messages = request.state.get("messages", []) if hasattr(request.state, "get") else []
        prior_signatures = {
            _call_signature(tool_call["name"], tool_call["args"])
            for message in messages
            for tool_call in (getattr(message, "tool_calls", None) or [])
            if tool_call["id"] != current_id
        }

        if current_signature in prior_signatures:
            return ToolMessage(
                content=(
                    f"[중복 호출 차단] '{request.tool_call['name']}'을 동일한 조건으로 이미 호출했습니다. "
                    "같은 조건으로 다시 호출하지 말고, 지금까지 얻은 결과 중 가장 적합한 것으로 응답을 완성하세요."
                ),
                tool_call_id=current_id,
            )
        return handler(request)
