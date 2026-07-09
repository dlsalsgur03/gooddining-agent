from typing import Annotated, Literal, TypedDict

from langchain.agents import AgentState as BaseInnerAgentState
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from app.schemas import CalorieTarget, MealPlan, UserProfile


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    profile: UserProfile | None
    calorie_target: CalorieTarget | None
    request_type: Literal["unplanned_meal", "delivery", "general"] | None
    needs_more_info: bool
    structured_response: MealPlan | None


class InnerAgentState(BaseInnerAgentState):
    """`create_agent`용 확장 상태 스키마.

    바깥쪽 `AgentState`의 `calorie_target`과 같은 이름/타입으로 맞춰서, 바깥 그래프의
    `calculate_targets` 노드가 채운 값이 안쪽 agent와 미들웨어까지 그대로 전달되게 한다.
    """

    calorie_target: CalorieTarget | None
