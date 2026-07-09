"""사용자가 자연어로 설명한 음식의 칼로리·매크로를 LLM으로 추정하는 `@tool`.

`structured_llm`을 주입받는 구조라, 테스트에서는 실제 OpenAI 호출 대신
`.invoke(prompt)`만 구현한 가짜 객체를 넣어 검증할 수 있다.
"""

from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from pydantic import BaseModel

from app.schemas import MacroTargets, MealNutritionEstimate


class _LLMMealEstimate(BaseModel):
    """LLM 구조화 출력 원본 (disclaimer는 여기 없음 — 코드에서 항상 고정 부착)."""

    calories: float
    macros: MacroTargets


def estimate_meal_nutrition_impl(
    description: str, structured_llm: Runnable
) -> MealNutritionEstimate:
    prompt = (
        "다음 음식/식사를 섭취했을 때 예상되는 총 칼로리(kcal)와 매크로(단백질/탄수화물/지방, g)를 추정해줘. "
        "알코올이 포함되어 있으면 알코올 열량도 총 칼로리에는 반영하되 매크로(단백질/탄수화물/지방)에는 넣지 마. "
        f"음식: {description}"
    )
    result = structured_llm.invoke(prompt)
    return MealNutritionEstimate(calories=result.calories, macros=result.macros)


_structured_llm: Runnable | None = None


def _get_structured_llm() -> Runnable:
    """운영용 진입점. 최초 호출 시 ChatOpenAI를 구성하고 이후 호출은 캐시된 인스턴스를 재사용한다."""
    global _structured_llm
    if _structured_llm is None:
        from langchain_openai import ChatOpenAI

        _structured_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(
            _LLMMealEstimate
        )
    return _structured_llm


@tool
def estimate_meal_nutrition(description: str) -> MealNutritionEstimate:
    """사용자가 설명한 음식의 칼로리·매크로를 추정한다(LLM 추정치, 정밀 수치 아님).

    예: '삼겹살 2인분, 소주 한 병'
    """
    return estimate_meal_nutrition_impl(description, _get_structured_llm())
