"""RAG(search_recipes/search_delivery_menu)에 없는 음식을 Tavily 웹 검색으로 보완하는 `@tool`.

검색 결과 스니펫을 구조화 LLM에 근거로 제공해 칼로리·매크로를 추출한다 — LLM의 순수 기억이 아니라
실제 검색된 텍스트에 근거한다는 점에서 `estimate_meal_nutrition`(LLM 추정)과 다르다.
`tavily_search`/`structured_llm`을 주입받는 구조라, 테스트에서는 실제 API 호출 없이
가짜 객체로 검증할 수 있다.
"""

from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from pydantic import BaseModel

from app.schemas import Dish, MacroTargets

WEB_FOOD_SOURCE_TAG = "웹검색기반추정"


class _WebFoodEstimate(BaseModel):
    """LLM 구조화 출력 원본. 검색 결과에서 못 찾으면 found=False."""

    found: bool
    name: str
    calories: float
    macros: MacroTargets


def search_web_food_impl(query: str, tavily_search: Runnable, structured_llm: Runnable) -> Dish | None:
    search_result = tavily_search.invoke({"query": f"{query} 칼로리 영양성분"})
    snippets = [item["content"] for item in search_result.get("results", []) if item.get("content")]
    if not snippets:
        return None

    context = "\n\n".join(snippets)
    prompt = (
        f"아래는 '{query}'에 대한 웹 검색 결과 스니펫이다. 이 내용에 실제로 칼로리/영양성분 정보가 있으면 "
        "그 값을 그대로 추출하고, 없으면 found=False로 답하라. 스니펫에 없는 값을 추측해서 채우지 마라.\n\n"
        f"{context}"
    )
    result = structured_llm.invoke(prompt)
    if not result.found:
        return None

    return Dish(
        name=result.name,
        brand=None,
        calories=result.calories,
        macros=result.macros,
        recipe=None,
        tags=[WEB_FOOD_SOURCE_TAG],
    )


_tavily_search: Runnable | None = None
_web_food_llm: Runnable | None = None


def _get_tavily_search() -> Runnable:
    global _tavily_search
    if _tavily_search is None:
        from langchain_tavily import TavilySearch

        _tavily_search = TavilySearch(max_results=3)
    return _tavily_search


def _get_web_food_llm() -> Runnable:
    global _web_food_llm
    if _web_food_llm is None:
        from langchain_openai import ChatOpenAI

        _web_food_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(
            _WebFoodEstimate
        )
    return _web_food_llm


@tool
def search_web_food(query: str) -> Dish | None:
    """search_recipes/search_delivery_menu에서 적절한 요리를 찾지 못했을 때, 웹 검색으로 음식의
    칼로리·영양성분을 찾아 보완한다. 결과를 찾지 못하면 None을 반환한다.
    (검색된 웹 페이지 기반 추정치이며 공식 영양성분표와 다를 수 있음)
    """
    return search_web_food_impl(query, _get_tavily_search(), _get_web_food_llm())
