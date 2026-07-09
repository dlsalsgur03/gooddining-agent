"""레시피 벡터스토어를 검색하는 `@tool` 인터페이스 레이어."""

from langchain_core.documents import Document
from langchain_core.tools import tool

from app.rag.vectorstore import get_recipe_vectorstore
from app.schemas import Dish, MacroTargets


def _document_to_dish(document: Document) -> Dish:
    metadata = document.metadata
    return Dish(
        name=metadata["name"],
        calories=metadata["calories"],
        macros=MacroTargets(
            protein_g=metadata["protein_g"],
            carbs_g=metadata["carbs_g"],
            fat_g=metadata["fat_g"],
        ),
        recipe=metadata.get("instructions"),
        tags=metadata.get("tags", []),
    )


@tool
def search_recipes(
    query: str,
    exclude_ingredients: list[str] | None = None,
    max_results: int = 3,
) -> list[Dish]:
    """목표 칼로리·영양성분에 맞는 레시피를 검색한다.

    query에는 원하는 특징(예: '저칼로리 고단백 닭가슴살 요리')을 자연어로 담는다.
    exclude_ingredients에 알러지·비선호 재료를 넣으면 해당 재료가 포함된 레시피는 제외된다.
    """
    exclude = set(exclude_ingredients or [])
    vectorstore = get_recipe_vectorstore()
    candidates = vectorstore.similarity_search(query, k=max_results * 3)

    matches: list[Dish] = []
    for doc in candidates:
        ingredients = set(doc.metadata.get("ingredients", []))
        if ingredients & exclude:
            continue
        matches.append(_document_to_dish(doc))
        if len(matches) >= max_results:
            break

    return matches
