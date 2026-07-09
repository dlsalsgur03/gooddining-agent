"""레시피 JSON 데이터셋을 검색용 Document로 변환하는 순수 함수."""

import json
from pathlib import Path

from langchain_core.documents import Document

RECIPES_JSON_PATH = Path(__file__).parent / "recipes.json"


def _to_page_content(recipe: dict) -> str:
    tags = ", ".join(recipe["tags"])
    ingredients = ", ".join(recipe["ingredients"])
    return (
        f"{recipe['name']} ({tags}). "
        f"재료: {ingredients}. "
        f"칼로리 {recipe['calories']}kcal, "
        f"단백질 {recipe['protein_g']}g, 탄수화물 {recipe['carbs_g']}g, 지방 {recipe['fat_g']}g."
    )


def load_recipe_documents(path: Path = RECIPES_JSON_PATH) -> list[Document]:
    """recipes.json을 읽어 임베딩·검색에 사용할 Document 리스트로 변환한다."""
    recipes = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Document(page_content=_to_page_content(recipe), metadata=recipe)
        for recipe in recipes
    ]
