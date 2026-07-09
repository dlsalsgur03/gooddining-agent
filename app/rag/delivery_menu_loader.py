"""배달 프랜차이즈 메뉴 JSON 데이터셋을 검색용 Document로 변환하는 순수 함수."""

import json
from pathlib import Path

from langchain_core.documents import Document

DELIVERY_MENUS_JSON_PATH = Path(__file__).parent / "delivery_menus.json"


def _to_page_content(item: dict) -> str:
    if item["category"] == "세트":
        return (
            f"{item['brand']} {item['name']} ({item['category']}). "
            f"구성에 따라 칼로리 {item['calories_min']}~{item['calories_max']}kcal."
        )
    return (
        f"{item['brand']} {item['name']} ({item['category']}). "
        f"칼로리 {item['calories']}kcal, 단백질 {item['protein_g']}g, "
        f"당류 {item['sugar_g']}g, 포화지방 {item['saturated_fat_g']}g."
    )


def load_delivery_menu_documents(path: Path = DELIVERY_MENUS_JSON_PATH) -> list[Document]:
    """delivery_menus.json을 읽어 임베딩·검색에 사용할 Document 리스트로 변환한다."""
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Document(page_content=_to_page_content(item), metadata=item)
        for item in items
    ]
