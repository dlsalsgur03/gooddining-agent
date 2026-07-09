"""배달 프랜차이즈 메뉴 벡터스토어를 검색하는 `@tool` 인터페이스 레이어."""

from langchain_core.tools import tool

from app.rag.vectorstore import get_delivery_menu_vectorstore
from app.schemas import DeliveryMenuItem


@tool
def search_delivery_menu(
    query: str,
    brand: str | None = None,
    max_results: int = 3,
) -> list[DeliveryMenuItem]:
    """배달 프랜차이즈 메뉴 중 잔여 칼로리·영양성분 예산에 맞는 메뉴를 검색한다.

    query에는 원하는 특징(예: '저칼로리 치킨 샌드위치')을 자연어로 담는다.
    brand를 지정하면(예: '버거킹', '서브웨이') 해당 브랜드로만 필터링한다.
    """
    vectorstore = get_delivery_menu_vectorstore()
    filter_fn = (lambda doc: doc.metadata.get("brand") == brand) if brand else None
    docs = vectorstore.similarity_search(query, k=max_results, filter=filter_fn)
    return [DeliveryMenuItem(**doc.metadata) for doc in docs]
