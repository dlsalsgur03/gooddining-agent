from langchain_core.embeddings import FakeEmbeddings

from app.rag.delivery_menu_loader import load_delivery_menu_documents
from app.rag.vectorstore import build_vectorstore
from app.tools import delivery_menu_search


class TestLoadDeliveryMenuDocuments:
    def test_loads_all_items_with_metadata(self):
        documents = load_delivery_menu_documents()

        assert len(documents) == 227
        for doc in documents:
            for field in ("brand", "category", "name"):
                assert field in doc.metadata
            assert doc.metadata["name"] in doc.page_content

    def test_set_items_use_calorie_range_in_page_content(self):
        documents = load_delivery_menu_documents()
        set_doc = next(d for d in documents if d.metadata["category"] == "세트")

        assert str(set_doc.metadata["calories_min"]) in set_doc.page_content
        assert str(set_doc.metadata["calories_max"]) in set_doc.page_content


class TestSearchDeliveryMenuTool:
    def _build_fake_vectorstore(self):
        documents = load_delivery_menu_documents()
        return build_vectorstore(documents, FakeEmbeddings(size=32))

    def test_filters_by_brand(self, monkeypatch):
        vectorstore = self._build_fake_vectorstore()
        monkeypatch.setattr(delivery_menu_search, "get_delivery_menu_vectorstore", lambda: vectorstore)

        results = delivery_menu_search.search_delivery_menu.invoke(
            {"query": "저칼로리 메뉴", "brand": "서브웨이", "max_results": 10}
        )

        assert len(results) > 0
        assert all(item.brand == "서브웨이" for item in results)

    def test_regular_item_has_calories_set_item_has_range(self, monkeypatch):
        vectorstore = self._build_fake_vectorstore()
        monkeypatch.setattr(delivery_menu_search, "get_delivery_menu_vectorstore", lambda: vectorstore)

        results = delivery_menu_search.search_delivery_menu.invoke(
            {"query": "버거킹 세트", "brand": "버거킹", "max_results": 200}
        )

        regular_items = [r for r in results if r.category != "세트"]
        set_items = [r for r in results if r.category == "세트"]
        assert regular_items
        assert set_items

        for item in regular_items:
            assert item.calories is not None
            assert item.calories_min is None

        for item in set_items:
            assert item.calories_min is not None
            assert item.calories_max is not None
            assert item.calories is None
