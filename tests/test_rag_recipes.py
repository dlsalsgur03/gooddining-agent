from langchain_core.embeddings import FakeEmbeddings

from app.rag.loader import load_recipe_documents
from app.rag.vectorstore import build_vectorstore
from app.tools import recipe_search


class TestLoadRecipeDocuments:
    def test_loads_all_recipes_with_metadata(self):
        documents = load_recipe_documents()

        assert len(documents) == 20
        for doc in documents:
            for field in ("name", "calories", "protein_g", "carbs_g", "fat_g", "tags", "ingredients"):
                assert field in doc.metadata
            assert doc.metadata["name"] in doc.page_content


class TestBuildVectorstore:
    def test_similarity_search_returns_documents(self):
        documents = load_recipe_documents()
        vectorstore = build_vectorstore(documents, FakeEmbeddings(size=32))

        results = vectorstore.similarity_search("고단백 저칼로리 요리", k=3)

        assert len(results) == 3


class TestSearchRecipesTool:
    def _build_fake_vectorstore(self):
        documents = load_recipe_documents()
        return build_vectorstore(documents, FakeEmbeddings(size=32))

    def test_excludes_recipes_with_excluded_ingredients(self, monkeypatch):
        vectorstore = self._build_fake_vectorstore()
        monkeypatch.setattr(recipe_search, "get_recipe_vectorstore", lambda: vectorstore)

        results = recipe_search.search_recipes.invoke(
            {"query": "아무 요리", "exclude_ingredients": ["새우"], "max_results": 20}
        )

        # 재료 목록에 "새우"가 들어간 레시피(새우 두부 볶음)는 결과에 없어야 한다
        assert all(dish.name != "새우 두부 볶음" for dish in results)

    def test_respects_max_results(self, monkeypatch):
        vectorstore = self._build_fake_vectorstore()
        monkeypatch.setattr(recipe_search, "get_recipe_vectorstore", lambda: vectorstore)

        results = recipe_search.search_recipes.invoke(
            {"query": "아무 요리", "exclude_ingredients": [], "max_results": 2}
        )

        assert len(results) == 2

    def test_returns_dish_with_macros_and_recipe(self, monkeypatch):
        vectorstore = self._build_fake_vectorstore()
        monkeypatch.setattr(recipe_search, "get_recipe_vectorstore", lambda: vectorstore)

        results = recipe_search.search_recipes.invoke(
            {"query": "아무 요리", "exclude_ingredients": [], "max_results": 1}
        )

        dish = results[0]
        assert dish.calories > 0
        assert dish.macros.protein_g >= 0
        assert dish.recipe is not None
