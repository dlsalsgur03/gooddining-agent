from app.schemas import MacroTargets
from app.tools.web_food_search import _WebFoodEstimate, search_web_food_impl


class FakeTavilySearch:
    def __init__(self, result):
        self._result = result

    def invoke(self, args):
        return self._result


class FakeStructuredLLM:
    def __init__(self, result):
        self._result = result

    def invoke(self, prompt):
        return self._result


class TestSearchWebFoodImpl:
    def test_returns_dish_when_found(self):
        tavily = FakeTavilySearch(
            {"results": [{"content": "고구마닭가슴살샐러드 270kcal, 단백질 18g, 탄수화물 37g, 지방 6g"}]}
        )
        llm = FakeStructuredLLM(
            _WebFoodEstimate(
                found=True,
                name="고구마닭가슴살샐러드",
                calories=270,
                macros=MacroTargets(protein_g=18, carbs_g=37, fat_g=6),
            )
        )

        dish = search_web_food_impl("고구마닭가슴살샐러드", tavily, llm)

        assert dish.name == "고구마닭가슴살샐러드"
        assert dish.calories == 270
        assert dish.tags == ["웹검색기반추정"]
        assert dish.recipe is None

    def test_returns_none_when_no_search_results(self):
        tavily = FakeTavilySearch({"results": []})
        llm = FakeStructuredLLM(
            _WebFoodEstimate(found=True, name="x", calories=1, macros=MacroTargets(protein_g=0, carbs_g=0, fat_g=0))
        )

        dish = search_web_food_impl("존재하지않는음식", tavily, llm)

        assert dish is None

    def test_returns_none_when_llm_reports_not_found(self):
        tavily = FakeTavilySearch({"results": [{"content": "관련 없는 내용"}]})
        llm = FakeStructuredLLM(
            _WebFoodEstimate(found=False, name="", calories=0, macros=MacroTargets(protein_g=0, carbs_g=0, fat_g=0))
        )

        dish = search_web_food_impl("존재하지않는음식", tavily, llm)

        assert dish is None
