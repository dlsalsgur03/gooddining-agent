from app.schemas import MacroTargets
from app.tools import meal_estimation
from app.tools.meal_estimation import _LLMMealEstimate, estimate_meal_nutrition_impl


class FakeStructuredLLM:
    def __init__(self, result: _LLMMealEstimate):
        self._result = result
        self.last_prompt = None

    def invoke(self, prompt):
        self.last_prompt = prompt
        return self._result


class TestEstimateMealNutritionImpl:
    def test_copies_calories_and_macros_from_llm(self):
        fake_llm = FakeStructuredLLM(
            _LLMMealEstimate(
                calories=1400,
                macros=MacroTargets(protein_g=45, carbs_g=20, fat_g=90),
            )
        )

        result = estimate_meal_nutrition_impl("삼겹살 2인분, 소주 한 병", fake_llm)

        assert result.calories == 1400
        assert result.macros.protein_g == 45
        assert result.macros.carbs_g == 20
        assert result.macros.fat_g == 90
        assert fake_llm.last_prompt is not None
        assert "삼겹살" in fake_llm.last_prompt

    def test_disclaimer_is_always_attached_regardless_of_llm_output(self):
        fake_llm = FakeStructuredLLM(
            _LLMMealEstimate(calories=500, macros=MacroTargets(protein_g=10, carbs_g=10, fat_g=10))
        )

        result = estimate_meal_nutrition_impl("아무 음식", fake_llm)

        assert "추정" in result.disclaimer


class TestEstimateMealNutritionTool:
    def test_invoke_uses_injected_llm(self, monkeypatch):
        fake_llm = FakeStructuredLLM(
            _LLMMealEstimate(calories=650, macros=MacroTargets(protein_g=30, carbs_g=50, fat_g=20))
        )
        monkeypatch.setattr(meal_estimation, "_get_structured_llm", lambda: fake_llm)

        result = meal_estimation.estimate_meal_nutrition.invoke({"description": "김치찌개 1인분"})

        assert result.calories == 650
        assert result.macros.protein_g == 30
        assert result.disclaimer
