import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fuzzy_processor import FuzzyQuantityProcessor
from standalone_predictor import StandaloneNutritionPredictor


class NutritionAnalyzerStandalone:
    def __init__(self):
        self.fuzzy = FuzzyQuantityProcessor()
        self.predictor = StandaloneNutritionPredictor()

    def analyze(self, meal_description):
        quantity_info = self.fuzzy.extract_quantity_info(meal_description)

        llm_nutrition = self.predictor.predict(meal_description)

        nutrition = {
            "carbohydrates": llm_nutrition["carbohydrates"],
            "protein": llm_nutrition["protein"],
            "fat": llm_nutrition["fat"],
            "calories": llm_nutrition["calories"],
            "source": "llm_standalone",
            "rag_matches": 0,
            "raw_output": llm_nutrition.get("raw_output", "")
        }

        if quantity_info["scale"] != 1.0:
            nutrition = self.fuzzy.scale_nutrition(nutrition, quantity_info["scale"])

        nutrition["quantity_info"] = quantity_info
        return nutrition
