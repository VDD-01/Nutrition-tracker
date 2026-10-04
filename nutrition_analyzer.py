import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_system import NutritionRAG
from fuzzy_processor import FuzzyQuantityProcessor
from inference.predict import NutritionLlama
from config import OLLAMA_MODEL, OLLAMA_HOST


class NutritionAnalyzer:
    def __init__(self):
        self.rag = NutritionRAG()
        self.fuzzy = FuzzyQuantityProcessor()
        self.llm = NutritionLlama(model_name=OLLAMA_MODEL, host=OLLAMA_HOST)

    def analyze(self, meal_description):
        quantity_info = self.fuzzy.extract_quantity_info(meal_description)

        rag_matches = self.rag.search_similar_meals(meal_description)
        rag_nutrition = self.rag.get_weighted_nutrition(rag_matches) if rag_matches else None

        llm_nutrition = self.llm.predict(meal_description)

        if rag_nutrition:
            nutrition = {
                "carbohydrates": round(0.4 * rag_nutrition["carbohydrates"] + 0.6 * llm_nutrition["carbohydrates"], 2),
                "protein": round(0.4 * rag_nutrition["protein"] + 0.6 * llm_nutrition["protein"], 2),
                "fat": round(0.4 * rag_nutrition["fat"] + 0.6 * llm_nutrition["fat"], 2),
                "calories": round(0.4 * rag_nutrition["calories"] + 0.6 * llm_nutrition["calories"], 2),
                "source": "rag+llm",
                "rag_matches": len(rag_matches),
                "raw_output": llm_nutrition.get("raw_output", "")
            }
        else:
            nutrition = {
                "carbohydrates": llm_nutrition["carbohydrates"],
                "protein": llm_nutrition["protein"],
                "fat": llm_nutrition["fat"],
                "calories": llm_nutrition["calories"],
                "source": "llm",
                "rag_matches": 0,
                "raw_output": llm_nutrition.get("raw_output", "")
            }

        if quantity_info["scale"] != 1.0:
            nutrition = self.fuzzy.scale_nutrition(nutrition, quantity_info["scale"])

        nutrition["quantity_info"] = quantity_info
        return nutrition
