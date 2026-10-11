import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grag_system import NutritionGRAG
from fuzzy_processor import FuzzyQuantityProcessor
from inference.predict import NutritionLlama
from config import OLLAMA_MODEL, OLLAMA_HOST


class NutritionAnalyzer:
    def __init__(self):
        self.grag = NutritionGRAG()
        self.fuzzy = FuzzyQuantityProcessor()
        self.llm = NutritionLlama(model_name=OLLAMA_MODEL, host=OLLAMA_HOST)

    def analyze(self, meal_description):
        corrected_description, corrections = self.fuzzy.correct_spelling(meal_description)

        quantity_info = self.fuzzy.extract_quantity_info(corrected_description)

        grag_result = self.grag.query_graph(corrected_description)

        llm_nutrition = self.llm.predict(corrected_description)

        if grag_result["found"]:
            nutrition = {
                "carbohydrates": round(0.4 * grag_result["carbohydrates"] + 0.6 * llm_nutrition["carbohydrates"], 2),
                "protein": round(0.4 * grag_result["protein"] + 0.6 * llm_nutrition["protein"], 2),
                "fat": round(0.4 * grag_result["fat"] + 0.6 * llm_nutrition["fat"], 2),
                "calories": round(0.4 * grag_result["calories"] + 0.6 * llm_nutrition["calories"], 2),
                "source": "grag+llm",
                "rag_matches": len(grag_result["nodes_traversed"]),
                "graph_paths": grag_result["graph_paths"],
                "nodes_traversed": grag_result["nodes_traversed"],
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
                "graph_paths": [],
                "nodes_traversed": [],
                "raw_output": llm_nutrition.get("raw_output", "")
            }

        if quantity_info["scale"] != 1.0:
            nutrition = self.fuzzy.scale_nutrition(nutrition, quantity_info["scale"])

        nutrition["quantity_info"] = quantity_info
        nutrition["interpreted_meal"] = corrected_description
        nutrition["corrections"] = corrections
        return nutrition
