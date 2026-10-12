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

        if not grag_result["found"]:
            return {
                "carbohydrates": 0.0,
                "protein": 0.0,
                "fat": 0.0,
                "calories": 0.0,
                "source": "not_found",
                "rag_matches": 0,
                "graph_paths": [],
                "nodes_traversed": [],
                "raw_output": "Food item not found in the database. Please check your input.",
                "status": "not_found",
                "message": "Food item not found in the database. Please check your input.",
                "quantity_info": quantity_info,
                "interpreted_meal": corrected_description,
                "corrections": corrections,
            }

        llm_nutrition = self.llm.predict(corrected_description)

        nutrition = {
            "carbohydrates": round(0.4 * grag_result["carbohydrates"] + 0.6 * llm_nutrition["carbohydrates"], 2),
            "protein": round(0.4 * grag_result["protein"] + 0.6 * llm_nutrition["protein"], 2),
            "fat": round(0.4 * grag_result["fat"] + 0.6 * llm_nutrition["fat"], 2),
            "calories": round(0.4 * grag_result["calories"] + 0.6 * llm_nutrition["calories"], 2),
            "source": "grag+llm",
            "rag_matches": len(grag_result["nodes_traversed"]),
            "graph_paths": grag_result["graph_paths"],
            "nodes_traversed": grag_result["nodes_traversed"],
            "raw_output": llm_nutrition.get("raw_output", ""),
            "status": "success",
            "message": "Food item analyzed successfully.",
        }

        if quantity_info["scale"] != 1.0:
            nutrition = self.fuzzy.scale_nutrition(nutrition, quantity_info["scale"])

        nutrition["quantity_info"] = quantity_info
        nutrition["interpreted_meal"] = corrected_description
        nutrition["corrections"] = corrections
        return nutrition
