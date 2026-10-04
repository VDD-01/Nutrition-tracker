import os
import sys
import re
import ollama

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OLLAMA_MODEL, OLLAMA_HOST


class NutritionLlama:
    def __init__(self, model_name=OLLAMA_MODEL, host=OLLAMA_HOST):
        self.model_name = model_name
        self.host = host
        self.client = ollama.Client(host=self.host)
        print(f"Connecting to Ollama model {self.model_name}...")
        self.client.show(self.model_name)
        print(f"Ollama model {self.model_name} ready")

    def predict(self, meal_description):
        system_prompt = (
            "You are a nutrition expert AI. Given a meal description, estimate the nutritional values accurately.\n"
            "Respond ONLY in this exact format:\n"
            "Carbohydrates: <number>g\n"
            "Protein: <number>g\n"
            "Fat: <number>g\n"
            "Calories: <number>kcal"
        )
        user_prompt = f"Meal: {meal_description}\n\nEstimate the nutritional values for this meal."

        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={"temperature": 0.1}
        )

        content = response["message"]["content"].strip()
        nutrition = self._parse_nutrition(content)
        return nutrition

    def _parse_nutrition(self, text):
        nutrition = {
            'carbohydrates': 0.0,
            'protein': 0.0,
            'fat': 0.0,
            'calories': 0.0,
            'raw_output': text
        }

        carb_match = re.search(r'Carbohydrates?:\s*(\d+\.?\d*)(?:\s*-\s*(\d+\.?\d*))?', text, re.IGNORECASE)
        protein_match = re.search(r'Protein:\s*(\d+\.?\d*)(?:\s*-\s*(\d+\.?\d*))?', text, re.IGNORECASE)
        fat_match = re.search(r'Fat:\s*(\d+\.?\d*)(?:\s*-\s*(\d+\.?\d*))?', text, re.IGNORECASE)
        cal_match = re.search(r'Calories?:\s*(\d+\.?\d*)(?:\s*-\s*(\d+\.?\d*))?', text, re.IGNORECASE)

        def extract_val(match):
            if not match:
                return 0.0
            val1 = float(match.group(1))
            if match.group(2):
                val2 = float(match.group(2))
                return (val1 + val2) / 2.0
            return val1

        nutrition['carbohydrates'] = extract_val(carb_match)
        nutrition['protein'] = extract_val(protein_match)
        nutrition['fat'] = extract_val(fat_match)
        nutrition['calories'] = extract_val(cal_match)

        return nutrition


def main():
    print("=" * 80)
    print("NUTRITION LLAMA - INFERENCE DEMO (OLLAMA)")
    print("=" * 80)

    model = NutritionLlama()

    test_meals = [
        "I ate a small bowl of rice with grilled chicken",
        "I had 250g of chicken breast",
        "large slice of pepperoni pizza",
        "chocolate ice cream",
        "handful of almonds"
    ]

    print("\nRunning predictions...\n")

    for meal in test_meals:
        print(f"Meal: {meal}")
        result = model.predict(meal)
        print(f"  Carbs: {result['carbohydrates']:.1f}g")
        print(f"  Protein: {result['protein']:.1f}g")
        print(f"  Fat: {result['fat']:.1f}g")
        print(f"  Calories: {result['calories']:.0f}kcal")
        print()


if __name__ == "__main__":
    main()
