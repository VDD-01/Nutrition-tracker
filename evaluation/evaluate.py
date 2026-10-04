import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.predict import NutritionLlama
from config import OLLAMA_MODEL, OLLAMA_HOST


def evaluate(num_samples=50):
    model = NutritionLlama(model_name=OLLAMA_MODEL, host=OLLAMA_HOST)

    test_meals = [
        {"meal": "100g chicken breast grilled", "carb": 0.0, "protein": 31.0, "fat": 3.6, "calories": 165.0},
        {"meal": "one cup cooked white rice", "carb": 45.0, "protein": 4.3, "fat": 0.4, "calories": 206.0},
        {"meal": "one medium banana", "carb": 27.0, "protein": 1.3, "fat": 0.4, "calories": 105.0},
        {"meal": "two scrambled eggs", "carb": 1.6, "protein": 12.0, "fat": 10.0, "calories": 148.0},
        {"meal": "one cup whole milk", "carb": 11.7, "protein": 8.0, "fat": 7.9, "calories": 149.0},
    ]

    results = []
    total_mae = {"carb": 0, "protein": 0, "fat": 0, "calories": 0}

    print(f"\nEvaluating on {len(test_meals)} samples...\n")

    for sample in test_meals:
        pred = model.predict(sample["meal"])

        mae = {
            "carb": abs(pred["carbohydrates"] - sample["carb"]),
            "protein": abs(pred["protein"] - sample["protein"]),
            "fat": abs(pred["fat"] - sample["fat"]),
            "calories": abs(pred["calories"] - sample["calories"]),
        }

        for k in total_mae:
            total_mae[k] += mae[k]

        results.append({
            "meal": sample["meal"],
            "predicted": pred,
            "actual": sample,
            "mae": mae
        })

        print(f"Meal: {sample['meal']}")
        print(f"  Predicted: carb={pred['carbohydrates']:.1f}g, protein={pred['protein']:.1f}g, fat={pred['fat']:.1f}g, cal={pred['calories']:.0f}kcal")
        print(f"  Actual:    carb={sample['carb']:.1f}g, protein={sample['protein']:.1f}g, fat={sample['fat']:.1f}g, cal={sample['calories']:.0f}kcal")
        print()

    n = len(test_meals)
    print("=" * 60)
    print("MEAN ABSOLUTE ERROR")
    print("=" * 60)
    print(f"  Carbs:    {total_mae['carb']/n:.2f}g")
    print(f"  Protein:  {total_mae['protein']/n:.2f}g")
    print(f"  Fat:      {total_mae['fat']/n:.2f}g")
    print(f"  Calories: {total_mae['calories']/n:.2f}kcal")

    output = {
        "num_samples": n,
        "mean_absolute_error": {k: round(v / n, 2) for k, v in total_mae.items()},
        "results": results
    }

    with open("evaluation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to evaluation_results.json")


if __name__ == "__main__":
    evaluate()
