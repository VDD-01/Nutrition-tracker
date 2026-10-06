from datetime import datetime


class DailyTracker:
    def __init__(self, user_profile):
        self.user_profile = user_profile
        self.meals = []
        self.totals = {
            "calories": 0.0,
            "protein": 0.0,
            "carbohydrates": 0.0,
            "fat": 0.0
        }

    def add_meal(self, meal_description, nutrition_data):
        meal_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": meal_description,
            "nutrition": {
                "calories": nutrition_data["calories"],
                "protein": nutrition_data["protein"],
                "carbohydrates": nutrition_data["carbohydrates"],
                "fat": nutrition_data["fat"]
            },
            "source": nutrition_data.get("source", "unknown"),
            "quantity_info": nutrition_data.get("quantity_info", {})
        }

        self.meals.append(meal_entry)

        self.totals["calories"] += nutrition_data["calories"]
        self.totals["protein"] += nutrition_data["protein"]
        self.totals["carbohydrates"] += nutrition_data["carbohydrates"]
        self.totals["fat"] += nutrition_data["fat"]

        return meal_entry

    def get_progress(self):
        targets = self.user_profile.daily_targets

        progress = {
            "calories": {
                "consumed": round(self.totals["calories"], 1),
                "target": targets["calories"],
                "remaining": round(targets["calories"] - self.totals["calories"], 1),
                "percentage": round((self.totals["calories"] / targets["calories"]) * 100, 1)
            },
            "protein": {
                "consumed": round(self.totals["protein"], 1),
                "target": targets["protein"],
                "remaining": round(targets["protein"] - self.totals["protein"], 1),
                "percentage": round((self.totals["protein"] / targets["protein"]) * 100, 1)
            },
            "carbohydrates": {
                "consumed": round(self.totals["carbohydrates"], 1),
                "target": targets["carbohydrates"],
                "remaining": round(targets["carbohydrates"] - self.totals["carbohydrates"], 1),
                "percentage": round((self.totals["carbohydrates"] / targets["carbohydrates"]) * 100, 1)
            },
            "fat": {
                "consumed": round(self.totals["fat"], 1),
                "target": targets["fat"],
                "remaining": round(targets["fat"] - self.totals["fat"], 1),
                "percentage": round((self.totals["fat"] / targets["fat"]) * 100, 1)
            }
        }

        return progress

    def get_summary(self):
        return {
            "total_meals": len(self.meals),
            "meals": self.meals,
            "totals": self.totals,
            "progress": self.get_progress()
        }

    def reset(self):
        self.meals = []
        self.totals = {
            "calories": 0.0,
            "protein": 0.0,
            "carbohydrates": 0.0,
            "fat": 0.0
        }
