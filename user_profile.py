class UserProfile:
    def __init__(self, age, weight_kg, height_cm, gender, goal="get_in_shape", activity_level="moderate"):
        self.age = age
        self.weight_kg = weight_kg
        self.height_cm = height_cm
        self.gender = gender
        self.goal = goal
        self.activity_level = activity_level
        self.daily_targets = self._calculate_daily_targets()

    def _calculate_bmr(self):
        if self.gender.lower() == "male":
            bmr = 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age + 5
        else:
            bmr = 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age - 161
        return bmr

    def _calculate_tdee(self):
        bmr = self._calculate_bmr()
        activity_multipliers = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very_active": 1.9
        }
        multiplier = activity_multipliers.get(self.activity_level, 1.55)
        return bmr * multiplier

    def _calculate_daily_targets(self):
        tdee = self._calculate_tdee()

        if self.goal == "get_in_shape":
            target_calories = tdee - 300
        elif self.goal == "lose_weight":
            target_calories = tdee - 500
        elif self.goal == "gain_muscle":
            target_calories = tdee + 300
        elif self.goal == "maintain":
            target_calories = tdee
        else:
            target_calories = tdee

        protein_grams = self.weight_kg * 2.0
        fat_grams = self.weight_kg * 0.8

        protein_calories = protein_grams * 4
        fat_calories = fat_grams * 9
        carb_calories = target_calories - protein_calories - fat_calories
        carb_grams = carb_calories / 4

        return {
            "calories": round(target_calories, 1),
            "protein": round(protein_grams, 1),
            "carbohydrates": round(carb_grams, 1),
            "fat": round(fat_grams, 1),
            "bmr": round(self._calculate_bmr(), 1),
            "tdee": round(tdee, 1)
        }

    def get_profile_summary(self):
        return {
            "age": self.age,
            "weight_kg": self.weight_kg,
            "height_cm": self.height_cm,
            "gender": self.gender,
            "goal": self.goal,
            "activity_level": self.activity_level,
            "bmr": self.daily_targets["bmr"],
            "tdee": self.daily_targets["tdee"],
            "daily_targets": {
                "calories": self.daily_targets["calories"],
                "protein": self.daily_targets["protein"],
                "carbohydrates": self.daily_targets["carbohydrates"],
                "fat": self.daily_targets["fat"]
            }
        }
