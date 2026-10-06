class RecommendationEngine:
    def __init__(self):
        self.food_recommendations = {
            "high_protein": [
                "chicken breast", "greek yogurt", "eggs", "salmon", "tofu",
                "cottage cheese", "turkey", "tuna", "lentils", "chickpeas"
            ],
            "high_carb": [
                "brown rice", "quinoa", "oats", "sweet potato", "whole wheat bread",
                "pasta", "banana", "berries", "beans"
            ],
            "healthy_fats": [
                "avocado", "nuts", "olive oil", "salmon", "chia seeds",
                "flaxseeds", "dark chocolate", "eggs"
            ],
            "low_calorie": [
                "cucumber", "celery", "leafy greens", "broccoli", "cauliflower",
                "zucchini", "tomatoes", "bell peppers", "mushrooms"
            ],
            "diabetes_friendly": [
                "leafy greens", "non-starchy vegetables", "whole grains",
                "lean proteins", "fatty fish", "nuts", "legumes"
            ],
            "heart_healthy": [
                "salmon", "oats", "berries", "dark leafy greens", "avocado",
                "walnuts", "olive oil", "beans"
            ]
        }

    def generate_recommendations(self, user_profile, nutrition_data):
        remaining = user_profile.get_remaining_needs()
        recommendations = []
        warnings = []

        if remaining["protein"] > 20:
            protein_foods = ", ".join(self.food_recommendations["high_protein"][:3])
            recommendations.append(f"You need {remaining['protein']}g more protein today. Consider: {protein_foods}")
        elif remaining["protein"] < -10:
            warnings.append(f"You've exceeded your protein target by {abs(remaining['protein'])}g")

        if remaining["carbohydrates"] > 30:
            carb_foods = ", ".join(self.food_recommendations["high_carb"][:3])
            recommendations.append(f"You need {remaining['carbohydrates']}g more carbs. Try: {carb_foods}")
        elif remaining["carbohydrates"] < -20:
            warnings.append(f"You've exceeded your carb target by {abs(remaining['carbohydrates'])}g")

        if remaining["fat"] > 15:
            fat_foods = ", ".join(self.food_recommendations["healthy_fats"][:3])
            recommendations.append(f"You need {remaining['fat']}g more healthy fats. Good sources: {fat_foods}")
        elif remaining["fat"] < -10:
            warnings.append(f"You've exceeded your fat target by {abs(remaining['fat'])}g")

        if remaining["calories"] > 500:
            recommendations.append(f"You have {remaining['calories']} calories remaining for today")
        elif remaining["calories"] < -200:
            warnings.append(f"You've exceeded your daily calorie goal by {abs(remaining['calories'])} calories")

        health_recommendations = self._get_health_specific_recommendations(user_profile)
        recommendations.extend(health_recommendations)

        return {
            "recommendations": recommendations,
            "warnings": warnings,
            "remaining_needs": remaining,
            "daily_requirements": user_profile.get_daily_requirements(),
            "current_intake": user_profile.daily_intake
        }

    def _get_health_specific_recommendations(self, user_profile):
        recommendations = []

        if "diabetes" in user_profile.health_conditions:
            recommendations.append("Focus on low-glycemic foods: leafy greens, legumes, and lean proteins")
            recommendations.append("Avoid refined sugars and processed carbohydrates")

        if "hypertension" in user_profile.health_conditions:
            recommendations.append("Limit sodium intake and focus on potassium-rich foods like bananas and leafy greens")

        if "high_cholesterol" in user_profile.health_conditions:
            recommendations.append("Include more omega-3 fatty acids from fish, walnuts, and flaxseeds")
            recommendations.append("Avoid trans fats and limit saturated fats")

        if "obesity" in user_profile.health_conditions or "weight_loss" in user_profile.goals:
            recommendations.append("Focus on high-fiber, low-calorie foods to feel full longer")
            recommendations.append("Consider smaller, frequent meals to maintain metabolism")

        if "muscle_gain" in user_profile.goals:
            recommendations.append("Aim for protein intake within 30 minutes post-workout")
            recommendations.append("Include complex carbs for sustained energy during training")

        return recommendations

    def get_meal_suggestions(self, user_profile):
        remaining = user_profile.get_remaining_needs()
        suggestions = []

        if remaining["calories"] > 300:
            if remaining["protein"] > 20:
                suggestions.append({
                    "meal": "Grilled chicken breast with quinoa and steamed broccoli",
                    "reasoning": "High in protein and balanced macros"
                })
            elif remaining["carbohydrates"] > 40:
                suggestions.append({
                    "meal": "Whole grain pasta with vegetables and lean protein",
                    "reasoning": "Good source of complex carbs"
                })
            else:
                suggestions.append({
                    "meal": "Salmon with sweet potato and mixed vegetables",
                    "reasoning": "Balanced meal with healthy fats"
                })

        return suggestions
