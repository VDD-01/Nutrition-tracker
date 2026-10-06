import os
import sys
from colorama import init, Fore, Style

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nutrition_analyzer import NutritionAnalyzer
from user_profile import UserProfile
from daily_tracker import DailyTracker

init(autoreset=True)


class NutritionAssistant:
    def __init__(self):
        self.analyzer = NutritionAnalyzer()
        self.user = UserProfile(
            age=20,
            weight_kg=75,
            height_cm=174,
            gender="male",
            goal="get_in_shape",
            activity_level="moderate"
        )
        self.tracker = DailyTracker(self.user)
        self._show_welcome()

    def _show_welcome(self):
        print("\n" + "=" * 70)
        print(Fore.CYAN + Style.BRIGHT + "NUTRITION TRACKING ASSISTANT")
        print("=" * 70)
        profile = self.user.get_profile_summary()
        print(f"\n{Fore.GREEN}User Profile:")
        print(f"  Age: {profile['age']} years")
        print(f"  Weight: {profile['weight_kg']} kg")
        print(f"  Height: {profile['height_cm']} cm")
        print(f"  Gender: {profile['gender'].title()}")
        print(f"  Goal: {profile['goal'].replace('_', ' ').title()}")
        print(f"  Activity Level: {profile['activity_level'].title()}")

        print(f"\n{Fore.YELLOW}Metabolic Information:")
        print(f"  BMR (Basal Metabolic Rate): {profile['bmr']} kcal/day")
        print(f"  TDEE (Total Daily Energy Expenditure): {profile['tdee']} kcal/day")

        print(f"\n{Fore.MAGENTA}Daily Nutrition Targets:")
        targets = profile['daily_targets']
        print(f"  Calories: {targets['calories']} kcal")
        print(f"  Protein: {targets['protein']} g")
        print(f"  Carbohydrates: {targets['carbohydrates']} g")
        print(f"  Fat: {targets['fat']} g")
        print("=" * 70 + "\n")

    def _format_progress_bar(self, percentage, width=30):
        filled = int((percentage / 100) * width)
        bar = "█" * filled + "░" * (width - filled)

        if percentage < 70:
            color = Fore.GREEN
        elif percentage < 100:
            color = Fore.YELLOW
        else:
            color = Fore.RED

        return f"{color}{bar}{Style.RESET_ALL}"

    def _display_nutrition_result(self, meal_description, nutrition):
        print(f"\n{Fore.CYAN}Meal Analyzed: {Style.BRIGHT}{meal_description}")
        print(f"{Fore.WHITE}Source: {nutrition.get('source', 'unknown').upper()}")

        qty_info = nutrition.get('quantity_info', {})
        if qty_info:
            method = qty_info.get('method', 'default')
            if method == 'crisp':
                print(f"Quantity: {qty_info.get('quantity')} {qty_info.get('unit')} ({qty_info.get('grams', 0):.1f}g)")
            elif method == 'mamdani':
                print(f"Quantity: Fuzzy estimate ~{qty_info.get('grams', 0):.0f}g (linguistic portion)")

        print(f"\n{Fore.GREEN}Nutrition Breakdown:")
        print(f"  Calories: {nutrition['calories']:.1f} kcal")
        print(f"  Protein: {nutrition['protein']:.1f} g")
        print(f"  Carbohydrates: {nutrition['carbohydrates']:.1f} g")
        print(f"  Fat: {nutrition['fat']:.1f} g")

    def _display_daily_progress(self):
        progress = self.tracker.get_progress()

        print(f"\n{Fore.YELLOW}{Style.BRIGHT}═══ DAILY PROGRESS ═══")

        for nutrient in ['calories', 'protein', 'carbohydrates', 'fat']:
            data = progress[nutrient]
            bar = self._format_progress_bar(data['percentage'])

            unit = "kcal" if nutrient == "calories" else "g"
            nutrient_display = nutrient.title()

            print(f"\n{Fore.CYAN}{nutrient_display}:")
            print(f"  {bar} {data['percentage']:.1f}%")
            print(f"  {data['consumed']:.1f}/{data['target']:.1f} {unit} "
                  f"({'+' if data['remaining'] < 0 else ''}{abs(data['remaining']):.1f} {unit} "
                  f"{'over' if data['remaining'] < 0 else 'remaining'})")

        total_meals = len(self.tracker.meals)
        print(f"\n{Fore.WHITE}Total meals logged today: {total_meals}")
        print("=" * 70)

    def _show_meal_history(self):
        if not self.tracker.meals:
            print(f"\n{Fore.YELLOW}No meals logged yet today.")
            return

        print(f"\n{Fore.CYAN}{Style.BRIGHT}═══ MEAL HISTORY ═══")
        for i, meal in enumerate(self.tracker.meals, 1):
            print(f"\n{Fore.GREEN}Meal #{i} - {meal['timestamp']}")
            print(f"  {meal['description']}")
            n = meal['nutrition']
            print(f"  {n['calories']:.1f} kcal | "
                  f"P: {n['protein']:.1f}g | "
                  f"C: {n['carbohydrates']:.1f}g | "
                  f"F: {n['fat']:.1f}g")
        print("=" * 70)

    def _show_help(self):
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Available Commands:")
        print(f"{Fore.WHITE}  - Type your meal description to log it")
        print(f"  - 'progress' or 'p' - Show daily progress")
        print(f"  - 'history' or 'h' - Show meal history")
        print(f"  - 'reset' or 'r' - Reset daily tracker")
        print(f"  - 'help' - Show this help message")
        print(f"  - 'quit' or 'exit' - Exit assistant")
        print()

    def run(self):
        self._show_help()

        while True:
            try:
                user_input = input(f"{Fore.YELLOW}Enter meal or command: {Style.RESET_ALL}").strip()

                if not user_input:
                    continue

                command = user_input.lower()

                if command in ['quit', 'exit', 'q']:
                    print(f"\n{Fore.GREEN}Goodbye! Stay healthy!")
                    break

                elif command in ['help']:
                    self._show_help()

                elif command in ['progress', 'p']:
                    self._display_daily_progress()

                elif command in ['history', 'h']:
                    self._show_meal_history()

                elif command in ['reset', 'r']:
                    self.tracker.reset()
                    print(f"\n{Fore.GREEN}Daily tracker has been reset!")

                else:
                    print(f"\n{Fore.YELLOW}Analyzing meal...")
                    nutrition = self.analyzer.analyze(user_input)
                    self.tracker.add_meal(user_input, nutrition)

                    self._display_nutrition_result(user_input, nutrition)
                    self._display_daily_progress()

            except KeyboardInterrupt:
                print(f"\n\n{Fore.GREEN}Goodbye! Stay healthy!")
                break
            except Exception as e:
                print(f"\n{Fore.RED}Error: {str(e)}")
                print(f"{Fore.YELLOW}Type 'help' for available commands.")


if __name__ == "__main__":
    assistant = NutritionAssistant()
    assistant.run()
