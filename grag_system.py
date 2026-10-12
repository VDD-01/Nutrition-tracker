import re
import networkx as nx


class NutritionGRAG:
    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_knowledge_graph()

    def _build_knowledge_graph(self):
        ingredients = {
            "chicken breast": {"carb": 0.0, "protein": 31.02, "fat": 3.57, "calories": 165.0, "category": "poultry"},
            "chicken thigh": {"carb": 0.0, "protein": 24.0, "fat": 8.0, "calories": 177.0, "category": "poultry"},
            "turkey breast": {"carb": 0.0, "protein": 29.0, "fat": 1.5, "calories": 135.0, "category": "poultry"},
            "ground beef": {"carb": 0.0, "protein": 17.17, "fat": 20.0, "calories": 254.0, "category": "meat"},
            "beef steak": {"carb": 0.0, "protein": 26.0, "fat": 15.0, "calories": 242.0, "category": "meat"},
            "pork": {"carb": 0.0, "protein": 25.7, "fat": 14.0, "calories": 231.0, "category": "meat"},
            "bacon": {"carb": 1.4, "protein": 37.0, "fat": 42.0, "calories": 541.0, "category": "meat"},
            "ham": {"carb": 1.5, "protein": 21.0, "fat": 6.0, "calories": 145.0, "category": "meat"},
            "sausage": {"carb": 1.6, "protein": 12.0, "fat": 27.0, "calories": 301.0, "category": "meat"},
            "salmon": {"carb": 0.0, "protein": 19.84, "fat": 6.34, "calories": 142.0, "category": "fish"},
            "tuna": {"carb": 0.0, "protein": 28.0, "fat": 1.0, "calories": 132.0, "category": "fish"},
            "shrimp": {"carb": 0.2, "protein": 24.0, "fat": 0.3, "calories": 99.0, "category": "seafood"},
            "egg": {"carb": 0.72, "protein": 12.56, "fat": 9.51, "calories": 143.0, "category": "dairy_egg"},
            "egg white": {"carb": 0.7, "protein": 11.0, "fat": 0.2, "calories": 52.0, "category": "dairy_egg"},
            "milk": {"carb": 4.8, "protein": 3.15, "fat": 3.25, "calories": 61.0, "category": "dairy"},
            "cheddar cheese": {"carb": 1.3, "protein": 25.0, "fat": 33.0, "calories": 403.0, "category": "dairy"},
            "paneer": {"carb": 1.2, "protein": 18.3, "fat": 20.8, "calories": 265.0, "category": "dairy"},
            "yogurt": {"carb": 4.7, "protein": 3.5, "fat": 3.3, "calories": 61.0, "category": "dairy"},
            "butter": {"carb": 0.06, "protein": 0.85, "fat": 81.11, "calories": 717.0, "category": "fat_oil"},
            "olive oil": {"carb": 0.0, "protein": 0.0, "fat": 100.0, "calories": 884.0, "category": "fat_oil"},
            "mayo": {"carb": 0.6, "protein": 1.0, "fat": 75.0, "calories": 680.0, "category": "condiment"},
            "white rice": {"carb": 28.17, "protein": 2.69, "fat": 0.28, "calories": 130.0, "category": "grain"},
            "brown rice": {"carb": 25.58, "protein": 2.74, "fat": 0.97, "calories": 123.0, "category": "grain"},
            "oats": {"carb": 66.27, "protein": 16.89, "fat": 6.9, "calories": 389.0, "category": "grain"},
            "cereal": {"carb": 75.0, "protein": 7.0, "fat": 2.5, "calories": 360.0, "category": "grain"},
            "white bread": {"carb": 49.42, "protein": 8.85, "fat": 3.33, "calories": 266.0, "category": "grain"},
            "whole wheat bread": {"carb": 41.0, "protein": 12.0, "fat": 3.5, "calories": 247.0, "category": "grain"},
            "burger bun": {"carb": 50.0, "protein": 9.0, "fat": 4.0, "calories": 270.0, "category": "grain"},
            "pasta": {"carb": 30.86, "protein": 5.8, "fat": 0.93, "calories": 158.0, "category": "grain"},
            "quinoa": {"carb": 21.3, "protein": 4.4, "fat": 1.9, "calories": 120.0, "category": "grain"},
            "dal": {"carb": 20.0, "protein": 9.0, "fat": 3.5, "calories": 147.0, "category": "legume"},
            "beans": {"carb": 22.8, "protein": 8.9, "fat": 0.5, "calories": 127.0, "category": "legume"},
            "peas": {"carb": 14.5, "protein": 5.4, "fat": 0.4, "calories": 81.0, "category": "legume"},
            "tofu": {"carb": 1.9, "protein": 8.1, "fat": 4.8, "calories": 76.0, "category": "legume"},
            "potato": {"carb": 20.13, "protein": 1.87, "fat": 0.1, "calories": 87.0, "category": "vegetable"},
            "sweet potato": {"carb": 20.1, "protein": 1.6, "fat": 0.1, "calories": 86.0, "category": "vegetable"},
            "carrot": {"carb": 9.6, "protein": 0.9, "fat": 0.2, "calories": 41.0, "category": "vegetable"},
            "onion": {"carb": 9.3, "protein": 1.1, "fat": 0.1, "calories": 40.0, "category": "vegetable"},
            "garlic": {"carb": 33.0, "protein": 6.4, "fat": 0.5, "calories": 149.0, "category": "vegetable"},
            "mushrooms": {"carb": 3.3, "protein": 3.1, "fat": 0.3, "calories": 22.0, "category": "vegetable"},
            "corn": {"carb": 19.0, "protein": 3.2, "fat": 1.2, "calories": 86.0, "category": "vegetable"},
            "spinach": {"carb": 3.63, "protein": 2.86, "fat": 0.39, "calories": 23.0, "category": "vegetable"},
            "tomato": {"carb": 3.89, "protein": 0.88, "fat": 0.2, "calories": 18.0, "category": "vegetable"},
            "lettuce": {"carb": 3.29, "protein": 1.36, "fat": 0.15, "calories": 15.0, "category": "vegetable"},
            "cucumber": {"carb": 3.63, "protein": 0.65, "fat": 0.11, "calories": 15.0, "category": "vegetable"},
            "broccoli": {"carb": 6.64, "protein": 2.82, "fat": 0.37, "calories": 34.0, "category": "vegetable"},
            "almonds": {"carb": 21.55, "protein": 21.15, "fat": 49.93, "calories": 579.0, "category": "nut"},
            "peanuts": {"carb": 16.13, "protein": 25.8, "fat": 49.24, "calories": 567.0, "category": "nut"},
            "banana": {"carb": 22.84, "protein": 1.09, "fat": 0.33, "calories": 89.0, "category": "fruit"},
            "apple": {"carb": 13.81, "protein": 0.26, "fat": 0.17, "calories": 52.0, "category": "fruit"},
            "orange": {"carb": 11.8, "protein": 0.9, "fat": 0.1, "calories": 47.0, "category": "fruit"},
            "mango": {"carb": 15.0, "protein": 0.8, "fat": 0.4, "calories": 60.0, "category": "fruit"},
            "grapes": {"carb": 18.1, "protein": 0.7, "fat": 0.2, "calories": 69.0, "category": "fruit"},
            "strawberries": {"carb": 7.7, "protein": 0.7, "fat": 0.3, "calories": 32.0, "category": "fruit"},
            "blueberries": {"carb": 14.5, "protein": 0.7, "fat": 0.3, "calories": 57.0, "category": "fruit"},
            "avocado": {"carb": 8.5, "protein": 2.0, "fat": 14.7, "calories": 160.0, "category": "fruit"},
            "watermelon": {"carb": 7.6, "protein": 0.6, "fat": 0.2, "calories": 30.0, "category": "fruit"},
            "pepperoni": {"carb": 1.18, "protein": 19.25, "fat": 44.0, "calories": 494.0, "category": "processed_meat"},
            "pizza crust": {"carb": 49.0, "protein": 7.9, "fat": 3.0, "calories": 260.0, "category": "grain"},
        }

        for name, data in ingredients.items():
            self.graph.add_node(name, node_type="ingredient", **data)

        dishes = {
            "chicken fried rice": [
                ("white rice", 0.60),
                ("chicken breast", 0.25),
                ("egg", 0.10),
                ("olive oil", 0.05),
            ],
            "pepperoni pizza": [
                ("pizza crust", 0.50),
                ("cheddar cheese", 0.25),
                ("pepperoni", 0.15),
                ("tomato", 0.10),
            ],
            "cheese pizza": [
                ("pizza crust", 0.55),
                ("cheddar cheese", 0.35),
                ("tomato", 0.10),
            ],
            "turkey sandwich": [
                ("white bread", 0.50),
                ("turkey breast", 0.35),
                ("cheddar cheese", 0.10),
                ("lettuce", 0.05),
            ],
            "chicken salad": [
                ("lettuce", 0.45),
                ("chicken breast", 0.40),
                ("tomato", 0.10),
                ("olive oil", 0.05),
            ],
            "dal rice": [
                ("white rice", 0.50),
                ("dal", 0.50),
            ],
            "scrambled eggs with toast": [
                ("egg", 0.55),
                ("white bread", 0.40),
                ("butter", 0.05),
            ],
            "oatmeal with milk": [
                ("oats", 0.40),
                ("milk", 0.60),
            ],
            "cereal with milk": [
                ("cereal", 0.40),
                ("milk", 0.60),
            ],
            "cheeseburger": [
                ("burger bun", 0.40),
                ("ground beef", 0.45),
                ("cheddar cheese", 0.15),
            ],
            "hamburger": [
                ("burger bun", 0.45),
                ("ground beef", 0.55),
            ],
        }

        for dish_name, components in dishes.items():
            self.graph.add_node(dish_name, node_type="composite_dish")
            for ing_name, weight in components:
                self.graph.add_edge(dish_name, ing_name, relation="CONTAINS", weight=weight)

        cooking_modifiers = {
            "fried": {"carb_delta": 0.0, "protein_delta": 0.0, "fat_delta": 6.0, "cal_delta": 54.0},
            "deep fried": {"carb_delta": 2.0, "protein_delta": 0.0, "fat_delta": 10.0, "cal_delta": 98.0},
            "grilled": {"carb_delta": 0.0, "protein_delta": 0.0, "fat_delta": -0.5, "cal_delta": -4.5},
            "scrambled": {"carb_delta": 0.0, "protein_delta": 0.0, "fat_delta": 3.0, "cal_delta": 27.0},
            "buttered": {"carb_delta": 0.0, "protein_delta": 0.0, "fat_delta": 8.0, "cal_delta": 72.0},
        }

        for method, deltas in cooking_modifiers.items():
            self.graph.add_node(method, node_type="cooking_modifier", **deltas)

        self.aliases = {
            "chicken": "chicken breast",
            "grilled chicken": "chicken breast",
            "chicken breast": "chicken breast",
            "chicken thigh": "chicken thigh",
            "rice": "white rice",
            "cooked rice": "white rice",
            "white rice": "white rice",
            "brown rice": "brown rice",
            "egg": "egg",
            "eggs": "egg",
            "boiled egg": "egg",
            "boiled eggs": "egg",
            "egg white": "egg white",
            "egg whites": "egg white",
            "milk": "milk",
            "whole milk": "milk",
            "beef": "ground beef",
            "ground beef": "ground beef",
            "steak": "beef steak",
            "pork": "pork",
            "bacon": "bacon",
            "ham": "ham",
            "sausage": "sausage",
            "sausages": "sausage",
            "potato": "potato",
            "potatoes": "potato",
            "boiled potato": "potato",
            "boiled potatoes": "potato",
            "sweet potato": "sweet potato",
            "sweet potatoes": "sweet potato",
            "carrot": "carrot",
            "carrots": "carrot",
            "onion": "onion",
            "onions": "onion",
            "garlic": "garlic",
            "mushrooms": "mushrooms",
            "mushroom": "mushrooms",
            "corn": "corn",
            "peas": "peas",
            "beans": "beans",
            "salmon": "salmon",
            "tuna": "tuna",
            "shrimp": "shrimp",
            "bread": "white bread",
            "toast": "white bread",
            "whole wheat bread": "whole wheat bread",
            "oats": "oats",
            "oatmeal": "oats",
            "cereal": "cereal",
            "almonds": "almonds",
            "almond": "almonds",
            "peanuts": "peanuts",
            "peanut": "peanuts",
            "apple": "apple",
            "apples": "apple",
            "banana": "banana",
            "bananas": "banana",
            "orange": "orange",
            "oranges": "orange",
            "mango": "mango",
            "mangoes": "mango",
            "grapes": "grapes",
            "grape": "grapes",
            "strawberries": "strawberries",
            "strawberry": "strawberries",
            "blueberries": "blueberries",
            "blueberry": "blueberries",
            "avocado": "avocado",
            "watermelon": "watermelon",
            "dal": "dal",
            "lentils": "dal",
            "paneer": "paneer",
            "tofu": "tofu",
            "cheese": "cheddar cheese",
            "cheddar": "cheddar cheese",
            "yogurt": "yogurt",
            "curd": "yogurt",
            "butter": "butter",
            "pasta": "pasta",
            "quinoa": "quinoa",
            "spinach": "spinach",
            "lettuce": "lettuce",
            "cucumber": "cucumber",
            "broccoli": "broccoli",
            "salad": "chicken salad",
            "chicken salad": "chicken salad",
            "turkey": "turkey breast",
            "pizza": "pepperoni pizza",
            "pepperoni pizza": "pepperoni pizza",
            "cheese pizza": "cheese pizza",
            "fried rice": "chicken fried rice",
            "chicken fried rice": "chicken fried rice",
            "turkey sandwich": "turkey sandwich",
            "sandwich": "turkey sandwich",
            "burger": "hamburger",
            "hamburger": "hamburger",
            "cheeseburger": "cheeseburger",
            "scrambled eggs": "egg",
        }

    def query_graph(self, meal_text):
        text = meal_text.lower()
        matched_dishes = []
        matched_ingredients = []
        matched_modifiers = []
        traversed_paths = []

        for dish in [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "composite_dish"]:
            if re.search(rf"\b{re.escape(dish)}\b", text):
                matched_dishes.append(dish)

        if not matched_dishes:
            for alias, target in self.aliases.items():
                if self.graph.nodes.get(target, {}).get("node_type") == "composite_dish":
                    if re.search(rf"\b{re.escape(alias)}\b", text):
                        matched_dishes.append(target)
                        break

        for mod in [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "cooking_modifier"]:
            if re.search(rf"\b{re.escape(mod)}\b", text):
                matched_modifiers.append(mod)

        if matched_dishes:
            dish = matched_dishes[0]
            carb, protein, fat, calories = 0.0, 0.0, 0.0, 0.0
            sub_paths = []

            for neighbor in self.graph.neighbors(dish):
                weight = self.graph[dish][neighbor].get("weight", 1.0)
                ing_data = self.graph.nodes[neighbor]
                carb += weight * ing_data["carb"]
                protein += weight * ing_data["protein"]
                fat += weight * ing_data["fat"]
                calories += weight * ing_data["calories"]
                sub_paths.append(f"{dish} -> CONTAINS -> {neighbor} (weight={weight})")

            traversed_paths.extend(sub_paths)

            for mod in matched_modifiers:
                mod_data = self.graph.nodes[mod]
                carb += mod_data["carb_delta"]
                protein += mod_data["protein_delta"]
                fat += mod_data["fat_delta"]
                calories += mod_data["cal_delta"]
                traversed_paths.append(f"{dish} -> PREPARED_BY -> {mod}")

            return {
                "carbohydrates": round(max(0.0, carb), 2),
                "protein": round(max(0.0, protein), 2),
                "fat": round(max(0.0, fat), 2),
                "calories": round(max(0.0, calories), 2),
                "nodes_traversed": [dish] + list(self.graph.neighbors(dish)) + matched_modifiers,
                "graph_paths": traversed_paths,
                "found": True,
                "message": "Food identified in knowledge graph.",
            }

        sorted_aliases = sorted(self.aliases.keys(), key=len, reverse=True)
        seen_targets = set()

        for alias in sorted_aliases:
            target = self.aliases[alias]
            if target in seen_targets:
                continue
            if self.graph.nodes.get(target, {}).get("node_type") == "ingredient":
                if re.search(rf"\b{re.escape(alias)}\b", text):
                    matched_ingredients.append(target)
                    seen_targets.add(target)

        if matched_ingredients:
            n_items = len(matched_ingredients)
            weight = 1.0 / n_items
            carb, protein, fat, calories = 0.0, 0.0, 0.0, 0.0

            for ing in matched_ingredients:
                ing_data = self.graph.nodes[ing]
                carb += weight * ing_data["carb"]
                protein += weight * ing_data["protein"]
                fat += weight * ing_data["fat"]
                calories += weight * ing_data["calories"]
                traversed_paths.append(f"Query -> MATCHED_INGREDIENT -> {ing} (weight={round(weight, 2)})")

            for mod in matched_modifiers:
                mod_data = self.graph.nodes[mod]
                carb += mod_data["carb_delta"]
                protein += mod_data["protein_delta"]
                fat += mod_data["fat_delta"]
                calories += mod_data["cal_delta"]
                traversed_paths.append(f"Ingredients -> PREPARED_BY -> {mod}")

            return {
                "carbohydrates": round(max(0.0, carb), 2),
                "protein": round(max(0.0, protein), 2),
                "fat": round(max(0.0, fat), 2),
                "calories": round(max(0.0, calories), 2),
                "nodes_traversed": matched_ingredients + matched_modifiers,
                "graph_paths": traversed_paths,
                "found": True,
                "message": "Food identified in knowledge graph.",
            }

        return {
            "carbohydrates": 0.0,
            "protein": 0.0,
            "fat": 0.0,
            "calories": 0.0,
            "nodes_traversed": [],
            "graph_paths": [],
            "found": False,
            "message": "Food item not found in the database. Please check your input.",
        }
