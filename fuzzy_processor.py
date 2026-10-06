import re


class FuzzyQuantityProcessor:
    """
    Hybrid quantity module:
      - Crisp rules for measured amounts (250g, 2 cups, ...)
      - Mamdani fuzzy inference for linguistic portions (small bowl, handful, ...)
    Does not change model outputs; only returns a scale factor.
    """

    UNIT_CONVERSIONS = {
        "g": 1.0, "gram": 1.0, "grams": 1.0,
        "kg": 1000.0, "kilogram": 1000.0, "kilograms": 1000.0,
        "oz": 28.35, "ounce": 28.35, "ounces": 28.35,
        "lb": 453.59, "pound": 453.59, "pounds": 453.59,
        "ml": 1.0, "milliliter": 1.0, "milliliters": 1.0,
        "l": 1000.0, "liter": 1000.0, "liters": 1000.0,
        "cup": 240.0, "cups": 240.0,
        "tbsp": 15.0, "tablespoon": 15.0, "tablespoons": 15.0,
        "tsp": 5.0, "teaspoon": 5.0, "teaspoons": 5.0,
        "glass": 240.0, "glasses": 240.0,
    }

    VOLUME_UNITS_ML = {
        "ml": 1.0, "milliliter": 1.0, "milliliters": 1.0,
        "l": 1000.0, "liter": 1000.0, "liters": 1000.0,
        "cup": 240.0, "cups": 240.0,
        "tbsp": 15.0, "tablespoon": 15.0, "tablespoons": 15.0,
        "tsp": 5.0, "teaspoon": 5.0, "teaspoons": 5.0,
        "glass": 240.0, "glasses": 240.0,
    }

    FOOD_DENSITY_CATEGORIES = {
        "cooked_grains": {
            "density_g_per_ml": 0.75,
            "keywords": (
                "rice", "pulao", "biryani", "poha", "upma", "quinoa",
                "oats", "porridge", "khichdi",
            ),
        },
        "leafy_salad": {
            "density_g_per_ml": 0.25,
            "keywords": (
                "salad", "lettuce", "spinach", "greens", "cabbage",
                "kale", "rocket", "arugula",
            ),
        },
        "liquid": {
            "density_g_per_ml": 1.0,
            "keywords": (
                "water", "milk", "juice", "tea", "coffee", "lassi",
                "smoothie", "soup", "broth", "dal", "sambar", "rasam",
            ),
        },
        "nuts_seeds": {
            "density_g_per_ml": 0.55,
            "keywords": (
                "nuts", "almond", "almonds", "cashew", "cashews",
                "peanut", "peanuts", "walnut", "walnuts", "seed", "seeds",
            ),
        },
        "cut_fruit": {
            "density_g_per_ml": 0.65,
            "keywords": (
                "fruit", "apple", "banana", "mango", "grapes", "berries",
                "orange", "papaya", "melon",
            ),
        },
        "cooked_vegetables": {
            "density_g_per_ml": 0.55,
            "keywords": (
                "vegetable", "vegetables", "sabzi", "curry", "potato",
                "beans", "peas", "carrot", "cauliflower", "broccoli",
            ),
        },
        "dense_protein": {
            "density_g_per_ml": 0.85,
            "keywords": (
                "paneer", "tofu", "chicken", "fish", "egg", "eggs",
                "meat", "mutton", "beef", "pork",
            ),
        },
    }

    WORD_NUMBERS = {
        "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "half": 0.5, "quarter": 0.25, "double": 2.0, "triple": 3.0,
    }

    OUTPUT_SETS = {
        "tiny": (0.0, 30.0, 60.0),
        "small": (40.0, 100.0, 160.0),
        "medium": (120.0, 200.0, 280.0),
        "large": (220.0, 350.0, 500.0),
        "xlarge": (350.0, 450.0, 500.0),
    }

    # Mamdani rules: (size_term or None, container_term, output_set, extra_fire)
    # None size = "any / unspecified size"
    FUZZY_RULES = [
        ("small", "handful", "tiny"),
        (None, "handful", "tiny"),
        ("large", "handful", "small"),
        ("small", "bowl", "small"),
        ("medium", "bowl", "medium"),
        ("large", "bowl", "large"),
        (None, "bowl", "medium"),
        ("small", "plate", "medium"),
        ("medium", "plate", "large"),
        ("large", "plate", "xlarge"),
        (None, "plate", "large"),
        ("small", "portion", "small"),
        ("medium", "portion", "medium"),
        ("large", "portion", "large"),
        (None, "portion", "medium"),
        ("small", "serving", "small"),
        ("medium", "serving", "medium"),
        ("large", "serving", "large"),
        (None, "cup", "medium"),
        ("small", "cup", "small"),
        ("large", "cup", "large"),
        (None, "glass", "medium"),
        (None, "slice", "small"),
        (None, "piece", "small"),
    ]

    CONTAINER_PATTERNS = {
        "handful": r"\bhandfuls?\b",
        "bowl": r"\bbowls?\b",
        "plate": r"\bplates?\b",
        "portion": r"\bportions?\b",
        "serving": r"\bservings?\b",
        "cup": r"\bcups?\b",
        "glass": r"\bglasses?\b",
        "slice": r"\bslices?\b",
        "piece": r"\bpieces?\b",
    }

    def extract_quantity_info(self, text):
        text_lower = text.lower()

        # --- Crisp path: measured qty + unit (unchanged behaviour) ---
        unit_alt = "|".join(self.UNIT_CONVERSIONS.keys())
        word_alt = "|".join(self.WORD_NUMBERS.keys())
        pattern = rf"(\d+(?:\.\d+)?|\b(?:{word_alt})\b)\s*({unit_alt})\b"
        match = re.search(pattern, text_lower)
        if match:
            qty_str, unit = match.group(1), match.group(2)
            qty = self.WORD_NUMBERS.get(qty_str, None)
            if qty is None:
                qty = float(qty_str)
            density_info = self._detect_food_density(text_lower)
            if unit in self.VOLUME_UNITS_ML:
                neutral_grams = qty * self.UNIT_CONVERSIONS[unit]
                grams = neutral_grams * density_info["density_g_per_ml"]
            else:
                neutral_grams = None
                grams = qty * self.UNIT_CONVERSIONS[unit]
            return {
                "quantity": qty,
                "unit": unit,
                "grams": grams,
                "scale": grams / 100.0,
                "method": "crisp",
                "food_category": density_info["category"],
                "density_g_per_ml": density_info["density_g_per_ml"] if unit in self.VOLUME_UNITS_ML else None,
                "matched_food_keyword": density_info["matched_keyword"],
                "base_grams_before_density": neutral_grams,
                "memberships": {},
            }

        # --- Fuzzy path: linguistic portion ---
        size_mu = self._fuzzify_size(text_lower)
        container_mu = self._fuzzify_container(text_lower)
        multiplier = self._crisp_multiplier(text_lower)
        has_container = any(v > 0 for v in container_mu.values())
        has_size = max(size_mu.values()) > 0
        if has_size and not has_container:
            container_mu["portion"] = 1.0

        if has_container or has_size:
            base_grams, aggregated = self._mamdani(size_mu, container_mu)
            density_info = self._detect_food_density(text_lower)
            grams = base_grams * density_info["density_g_per_ml"]
            grams *= multiplier
            return {
                "quantity": round(grams, 2),
                "unit": "grams_fuzzy",
                "grams": round(grams, 2),
                "scale": round(grams / 100.0, 4),
                "method": "mamdani",
                "food_category": density_info["category"],
                "density_g_per_ml": density_info["density_g_per_ml"],
                "matched_food_keyword": density_info["matched_keyword"],
                "base_grams_before_density": round(base_grams, 2),
                "memberships": {
                    "size": size_mu,
                    "container": container_mu,
                    "output": aggregated,
                },
                "multiplier": multiplier,
            }

        for word, val in self.WORD_NUMBERS.items():
            if word in ("a", "an"):
                continue
            if re.search(rf"\b{word}\b", text_lower):
                return {
                    "quantity": val,
                    "unit": "serving",
                    "grams": None,
                    "scale": val,
                    "method": "crisp_serving",
                    "memberships": {},
                }

        return {
            "quantity": 1.0,
            "unit": "serving",
            "grams": None,
            "scale": 1.0,
            "method": "default",
            "memberships": {},
        }

    def scale_nutrition(self, nutrition, scale):
        """Multiply numeric model outputs. Leaves source/raw_output/rag_matches intact."""
        if scale == 1.0:
            return nutrition
        skip_keys = {"raw_output", "source", "rag_matches", "quantity_info"}
        scaled = {}
        nutrient_keys = {"carbohydrates", "protein", "fat", "calories"}
        for k, v in nutrition.items():
            if k in skip_keys or k not in nutrient_keys:
                scaled[k] = v
                continue
            try:
                scaled[k] = round(float(v) * scale, 2)
            except (TypeError, ValueError):
                scaled[k] = v
        return scaled

    # ----- fuzzification -----

    @staticmethod
    def triangle(x, a, b, c):
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if x < b:
            return (x - a) / (b - a) if b != a else 1.0
        return (c - x) / (c - b) if c != b else 1.0

    def _fuzzify_size(self, text):
        """
        Overlapping linguistic size. Detected word gets 1.0; neighbours get a small overlap
        so this is actual fuzzy sets, not a one-hot label.
        """
        mu = {"small": 0.0, "medium": 0.0, "large": 0.0}
        if re.search(r"\b(small|tiny|little)\b", text):
            mu.update({"small": 1.0, "medium": 0.25, "large": 0.0})
        elif re.search(r"\b(medium|regular|normal|average)\b", text):
            mu.update({"small": 0.25, "medium": 1.0, "large": 0.25})
        elif re.search(r"\b(large|big|huge|generous)\b", text):
            mu.update({"small": 0.0, "medium": 0.25, "large": 1.0})
        return mu

    def _fuzzify_container(self, text):
        mu = {name: 0.0 for name in self.CONTAINER_PATTERNS}
        for name, pat in self.CONTAINER_PATTERNS.items():
            if re.search(pat, text):
                mu[name] = 1.0
        # mild overlap: bowl ~ plate as serving vessels
        if mu["bowl"] == 1.0:
            mu["plate"] = max(mu["plate"], 0.15)
        if mu["plate"] == 1.0:
            mu["bowl"] = max(mu["bowl"], 0.15)
        return mu

    def _crisp_multiplier(self, text):
        if re.search(r"\bhalf\b", text):
            return 0.5
        if re.search(r"\bquarter\b", text):
            return 0.25
        if re.search(r"\bdouble\b", text):
            return 2.0
        if re.search(r"\btriple\b", text):
            return 3.0
        return 1.0

    def _detect_food_density(self, text):
        for category, config in self.FOOD_DENSITY_CATEGORIES.items():
            for keyword in config["keywords"]:
                if re.search(rf"\b{re.escape(keyword)}\b", text):
                    return {
                        "category": category,
                        "density_g_per_ml": config["density_g_per_ml"],
                        "matched_keyword": keyword,
                    }

        return {
            "category": "generic",
            "density_g_per_ml": 1.0,
            "matched_keyword": None,
        }

    # ----- Mamdani inference + centroid defuzzification -----

    def _mamdani(self, size_mu, container_mu, step=1.0):
        universe = [i * step for i in range(0, int(500 / step) + 1)]
        aggregated = [0.0] * len(universe)

        for size_term, container, out_set in self.FUZZY_RULES:
            if size_term is None:
                # container-only rule: fire with container membership
                fire = container_mu.get(container, 0.0)
            else:
                fire = min(size_mu.get(size_term, 0.0), container_mu.get(container, 0.0))
            if fire <= 0:
                continue

            a, b, c = self.OUTPUT_SETS[out_set]
            for i, x in enumerate(universe):
                clipped = min(self.triangle(x, a, b, c), fire)
                aggregated[i] = max(aggregated[i], clipped)  # max aggregation

        num = sum(x * mu for x, mu in zip(universe, aggregated))
        den = sum(aggregated)
        if den == 0:
            grams = 100.0
        else:
            grams = num / den  # centroid

        # compact membership snapshot for viva / debugging (every 50g)
        snapshot = {str(int(x)): round(aggregated[int(x / step)], 3)
                    for x in range(0, 501, 50)}
        return grams, snapshot
