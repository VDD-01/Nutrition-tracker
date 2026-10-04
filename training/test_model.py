import os
import sys
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_DIR = Path(__file__).resolve().parent.parent
ADAPTER_DIR = str(BASE_DIR / "trained_models" / "llama_nutrition")
BASE_MODEL = "unsloth/Llama-3.2-3B"
MAX_NEW_TOKENS = 100

QUANTITY_MAP = [
    (r'\bhalf\s+a\s+cup\b',                                         0.5,   "half a cup",           "~120ml"),
    (r'\bquarter\s+a?\s*cup\b',                                     0.25,  "quarter cup",          "~60ml"),
    (r'\ba\s+cup\b|\bone\s+cup\b',                                  1.0,   "1 cup",                "~240ml"),
    (r'\btwo\s+cups?\b',                                            2.0,   "2 cups",               "~480ml"),
    (r'\bthree\s+cups?\b',                                          3.0,   "3 cups",               "~720ml"),
    (r'\ba\s+glass\b|\bone\s+glass\b',                              1.0,   "1 glass",              "~240ml"),

    (r'\bsmall\s+(?:sized\s+)?bowl\b',                             1.5,   "small bowl",           "~150g"),
    (r'\bmedium\s+(?:sized\s+)?bowl\b',                            2.5,   "medium bowl",          "~250g"),
    (r'\blarge\s+(?:sized\s+)?bowl\b|\bbig\s+bowl\b',             3.5,   "large bowl",           "~350g"),
    (r'\ba\s+bowl\b|\bone\s+bowl\b',                               2.0,   "bowl",                 "~200g"),

    (r'\bsmall\s+(?:sized\s+)?plate\b',                            2.0,   "small plate",          "~200g"),
    (r'\bmedium\s+(?:sized\s+)?plate\b',                           3.0,   "medium plate",         "~300g"),
    (r'\blarge\s+(?:sized\s+)?plate\b|\bbig\s+plate\b',           4.0,   "large plate",          "~400g"),
    (r'\ba\s+plate\b|\bone\s+plate\b',                             2.5,   "plate",                "~250g"),

    (r'\bsmall\s+(?:sized\s+)?portion\b|\bsmall\s+serving\b',     0.75,  "small portion",        "~75g"),
    (r'\bmedium\s+(?:sized\s+)?portion\b|\bmedium\s+serving\b',   1.0,   "medium portion",       "~100g"),
    (r'\blarge\s+(?:sized\s+)?portion\b|\blarge\s+serving\b',     1.5,   "large portion",        "~150g"),

    (r'\bhandful\b',                                               0.3,   "handful",              "~30g"),
    (r'\bsmall\s+handful\b',                                       0.2,   "small handful",        "~20g"),
    (r'\blarge\s+handful\b',                                       0.4,   "large handful",        "~40g"),

    (r'\bslice\b|\bpiece\b',                                       0.8,   "1 slice/piece",        "~80g"),
    (r'\b(\d+)\s+slices?\b|\b(\d+)\s+pieces?\b',                  None,  None,                   "N slices"),

    (r'\b(\d+(?:\.\d+)?)\s+cups?\b',                              None,  None,                   "cup x N"),
    (r'\b(\d+(?:\.\d+)?)\s*(?:tbsp|tablespoons?)\b',              None,  None,                   "tbsp x N"),
    (r'\b(\d+(?:\.\d+)?)\s*(?:tsp|teaspoons?)\b',                 None,  None,                   "tsp x N"),
    (r'\b(\d+(?:\.\d+)?)\s*(?:g|grams?)\b',                       None,  None,                   "g"),
    (r'\b(\d+(?:\.\d+)?)\s*(?:ml|millilitres?|milliliters?)\b',   None,  None,                   "ml"),
    (r'\b(\d+(?:\.\d+)?)\s*(?:oz|ounces?)\b',                     None,  None,                   "oz"),

    (r'\bhalf\b',                                                  0.5,   "half",                 "0.5x serving"),
    (r'\bquarter\b',                                               0.25,  "quarter",              "0.25x serving"),
    (r'\bdouble\b',                                                2.0,   "double",               "2x serving"),
    (r'\b(\d+)\b',                                                 None,  None,                   "N servings"),
]

UNIT_TO_SCALE = {
    'cup': 240.0 / 100.0,
    'tbsp': 15.0 / 100.0,
    'tsp': 5.0 / 100.0,
    'g': 1.0 / 100.0,
    'ml': 1.0 / 100.0,
    'oz': 28.35 / 100.0,
}


def extract_quantity(text):
    t = text.lower()

    for pattern, fixed_scale, label, unit_desc in QUANTITY_MAP:
        match = re.search(pattern, t)
        if not match:
            continue

        if fixed_scale is not None:
            return {
                "detected": match.group(0).strip(),
                "label": label,
                "unit_desc": unit_desc,
                "scale": fixed_scale,
                "ref_amount": f"model reference x {fixed_scale}"
            }

        numeric = float(match.group(1))

        for unit_key, scale_per_unit in UNIT_TO_SCALE.items():
            if unit_key in pattern:
                scale = round(numeric * scale_per_unit, 4)
                return {
                    "detected": match.group(0).strip(),
                    "label": f"{numeric} {unit_key}",
                    "unit_desc": unit_desc,
                    "scale": scale,
                    "ref_amount": f"{numeric * (100 * scale_per_unit):.0f}g / 100g reference"
                }

        return {
            "detected": match.group(0).strip(),
            "label": f"x{numeric}",
            "unit_desc": unit_desc,
            "scale": numeric,
            "ref_amount": f"model reference x {numeric}"
        }

    return {
        "detected": "none",
        "label": "1 serving",
        "unit_desc": "model reference amount (approx 100g)",
        "scale": 1.0,
        "ref_amount": "model reference x 1.0"
    }


def load_model():
    print("Loading base model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, clean_up_tokenization_spaces=False)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading fine-tuned LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = [tokenizer.eos_token_id]
    if eot_id and eot_id != tokenizer.eos_token_id:
        eos_ids.append(eot_id)

    print("Model ready\n")
    return model, tokenizer, eos_ids


def build_prompt(meal_description):
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a nutrition expert AI. Given a meal description, estimate the nutritional values accurately.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Meal: {meal_description}\n\n"
        "Estimate the nutritional values for this meal.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def truncate_after_last_nutrition(text):
    cal_match = re.search(r'Calories?:\s*\d+\.?\d*\s*kcal?', text, re.IGNORECASE)
    if cal_match:
        return text[:cal_match.end()].strip()
    return text.strip()


def clean_output(text):
    text = re.sub(r'<\|.*?\|>', '', text)
    text = re.sub(r'\bassistant\b', '', text, flags=re.IGNORECASE)
    text = truncate_after_last_nutrition(text)
    return text.strip()


def parse_output(text):
    carb = re.search(r'Carbohydrates?:\s*(\d+\.?\d*)', text, re.IGNORECASE)
    protein = re.search(r'Protein:\s*(\d+\.?\d*)', text, re.IGNORECASE)
    fat = re.search(r'Fat:\s*(\d+\.?\d*)', text, re.IGNORECASE)
    cal = re.search(r'Calories?:\s*(\d+\.?\d*)', text, re.IGNORECASE)
    return {
        'carbohydrates': float(carb.group(1)) if carb else None,
        'protein': float(protein.group(1)) if protein else None,
        'fat': float(fat.group(1)) if fat else None,
        'calories': float(cal.group(1)) if cal else None,
    }


def predict(model, tokenizer, eos_ids, meal_description):
    qty = extract_quantity(meal_description)
    prompt = build_prompt(meal_description)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=eos_ids,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    generated = clean_output(generated)

    base_nutrients = parse_output(generated)

    scaled_nutrients = None
    if qty["scale"] != 1.0 and all(v is not None for v in base_nutrients.values()):
        scaled_nutrients = {k: round(v * qty["scale"], 2) for k, v in base_nutrients.items()}

    return generated, base_nutrients, scaled_nutrients, qty


def main():
    print("=" * 60)
    print("FINE-TUNED NUTRITION MODEL - TEST")
    print(f"Adapter: {ADAPTER_DIR}")
    print("=" * 60 + "\n")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    if not Path(ADAPTER_DIR).exists():
        print(f"ERROR: No trained model found at {ADAPTER_DIR}")
        print("Run training/train.py first.")
        sys.exit(1)

    model, tokenizer, eos_ids = load_model()

    print("Type a meal description and press Enter.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            meal = input("Meal: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not meal:
            continue
        if meal.lower() in ("quit", "exit", "q"):
            break

        raw, base, scaled, qty = predict(model, tokenizer, eos_ids, meal)

        print("\n--- Fuzzy Interpretation ---")
        print(f"  Quantity detected : {qty['detected']}")
        print(f"  Interpreted as    : {qty['label']} ({qty['unit_desc']})")
        print(f"  Scale applied     : x{qty['scale']}")
        print(f"  Reference amount  : {qty['ref_amount']}")

        print("\n--- Model raw output (per reference amount) ---")
        print(raw)

        if base and all(v is not None for v in base.values()):
            if scaled and qty["scale"] != 1.0:
                print(f"\n--- Scaled result (x{qty['scale']}) ---")
                print(f"  Carbs:    {scaled['carbohydrates']:.1f}g")
                print(f"  Protein:  {scaled['protein']:.1f}g")
                print(f"  Fat:      {scaled['fat']:.1f}g")
                print(f"  Calories: {scaled['calories']:.0f}kcal")
            else:
                print("\n--- Result (no scaling applied) ---")
                print(f"  Carbs:    {base['carbohydrates']:.1f}g")
                print(f"  Protein:  {base['protein']:.1f}g")
                print(f"  Fat:      {base['fat']:.1f}g")
                print(f"  Calories: {base['calories']:.0f}kcal")
        else:
            print("\n(Could not parse all values from output)")

        print()


if __name__ == "__main__":
    main()
