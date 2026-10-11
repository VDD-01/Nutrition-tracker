import os
import sys
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fuzzy_processor import FuzzyQuantityProcessor

ADAPTER_DIR = str(BASE_DIR / "trained_models" / "llama_nutrition")
BASE_MODEL = "unsloth/Llama-3.2-3B"
MAX_NEW_TOKENS = 100


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
    return generated, base_nutrients


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

    fuzzy = FuzzyQuantityProcessor()
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

        corrected_meal, corrections = fuzzy.correct_spelling(meal)

        if corrections:
            print("\n--- Typo Detected ---")
            print(f"  Original input  : {meal}")
            print(f"  Interpreted as  : {corrected_meal}")
            print("  Corrections     : " + ", ".join(f"{k} -> {v}" for k, v in corrections.items()))

        qty_info = fuzzy.extract_quantity_info(corrected_meal)

        print("\n--- Fuzzy Interpretation ---")
        print(f"  Method          : {qty_info.get('method', 'default')}")
        if qty_info.get("grams") is not None:
            print(f"  Estimated mass  : {qty_info['grams']}g")
        if qty_info.get("food_category"):
            print(f"  Category        : {qty_info['food_category']}")
        print(f"  Scale applied   : x{qty_info['scale']}")

        raw, base = predict(model, tokenizer, eos_ids, corrected_meal)

        print("\n--- Model raw output (per 100g reference) ---")
        print(raw)

        if base and all(v is not None for v in base.values()):
            if qty_info["scale"] != 1.0:
                scaled = fuzzy.scale_nutrition(base, qty_info["scale"])
                print(f"\n--- Scaled result (x{qty_info['scale']}) ---")
                print(f"  Carbs:    {scaled['carbohydrates']:.1f}g")
                print(f"  Protein:  {scaled['protein']:.1f}g")
                print(f"  Fat:      {scaled['fat']:.1f}g")
                print(f"  Calories: {scaled['calories']:.0f}kcal")
            else:
                print("\n--- Result (1.0x serving) ---")
                print(f"  Carbs:    {base['carbohydrates']:.1f}g")
                print(f"  Protein:  {base['protein']:.1f}g")
                print(f"  Fat:      {base['fat']:.1f}g")
                print(f"  Calories: {base['calories']:.0f}kcal")
        else:
            print("\n(Could not parse all values from output)")

        print()


if __name__ == "__main__":
    main()
