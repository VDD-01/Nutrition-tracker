import os
import sys
import re
import json
import warnings
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import logging as hf_logging
from peft import PeftModel

warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

BASE_DIR = Path(__file__).resolve().parent.parent
ADAPTER_DIR = str(BASE_DIR / "trained_models" / "llama_nutrition")
BASE_MODEL = "unsloth/Llama-3.2-3B"
OUTPUT_FILE = str(BASE_DIR / "evaluation_results_lora.json")
MAX_NEW_TOKENS = 100

BENCHMARK_DATA = [
    {"meal": "chicken breast, grilled", "carb": 0.0, "protein": 31.02, "fat": 3.57, "energy": 165.0},
    {"meal": "white rice, cooked", "carb": 28.17, "protein": 2.69, "fat": 0.28, "energy": 130.0},
    {"meal": "egg, whole, raw, fresh", "carb": 0.72, "protein": 12.56, "fat": 9.51, "energy": 143.0},
    {"meal": "milk, whole, 3.25% milkfat", "carb": 4.80, "protein": 3.15, "fat": 3.25, "energy": 61.0},
    {"meal": "bananas, raw", "carb": 22.84, "protein": 1.09, "fat": 0.33, "energy": 89.0},
    {"meal": "apples, raw, with skin", "carb": 13.81, "protein": 0.26, "fat": 0.17, "energy": 52.0},
    {"meal": "bread, white, commercially prepared", "carb": 49.42, "protein": 8.85, "fat": 3.33, "energy": 266.0},
    {"meal": "beef, ground, 80% lean meat / 20% fat, raw", "carb": 0.0, "protein": 17.17, "fat": 20.0, "energy": 254.0},
    {"meal": "potatoes, boiled, cooked in skin, flesh, without salt", "carb": 20.13, "protein": 1.87, "fat": 0.10, "energy": 87.0},
    {"meal": "salmon, Atlantic, wild, raw", "carb": 0.0, "protein": 19.84, "fat": 6.34, "energy": 142.0},
    {"meal": "oats, regular and quick, unenriched, dry", "carb": 66.27, "protein": 16.89, "fat": 6.90, "energy": 389.0},
    {"meal": "almonds, raw", "carb": 21.55, "protein": 21.15, "fat": 49.93, "energy": 579.0},
]


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
    base_model.generation_config.max_length = None

    print("Loading fine-tuned LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    model.generation_config.max_length = None

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = [tokenizer.eos_token_id]
    if eot_id and eot_id != tokenizer.eos_token_id:
        eos_ids.append(eot_id)

    print("Model ready for evaluation\n")
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
    ).strip()

    return parse_output(generated), generated


def compute_sample_accuracy(predicted, actual):
    nutrients = [("carb", "carbohydrates"), ("protein", "protein"), ("fat", "fat"), ("energy", "calories")]
    acc_scores = []
    abs_errors = {}

    for act_key, pred_key in nutrients:
        act_val = actual[act_key]
        pred_val = predicted[pred_key]
        if pred_val is None:
            pred_val = 0.0

        err = abs(pred_val - act_val)
        abs_errors[act_key] = err
        denom = max(act_val, 5.0)
        acc = max(0.0, 100.0 * (1.0 - (err / denom)))
        acc_scores.append(acc)

    return abs_errors, (sum(acc_scores) / len(acc_scores))


def main():
    print("=" * 80)
    print("LORA FINE-TUNED MODEL ACCURACY BENCHMARK")
    print(f"Adapter: {ADAPTER_DIR}")
    print(f"Test samples: {len(BENCHMARK_DATA)}")
    print("=" * 80 + "\n")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    if not Path(ADAPTER_DIR).exists():
        print(f"ERROR: Model adapter not found at {ADAPTER_DIR}")
        sys.exit(1)

    model, tokenizer, eos_ids = load_model()

    results = []
    total_mae = {"carb": 0.0, "protein": 0.0, "fat": 0.0, "energy": 0.0}
    total_acc = 0.0
    valid_predictions = 0

    print(f"{'Meal':<35} | {'Actual (C/P/F/Kcal)':<24} | {'Pred (C/P/F/Kcal)':<24} | {'Acc':<6}")
    print("-" * 96)

    for item in BENCHMARK_DATA:
        meal = item["meal"]
        pred, raw_text = predict(model, tokenizer, eos_ids, meal)

        act_str = f"{item['carb']:.1f}/{item['protein']:.1f}/{item['fat']:.1f}/{item['energy']:.0f}"

        if None in pred.values():
            pred_str = "PARSE ERROR"
            sample_acc = 0.0
            abs_err = {"carb": item['carb'], "protein": item['protein'], "fat": item['fat'], "energy": item['energy']}
        else:
            valid_predictions += 1
            pred_str = f"{pred['carbohydrates']:.1f}/{pred['protein']:.1f}/{pred['fat']:.1f}/{pred['calories']:.0f}"
            abs_err, sample_acc = compute_sample_accuracy(pred, item)
            for k in total_mae:
                total_mae[k] += abs_err[k]
            total_acc += sample_acc

        print(f"{meal:<35} | {act_str:<24} | {pred_str:<24} | {sample_acc:.1f}%")

        results.append({
            "meal": meal,
            "actual": item,
            "predicted": pred,
            "absolute_error": abs_err,
            "sample_accuracy_pct": round(sample_acc, 2),
            "raw_output": raw_text
        })

    n = len(BENCHMARK_DATA)
    count = max(valid_predictions, 1)

    mae_summary = {k: round(v / count, 2) for k, v in total_mae.items()}
    avg_accuracy = round(total_acc / count, 2)

    print("-" * 96)
    print("\n" + "=" * 80)
    print("ACCURACY SUMMARY")
    print("=" * 80)
    print(f"Total samples tested : {n}")
    print(f"Valid predictions    : {valid_predictions}/{n}")
    print(f"Mean Absolute Error  :")
    print(f"  Carbohydrates      : {mae_summary['carb']} g")
    print(f"  Protein            : {mae_summary['protein']} g")
    print(f"  Fat                : {mae_summary['fat']} g")
    print(f"  Calories           : {mae_summary['energy']} kcal")
    print(f"Overall Model Accuracy: {avg_accuracy}%")
    print("=" * 80)

    summary_data = {
        "samples_tested": n,
        "valid_predictions": valid_predictions,
        "mean_absolute_error": mae_summary,
        "overall_accuracy_percentage": avg_accuracy,
        "results": results
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nDetailed evaluation results saved to: {OUTPUT_FILE}\n")


if __name__ == "__main__":
    main()
