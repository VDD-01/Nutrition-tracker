import os
import sys
import json
import torch
import pandas as pd
import datasets as hf_datasets
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = str(BASE_DIR / "trained_models" / "llama_nutrition")
MODEL_NAME = "unsloth/Llama-3.2-3B"

BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_LENGTH = 128

hf_datasets.disable_caching()

print("=" * 80)
print("LLAMA NUTRITION FINE-TUNING")
print("=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Data directory: {DATA_DIR}")
print(f"Output: {OUTPUT_DIR}")
print("Strategy: QLoRA 4-bit")
print("=" * 80)


def parse_fdc_item(item):
    if not isinstance(item, dict):
        return None
    desc = item.get('description', '').strip()
    if not desc:
        return None

    nutrients = {'carb': None, 'protein': None, 'fat': None, 'energy': None}

    for fn in item.get('foodNutrients', []):
        n_info = fn.get('nutrient', {})
        n_name = n_info.get('name', '').lower()
        n_unit = n_info.get('unitName', '').lower()
        n_num = str(n_info.get('number', ''))
        amt = fn.get('amount', None)

        if amt is None:
            continue

        if n_num == '205' or 'carbohydrate, by difference' in n_name:
            nutrients['carb'] = float(amt)
        elif n_num == '203' or n_name == 'protein':
            nutrients['protein'] = float(amt)
        elif n_num == '204' or 'total lipid (fat)' in n_name:
            nutrients['fat'] = float(amt)
        elif (n_num == '208' or n_name == 'energy') and ('kcal' in n_unit or n_unit == 'kcal'):
            nutrients['energy'] = float(amt)

    if None in nutrients.values():
        return None

    return {
        'meal_description': desc,
        'carb': nutrients['carb'],
        'protein': nutrients['protein'],
        'fat': nutrients['fat'],
        'energy': nutrients['energy']
    }


def parse_generic_item(item):
    if not isinstance(item, dict):
        return None
    desc_key = next((k for k in ['meal_description', 'description', 'text', 'food', 'name'] if k in item), None)
    if not desc_key or not item[desc_key]:
        return None

    def get_num(keys):
        for k in keys:
            if k in item and item[k] is not None:
                try:
                    return float(item[k])
                except (ValueError, TypeError):
                    pass
        return None

    carb = get_num(['carb', 'carbs', 'carbohydrates', 'carbohydrate'])
    protein = get_num(['protein', 'prot'])
    fat = get_num(['fat', 'total_fat', 'fats'])
    energy = get_num(['energy', 'calories', 'calorie', 'kcal'])

    if None in (carb, protein, fat, energy):
        return None

    return {
        'meal_description': str(item[desc_key]).strip(),
        'carb': carb,
        'protein': protein,
        'fat': fat,
        'energy': energy
    }


def load_dataset_records(data_dir=DATA_DIR):
    records = []
    files = list(data_dir.rglob('*.json')) + list(data_dir.rglob('*.csv')) + list(data_dir.rglob('*.parquet'))
    print(f"\nScanning data files in {data_dir}...")
    print(f"Found {len(files)} data file(s)")

    for fpath in files:
        suffix = fpath.suffix.lower()
        print(f"Loading {fpath.name}...")
        if suffix == '.json':
            try:
                with open(fpath, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
                continue

            if isinstance(data, dict):
                fdc_matched = False
                for key in ['SRLegacyFoods', 'SurveyFoods', 'FoundationFoods', 'BrandedFoods']:
                    if key in data and isinstance(data[key], list):
                        fdc_matched = True
                        for item in data[key]:
                            rec = parse_fdc_item(item)
                            if rec:
                                records.append(rec)
                        break
                if not fdc_matched:
                    for k, v in data.items():
                        if isinstance(v, list):
                            for item in v:
                                rec = parse_generic_item(item) or parse_fdc_item(item)
                                if rec:
                                    records.append(rec)
            elif isinstance(data, list):
                for item in data:
                    rec = parse_generic_item(item) or parse_fdc_item(item)
                    if rec:
                        records.append(rec)
        elif suffix == '.parquet':
            df = pd.read_parquet(fpath)
            for _, row in df.iterrows():
                rec = parse_generic_item(row.to_dict())
                if rec:
                    records.append(rec)
        elif suffix == '.csv':
            df = pd.read_csv(fpath)
            for _, row in df.iterrows():
                rec = parse_generic_item(row.to_dict())
                if rec:
                    records.append(rec)

    df_out = pd.DataFrame(records)
    print(f"Successfully loaded and validated {len(df_out)} training examples")
    return df_out


def prepare_nutrition_data():
    df = load_dataset_records(DATA_DIR)
    if df.empty:
        raise ValueError(f"No valid data found in {DATA_DIR}")

    train_size = int(0.9 * len(df))
    train_df = df[:train_size]
    val_df = df[train_size:]
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}")

    def format_nutrition_prompt(row):
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a nutrition expert AI. Given a meal description, estimate the nutritional values accurately.<|eot_id|><|start_header_id|>user<|end_header_id|>

Meal: {row['meal_description']}

Estimate the nutritional values for this meal.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Carbohydrates: {row['carb']:.2f}g
Protein: {row['protein']:.2f}g
Fat: {row['fat']:.2f}g
Calories: {row['energy']:.2f}kcal<|eot_id|>"""
        return prompt

    print("\nFormatting data...")
    train_texts = [format_nutrition_prompt(row) for _, row in tqdm(train_df.iterrows(), total=len(train_df))]
    val_texts = [format_nutrition_prompt(row) for _, row in tqdm(val_df.iterrows(), total=len(val_df))]

    train_dataset = Dataset.from_dict({"text": train_texts})
    val_dataset = Dataset.from_dict({"text": val_texts})

    return train_dataset, val_dataset


def setup_model_and_tokenizer():
    print("\nLoading model with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print("Model loaded successfully")
    print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total params: {total_params:,}")

    return model, tokenizer


def train_model(model, tokenizer, train_dataset, val_dataset):
    print("\nStarting training...")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION}")

    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        outputs["labels"] = [ids.copy() for ids in outputs["input_ids"]]
        return outputs

    print("\nTokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"], keep_in_memory=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"], keep_in_memory=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        save_total_limit=2,
        eval_strategy="steps",
        load_best_model_at_end=True,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("\n" + "=" * 80)
    print("TRAINING STARTED")
    print("=" * 80)
    print("Training will take 2-3 hours")
    print("=" * 80 + "\n")

    trainer.train()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print("\nSaving model...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to: {OUTPUT_DIR}")

    return trainer


def quick_test(model, tokenizer):
    print("\nRunning quick test...")

    test_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a nutrition expert AI. Given a meal description, estimate the nutritional values accurately.<|eot_id|><|start_header_id|>user<|end_header_id|>

Meal: I ate a small bowl of rice with grilled chicken breast

Estimate the nutritional values for this meal.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nTest Input: 'I ate a small bowl of rice with grilled chicken breast'")
    print("\nModel Output:")
    print(result.split("assistant")[-1].strip())


def main():
    print("\nTraining initialization\n")

    train_dataset, val_dataset = prepare_nutrition_data()
    model, tokenizer = setup_model_and_tokenizer()
    trainer = train_model(model, tokenizer, train_dataset, val_dataset)
    quick_test(model, tokenizer)

    metadata = {
        "model": MODEL_NAME,
        "dataset_directory": str(DATA_DIR),
        "training_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "status": "completed"
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("SUCCESS - MODEL READY")
    print("=" * 80)
    print(f"\nTrained model saved at: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Run evaluation: python evaluation/evaluate.py")
    print("  2. Run inference: python inference/predict.py")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        print("Install CUDA-enabled PyTorch")
        exit(1)

    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")

    main()
