import torch
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


class StandaloneNutritionPredictor:
    def __init__(self, adapter_dir=None):
        if adapter_dir is None:
            base_dir = Path(__file__).resolve().parent
            adapter_dir = str(base_dir / "trained_models" / "llama_nutrition")

        self.adapter_dir = adapter_dir
        self.base_model_name = "unsloth/Llama-3.2-3B"
        self.max_new_tokens = 100

        self.model = None
        self.tokenizer = None
        self.eos_ids = None

        self._load_model()

    def _load_model(self):
        print("Loading nutrition model (this may take a moment)...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_dir,
            clean_up_tokenization_spaces=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model = PeftModel.from_pretrained(base_model, self.adapter_dir)
        self.model.eval()

        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        self.eos_ids = [self.tokenizer.eos_token_id]
        if eot_id and eot_id != self.tokenizer.eos_token_id:
            self.eos_ids.append(eot_id)

        print("Model loaded successfully!\n")

    def _build_prompt(self, meal_description):
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a nutrition expert AI. Given a meal description, estimate the nutritional values accurately.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"Meal: {meal_description}\n\n"
            "Estimate the nutritional values for this meal.<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def _clean_output(self, text):
        text = re.sub(r'<\|.*?\|>', '', text)
        text = re.sub(r'\bassistant\b', '', text, flags=re.IGNORECASE)

        cal_match = re.search(r'Calories?:\s*\d+\.?\d*\s*kcal?', text, re.IGNORECASE)
        if cal_match:
            text = text[:cal_match.end()].strip()

        return text.strip()

    def _parse_output(self, text):
        carb = re.search(r'Carbohydrates?:\s*(\d+\.?\d*)', text, re.IGNORECASE)
        protein = re.search(r'Protein:\s*(\d+\.?\d*)', text, re.IGNORECASE)
        fat = re.search(r'Fat:\s*(\d+\.?\d*)', text, re.IGNORECASE)
        cal = re.search(r'Calories?:\s*(\d+\.?\d*)', text, re.IGNORECASE)

        return {
            'carbohydrates': float(carb.group(1)) if carb else 0.0,
            'protein': float(protein.group(1)) if protein else 0.0,
            'fat': float(fat.group(1)) if fat else 0.0,
            'calories': float(cal.group(1)) if cal else 0.0,
            'raw_output': text
        }

    def predict(self, meal_description):
        prompt = self._build_prompt(meal_description)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.eos_ids,
            )

        generated = self.tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        cleaned = self._clean_output(generated)
        nutrition = self._parse_output(cleaned)

        return nutrition
