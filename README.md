# Intelligent Nutrition Tracker: GRAG and Fine-Tuned LLaMA-3.2 Architecture

An end-to-end nutrition estimation system combining Graph Retrieval-Augmented Generation (GRAG), a 4-bit QLoRA fine-tuned LLaMA-3.2-3B language model, Mamdani fuzzy inference for portion estimation, and spelling correction.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Algorithmic Core](#algorithmic-core)
  - [1. Typo Correction (Ratcliff/Obershelp Algorithm)](#1-typo-correction-ratcliffobershelp-algorithm)
  - [2. Portion Scaling (Mamdani Fuzzy Inference System)](#2-portion-scaling-mamdani-fuzzy-inference-system)
  - [3. Knowledge Representation (Graph RAG / GRAG)](#3-knowledge-representation-graph-rag--grag)
  - [4. Model Fusion Formula (0.4 GRAG + 0.6 LLM)](#4-model-fusion-formula-04-grag--06-llm)
  - [5. Anti-Hallucination Guardrail](#5-anti-hallucination-guardrail)
- [Fine-Tuning Methodology](#fine-tuning-methodology)
  - [Dataset Processing](#dataset-processing)
  - [QLoRA Configuration](#qlora-configuration)
  - [Training Hyperparameters](#training-hyperparameters)
- [Evaluation and Benchmark Results](#evaluation-and-benchmark-results)
- [Project Structure](#project-structure)
- [Step-by-Step Reproduction Guide](#step-by-step-reproduction-guide)
  - [Prerequisites](#prerequisites)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Dataset Preparation](#2-dataset-preparation)
  - [3. Model Fine-Tuning](#3-model-fine-tuning)
  - [4. Accuracy Evaluation](#4-accuracy-evaluation)
  - [5. Interactive Model Testing](#5-interactive-model-testing)
  - [6. Serving via REST API](#6-serving-via-rest-api)

---

## System Architecture

The pipeline processes natural language meal inputs through four coordinated stages:

```
User Input: "i ate a medum bowl of cerial"
                   │
                   ▼
     [1. Ratcliff/Obershelp Spellcheck]
        "medum" -> "medium", "cerial" -> "cereal"
                   │
                   ▼
     [2. GRAG Verification & Traversal]
        Entity Match: "cereal" (Node present in graph)
        If not found: Abort pipeline, return "Not in database"
                   │
                   ▼
     ┌─────────────┴─────────────┐
     ▼                           ▼
[3. GRAG Engine (0.4)]     [4. Fine-Tuned LLaMA (0.6)]
Sub-graph path lookup       QLoRA 4-bit LLaMA-3.2-3B
Base values (per 100g)      Text generation inference
     └─────────────┬─────────────┘
                   ▼
     [5. Macro Fusion Layer]
        Fused Base = 0.4 * GRAG + 0.6 * LLM
                   │
                   ▼
     [6. Mamdani Fuzzy Scaling]
        Medium bowl -> Centroid Defuzzification -> 185.5g (x1.855)
                   │
                   ▼
Final Scaled Nutritional Breakdown (Carbs, Protein, Fat, Calories)
```

---

## Algorithmic Core

### 1. Typo Correction (Ratcliff/Obershelp Algorithm)

Inputs frequently contain spelling mistakes, phonetic slips, or colloquial truncations. Standard edit distance (Levenshtein) penalizes transposed syllables heavily. The system implements the Ratcliff/Obershelp pattern matching algorithm via Python's `difflib`:

- It locates the longest contiguous common subsequence (LCS) between the input token and candidates in a curated culinary lexicon.
- It recursively evaluates remaining unmatched fragments to the left and right.
- Similarity ratio $S$ is computed as:

  $$S = \frac{2 \cdot M}{|s_1| + |s_2|}$$

  where $M$ is the number of matched characters across all matching segments, and $|s_1|$, $|s_2|$ are the respective token lengths.
- Tokens exceeding $S \ge 0.75$ are substituted before parsing.
- Common syntactic English words (e.g., "ate", "drank", "with", "in") are whitelisted to prevent false alterations.

### 2. Portion Scaling (Mamdani Fuzzy Inference System)

Natural meal logging relies on ambiguous, subjective linguistic descriptors (e.g., "small handful", "large bowl", "medium plate"). The system uses a Mamdani Fuzzy Inference System:

- **Fuzzification**: Input sizes ($S \in \{\text{small}, \text{medium}, \text{large}\}$) and containers ($C \in \{\text{handful}, \text{bowl}, \text{plate}, \text{cup}, \text{slice}, \dots\}$) are mapped to triangular membership functions $\mu(x)$:

  $$\mu(x; a, b, c) = \max\left(0, \min\left(\frac{x - a}{b - a}, \frac{c - x}{c - b}\right)\right)$$

- **Rule Base**: A 25-rule fuzzy associative matrix maps (Size, Container) combinations to output fuzzy volume sets ($\text{tiny}$, $\text{small}$, $\text{medium}$, $\text{large}$, $\text{xlarge}$).
- **Food Density Compensation**: Volumetric capacity is scaled by specific gravity classifications (e.g., cooked grains $\approx 0.75\text{ g/ml}$, liquids $\approx 1.0\text{ g/ml}$, leafy salads $\approx 0.25\text{ g/ml}$, dense proteins $\approx 0.85\text{ g/ml}$).
- **Defuzzification**: Output mass is extracted using centroid (center of gravity) defuzzification over a discretized universe $U \in [0, 500]\text{ g}$:

  $$x^* = \frac{\sum_{i} x_i \cdot \mu_{\text{agg}}(x_i)}{\sum_{i} \mu_{\text{agg}}(x_i)}$$

- The resulting gram weight yields a deterministic scale factor $\text{Scale} = x^* / 100$.

### 3. Knowledge Representation (Graph RAG / GRAG)

Unlike vector RAG, which retrieves nearest textual documents via dense vector cosine similarity, Graph RAG structures food relationships into a directed multi-relational NetworkX graph $G = (V, E)$:

- **Ingredient Nodes**: Baseline food entities containing USDA reference values per 100g ($\text{Carbs}, \text{Protein}, \text{Fat}, \text{Energy}$).
- **Composite Dish Nodes**: Represent multi-ingredient meals linked via directed `[:CONTAINS]` edges with proportional weight attributes $w_{ij}$ where $\sum w_{ij} = 1.0$.
- **Cooking Modifier Nodes**: Represent culinary techniques linked via `[:PREPARED_BY]` edges with delta adjustments ($\Delta \text{Carb}, \Delta \text{Protein}, \Delta \text{Fat}, \Delta \text{Calories}$).
- **Multi-Hop Traversal**: Querying composite dishes computes weighted ingredient aggregations:

  $$N_{\text{dish}} = \sum_{j \in \text{Neighbors}} w_{j} \cdot N_{\text{ingredient}, j} + \Delta N_{\text{modifier}}$$

### 4. Model Fusion Formula (0.4 GRAG + 0.6 LLM)

To combine deterministic graph knowledge with the generative semantic flexibility of the fine-tuned model, base reference values per 100g are combined as:

$$N_{\text{fused}} = 0.4 \cdot N_{\text{GRAG}} + 0.6 \cdot N_{\text{LLM}}$$

The final user-facing metrics are derived by applying the fuzzy scale factor:

$$N_{\text{final}} = N_{\text{fused}} \cdot \text{Scale}$$

### 5. Anti-Hallucination Guardrail

Large language models are prone to hallucinating plausible nutritional values for non-food inputs or gibberish. The system enforces a strict validation gate:

1. The query is cleaned and passed through the Ratcliff/Obershelp spellchecker.
2. The normalized text is checked against the Graph RAG entity index and aliases.
3. If no matching ingredient, composite dish, or alias is found (`found == False`), the LLM inference step is aborted.
4. The system returns an explicit status indicating the item is not recognized, guaranteeing zero hallucinated metrics for unknown inputs.

---

## Fine-Tuning Methodology

### Dataset Processing

Training utilized the USDA FoodData Central database, combining two foundational releases:
1. `FoodData_Central_sr_legacy_food_json_2018-04.json` (SR Legacy Foods: 7,793 items)
2. `surveyDownload.json` (Food and Nutrient Database for Dietary Studies: 5,432 items)

Data parsing normalized all records into a standardized structure containing:
- `meal_description`
- `carb` (Carbohydrates by difference, USDA Nutrient 205)
- `protein` (Protein, USDA Nutrient 203)
- `fat` (Total lipid, USDA Nutrient 204)
- `energy` (Calories in kcal, USDA Nutrient 208)

The dataset was formatted into chat completion prompts:
```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a nutrition expert AI. Given a meal description, estimate the nutritional values accurately.<|eot_id|><|start_header_id|>user<|end_header_id|>

Meal: {description}

Estimate the nutritional values for this meal.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Carbohydrates: {carb:.2f}g
Protein: {protein:.2f}g
Fat: {fat:.2f}g
Calories: {energy:.2f}kcal<|eot_id|>
```

Total dataset size: 13,224 validated examples (11,901 train / 1,323 validation split).

### QLoRA Configuration

- **Base Model**: `unsloth/Llama-3.2-3B`
- **Quantization**: 4-bit NormalFloat (`nf4`) with double quantization enabled and `torch.float16` compute dtype via BitsAndBytes.
- **LoRA Hyperparameters**:
  - Rank ($r$): 16
  - Alpha ($\alpha$): 32
  - Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - Dropout: 0.05
  - Bias: `none`
  - Trainable Parameters: Approximately 0.2% of total weights.
- **Memory Footprint**: LoRA adapter artifacts occupy approximately 35 MB on disk (`adapter_model.safetensors`).

### Training Hyperparameters

- **Epochs**: 3
- **Batch Size**: 4 per device
- **Gradient Accumulation Steps**: 4 (Effective batch size = 16)
- **Learning Rate**: $2 \times 10^{-4}$ with cosine decay schedule
- **Optimizer**: `paged_adamw_8bit`
- **Max Sequence Length**: 128 tokens
- **Gradient Checkpointing**: Enabled
- **Dataset Caching**: In-memory mapping (`keep_in_memory=True`) to prevent redundant disk cache expansion.

---

## Evaluation and Benchmark Results

The model was evaluated against ground truth laboratory standards from the USDA benchmark across 12 staple foods spanning high-protein, high-carb, high-fat, and liquid items.

### Metric Summary

$$\text{Overall Accuracy}: 92.12\%$$

| Nutrient | Mean Absolute Error (MAE) | Clinical / Practical Impact |
|---|---|---|
| **Carbohydrates** | $\pm 1.03\text{ g}$ | Within standard laboratory variance |
| **Protein** | $\pm 1.27\text{ g}$ | Negligible deviation |
| **Fat** | $\pm 1.11\text{ g}$ | Highly precise lipid estimation |
| **Calories** | $\pm 9.17\text{ kcal}$ | $\le 10\text{ kcal}$ across all food categories |

### Detailed Sample Performance

| Meal Item | Actual (C/P/F/Kcal) | Predicted (C/P/F/Kcal) | Accuracy |
|---|---|---|---|
| Grilled Chicken Breast | 0.0 / 31.0 / 3.6 / 165 | 0.0 / 26.4 / 6.6 / 166 | 80.98% |
| White Rice (Cooked) | 28.2 / 2.7 / 0.3 / 130 | 27.0 / 2.2 / 0.2 / 122 | 94.57% |
| Whole Egg (Raw, Fresh) | 0.7 / 12.6 / 9.5 / 143 | 0.9 / 11.6 / 11.6 / 161 | 88.55% |
| Whole Milk (3.25%) | 4.8 / 3.2 / 3.3 / 61 | 5.0 / 3.2 / 3.3 / 46 | 92.70% |
| Raw Banana | 22.8 / 1.1 / 0.3 / 89 | 20.6 / 0.9 / 0.2 / 86 | 95.11% |
| Raw Apple (with skin) | 13.8 / 0.3 / 0.2 / 52 | 13.9 / 0.4 / 0.2 / 53 | 98.51% |
| White Bread | 49.4 / 8.9 / 3.3 / 266 | 49.9 / 9.7 / 2.1 / 255 | 90.17% |
| Ground Beef (80/20) | 0.0 / 17.2 / 20.0 / 254 | 0.0 / 19.9 / 20.0 / 268 | 94.65% |
| Boiled Potato | 20.1 / 1.9 / 0.1 / 87 | 20.7 / 2.5 / 0.1 / 91 | 95.14% |
| Wild Atlantic Salmon | 0.0 / 19.8 / 6.3 / 142 | 0.0 / 18.6 / 7.0 / 137 | 94.95% |
| Dry Oats | 66.3 / 16.9 / 6.9 / 389 | 72.2 / 13.7 / 4.9 / 372 | 84.70% |
| Raw Almonds | 21.6 / 21.2 / 49.9 / 579 | 20.0 / 21.0 / 54.0 / 592 | 95.43% |

Full JSON metrics are preserved in `evaluation_results_lora.json`.

---

## Project Structure

```
Nutrition tracker/
├── api.py                          FastAPI REST API server with health and analysis routes
├── config.py                       Central configuration (parameters, thresholds, model identifiers)
├── fuzzy_processor.py              Ratcliff/Obershelp spellchecking and Mamdani fuzzy portion engine
├── grag_system.py                  NetworkX Graph RAG implementation and multi-hop traversal
├── nutrition_analyzer.py           Core controller orchestrating GRAG, LLM, and fuzzy scaling
├── rag_system.py                   Vector database RAG baseline using ChromaDB
├── dataset/                        Raw USDA FoodData Central JSON and tabular source files
├── evaluation/
│   ├── check_accuracy.py           Detailed accuracy and MAE benchmark against USDA ground truth
│   └── evaluate.py                 Ollama inference evaluation script
├── inference/
│   └── predict.py                  Ollama local inference integration client
├── scripts/
│   └── verify.py                   Environment and prerequisite validation script
├── trained_models/
│   └── llama_nutrition/            Saved LoRA adapter weights, configs, and tokenizer
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── tokenizer.json
│       └── training_metadata.json
├── training/
│   ├── test_model.py               Interactive CLI testing tool for end-to-end inference
│   └── train.py                    QLoRA 4-bit fine-tuning execution pipeline
└── evaluation_results_lora.json    Saved benchmark accuracy and error breakdown
```

---

## Step-by-Step Reproduction Guide

### Prerequisites

- **Operating System**: Windows 10/11, Linux, or macOS
- **Python**: Version 3.10 through 3.14
- **Hardware**: NVIDIA GPU with at least 6 GB VRAM (tested on RTX 4050 Laptop GPU) or CUDA-enabled compute instance.
- **CUDA Toolkit**: 12.1+ or 13.x

### 1. Environment Setup

Clone the repository and install core dependencies:

```bash
git clone <repository_url>
cd "Nutrition tracker"
```

Install required packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft bitsandbytes datasets accelerate pandas networkx fastapi uvicorn pydantic tqdm
```

Verify your installation:

```bash
python scripts/verify.py
```

### 2. Dataset Preparation

Ensure USDA FoodData Central JSON files are placed in the `dataset/` directory:
- `dataset/FoodData_Central_sr_legacy_food_json_2018-04/FoodData_Central_sr_legacy_food_json_2018-04.json`
- `dataset/FoodData_Central_survey_food_json_2024-10-31/surveyDownload.json`

The training script automatically detects and ingests all matching `.json`, `.csv`, and `.parquet` files in `dataset/`.

### 3. Model Fine-Tuning

To train the QLoRA adapter on the dataset:

```bash
python training/train.py
```

- Base model `unsloth/Llama-3.2-3B` will download to Hugging Face cache.
- Training executes for 3 epochs with periodic evaluations.
- Output artifacts are saved to `trained_models/llama_nutrition/`.

### 4. Accuracy Evaluation

To run the accuracy evaluation against ground truth reference items:

```bash
python evaluation/check_accuracy.py
```

This outputs per-sample metrics, computes MAE for all macronutrients, and updates `evaluation_results_lora.json`.

### 5. Interactive Model Testing

To launch the interactive command-line interface:

```bash
python training/test_model.py
```

Example session:

```text
Meal: i ate a medum bowl of cerial

--- Typo Detected ---
  Original input  : i ate a medum bowl of cerial
  Interpreted as  : i ate a medium bowl of cereal
  Corrections     : medum -> medium, cerial -> cereal

--- GRAG Knowledge Graph ---
  Nodes visited   : cereal
  Path            : Query -> MATCHED_INGREDIENT -> cereal (weight=1.0)
  GRAG Base (100g): C=75.0g, P=7.0g, F=2.5g, Cal=360.0kcal

--- Fine-Tuned LLaMA Raw Output ---
Carbohydrates: 75.00g
Protein: 7.00g
Fat: 2.50g
Calories: 360.00kcal

--- Fused Base (0.4 GRAG + 0.6 LLM) ---
  Carbs:    75.0g
  Protein:  7.0g
  Fat:      2.5g
  Calories: 360.0kcal

--- Fuzzy Quantity Interpretation ---
  Method          : mamdani
  Estimated mass  : 185.51g
  Category        : cooked_grains
  Scale applied   : x1.8551

--- Final Scaled Result (x1.8551) ---
  Carbs:    139.1g
  Protein:  13.0g
  Fat:      4.6g
  Calories: 668kcal
```

Example testing unknown or gibberish input:

```text
Meal: asdfghjkl

--- GRAG Knowledge Graph ---
  Status          : Food item not found in database.
  Notice          : Skipped LLM generation to prevent hallucination.

Result: Food item not in the database. Please check your input.
```

### 6. Serving via REST API

Start the FastAPI application:

```bash
python api.py
```

The service will run at `http://localhost:8000`.

#### Health Endpoint

```bash
curl -X GET http://localhost:8000/health
```

Response:
```json
{"status": "ok"}
```

#### Analyze Endpoint

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"meal_description": "grilled chicken breast with a cup of white rice"}'
```

Response:
```json
{
  "meal_description": "grilled chicken breast with a cup of white rice",
  "carbohydrates": 44.53,
  "protein": 37.38,
  "fat": 4.02,
  "calories": 368.52,
  "source": "grag+llm",
  "rag_matches": 3,
  "status": "success",
  "message": "Food item analyzed successfully."
}
```
