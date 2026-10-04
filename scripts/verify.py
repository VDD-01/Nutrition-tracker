import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OLLAMA_MODEL, OLLAMA_HOST, CHROMA_DB_PATH

BASE_DIR = Path(__file__).resolve().parent.parent


def check_ollama():
    try:
        import ollama
        client = ollama.Client(host=OLLAMA_HOST)
        models = client.list()
        names = [m["model"] for m in models.get("models", [])]
        if OLLAMA_MODEL in names or any(OLLAMA_MODEL in n for n in names):
            print(f"Ollama: {OLLAMA_MODEL} found")
            return True
        else:
            print(f"Ollama running but model '{OLLAMA_MODEL}' not found. Available: {names}")
            print(f"  Run: ollama pull {OLLAMA_MODEL}")
            return False
    except Exception as e:
        print(f"Ollama connection failed: {e}")
        print(f"  Make sure Ollama is running: ollama serve")
        return False


def check_dataset():
    data_dir = BASE_DIR / "dataset"
    if not data_dir.exists():
        print(f"Dataset directory missing: {data_dir}")
        return False
    files = list(data_dir.rglob("*.json")) + list(data_dir.rglob("*.csv")) + list(data_dir.rglob("*.parquet"))
    if files:
        print(f"Dataset: {len(files)} file(s) found in {data_dir}")
        return True
    else:
        print(f"Dataset directory exists but is empty: {data_dir}")
        return False


def check_vector_db():
    db_path = Path(CHROMA_DB_PATH)
    if db_path.exists():
        print(f"Vector DB: found at {db_path}")
        return True
    else:
        print(f"Vector DB: not found at {db_path} (will be created on first run)")
        return True


def check_trained_model():
    adapter_path = BASE_DIR / "trained_models" / "llama_nutrition" / "adapter_model.safetensors"
    if adapter_path.exists():
        size_mb = adapter_path.stat().st_size / (1024 * 1024)
        print(f"Trained model: found ({size_mb:.1f} MB)")
        return True
    else:
        print("Trained model: not found (run training/train.py to train)")
        return False


def main():
    print("=" * 60)
    print("NUTRITION TRACKER - SYSTEM VERIFY")
    print("=" * 60 + "\n")

    checks = [
        ("Ollama", check_ollama),
        ("Dataset", check_dataset),
        ("Vector DB", check_vector_db),
        ("Trained Model", check_trained_model),
    ]

    passed = 0
    for name, fn in checks:
        ok = fn()
        if ok:
            passed += 1
        print()

    print("=" * 60)
    print(f"Results: {passed}/{len(checks)} checks passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
