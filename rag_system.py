import os
import sys
import json
import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K_RESULTS, SIMILARITY_THRESHOLD


class NutritionRAG:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def search_similar_meals(self, query, n_results=TOP_K_RESULTS):
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        if not results["documents"][0]:
            return []

        matches = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            similarity = 1.0 - dist
            if similarity >= SIMILARITY_THRESHOLD:
                matches.append({
                    "meal": doc,
                    "nutrition": meta,
                    "similarity": similarity
                })

        return matches

    def get_weighted_nutrition(self, matches):
        if not matches:
            return None

        total_weight = sum(m["similarity"] for m in matches)
        result = {"carbohydrates": 0.0, "protein": 0.0, "fat": 0.0, "calories": 0.0}

        for m in matches:
            w = m["similarity"] / total_weight
            meta = m["nutrition"]
            result["carbohydrates"] += w * float(meta.get("carbohydrates", 0))
            result["protein"] += w * float(meta.get("protein", 0))
            result["fat"] += w * float(meta.get("fat", 0))
            result["calories"] += w * float(meta.get("calories", 0))

        return {k: round(v, 2) for k, v in result.items()}

    def add_meal(self, meal_description, nutrition, meal_id=None):
        if meal_id is None:
            meal_id = str(self.collection.count() + 1)
        self.collection.add(
            documents=[meal_description],
            metadatas=[{
                "carbohydrates": str(nutrition.get("carbohydrates", 0)),
                "protein": str(nutrition.get("protein", 0)),
                "fat": str(nutrition.get("fat", 0)),
                "calories": str(nutrition.get("calories", 0)),
            }],
            ids=[meal_id]
        )

    def count(self):
        return self.collection.count()
