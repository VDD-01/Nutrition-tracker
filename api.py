import os
import sys
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nutrition_analyzer import NutritionAnalyzer

app = FastAPI(title="Nutrition Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = None


@app.on_event("startup")
async def startup_event():
    global analyzer
    analyzer = NutritionAnalyzer()


class MealRequest(BaseModel):
    meal_description: str


class NutritionResponse(BaseModel):
    meal_description: str
    carbohydrates: float
    protein: float
    fat: float
    calories: float
    source: str
    rag_matches: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=NutritionResponse)
def analyze_meal(request: MealRequest):
    if not request.meal_description.strip():
        raise HTTPException(status_code=400, detail="meal_description cannot be empty")

    result = analyzer.analyze(request.meal_description)

    return NutritionResponse(
        meal_description=request.meal_description,
        carbohydrates=result["carbohydrates"],
        protein=result["protein"],
        fat=result["fat"],
        calories=result["calories"],
        source=result["source"],
        rag_matches=result["rag_matches"],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
