"""Simple FastAPI app for ML predictions"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ml_service import MLService
from typing import List

app = FastAPI(title="Diabetes Prediction API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML service
ml_service = MLService()

class PredictionInput(BaseModel):
    features: List[float]

class PredictionOutput(BaseModel):
    prediction: float
    metrics: dict

@app.get("/")
def root():
    return {"message": "Diabetes Prediction API"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    prediction = ml_service.predict(input_data.features)
    metrics = ml_service.get_metrics()
    return PredictionOutput(prediction=prediction, metrics=metrics)

@app.get("/metrics")
def get_metrics():
    return ml_service.get_metrics() 