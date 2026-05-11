"""
service.py
----------
BentoML service that loads the best model from MLflow
and serves it as a REST API.

Usage:
    bentoml serve src/service.py:svc --reload --port 3000
"""

import bentoml
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from bentoml.io import JSON
from pydantic import BaseModel

# ── CONFIG ────────────────────────────────────────────────────
MLFLOW_URI   = "http://localhost:5000"
MODEL_NAME   = "house-price-model"
MODEL_VERSION = "1"

# ── LOAD MODEL FROM MLFLOW ────────────────────────────────────
print(f"Loading model {MODEL_NAME} v{MODEL_VERSION} from MLflow...")
mlflow.set_tracking_uri(MLFLOW_URI)

model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model     = mlflow.sklearn.load_model(model_uri)
print("Model loaded successfully")

# ── INPUT SCHEMA ──────────────────────────────────────────────
class HouseFeatures(BaseModel):
    MedInc:                 float  # Median income in block group
    HouseAge:               float  # Median house age
    AveRooms:               float  # Average rooms per household
    AveBedrms:              float  # Average bedrooms per household
    Population:             float  # Block group population
    AveOccup:               float  # Average household occupancy
    Latitude:               float  # Block group latitude
    Longitude:              float  # Block group longitude
    RoomsPerHousehold:      float  # Engineered: AveRooms / AveOccup
    BedroomsRatio:          float  # Engineered: AveBedrms / AveRooms
    PopulationPerHousehold: float  # Engineered: Population / AveOccup
    DistFromCenter:         float  # Engineered: distance from LA center

# ── BENTOML SERVICE ───────────────────────────────────────────
svc = bentoml.Service("house_price_predictor")

@svc.api(input=JSON(pydantic_model=HouseFeatures), output=JSON())
def predict(features: HouseFeatures) -> dict:
    """
    Predict house price from input features.
    Returns predicted price in $100,000 units.
    """

    # Convert input to DataFrame
    input_df = pd.DataFrame([features.dict()])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Return result
    return {
        "predicted_price_100k": round(float(prediction), 4),
        "predicted_price_usd":  round(float(prediction) * 100_000, 2),
        "model":                MODEL_NAME,
        "version":              MODEL_VERSION
    }
