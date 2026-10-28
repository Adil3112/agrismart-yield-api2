
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="AgriSmart Crop Yield API")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yield_model.joblib")

data = None
model = None
crop_encoder = None
state_encoder = None
feature_columns = None


@app.on_event("startup")
def load_model():
    global data, model, crop_encoder, state_encoder, feature_columns

    data = joblib.load(MODEL_PATH)
    model = data["model"]
    crop_encoder = data["label_encoder_crop"]
    state_encoder = data["label_encoder_state"]
    feature_columns = data["feature_columns"]


class CropInput(BaseModel):
    crop: str
    state: str
    area: float
    fertilizer: float
    pesticide: float
    avg_temp_c: float
    total_rainfall_mm: float
    avg_humidity_percent: float
    N: float
    P: float
    K: float
    pH: float


@app.get("/")
def home():
    return {"message": "AgriSmart Crop Yield API is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(payload: CropInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        values = payload.model_dump()

        values["crop"] = crop_encoder.transform([values["crop"]])[0]
        values["state"] = state_encoder.transform([values["state"]])[0]

        input_df = pd.DataFrame([values])
        input_df = input_df[feature_columns]

        prediction = model.predict(input_df)[0]

        return {"predicted_yield": round(float(prediction), 2)}

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid crop or state, or invalid input: {exc}"
        )
