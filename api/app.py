import os
import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR.parent

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "rul_predictor_lstm.h5"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

print(f"Carregando modelo de: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo carregado.")

print(f"Carregando scaler de: {SCALER_PATH}")
scaler = joblib.load(SCALER_PATH)
print("Scaler carregado.")

SEQUENCE_LENGTH = 50
NUM_FEATURES = 19  

print("Carregando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo carregado.")

print("Carregando scaler...")
scaler = joblib.load(SCALER_PATH)
print("Scaler carregado.")

app = FastAPI(
    title="API de Manutenção Preditiva (PdM)",
    description="Prevê o RUL (Remaining Useful Life) de um motor.",
    version="1.0.0"
)

class MotorData(BaseModel):
    cycles: List[List[float]]


@app.get("/")
def read_root():
    return {"status": "API de Previsão de RUL está online."}


@app.post("/predict")
def predict_rul(data: MotorData):
    """
    Recebe os dados de ciclo de um motor e retorna o RUL previsto.

    Os dados devem ser os últimos ciclos observados, com as 19 features
    usadas no treinamento, na ordem correta.
    """
    try:
        raw_data = np.array(data.cycles)

        if raw_data.shape[1] != NUM_FEATURES:
            raise HTTPException(
                status_code=400,
                detail=f"Entrada inválida. Esperado {NUM_FEATURES} features por ciclo, mas recebi {raw_data.shape[1]}."
            )

        padded_data = np.zeros((SEQUENCE_LENGTH, NUM_FEATURES))
        padded_data[-raw_data.shape[0]:] = raw_data[-SEQUENCE_LENGTH:]

        scaled_data = scaler.transform(padded_data)

        model_input = np.reshape(
            scaled_data, (1, SEQUENCE_LENGTH, NUM_FEATURES))

        prediction = model.predict(model_input)

        rul = float(prediction[0][0])

        if rul < 0:
            rul = 0.0

        return {"predicted_RUL": round(rul, 2)}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erro no processamento: {str(e)}")
