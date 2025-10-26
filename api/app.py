import os
import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path  # <<< ADICIONE ESTA IMPORTAÇÃO

# --- Configuração (À Prova de Falhas) ---

# Obter o caminho absoluto para este arquivo (app.py)
SCRIPT_DIR = Path(__file__).resolve().parent

# Ir um nível "acima" (para a raiz do projeto)
PROJECT_ROOT = SCRIPT_DIR.parent

# Definir os caminhos a partir da raiz do projeto
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "rul_predictor_lstm.h5"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# --- Carregar Artefatos ---
print(f"Carregando modelo de: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo carregado.")

print(f"Carregando scaler de: {SCALER_PATH}")
scaler = joblib.load(SCALER_PATH)
print("Scaler carregado.")

# Constantes do modelo (devem ser as mesmas do treino)
SEQUENCE_LENGTH = 50
NUM_FEATURES = 19  # Usamos 19 features no treino

# --- Carregar Artefatos ---
print("Carregando modelo...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo carregado.")

print("Carregando scaler...")
scaler = joblib.load(SCALER_PATH)
print("Scaler carregado.")

# --- Inicializar a API ---
app = FastAPI(
    title="API de Manutenção Preditiva (PdM)",
    description="Prevê o RUL (Remaining Useful Life) de um motor.",
    version="1.0.0"
)

# --- Definir Modelo de Entrada (Pydantic) ---
# Esperamos uma lista de ciclos, onde cada ciclo é uma lista de floats


class MotorData(BaseModel):
    # Ex: [[f1, f2, ...], [f1, f2, ...]]
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
        # --- 1. Validar e Processar Entrada ---
        raw_data = np.array(data.cycles)

        # Validar o número de features
        if raw_data.shape[1] != NUM_FEATURES:
            raise HTTPException(
                status_code=400,
                detail=f"Entrada inválida. Esperado {NUM_FEATURES} features por ciclo, mas recebi {raw_data.shape[1]}."
            )

        # --- 2. Aplicar "Padding" ---
        # (Exatamente como fizemos no Passo 5, Célula v2)
        padded_data = np.zeros((SEQUENCE_LENGTH, NUM_FEATURES))
        # Pega os últimos 50 (ou menos)
        padded_data[-raw_data.shape[0]:] = raw_data[-SEQUENCE_LENGTH:]

        # --- 3. Aplicar "Scaling" ---
        # O scaler foi treinado em 2D (samples, features)
        scaled_data = scaler.transform(padded_data)

        # --- 4. Reshape para a LSTM ---
        # O modelo espera (batch_size, sequence_length, num_features)
        model_input = np.reshape(
            scaled_data, (1, SEQUENCE_LENGTH, NUM_FEATURES))

        # --- 5. Fazer a Previsão ---
        prediction = model.predict(model_input)

        # A saída é um array 2D (ex: [[120.5]]), então pegamos o primeiro item
        rul = float(prediction[0][0])

        # Segurança: RUL não pode ser negativo
        if rul < 0:
            rul = 0.0

        return {"predicted_RUL": round(rul, 2)}

    except Exception as e:
        # Se algo der errado (ex: dados mal formatados), retorna um erro claro
        raise HTTPException(
            status_code=500, detail=f"Erro no processamento: {str(e)}")

# --- Fim do app.py ---
