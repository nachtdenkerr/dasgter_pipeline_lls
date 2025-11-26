import json
import mlflow
import pickle
import numpy as np
import joblib
from tensorflow.keras.models import model_from_json
import tensorflow.keras as keras

MODEL_JSON_PATH = "models/json_model.json"
SCALER_X_PATH = "models/scaler_X.pkl"
SCALER_Y_PATH = "models/scaler_y.pkl"


def load_model_and_scalers():
    # Load model architecture from JSON
    #model = mlflow.pyfunc.load_model("models:/forklift-lstm/Production")
    model = keras.models.load_model('models/latest_model.keras')
    # Load trained weights (must exist: json_model.weights.h5)
    #model.load_weights("models/json_model.weights.h5")

    # Load scalers
    scaler_X = joblib.load("models/scaler_X.pkl")
    scaler_y = joblib.load("models/scaler_y.pkl")

    return model, scaler_X, scaler_y
