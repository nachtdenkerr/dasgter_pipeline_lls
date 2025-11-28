import uvicorn
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from .model_loader import load_model_and_scalers
from pathlib import Path
from fastapi.responses import FileResponse
from src.lls_dagster_pipeline.defs.train_model import weighted_mse
# Load model + scalers on startup
model, scaler_x, scaler_y = load_model_and_scalers()

app = FastAPI(title="Forklift Prediction API", version="1.0")

# For templates (optional, easier for dynamic HTML)
templates = Jinja2Templates(directory="static")


current_dir = Path(__file__).parent  # api/
static_dir = current_dir / "static"
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

class PredictRequest(BaseModel):
    date: str      # YYYY-MM-DD
    time: str      # HH:MM

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust for your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return FileResponse(static_dir / "index.html")

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Run inference on a single timestep for 3 zones using historical data.
    """
    import pandas as pd
    import numpy as np

    # ------------------------------
    # Configuration
    # ------------------------------
    FEATURE_COLUMNS = [
        '15_ElectroForklift', '2A_ElectroForklift', '2B_ElectroForklift', '3B_ElectroForklift',
        'NearMuseum_ElectroForklift', 'Tor2_ElectroForklift', 'Tor8_ElectroForklift',
        '15_DieselForklift', '2A_DieselForklift', '2B_DieselForklift', '3B_DieselForklift',
        'NearMuseum_DieselForklift', 'Tor2_DieselForklift', 'Tor8_DieselForklift',
        '15_TotalForklifts', '2A_TotalForklifts', '2B_TotalForklifts', '3B_TotalForklifts',
        'NearMuseum_TotalForklifts', 'Tor2_TotalForklifts', 'Tor8_TotalForklifts',
        'HourSin', 'HourCos', 'WeekdaySin', 'WeekdayCos',
        'IsWeekend', 'IsHoliday', 'IsWorkingHour',
        'forklift_lag_1', 'forklift_lag_6', 'forklift_lag_12', 'forklift_lag_36'
    ]
    LSTM_WINDOW_SIZE = 144
    zone_names = ["2A", "2B", "Tor2"]  # model output order

    # ------------------------------
    # Load data
    # ------------------------------
    df_features = pd.read_parquet("data/intermediate/df.parquet")
    df_features["Timebin"] = pd.to_datetime(df_features["Timebin"])

    # ------------------------------
    # Compute time-based features
    # ------------------------------
    hour = int(req.time.split(":")[0])
    minute = int(req.time.split(":")[1])
    frontend_ts = pd.to_datetime(f"{req.date} {req.time}")
    timebin = hour + minute / 60
    weekday = frontend_ts.weekday()

    hour_sin = np.sin(2 * np.pi * timebin / 24)
    hour_cos = np.cos(2 * np.pi * timebin / 24)
    weekday_sin = np.sin(2 * np.pi * weekday / 7)
    weekday_cos = np.cos(2 * np.pi * weekday / 7)
    is_weekend = int(weekday >= 5)
    is_holiday = 0  # implement holiday calendar if needed
    is_working_hour = int(5 <= hour <= 17 and hour not in [9, 12])

    # ------------------------------
    # Slice historical data
    # ------------------------------
    df_hist = df_features[df_features["Timebin"] < frontend_ts].copy()
    if len(df_hist) < LSTM_WINDOW_SIZE:
        raise ValueError(f"Not enough historical data for LSTM (needed {LSTM_WINDOW_SIZE}, got {len(df_hist)})")

    # Take last LSTM_WINDOW_SIZE rows
    df_seq = df_hist.iloc[-LSTM_WINDOW_SIZE:].copy()

    # ------------------------------
    # Update last row with frontend time features
    # ------------------------------
    df_seq.iloc[-1, df_seq.columns.get_indexer(["HourSin", "HourCos", "WeekdaySin", "WeekdayCos",
                                                "IsWeekend", "IsHoliday", "IsWorkingHour"])] = [
        hour_sin, hour_cos, weekday_sin, weekday_cos, is_weekend, is_holiday, is_working_hour
    ]

    # ------------------------------
    # Compute lag features dynamically
    # ------------------------------
    for lag in [1, 6, 12, 36]:
        df_seq[f"forklift_lag_{lag}"] = df_seq["Tor2_TotalForklifts"].shift(lag)
    df_seq.fillna(0, inplace=True)  # just in case early rows

    # ------------------------------
    # Build feature matrix
    # ------------------------------

    X_seq = df_seq[FEATURE_COLUMNS].values  # shape (LSTM_WINDOW_SIZE, num_features)
    X_seq_df = pd.DataFrame(X_seq, columns=FEATURE_COLUMNS)
    X_seq_scaled = scaler_x.transform(X_seq_df).astype("float32").reshape(1, LSTM_WINDOW_SIZE, len(FEATURE_COLUMNS))

    # ------------------------------
    # Predict
    # ------------------------------
    y_scaled = model.predict(X_seq_scaled)  # shape (1, 3)
    y_pred_real = scaler_y.inverse_transform(y_scaled)  # shape (1, 3)
    y_pred_rounded = np.rint(y_pred_real).astype(int)
    # ------------------------------
    # Return dict for front-end
    # ------------------------------
    prediction_dict = {zone: int(y_pred_rounded[0, idx]) for idx, zone in enumerate(zone_names)}
    print(prediction_dict)
    return {"prediction": prediction_dict}


