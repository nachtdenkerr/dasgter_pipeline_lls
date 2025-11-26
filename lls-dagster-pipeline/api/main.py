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
    area: str      # e.g., "Tor2"


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return FileResponse(static_dir / "index.html")

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Run inference on a single timestep.
    """
    # Convert inputs to model features
    # Example: convert date/time -> Timebin, Weekday, Hour, sin/cos features
    # Flatten features for LSTM (single timestep)
    # Example:
    TARGET_COLUMNS = [
        '15_ElectroForklift', '2A_ElectroForklift', '2B_ElectroForklift', '3B_ElectroForklift', 
        'NearMuseum_ElectroForklift', 'Tor2_ElectroForklift', 'Tor8_ElectroForklift', 
        '15_DieselForklift', '2A_DieselForklift', '2B_DieselForklift', '3B_DieselForklift', 
        'NearMuseum_DieselForklift', 'Tor2_DieselForklift', 'Tor8_DieselForklift', 
        '15_TotalForklifts', '2A_TotalForklifts', '2B_TotalForklifts', '3B_TotalForklifts', 
        'NearMuseum_TotalForklifts', 'Tor2_TotalForklifts', 'Tor8_TotalForklifts'
    ]

    # Feature columns
    FEATURE_COLUMNS = [
        '15_ElectroForklift', '2A_ElectroForklift', '2B_ElectroForklift', 
        '3B_ElectroForklift', 'NearMuseum_ElectroForklift', 'Tor2_ElectroForklift',
        'Tor8_ElectroForklift',
        '15_DieselForklift', '2A_DieselForklift', '2B_DieselForklift',
        '3B_DieselForklift', 'NearMuseum_DieselForklift', 'Tor2_DieselForklift',
        'Tor8_DieselForklift',
        '15_TotalForklifts', '2A_TotalForklifts', '2B_TotalForklifts',
        '3B_TotalForklifts', 'NearMuseum_TotalForklifts', 'Tor2_TotalForklifts',
        'Tor8_TotalForklifts',
        'HourSin', 'HourCos', 'WeekdaySin', 'WeekdayCos',
        'IsWeekend', 'IsHoliday','IsWorkingHour'
           ]
    hour = int(req.time.split(":")[0])
    minute = int(req.time.split(":")[1])
    timebin = hour + minute/60
    weekday = pd.to_datetime(req.date).weekday()

    # Compute sin/cos encoding
    hour_sin = np.sin(2 * np.pi * timebin / 24)
    hour_cos = np.cos(2 * np.pi * timebin / 24)
    weekday_sin = np.sin(2 * np.pi * weekday / 7)
    weekday_cos = np.cos(2 * np.pi * weekday / 7)
    is_weekend = int(weekday >= 5)
    is_holiday = 0  # if you have holiday calendar, implement it

    # Combine all features into a flat list
    features_dict = dict.fromkeys(FEATURE_COLUMNS + TARGET_COLUMNS, 0.0)

    # Set time-based features
    features_dict.update({
        'HourSin': hour_sin,
        'HourCos': hour_cos,
        'WeekdaySin': weekday_sin,
        'WeekdayCos': weekday_cos,
        'IsWeekend': is_weekend,
        'IsHoliday': is_holiday,
    })
    for lag in ['forklift_lag_1', 'forklift_lag_6', 'forklift_lag_12', 'forklift_lag_36']:
        features_dict[lag] = 0  # placeholder
    # pseudo-code at API startup

    parquet_path = "data/intermediate/df.parquet"
    df_features_scaled = pd.read_parquet(parquet_path)

    # Create new row with zeros for features frontend cannot provide
    new_step_dict = {col: 0 for col in FEATURE_COLUMNS}

    # Fill in the features we know from frontend
    new_step_dict.update({
        'HourSin': hour_sin,
        'HourCos': hour_cos,
        'WeekdaySin': weekday_sin,
        'WeekdayCos': weekday_cos,
        'IsWeekend': 1 if weekday >= 5 else 0,
        'IsWorkingHour': 1 if hour
        # optionally fill area-specific lag or other features
    })

    # Convert dict to array in same order as FEATURE_COLUMNS
    new_step = np.array([new_step_dict[col] for col in FEATURE_COLUMNS]).reshape(1, -1)  # (1, 31)

    # Last 287 rows from parquet
    sequence = df_features_scaled[FEATURE_COLUMNS].values[-287:]  # (287, 31)

    # Concatenate
    sequence = np.vstack([sequence, new_step])  # (288, 31)

    # Scale
    sequence_scaled = scaler_x.transform(sequence).reshape(1, 288, 31)
    sequence_scaled = sequence_scaled.astype("float32")

    # Predict
    y_scaled = model.predict(sequence_scaled)
    y_pred_real_test = scaler_y.inverse_transform(y_scaled)

    y_pred_rounded = np.round(y_pred_real_test).astype(int)

    return {"prediction": int(y_pred_rounded)}
