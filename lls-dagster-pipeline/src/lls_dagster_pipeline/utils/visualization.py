import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

def extract_single_output(y_pred):
    # If prediction is 3D, take last time step
    if y_pred.ndim == 3:
        y_pred = y_pred[:, -1, 0]   # last timestep, first feature
    elif y_pred.ndim == 2:
        y_pred = y_pred[:, 0]
    return y_pred

def plot_predict_vs_test(y_pred, y_test):
    # Getting final test predictions
    out_path = "plots/predict_vs_real.png"
        # ---- Build a time index (5-minute intervals) ----
    y_pred = extract_single_output(y_pred)
    num_points = len(y_test)
    time_index = pd.date_range(
        start="2024-01-01 00:00:00",
        periods=num_points,
        freq="5min"
    )

    # ---- Plot ----
    plt.figure(figsize=(15, 6))
    plt.plot(time_index, y_test, label="Real", linewidth=2)
    plt.plot(time_index, y_pred, '--', label="Predicted", linewidth=2)

    plt.xlabel("Time (5 min intervals)", fontsize=12)
    plt.ylabel("Target Value", fontsize=12)
    plt.title("Real vs Predicted Values Over Time", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path


def plot_predict_vs_test_last_5th_day(y_pred, y_test_real, x_test_real, hour_col_index):
    out_path = "plots/predict_vs_real_5th_day.png"

    # ---- 1) Clean 1D arrays ----
    y_pred = extract_single_output(y_pred).reshape(-1)
    y_real = extract_single_output(y_test_real).reshape(-1)

    # ---- 2) Extract hour from x_test_real ----
    # x_test_real shape: (N, window_size, num_features)
    # hour for each prediction = last timestep of window
    hours = x_test_real[hour_col_index].astype(int)

    # ---- 3) Find day boundaries (23 → 0 transition) ----
    day_starts = []
    for i in range(len(hours)):
        if hours[i] == 0 and (i == 0 or hours[i - 1] == 23):
            day_starts.append(i)

    if len(day_starts) < 5:
        raise ValueError(f"Expected >=5 full days, found only {len(day_starts)}")

    # 5th last day start
    start_idx = day_starts[-5]

    # End at next day start, or end of data
    end_idx = day_starts[-4] if len(day_starts) >= 4 else len(hours)

    # ---- 4) Slice data for that day ----
    y_real_day = y_real[start_idx:end_idx]
    y_pred_day = y_pred[start_idx:end_idx]

    # ---- 5) Build real time-of-day axis from hour + 5min steps ----
    # construct timedelta from first hour value
    start_hour = int(hours[start_idx])
    time_index = pd.date_range(
        start=f"2024-01-01 {start_hour:02d}:00:00",
        periods=len(y_real_day),
        freq="5min"
    )

    # ---- 6) Plot ----
    plt.figure(figsize=(15, 6))
    plt.plot(time_index, y_real_day, label="Real (5th last day)", linewidth=2)
    plt.plot(time_index, y_pred_day, "--", label="Predicted", linewidth=2)

    plt.xlabel("Time")
    plt.ylabel("Target")
    plt.title("Real vs Predicted – 5th Last Day")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path
