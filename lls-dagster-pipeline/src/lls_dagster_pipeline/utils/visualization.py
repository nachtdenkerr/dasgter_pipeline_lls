import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def extract_single_output(y_pred):
    # If prediction is 3D, take last time step
    if y_pred.ndim == 3:
        y_pred = y_pred[:, -1, 0]   # last timestep, first feature
    elif y_pred.ndim == 2:
        y_pred = y_pred[:, 0]
    return y_pred

def plot_predict_vs_test(y_pred, x_test, y_test):
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