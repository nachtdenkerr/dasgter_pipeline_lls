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

def plot_predictions(timestamps, y_real, y_pred, out_path):
    plt.figure(figsize=(15, 6))
    plt.plot(timestamps, y_real, label="Real", linewidth=2)
    plt.plot(timestamps, y_pred, "--", label="Predicted", linewidth=2)
    
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Forklifts", fontsize=12)
    plt.title("Real vs Predicted", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_predict_vs_test_zone(y_pred, y_test, zone_name):
    """
    Plot predictions vs real values for a single zone.

    Args:
        y_pred: np.ndarray of shape (N, num_zones)
        y_test: np.ndarray of shape (N, num_zones)
        zone_name: str, one of ["2A", "2B", "Tor2"]
    """
    df_test = pd.read_parquet("data/intermediate/train_set.parquet")
    test_indices = pd.read_parquet("data/intermediate/test_indices.parquet")["test_indices"].tolist()

    timestamps = pd.to_datetime(df_test.loc[test_indices, "Timebin"])

    # Map zone_name to column index
    zone_map = {"2A": 0, "2B": 1, "Tor2": 2}
    zone_idx = zone_map[zone_name]

    # --------- Full test set ----------
    plot_predictions(
        timestamps,
        y_test,
        y_pred,
        f"plots/test_full_{zone_name}.png"
    )

    # --------- Specific days ----------
    df_test["day"] = df_test["Timebin"].dt.date
    unique_days = sorted(df_test["day"].unique())
    day_minus_6  = unique_days[-6]
    day_minus_12 = unique_days[-12]

    test_days = df_test.loc[test_indices, "day"]

    mask_day_6  = test_days == day_minus_6
    mask_day_12 = test_days == day_minus_12

    plot_predictions(
        timestamps[mask_day_6],
        y_test[mask_day_6],
        y_pred[mask_day_6],
        f"plots/test_day_minus_6_{zone_name}.png"
    )

    plot_predictions(
        timestamps[mask_day_12],
        y_test[mask_day_12],
        y_pred[mask_day_12],
        f"plots/test_day_minus_12_{zone_name}.png"
    )
    return
