import sys
import joblib
import numpy as np
import pandas as pd
import dagster as dg
from sklearn.preprocessing import StandardScaler
from lls_dagster_pipeline.resources.lstm_config import lstm_config_resource

def create_sequence(context: dg.AssetExecutionContext, df: pd.DataFrame):
	lstm_params = context.resources.lstm_config
	LSTM_WINDOW_SIZE = lstm_params['LSTM_WINDOW_SIZE']
	LSTM_FUTURE_OFFSET = lstm_params['LSTM_FUTURE_OFFSET']
	all_cols = df.columns.to_list()
	not_use_cols = ['Timebin', 'Hour', 'Weekday']
	feature_cols = [col for col in all_cols if col not in not_use_cols]
	target_col = ['2A_TotalForklifts']
	context.log.info(f'Features: {feature_cols}')
	# Ensure data is sorted by time
	df = df.sort_values("Timebin").reset_index(drop=True)

	scaler_X = StandardScaler()
	scaler_y = StandardScaler()
	df_features_scaled = pd.DataFrame(
		scaler_X.fit_transform(df[feature_cols]),
		columns=feature_cols,
		index=df.index
	)
	df['target_scaled'] = scaler_y.fit_transform(df[target_col])
	joblib.dump(scaler_X, "models/scaler_X.pkl")
	joblib.dump(scaler_y, "models/scaler_y.pkl")
	# Create sequences
	X, y= [], []

	for i in range(len(df) - LSTM_WINDOW_SIZE - (LSTM_FUTURE_OFFSET-1)):
		# Feature window
		X_window = df_features_scaled.iloc[i:i+LSTM_WINDOW_SIZE].values
		
		# Target variable of the next record
		y_next = df['target_scaled'].iloc[i + LSTM_WINDOW_SIZE + (LSTM_FUTURE_OFFSET-1)]
		
		X.append(X_window)
		y.append(y_next)
	y = np.array(y).reshape(-1, 1).astype(np.float32)
	sys.stdout.write(f"Y shape {y.shape}")
	return np.array(X).astype(np.float32), y