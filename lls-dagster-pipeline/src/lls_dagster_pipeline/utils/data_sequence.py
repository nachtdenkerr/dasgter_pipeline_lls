import sys
import numpy as np
import pandas as pd
import dagster as dg
from sklearn.preprocessing import MinMaxScaler

def create_sequence(context: dg.AssetExecutionContext, df: pd.DataFrame, LSTM_WINDOW_SIZE):
	all_cols = df.columns.to_list()
	not_use_cols = ['Timebin', 'Hour', 'Weekday']
	feature_cols = [col for col in all_cols if col not in not_use_cols]
	#feature_cols = ['HourSin']
	target_col = ['2A_TotalForklifts']
	context.log.info(f'Features: {feature_cols}')
	# Ensure data is sorted by time
	df = df.sort_values("Timebin").reset_index(drop=True)

	scaler_X = MinMaxScaler()
	scaler_y = MinMaxScaler()
	df_features_scaled = pd.DataFrame(
		scaler_X.fit_transform(df[feature_cols]),
		columns=feature_cols,
		index=df.index
	)
	df['target_scaled'] = scaler_y.fit_transform(df[target_col])

	# Create sequences
	X, y, indices = [], [], []

	for i in range(len(df) - LSTM_WINDOW_SIZE):
		# Feature window
		X_window = df_features_scaled.iloc[i:i+LSTM_WINDOW_SIZE].values
		
		# Target variable of the next record
		y_next = df['target_scaled'].iloc[i+LSTM_WINDOW_SIZE]
		
		X.append(X_window)
		y.append(y_next)
		indices.append(i + LSTM_WINDOW_SIZE)
	y = np.array(y).reshape(-1, 1).astype(np.float32)
	sys.stdout.write(f"Y shape {y.shape}")
	return np.array(X).astype(np.float32), y