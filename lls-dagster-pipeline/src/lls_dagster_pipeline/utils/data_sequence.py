import sys
import numpy as np
import pandas as pd

def create_sequence(context, df: pd.DataFrame, scaler_X, scaler_y):
	lstm_params = context.resources.lstm_config
	LSTM_WINDOW_SIZE = lstm_params['LSTM_WINDOW_SIZE']
	LSTM_FUTURE_OFFSET = lstm_params['LSTM_FUTURE_OFFSET']

	all_cols = df.columns.to_list()
	not_use_cols = ['Timebin', 'Hour', 'Weekday']
	feature_cols = [col for col in all_cols if col not in not_use_cols]
	target_col = ['2A_TotalForklifts', '2B_TotalForklifts', 'Tor2_TotalForklifts']

	df = df.sort_values("Timebin").reset_index(drop=True).copy()

	context.log.info(f"Features: {feature_cols}")
	sys.stdout.write(f"Real y max/min before scale: {df[target_col].max()}, {df[target_col].min()}\n")

	# IMPORTANT: only transform; scalers are already fit
	df_features_scaled = pd.DataFrame(
		scaler_X.transform(df[feature_cols]),
		columns=feature_cols,
		index=df.index
	)

	#df['target_scaled'] = scaler_y.transform(df[target_col])
	df_target_scaled = pd.DataFrame(
		scaler_y.transform(df[target_col]),
		columns=target_col,
		index=df.index
	)
	X, y = [], []
	indices = []

	context.log.info(f"Df shape: {df.shape}")
	context.log.info(f"Df columns: {df.columns}")
	context.log.info(f"Target_Scaled columns: {df_target_scaled.columns}")
	for i in range(len(df) - LSTM_WINDOW_SIZE - (LSTM_FUTURE_OFFSET - 1)):
		if df["Weekday"].iloc[i] == df["Weekday"].iloc[i + LSTM_WINDOW_SIZE]:
			X_window = df_features_scaled.iloc[i:i + LSTM_WINDOW_SIZE].values
			y_next   = df_target_scaled.iloc[i + LSTM_WINDOW_SIZE + (LSTM_FUTURE_OFFSET - 1)].values
			X.append(X_window)
			y.append(y_next)
			indices.append(i + LSTM_WINDOW_SIZE + (LSTM_FUTURE_OFFSET - 1))

	context.log.info(f"X shape: {np.array(X).astype(np.float32).shape}")
	y = np.array(y).astype(np.float32)
	sys.stdout.write(f"Y shape {y.shape}\n")

	return np.array(X).astype(np.float32), y, indices
