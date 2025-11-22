import sys
import pandas as pd
import dagster as dg
from dagster import AssetKey, multi_asset, AssetOut
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from lls_dagster_pipeline.utils.data_sequence import create_sequence


# MARK: split train test data
@multi_asset(
    outs={
        "train_set": AssetOut(),
        "test_set": AssetOut(),
    },
	required_resource_keys={"lstm_config"},
)
def s11_split_data(context: dg.AssetExecutionContext, df: pd.DataFrame) :
	"""
	Split processed_data into train + validation / test.
		- 80% train and validation, 20% test
	"""
	ltsm_patams = context.resources.lstm_config
	sys.stdout.write(f"column {df.columns}")
	df.sort_values(by=["Timebin"])
	train_val, test = train_test_split(
		df,
		test_size=ltsm_patams['test_size'],
		shuffle=False,
		random_state=0,
		stratify=None
	)
	context.log.info(f'Train data shape: {train_val.shape}, test data shape {test.shape}')
	return train_val, test


# MARK: lstm train sequence
@dg.multi_asset(
    outs={
        "x_train": AssetOut(),
        "y_train": AssetOut(),
    },
	required_resource_keys={"lstm_config"},
)
def create_lstm_sequences(context: dg.AssetExecutionContext, train_set: pd.DataFrame):
	"""
	Convert dataframe into sequences for LSTM.
	Input: df -> pd.DataFrame - Input dataframe sorted by time.
	Returns:
	X : np.ndarray
		Array of shape (num_sequences, seq_len, num_features)
	y : np.ndarray
		Array of shape (num_sequences,)
	"""
	ltsm_patams = context.resources.lstm_config
	LSTM_WINDOW_SIZE = ltsm_patams['LSTM_WINDOW_SIZE']

	return create_sequence(context=context, df=train_set, LSTM_WINDOW_SIZE=LSTM_WINDOW_SIZE)


# MARK: lstm test sequence
@dg.multi_asset(
    outs={
        "x_test": AssetOut(),
        "y_test": AssetOut(),
    },
	required_resource_keys={"lstm_config"},
)
def create_lstm_test_sequences(context: dg.AssetExecutionContext, test_set: pd.DataFrame):
	"""
	Convert dataframe into sequences for LSTM.
	Input: df -> pd.DataFrame - Input dataframe sorted by time.
	Returns:
	X : np.ndarray
		Array of shape (num_sequences, seq_len, num_features)
	y : np.ndarray
		Array of shape (num_sequences,)
	"""
	ltsm_params = context.resources.lstm_config
	LSTM_WINDOW_SIZE = ltsm_params['LSTM_WINDOW_SIZE']

	return create_sequence(context=context, df=test_set, LSTM_WINDOW_SIZE=LSTM_WINDOW_SIZE)


# MARK: train LSTM model
@dg.asset(
	key=AssetKey(["trained_model"]),
	required_resource_keys={"lstm_config"},
)
def train_lstm_model(context: dg.AssetExecutionContext, x_train, y_train):
	"""
	Train LSTM model with Early Stopping
	model : keras Model
	X_train, y_train : numpy arrays
	history : History object
	"""
	ltsm_params = context.resources.lstm_config

	input_shape = (ltsm_params['LSTM_WINDOW_SIZE'], x_train.shape[2])
	model = Sequential()
	model.add(LSTM(32, input_shape=input_shape, return_sequences=False))
	model.add(Dropout(0.2))
	model.add(Dense(1))
	model.compile(
		optimizer='adam',
		loss='mse',
		metrics=['mae']
	)

	early_stop = EarlyStopping(
		monitor='val_loss',
		patience=ltsm_params['LSTM_PATIENCE'],
		restore_best_weights=True,
		verbose=1
	)

	model.fit(
		x_train[:100], y_train[:100],
		epochs=ltsm_params['LSTM_EPOCHS'],
		batch_size=ltsm_params['LSTM_BATCH_SIZE'],
		validation_split=ltsm_params['validation_split'],
		callbacks=[early_stop],
		verbose=1
	)

	return model
