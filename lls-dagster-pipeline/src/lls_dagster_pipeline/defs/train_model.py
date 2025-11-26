import sys
import pandas as pd
import dagster as dg
import mlflow.tensorflow
from dagster import multi_asset, AssetOut
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from lls_dagster_pipeline.utils.data_sequence import create_sequence
from lls_dagster_pipeline.resources.lstm_config import lstm_config_resource


# MARK: split train test data
@multi_asset(
    outs={
        "train_set": AssetOut(io_manager_key="parquet_io_manager"),
        "test_set": AssetOut(io_manager_key="parquet_io_manager"),
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
        "x_train": AssetOut(io_manager_key='numpy_io_manager'),
        "y_train": AssetOut(io_manager_key='numpy_io_manager'),
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
	return create_sequence(context=context, df=train_set)


# MARK: lstm test sequence
@dg.multi_asset(
    outs={
        "x_test": AssetOut(io_manager_key='numpy_io_manager'),
        "y_test": AssetOut(io_manager_key='numpy_io_manager'),
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
	return create_sequence(context=context, df=test_set)


# MARK: train LSTM model
@dg.multi_asset(
       outs={
           "trained_model": AssetOut(),
           "training_history": AssetOut(),
       },
	required_resource_keys={"lstm_config"},
	can_subset=True,
)
def train_lstm_model(context: dg.AssetExecutionContext, x_train, y_train):
	"""
	Train LSTM model with Early Stopping
	model : keras Model
	X_train, y_train : numpy arrays
	history : History object
	"""

	lstm_params = context.resources.lstm_config
	input_shape = (lstm_params['LSTM_WINDOW_SIZE'], x_train.shape[2])
	model = Sequential()
	model.add(Input(shape=input_shape))
	model.add(LSTM(64, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(32, return_sequences=False))
	model.add(Dropout(0.2))
	model.add(Dense(16, activation="relu"))
	model.add(Dense(1))

	model.compile(
		optimizer='adam',
		loss='mse',
		metrics=['mae']
	)

	early_stop = EarlyStopping(
		monitor='val_loss',
		patience=lstm_params['LSTM_PATIENCE'],
		restore_best_weights=True,
		verbose=1
	)
	context.log.info(f"{x_train.shape}")
	context.log.info(f"{y_train.shape}")
	context.log.info(f"{x_train.dtype}, {y_train.dtype}")
	history = model.fit(
		x_train, y_train,
		epochs=lstm_params['LSTM_EPOCHS'],
		batch_size=lstm_params['LSTM_BATCH_SIZE'],
		validation_split=lstm_params['validation_split'],
		callbacks=[early_stop],
		verbose=1
	)
	#checkpoint_path = "models/best_model.keras"
	#checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss")
	
	context.log.info(f"Training completed. Final loss: {history.history['loss'][-1]:.4f}")

	return model, history
