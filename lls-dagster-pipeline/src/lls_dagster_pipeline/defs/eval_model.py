import sys
import joblib
import mlflow
import pandas as pd
import numpy as np
import dagster as dg
from mlflow import MlflowException
from mlflow.models import infer_signature
from mlflow.entities import model_registry
from mlflow.tracking import MlflowClient
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from lls_dagster_pipeline.utils.visualization import plot_predict_vs_test_zone
#from tensorflow.keras.models import model_to_json

@dg.asset(
	required_resource_keys={"lstm_config", "mlflow"},
	group_name='ml_model',

)
def eval_model(context: dg.AssetExecutionContext, trained_model, training_history, x_train, y_train, x_test, y_test):
	"""
	Evaluate model and log everything to MLflow.
	"""
	context.resources.mlflow  # ensure resource is initialized
	lstm_params = context.resources.lstm_config	
	with mlflow.start_run(run_name="lstm_evaluation") as run:
		context.log.info("Starting MLflow run for model evaluation...")
		mlflow.log_param("lstm_window_size", lstm_params['LSTM_WINDOW_SIZE'])
		mlflow.log_param("epochs", lstm_params['LSTM_EPOCHS'])
		mlflow.log_param("batch_size", lstm_params['LSTM_BATCH_SIZE'])
		mlflow.log_param("patience", lstm_params['LSTM_PATIENCE'])
		mlflow.log_param("validation_split", lstm_params['validation_split'])
		mlflow.log_param("test_size", lstm_params['test_size'])
		scaler_X = joblib.load("models/scaler_X.pkl")
		scaler_y = joblib.load("models/scaler_y.pkl")
	
		# Zone names for logging
		zone_names = ['2A', '2B', 'Tor2']

		# 5. Calculate evaluation metrics for each zone
		context.log.info("=" * 60)
		context.log.info("CALCULATING METRICS FOR EACH ZONE:")
		# 3. Make predictions
		y_pred_train = trained_model.predict(x_train)
		y_pred_test = trained_model.predict(x_test)

		# Clip negative predictions (can't have negative equipment count)
		y_pred_train = np.clip(y_pred_train, 0, None)
		y_pred_test = np.clip(y_pred_test, 0, None)

		# 4. Inverse transform
		
		# Inverse transform to get REAL values (in forklift counts)
		y_train_real = scaler_y.inverse_transform(y_train)
		y_test_real = scaler_y.inverse_transform(y_test)
		y_pred_train_real = scaler_y.inverse_transform(y_pred_train)
		y_pred_test_real = scaler_y.inverse_transform(y_pred_test)

		# Clip negatives (если ещё не сделали)
		y_pred_train_real = np.clip(y_pred_train_real, 0, None)
		y_pred_test_real = np.clip(y_pred_test_real, 0, None)
		
		context.log.info("✅ Inverse transform completed - all values in real forklift counts")

		# Convert to integer counts
		#y_train_real_int = np.round(y_train_real).astype(int)
		#y_test_real_int = np.round(y_test_real).astype(int)
		#y_pred_train_real_int = np.round(y_pred_train_real).astype(int)
		#y_pred_test_real_int = np.round(y_pred_test_real).astype(int)

		y_train_real_int =y_train_real
		y_test_real_int = y_test_real
		y_pred_train_real_int = y_pred_train_real
		y_pred_test_real_int =y_pred_test_real

		# For each zone
		for zone_idx, zone_name in enumerate(zone_names):
			# Extract predictions for the specific zone
			y_train_zone = y_train_real_int[:, zone_idx]
			y_test_zone = y_test_real_int[:, zone_idx]
			y_pred_train_zone = y_pred_train_real_int[:, zone_idx]
			y_pred_test_zone = y_pred_test_real_int[:, zone_idx]

			# Clip negative predictions to 0 (can't have negative equipment count)
			y_pred_train_zone = np.clip(y_pred_train_zone, 0, None)
			y_pred_test_zone = np.clip(y_pred_test_zone, 0, None)

			# Calculate evaluation metrics
			train_mae = mean_absolute_error(y_train_zone, y_pred_train_zone)
			train_rmse = np.sqrt(mean_squared_error(y_train_zone, y_pred_train_zone))
			train_r2 = r2_score(y_train_zone, y_pred_train_zone)
			#train_da = calculate_direction_accuracy(y_train_zone, y_pred_train_zone)

			test_mae = mean_absolute_error(y_test_zone, y_pred_test_zone)
			test_rmse = np.sqrt(mean_squared_error(y_test_zone, y_pred_test_zone))
			test_r2 = r2_score(y_test_zone, y_pred_test_zone)
			#test_da = calculate_direction_accuracy(y_test_zone, y_pred_test_zone)

			# Log evaluation metrics to MLflow
			mlflow.log_metric(f"{zone_name}_train_rmse", train_rmse)
			mlflow.log_metric(f"{zone_name}_train_mae", train_mae)
			mlflow.log_metric(f"{zone_name}_train_r2", train_r2)
			#mlflow.log_metric(f"{zone_name}_train_direction_accuracy", train_da)
			mlflow.log_metric(f"{zone_name}_test_rmse", test_rmse)
			mlflow.log_metric(f"{zone_name}_test_mae", test_mae)
			mlflow.log_metric(f"{zone_name}_test_r2", test_r2)
			plot_predict_vs_test_zone(y_pred_test_zone, y_test_real_int, zone_name)
			#mlflow.log_artifact(plot_path, artifact_path="plots")
	
		# 6. Log the trained model
		mlflow.tensorflow.log_model(
			trained_model, 
			name="model", 
			input_example=x_test[:1],
		)
		run_id = run.info.run_id
		model_uri = f"runs:/{run_id}/model"  # points to the logged model

		try:
			result = mlflow.register_model(model_uri, "forklift-lstm")
			print(f"Model registered: {result.name}, version: {result.version}")
		except MlflowException as e:
			print(f"Failed to register model: {e}")
		client = MlflowClient()
		client.transition_model_version_stage(
			name="forklift-lstm",
			version=result.version,
			stage="Production",
			archive_existing_versions=True  # optional: archive old prod versions
		)
		context.log.info("Model logged to MLflow.")

		model_json = trained_model.to_json()
		with open("models/json_model.json", "w") as json_file:
			json_file.write(model_json)
		trained_model.save_weights("models/json_model.weights.h5")
		# create plot
		# plot_path = plot_predict_vs_test(y_pred_test_round, x_test, y_test_real_round, 1)
		# log to MLflow
		#mlflow.log_artifact(plot_path, artifact_path="plots")

		#context.log.info(f"Saved prediction plot to {plot_path}")

	context.log.info("MLflow run completed successfully.")
