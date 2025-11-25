import sys
import mlflow
import numpy as np
import dagster as dg
from mlflow.models import infer_signature
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


@dg.asset(
	required_resource_keys={"lstm_config", "mlflow"},

)
def eval_model(context: dg.AssetExecutionContext, trained_model, training_history, x_train, y_train, x_test, y_test):
	"""
	Evaluate model and log everything to MLflow.
	"""
	context.resources.mlflow  # ensure resource is initialized
	lstm_params = context.resources.lstm_config	
	with mlflow.start_run(run_name="lstm_evaluation"):
		context.log.info("Starting MLflow run for model evaluation...")
		mlflow.log_param("lstm_window_size", lstm_params['LSTM_WINDOW_SIZE'])
		mlflow.log_param("epochs", lstm_params['LSTM_EPOCHS'])
		mlflow.log_param("batch_size", lstm_params['LSTM_BATCH_SIZE'])
		mlflow.log_param("patience", lstm_params['LSTM_PATIENCE'])
		mlflow.log_param("validation_split", lstm_params['validation_split'])
		mlflow.log_param("test_size", lstm_params['test_size'])

		# 2. Log training history metrics
		if hasattr(training_history, 'history'):
			# Log final training metrics
			final_train_loss = training_history.history['loss'][-1]
			final_val_loss = training_history.history['val_loss'][-1]
			final_train_mae = training_history.history['mae'][-1]
			final_val_mae = training_history.history['val_mae'][-1]
			
			mlflow.log_metric("final_train_loss", final_train_loss)
			mlflow.log_metric("final_val_loss", final_val_loss)
			mlflow.log_metric("final_train_mae", final_train_mae)
			mlflow.log_metric("final_val_mae", final_val_mae)
			
			context.log.info(f"Training metrics - Loss: {final_train_loss:.4f}, MAE: {final_train_mae:.4f}")
			context.log.info(f"Validation metrics - Loss: {final_val_loss:.4f}, MAE: {final_val_mae:.4f}")
			
		# 3. Make predictions
		y_pred_train = trained_model.predict(x_train)
		y_pred_test = trained_model.predict(x_test)

		# Clip negative predictions to 0 (can't have negative equipment count)
		y_pred_train = np.clip(y_pred_train, 0, None)
		y_pred_test = np.clip(y_pred_test, 0, None)

		# Calculate evaluation metrics
		train_mae = mean_absolute_error(y_train, y_pred_train)
		train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
		train_r2 = r2_score(y_train, y_pred_train)
		#train_da = calculate_direction_accuracy(y_train, y_pred_train)

		test_mae = mean_absolute_error(y_test, y_pred_test)
		test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
		test_r2 = r2_score(y_test, y_pred_test)
		#test_da = calculate_direction_accuracy(y_test, y_pred_test)

		# 5. Log evaluation metrics to MLflow
		mlflow.log_metric("train_rmse", train_rmse)
		mlflow.log_metric("train_mae", train_mae)
		mlflow.log_metric("train_r2", train_r2)
		#mlflow.log_metric("train_direction_accuracy", train_da)
		mlflow.log_metric("test_rmse", test_rmse)
		mlflow.log_metric("test_mae", test_mae)	
		mlflow.log_metric("test_r2", test_r2)
		#mlflow.log_metric("test_direction_accuracy", test_da)

		# 6. Log the trained model
		mlflow.tensorflow.log_model(trained_model, artifact_path="model")
		context.log.info("Model logged to MLflow.")

		# 7. Log to console
		context.log.info("=" * 60)
		context.log.info("TRAIN METRICS:")
		context.log.info(f"  MAE:  {train_mae:.4f}")
		context.log.info(f"  RMSE: {train_rmse:.4f}")
		context.log.info(f"  R²:   {train_r2:.4f}")
		#context.log.info(f"  Direction Accuracy: {train_da:.4f} ({train_da*100:.2f}%)")
		context.log.info("TEST METRICS:")
		context.log.info(f"  MAE:  {test_mae:.4f}")
		context.log.info(f"  RMSE: {test_rmse:.4f}")
		context.log.info(f"  R²:   {test_r2:.4f}")
		#context.log.info(f"  Direction Accuracy: {test_da:.4f} ({test_da*100:.2f}%)")
		context.log.info("=" * 60)

		# Also print to stdout for compatibility
		sys.stdout.write(f"Train MAE: {train_mae:.4f}\n")
		sys.stdout.write(f"Train RMSE: {train_rmse:.4f}\n")
		sys.stdout.write(f"Train R2: {train_r2:.4f}\n")
		#sys.stdout.write(f"Train Direction Accuracy: {train_da:.4f} ({train_da*100:.2f}%)\n")
		sys.stdout.write(f"Test MAE: {test_mae:.4f}\n")
		sys.stdout.write(f"Test RMSE: {test_rmse:.4f}\n")
		sys.stdout.write(f"Test R2: {test_r2:.4f}\n")
		#sys.stdout.write(f"Test Direction Accuracy: {test_da:.4f} ({test_da*100:.2f}%)\n")

		#signature = infer_signature(x_test, y_pred_test)

		#mlflow.tensorflow.log_model(
		#	trained_model,
		#	artifact_path="lstm_model",
		#	signature=signature,
		#)
	context.log.info("MLflow run completed successfully.")
