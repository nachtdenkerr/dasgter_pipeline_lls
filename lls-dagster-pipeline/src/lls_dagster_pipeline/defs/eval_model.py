import sys
import numpy as np
import dagster as dg
import mlflow
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


@dg.asset(
	required_resource_keys={"lstm_config", "mlflow"},

)
def eval_model(context: dg.AssetExecutionContext, trained_model, x_train, y_train, x_test, y_test):
	context.resources.mlflow  # makes sure resource is initialized

	y_pred_train = trained_model.predict(x_train)
	y_pred_test = trained_model.predict(x_test)

	# Clip negative predictions to 0 (can't have negative equipment count)
	y_pred_train = np.clip(y_pred_train, 0, None)
	y_pred_test = np.clip(y_pred_test, 0, None)

	# Metrics
	train_mae = mean_absolute_error(y_train, y_pred_train)
	train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
	train_r2 = r2_score(y_train, y_pred_train)

	test_mae = mean_absolute_error(y_test, y_pred_test)
	test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
	test_r2 = r2_score(y_test, y_pred_test)

	sys.stdout.write(f"Train MAE: {train_mae}\n")
	sys.stdout.write(f"Train RMAE: {train_rmse}\n")
	sys.stdout.write(f"Train R2: {train_r2}\n")
	sys.stdout.write(f"Test MAE: {test_mae}\n")
	sys.stdout.write(f"Test MRMAEAE: {test_rmse}\n")
	sys.stdout.write(f"Test R2: {test_r2}\n")

	mlflow.log_metric("train_rmse", train_rmse)
	mlflow.log_metric("train_mae", train_mae)
	mlflow.log_metric("train_r2", train_r2)
	mlflow.log_metric("test_rmse", test_rmse)
	mlflow.log_metric("test_mae", test_mae)	
	mlflow.log_metric("test_r2", test_r2)

	context.log.info(f"RMSE: {test_rmse}, MAE: {test_mae}, R2: {test_r2}")
