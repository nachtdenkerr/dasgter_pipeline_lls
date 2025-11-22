import sys
import numpy as np
import dagster as dg
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


@dg.asset
def eval_model(trained_model, x_train, y_train, x_test, y_test):
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
	sys.stdout.write(f"Train MAE: {train_mae}")
	sys.stdout.write(f"Train RMAE: {train_rmse}")
	sys.stdout.write(f"Train R2: {train_r2}")
	sys.stdout.write(f"Test MAE: {test_mae}")
	sys.stdout.write(f"Test MRMAEAE: {test_rmse}")
	sys.stdout.write(f"Test R2: {test_r2}")
