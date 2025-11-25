from dagster import Definitions, load_assets_from_package_module, define_asset_job, AssetSelection

from lls_dagster_pipeline import defs as asset_pkg
from lls_dagster_pipeline.resources.lstm_config import lstm_config_resource
from lls_dagster_pipeline.resources.area_config import area_config_resource
from lls_dagster_pipeline.resources.vehicle_config import vehicle_config_resource
from lls_dagster_pipeline.io_managers.parquet_io_manager import local_parquet_io_manager
from lls_dagster_pipeline.io_managers.numpy_io_manager import numpy_io_manager
from lls_dagster_pipeline.resources.mlflow_resource import MLflowResource
from lls_dagster_pipeline.defs.preprocess import s1_read_csv as raw_data
from lls_dagster_pipeline.sensors import new_batch_sensor
# This job runs the full pipeline
incoming_data_job = define_asset_job(
    "incoming_data_job",
    selection="*",
)

defs = Definitions(
    assets=[
		*load_assets_from_package_module(asset_pkg),
	],
    jobs=[incoming_data_job],
    sensors=[ new_batch_sensor ],   # ← add this
    resources={
        "parquet_io_manager": local_parquet_io_manager,
		"area_config": area_config_resource,
        "vehicle_config": vehicle_config_resource,
		"lstm_config": lstm_config_resource,
		"numpy_io_manager": numpy_io_manager,
        "mlflow": MLflowResource(
            tracking_uri="file:./mlruns",   # local MLflow directory
            experiment_name="forklift-prediction"
        ),
    },
)