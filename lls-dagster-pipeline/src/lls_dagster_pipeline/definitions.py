from dagster import Definitions, load_assets_from_package_module

from lls_dagster_pipeline import defs
from lls_dagster_pipeline.resources.lstm_config import lstm_config_resource
from lls_dagster_pipeline.resources.area_config import area_config_resource
from lls_dagster_pipeline.resources.vehicle_config import vehicle_config_resource
from lls_dagster_pipeline.io_managers.parquet_io_manager import local_parquet_io_manager

defs = Definitions(
    assets=[
		*load_assets_from_package_module(defs),
	],

    resources={
        "parquet_io_manager": local_parquet_io_manager,
		"area_config": area_config_resource,
        "vehicle_config": vehicle_config_resource,
		"lstm_config": lstm_config_resource
    },
)