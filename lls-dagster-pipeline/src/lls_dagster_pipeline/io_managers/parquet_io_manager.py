import os
import pandas as pd
from dagster import IOManager, io_manager

class LocalParquetIOManager(IOManager):
    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def _get_path(self, context):
        # asset key becomes folder/name.parquet
        key = "/".join(context.asset_key.path)
        return os.path.join(self.base_path, f"{key}.parquet")

    def handle_output(self, context, obj):
        path = self._get_path(context)
        context.log.info(f"Saving to: {path}")
        obj.to_parquet(path, index=False)

    def load_input(self, context):
        upstream_output_context = context.upstream_output
        path = self._get_path(upstream_output_context)
        context.log.info(f"Loading from: {path}")
        return pd.read_parquet(path)

@io_manager
def local_parquet_io_manager(_):
    return LocalParquetIOManager(base_path="data/intermediate")
