import os
import numpy as np
import pandas as pd
from dagster import IOManager, io_manager


class NumpyArrayIOManager(IOManager):
    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def _get_path(self, context):
        key = "/".join(context.asset_key.path)
        return os.path.join(self.base_path, f"{key}.npy")

    def handle_output(self, context, obj):
        # obj is expected to be a numpy array
        path = self._get_path(context)
        context.log.info(f"Saving numpy array to: {path}")

        if not isinstance(obj, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(obj)}")

        np.save(path, obj)

    def load_input(self, context):
        # load upstream output which may be DataFrame or numpy
        upstream_output_context = context.upstream_output
        path = self._get_path(upstream_output_context)

        context.log.info(f"Loading data from: {path}")

        # Detect type by extension
        if path.endswith(".npy"):
            return np.load(path)
        else:
            # fallback for DF loaded by another IOManager
            import pandas as pd
            return pd.read_parquet(path)


@io_manager
def numpy_io_manager(_):
    return NumpyArrayIOManager(base_path="data/intermediate")
