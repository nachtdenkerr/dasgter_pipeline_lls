import mlflow
from dagster import ConfigurableResource

class MLflowResource(ConfigurableResource):
    tracking_uri: str
    experiment_name: str = "default"

    def setup_for_execution(self, context):
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        context.log.info(f"MLflow tracking at: {self.tracking_uri}")
        return self
