import yaml
from dagster import resource

@resource
def lstm_config_resource(context):
    config_path = "src/lls_dagster_pipeline/config/lstm.yaml"

    # Load YAML
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    lstm_params = yaml_data["lstm_params"]

    context.log.info(f"Loaded vehicle categories: {list(lstm_params.keys())}")

    return lstm_params
