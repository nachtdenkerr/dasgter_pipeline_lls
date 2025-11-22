import yaml
from dagster import resource

@resource
def vehicle_config_resource(context):
    config_path = "src/lls_dagster_pipeline/config/vehicles.yaml"

    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    vehicles = yaml_data["vehicles"]

    context.log.info(f"Loaded vehicle categories: {list(vehicles.keys())}")

    return vehicles
