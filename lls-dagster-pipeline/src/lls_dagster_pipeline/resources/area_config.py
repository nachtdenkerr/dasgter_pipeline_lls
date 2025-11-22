import yaml
from dagster import resource
from shapely.geometry import Polygon

@resource
def area_config_resource(context):
    config_path = "src/lls_dagster_pipeline/config/areas.yaml"

    # Load YAML
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    area_coords = yaml_data["areas"]

    # Convert each area to a Shapely Polygon
    polygons = {
        area_name: Polygon(coords)
        for area_name, coords in area_coords.items()
    }

    context.log.info(f"Loaded {len(polygons)} area polygons from {config_path}")

    return polygons
