from dagster import sensor, SkipReason, RunRequest, DefaultSensorStatus
import os

DATASET_DIR = "data/incoming"

@sensor(
    job_name="incoming_data_job",
    default_status=DefaultSensorStatus.RUNNING,
)
def new_batch_sensor(context):
    """Detects NEW batch folders in data/incoming."""
    
     # defensive: ensure folder exists
    if not os.path.exists(DATASET_DIR):
        context.log.warning(f"Incoming dir does not exist: {DATASET_DIR}")
        yield SkipReason("Incoming dir not found.")
        return

    # list only directories
    try:
        batches = sorted(
            [
                d
                for d in os.listdir(DATASET_DIR)
                if os.path.isdir(os.path.join(DATASET_DIR, d))
            ]
        )
    except Exception as exc:
        context.log.error(f"Error listing {DATASET_DIR}: {exc}")
        yield SkipReason("Failed to list incoming dir.")
        return

    last_processed = context.cursor or ""
    # select batches strictly greater than cursor (lexicographic)
    new_batches = [b for b in batches if b > last_processed]

    if not new_batches:
        #context.log.debug("No new batch found.")
        #yield SkipReason("No new batch found in data/incoming.")
        return

    # take first new batch (oldest unprocessed)
    next_batch = new_batches[0]
    context.log.info(f"Detected new batch: {next_batch}")


    yield RunRequest(
        run_key=next_batch,
        run_config={
            "ops": {
                "raw_data": {
                    "config": {"batch_name": next_batch}
                },
                "archive_batch": {"config": {"batch_name": next_batch}},
            },
        }
    )

    context.update_cursor(next_batch)