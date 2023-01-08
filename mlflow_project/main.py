import click
import os

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint

from mlflow.tracking.fluent import _get_experiment_id


def run(entrypoint, parameters):
    print("Launching new run for entrypoint={} and parameters={}".format(entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters, env_manager="local")
    return MlflowClient().get_run(submitted_run.run_id)

@click.command()
def pipeline():
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run() as active_run:
        # Run the download_data task
        load_raw_data_run = run("download_data", {})
        # Extract the raw data location
        dset_path = load_raw_data_run.data.params['raw-data-dir']
        # Run the preprocess_data task
        prep_data_run = run("preprocess_data", {"dset-path": dset_path})
        # Extract the train/val/test data locations
        train_path = prep_data_run.data.params['train-data-dir']
        val_path = prep_data_run.data.params['val-data-dir']
        test_path = prep_data_run.data.params['test-data-dir']
        print(train_path, val_path, test_path)

if __name__ == "__main__":
    pipeline()