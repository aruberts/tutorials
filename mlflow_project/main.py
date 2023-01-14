import click

import mlflow
from mlflow.tracking import MlflowClient


def run(entrypoint, parameters):
    print(
        "Launching new run for entrypoint={} and parameters={}".format(
            entrypoint, parameters
        )
    )
    submitted_run = mlflow.run(
        ".", entrypoint, parameters=parameters, env_manager="local"
    )
    return MlflowClient().get_run(submitted_run.run_id)


@click.command()
@click.option("--n-trials", default=10, type=int)
def pipeline(n_trials):
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run() as active_run:
        # Run the download_data task
        load_raw_data_run = run("download_data", {})
        # Extract the raw data location
        dset_path = load_raw_data_run.data.params["raw-data-dir"]
        # Run the preprocess_data task
        prep_data_run = run("preprocess_data", {"dset-path": dset_path})
        # Extract the train/val/test data locations
        train_path = prep_data_run.data.params["train-data-dir"]
        val_path = prep_data_run.data.params["val-data-dir"]
        test_path = prep_data_run.data.params["test-data-dir"]
        # Do HP search
        model_tuning_run = run(
            "tune_model", {
                "parent-run": active_run.info.run_id,
                "train-path": train_path,
                "val-path": val_path,
                "n-trials": n_trials
            }
        )
        print(model_tuning_run.data.metrics)
        print(model_tuning_run.data.params)


if __name__ == "__main__":
    pipeline()
