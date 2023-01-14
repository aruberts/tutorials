import os
import zipfile

import click
import kaggle
import mlflow


@click.command(
    help="Downloads a fraud dataset and saves it as an MLlow artifact called 'fraud-csv-dir'."
)
@click.option("--dset-name", default="sgpjesus/bank-account-fraud-dataset-neurips-2022")
def load_raw_data(dset_name):
    with mlflow.start_run(run_name='download_data') as mlrun:
        zip_destination_folder = "./data/"
        raw_destination_folder = os.path.join(zip_destination_folder, "raw")

        # Check if the Kaggle API key was created
        if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
            raise Exception(
                "Kaggle API key not found. Make sure to follow the instructions to set up your Kaggle API key."
            )

        # Download the dataset into a current folder
        kaggle.api.dataset_download_files(
            "sgpjesus/bank-account-fraud-dataset-neurips-2022",
            path=zip_destination_folder,
        )

        # Check if the destination folder exists, and create it if it does not
        if not os.path.exists(raw_destination_folder):
            os.makedirs(raw_destination_folder)

        # Open the zip file in read mode
        zip_name = os.path.join(
            zip_destination_folder, "bank-account-fraud-dataset-neurips-2022.zip"
        )
        with zipfile.ZipFile(zip_name, "r") as zip_ref:
            # Extract all the files to the destination folder
            zip_ref.extractall(raw_destination_folder)

        # TODO: make file name a param as well
        csv_location = os.path.join(raw_destination_folder, "Base.csv")
        # Save location of raw data 
        mlflow.log_param("raw-data-dir", csv_location)


if __name__ == "__main__":
    load_raw_data()
