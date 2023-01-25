import os
import zipfile

import kaggle


def load_raw_data(dset_name):
    zip_destination_folder = "./data/"
    raw_destination_folder = os.path.join(zip_destination_folder, "raw")

    # Check if the Kaggle API key was created
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        raise Exception(
            "Kaggle API key not found. Make sure to follow the instructions to set up your Kaggle API key."
        )

    # Download the dataset into a current folder
    kaggle.api.dataset_download_files(
        dset_name,
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

    return csv_location