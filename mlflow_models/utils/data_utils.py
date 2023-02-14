import os
import zipfile

import kaggle


def load_raw_data(dset_name: str, file_name: str,):
    """Downloads and unpacks Kaggle data

    Args:
        dset_name (str, optional): name of kaggle dataset.
            Follows the format - username/dataset-name.
            For example - "sgpjesus/bank-account-fraud-dataset-neurips-2022".
        file_name (str, optional): name of the extracted file.
            Should be specified in case there are many files in the zip archive

    Raises:
        Exception: if kaggle API was not setup

    Returns:
        str: location of the downlaoded and extracted csv file
    """
    zip_destination_folder = "./data/"
    raw_destination_folder = os.path.join(zip_destination_folder, "raw")

    # Check if the Kaggle API key was created
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        raise Exception(
            """
            Kaggle API key not found.
            Make sure to follow the instructions to set up your Kaggle API key.
            """
        )

    # Download the dataset into a current folder
    kaggle.api.dataset_download_files(dset_name, path=zip_destination_folder)

    # Check if the destination folder exists, and create it if it does not
    if not os.path.exists(raw_destination_folder):
        os.makedirs(raw_destination_folder)

    # Open the zip file in read mode
    zip_name = os.path.join(
        zip_destination_folder,
        f"{dset_name.split('/')[1]}.zip"
    )
    with zipfile.ZipFile(zip_name, "r") as zip_ref:
        # Extract all the files to the destination folder
        zip_ref.extractall(raw_destination_folder)

    csv_location = os.path.join(raw_destination_folder, file_name)

    return csv_location
