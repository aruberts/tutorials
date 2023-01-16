import os

import click
import mlflow
import polars as pl


def process_nans(df: pl.DataFrame, drop_thr: float = 0.95) -> pl.DataFrame:
    for col in df.get_columns():
        nulls_prop = col.is_null().mean()
        print(f"{col.name} - {nulls_prop * 100}% missing")
        # drop if missing more than a threshold
        if nulls_prop >= drop_thr:
            print("Dropping", col.name)
            df = df.select([pl.exclude(col.name)])
        # If some values are missing
        elif nulls_prop > 0:
            print("Imputing", col.name)
            # If numeric, impute with median
            if col.is_numeric():
                fill_value = col.median()
            else:
                # Else, impute with mode
                fill_value = col.mode()
            df = df.select(
                [
                    # Exclude the original column
                    pl.exclude(col.name),
                    # Include the imputed one
                    pl.col(col.name).fill_null(value=fill_value),
                ]
            )

    return df

def drop_static(df:pl.DataFrame) -> pl.DataFrame:
    for col in df.get_columns():
        std = col.std()
        # drop if missing more than a threshold
        if std == 0:
            print("Dropping", col.name)
            df = df.select([pl.exclude(col.name)])
    
    return df


def train_val_test_split(df, test_size=0.2, val_size=0.2):
    df_train = df.filter(
        pl.col("month") < df['month'].quantile(0.8)
    )

    df_test = df.filter(
        pl.col("month") >= df['month'].quantile(0.8)
    )

    df_val = df_train.filter(
        pl.col("month") >= df_train['month'].quantile(0.8)
    )

    df_train = df_train.filter(
        pl.col("month") < df_train['month'].quantile(0.8)
    )

    return df_train, df_val, df_test



@click.command(
    help="""Given a path to raw csv file, this steps fills in missing data, 
    drops static columns, and splits the data into train/val/test sets"""
)
@click.option("--dset-path")
@click.option("--missing-thr", default=0.95)
def preprocess_data(dset_path, missing_thr):
    with mlflow.start_run(run_name='preprocess_data') as mlrun:
        df = pl.read_csv(dset_path)
        # Preprocess nulls
        df = process_nans(df, missing_thr)
        # Drop static
        df = drop_static(df)
        # Train/val/test split 
        train_df, val_df, test_df = train_val_test_split(df)
        # Save data
        split_destination_folder = './data/processed'
        if not os.path.exists(split_destination_folder):
            os.makedirs(split_destination_folder)

        train_df.write_parquet('./data/processed/train.parquet')
        val_df.write_parquet('./data/processed/validation.parquet')
        test_df.write_parquet('./data/processed/test.parquet')

        file_locations = {
            'train-data-dir': './data/processed/train.parquet',
            'val-data-dir': './data/processed/validation.parquet',
            'test-data-dir': './data/processed/test.parquet',
        }
        
        mlflow.log_params(file_locations)


if __name__ == "__main__":
    preprocess_data()