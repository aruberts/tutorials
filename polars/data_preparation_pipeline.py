"""Pipeline script to prepare and save data for modelling"""
import time

import polars as pl
import yaml

from data_utils.feature_engineering import (
    add_period_features,
    add_rolling_features,
    basic_feature_engineering,
)
from data_utils.processing import clean_data, read_category_mappings
from data_utils.transfomation import create_target_df


def pipeline():
    """Pipeline that reads, cleans, and transofrms data into
    the format we need for modelling
    """
    # Read and unwrap the config
    with open("pipe_config.yaml", "r") as file:
        pipe_config = yaml.safe_load(file)

    date_column_format = pipe_config["date_column_format"]
    ratios_config = pipe_config["ratio_features"]
    diffs_config = pipe_config["difference_features"]
    dates_config = pipe_config["date_features"]

    id_to_category = read_category_mappings(pipe_config["category_map_path"])
    col_mappings = {"category_id": id_to_category}

    output_data = (
        pl.scan_csv(pipe_config["data_path"])
        .pipe(clean_data, date_column_format, col_mappings)
        .pipe(basic_feature_engineering, ratios_config, diffs_config, dates_config)
        .pipe(
            create_target_df,
            time_to_trending_thr=pipe_config["max_time_to_trending"],
            original_join_cols=pipe_config["join_columns"],
            other_cols=pipe_config["base_columns"],
        )
        .pipe(
            add_rolling_features,
            "first_day_in_trending",
            pipe_config["aggregate_windows"],
        )
        .pipe(
            add_period_features,
            "first_day_in_trending",
            pipe_config["aggregate_windows"],
        )
    ).collect()

    return output_data


if __name__ == "__main__":
    t0 = time.time()
    output = pipeline()
    t1 = time.time()
    print("Pipeline took", t1 - t0, "seconds")
    print("Output shape", output.shape)
    print("Output columns:", output.columns)
    output.write_parquet("./data/modelling_data.parquet")
