"""Pipeline script to prepare and save data for modelling"""
import polars as pl
import yaml

from data_utils.feature_engineering import (
    add_period_features,
    add_rolling_features,
    basic_feature_engineering,
)
from data_utils.processing import clean_data, read_category_mappings
from data_utils.transfomation import create_target_df

# Configs
with open("pipe_config.yaml", "r") as file:
    pipe_config = yaml.safe_load(file)


date_column_format = pipe_config["date_column_format"]
ratios_config = pipe_config["ratio_features"]
diffs_config = pipe_config["difference_features"]
dates_config = pipe_config["date_features"]


def pipeline():
    """Pipeline that reads, cleans, and transofrms data into
    the format we need for modelling
    """
    days = [7, 30, 180]
    join_cols = ["video_id", "trending_date"]

    base_features = [
        "views",
        "likes",
        "dislikes",
        "comment_count",
        "comments_disabled",
        "ratings_disabled",
        "video_error_or_removed",
        "likes_to_dislikes",
        "likes_to_views",
        "comments_to_views",
        "trending_date_weekday",
        "channel_title",
        "tags",
        "description",
        "category_id",
    ]

    id_to_category = read_category_mappings(pipe_config["category_map_path"])
    col_mappings = {"category_id": id_to_category}

    output_data = (
        pl.read_csv(pipe_config["data_path"])
        .pipe(clean_data, date_column_format, col_mappings)
        .pipe(basic_feature_engineering, ratios_config, diffs_config, dates_config)
        .pipe(
            create_target_df,
            time_to_trending_thr=60,
            original_join_cols=join_cols,
            other_cols=base_features,
        )
        .pipe(add_rolling_features, "first_day_in_trending", days)
        .pipe(add_period_features, "first_day_in_trending", days)
    )

    return output_data


if __name__ == "__main__":
    pipeline().write_parquet("./data/modelling_data.parquet")
