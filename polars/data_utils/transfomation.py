from typing import Dict, List

import polars as pl


def join_original_features(
    main: pl.LazyFrame,
    original: pl.LazyFrame,
    main_join_cols: List[str],
    original_join_cols: List[str],
    other_cols: List[str],
) -> pl.LazyFrame:
    original_features = original.select(original_join_cols + other_cols).unique(
        original_join_cols
    )  # unique ensures one row per video + date
    main = main.join(
        original_features,
        left_on=main_join_cols,
        right_on=original_join_cols,
        how="left",
    )

    return main


def create_target_df(
    df: pl.LazyFrame,
    time_to_trending_thr: int,
    original_join_cols: List[str],
    other_cols: List[str],
) -> pl.LazyFrame:
    # Create a DF with video ID per row and corresponding days to trending and days in trending (target)
    target = (
        df.groupby(["video_id"])
        .agg(
            pl.col("days_to_trending").min().dt.days(),
            pl.col("trending_date").min().dt.date().alias("first_day_in_trending"),
            pl.col("trending_date").max().dt.date().alias("last_day_in_trending"),
            (pl.col("trending_date").max() - pl.col("trending_date").min())
            .dt.days()
            .alias("days_in_trending"),
        )
        .filter(pl.col("days_to_trending") <= time_to_trending_thr)
    )

    # Join features to the aggregates
    target = join_original_features(
        main=target,
        original=df,
        main_join_cols=["video_id", "first_day_in_trending"],
        original_join_cols=original_join_cols,
        other_cols=other_cols,
    )

    return target
