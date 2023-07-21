from typing import Dict, List

import polars as pl


def ratio_features(features_config: Dict[str, List[str]]) -> List[pl.Expr]:
    expressions = []
    for name, cols in features_config.items():
        expressions.append((pl.col(cols[0]) / pl.col(cols[1])).alias(name))

    return expressions


def diff_features(features_config: Dict[str, List[str]]) -> List[pl.Expr]:
    expressions = []
    for name, cols in features_config.items():
        expressions.append((pl.col(cols[0]) - pl.col(cols[1])).alias(name))

    return expressions


def date_features(features_config: Dict[str, List[str]]) -> List[pl.Expr]:
    expressions = []
    for col, features in features_config.items():
        if "weekday" in features:
            expressions.append(pl.col(col).dt.weekday().alias(f"{col}_weekday"))
        if "month" in features:
            expressions.append(pl.col(col).dt.month().alias(f"{col}_month"))
        if "year" in features:
            expressions.append(pl.col(col).dt.year().alias(f"{col}_year"))

    return expressions


def basic_feature_engineering(
    data: pl.LazyFrame,
    ratios_config: Dict[str, List[str]],
    diffs_config: Dict[str, List[str]],
    dates_config: Dict[str, List[str]],
) -> pl.LazyFrame:
    ratio_expressions = ratio_features(ratios_config)
    date_diff_expressions = diff_features(diffs_config)
    date_expressions = date_features(dates_config)

    data = data.with_columns(
        ratio_expressions + date_diff_expressions + date_expressions
    )
    return data


def build_channel_rolling(df: pl.LazyFrame, date_col: str, period: int) -> pl.LazyFrame:
    channel_aggs = (
        df.sort(date_col)
        .groupby_rolling(
            index_column=date_col,
            period=f"{period}d",
            by="channel_title",
            closed="left",  #
        )
        .agg(
            pl.col("video_id")
            .n_unique()
            .alias(f"channel_num_trending_videos_last_{period}_days"),
            pl.col("days_in_trending")
            .max()
            .alias(f"channel_max_days_in_trending_{period}_days"),
            pl.col("days_in_trending")
            .mean()
            .alias(f"channel_avg_days_in_trending_{period}_days"),
        )
        .fill_null(0)
    )

    return channel_aggs


def add_rolling_features(
    df: pl.LazyFrame, date_col: str, periods: List[int]
) -> pl.LazyFrame:
    for period in periods:
        rolling_features = build_channel_rolling(df, date_col, period)
        df = df.join(rolling_features, on=["channel_title", "first_day_in_trending"])

    return df


def build_period_features(df: pl.LazyFrame, date_col: str, period: int) -> pl.LazyFrame:
    general_aggs = (
        df.sort(date_col)
        .groupby_dynamic(
            index_column=date_col,
            every="1d",
            period=f"{period}d",
            closed="left",
        )
        .agg(
            pl.col("video_id")
            .n_unique()
            .alias(f"general_num_trending_videos_last_{period}_days"),
            pl.col("days_in_trending")
            .max()
            .alias(f"general_max_days_in_trending_{period}_days"),
            pl.col("days_in_trending")
            .mean()
            .alias(f"general_avg_days_in_trending_{period}_days"),
        )
        .with_columns(
            # shift match values with previous period
            pl.col(f"general_num_trending_videos_last_{period}_days").shift(period),
            pl.col(f"general_max_days_in_trending_{period}_days").shift(period),
            pl.col(f"general_avg_days_in_trending_{period}_days").shift(period),
        )
        .fill_null(0)
    )

    return general_aggs


def add_period_features(
    df: pl.LazyFrame, date_col: str, periods: List[int]
) -> pl.LazyFrame:
    for period in periods:
        rolling_features = build_period_features(df, date_col, period)
        df = df.join(rolling_features, on=["first_day_in_trending"])

    return df
