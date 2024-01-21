import polars as pl


def count_missing(data: pl.DataFrame) -> pl.DataFrame:
    """Return a polars dataframe with missing counts per columns

    Args:
        data (pl.DataFrame): input dataframe to be analysed

    Returns:
        pl.DataFrame: dataframe with missing counts
    """
    missing = data.select(
        pl.col(c).is_null().sum().alias(f"{c}_missing") for c in data.columns
    )

    return missing
