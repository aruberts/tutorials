import pyspark.sql.functions as F
from pyspark.sql import DataFrame


def get_static(data: DataFrame, cols_to_analyse: list[str]) -> list[str]:
    """Return the list of static columns

    Args:
        data (DataFrame): input PySpark dataframe
        cols_to_analyse (list[str]): list of columns to analyse

    Returns:
        list[str]: list of static columns
    """
    unique_counts = data.agg(
        *(F.countDistinct(F.col(c)).alias(c) for c in cols_to_analyse)
    ).first()
    static_cols = [c for c in unique_counts.asDict() if unique_counts[c] == 1]
    print("Static columns:", static_cols)
    return static_cols


def remove_rare_categories(
    data: DataFrame, columns: list[str], min_count: int = 100
) -> DataFrame:
    """Removes rare categories in categorical features by substituting
    them with 'Other'

    Args:
        data (DataFrame): input PySpark dataframe
        columns (list[str]): list of categorical features to process
        min_count (int, optional): minimum number of times for category
        to appear to not be considered rare. Defaults to 100.

    Returns:
        DataFrame: processed PySpark dataframe
    """
    categorical_valid_values = {}

    for c in columns:
        # Find frequent values
        categorical_valid_values[c] = (
            data.groupby(c)
            .count()
            .filter(F.col("count") > min_count)
            .select(c)
            .toPandas()
            .values.ravel()
        )

        data = data.withColumn(
            c,
            F.when(
                F.col(c).isin(list(categorical_valid_values[c])), F.col(c)
            ).otherwise(F.lit("Other").alias(c)),
        )

    return data
