import pyspark.sql.functions as F
from pyspark.sql import Column, Window, WindowSpec


def mins_to_secs(mins: int) -> int:
    """Transforms minutes to seconds

    Args:
        mins (int): number of minutes to be transformed

    Returns:
        int: numeber of seconds
    """
    return mins * 60


def generate_window(
    window_in_minutes: int, partition_by: str, timestamp_col: str
) -> WindowSpec:
    """Generates window expressions for PySpark

    Args:
        window_in_minutes (int): Number of minutes you want in the rolling window
        partition_by (str): Column to partition by e.g. IP or user account
        timestamp_col (str): Column with timestamp data type

    Returns:
        _type_: _description_
    """
    window = (
        Window()
        .partitionBy(F.col(partition_by))
        .orderBy(F.col(timestamp_col).cast("long"))
        .rangeBetween(-mins_to_secs(window_in_minutes), -1)
    )

    return window


def generate_rolling_aggregate(
    col: str,
    partition_by: str | None = None,
    operation: str = "count",
    timestamp_col: str = "dt",
    window_in_minutes: int = 1,
) -> Column:
    """Rolling aggregate experession constructor

    Args:
        col (str): Name of column to aggregate
        partition_by (str | None, optional): Column to partition by. Defaults to None.
        operation (str, optional): What type of aggregation should be done. Defaults to "count".
        timestamp_col (str, optional): Timestamp column in your PySpark DF. Defaults to "dt".
        window_in_minutes (int, optional): Number of minutes for the window. Defaults to 1.

    Raises:
        ValueError: _description_

    Returns:
        Column: _description_
    """
    if partition_by is None:
        partition_by = col

    match operation:
        case "count":
            return F.count(col).over(
                generate_window(
                    window_in_minutes=window_in_minutes,
                    partition_by=col,
                    timestamp_col=timestamp_col,
                )
            )
        case "sum":
            return F.sum(col).over(
                generate_window(
                    window_in_minutes=window_in_minutes,
                    partition_by=col,
                    timestamp_col=timestamp_col,
                )
            )
        case "avg":
            return F.avg(col).over(
                generate_window(
                    window_in_minutes=window_in_minutes,
                    partition_by=col,
                    timestamp_col=timestamp_col,
                )
            )
        case _:
            raise ValueError(f"Operation {operation} is not defined")
