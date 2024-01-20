import plotly.express as px
import plotly.graph_objects as go

import polars as pl


def bar_plot(data: pl.DataFrame, column: str, title: str) -> go.Figure:
    """Creates a plotly barplot from Polars column

    Args:
        data (pl.DataFrame): input dataframe
        column (str): column to plot
        title (str): title for the plot

    Returns:
        go.Figure: resulting barplot as plotly Figure
    """
    counts = data[column].value_counts(sort=True)
    fig = px.bar(
        x=counts[column].to_list(),
        y=counts["counts"].to_list(),
        text_auto=True,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Antique,
        labels={
            "x": column,
            "y": "Counts",
        },
    )
    fig.update_traces(
        textfont_size=12, textangle=0, textposition="outside", cliponaxis=False
    )

    return fig


def proportion_plot(
    data: pl.DataFrame, column: str, target: str, title: str
) -> go.Figure:
    """Creates a plotly barplot with proportions

    Args:
        data (pl.DataFrame): input dataframe
        column (str): column to analyse
        target (str): a discrete target
        title (str): title for the plot

    Returns:
        go.Figure: resulting barplot as plotly Figure
    """
    counts = data.groupby(column, target).agg(pl.count())
    target_counts = counts.groupby(column).agg(pl.col("count").sum().alias("total"))
    proportions = counts.join(target_counts, on=column)
    proportions = proportions.with_columns(
        proportion=pl.col("count") / pl.col("total")
    ).sort((column, target))
    fig = px.bar(
        x=proportions[column].to_list(),
        y=proportions["proportion"].to_list(),
        color=proportions[target].to_list(),
        color_discrete_sequence=px.colors.qualitative.Antique,
        labels={
            "x": column,
            "y": f"{target} proportion",
        },
        title=title,
    )
    fig.update_traces(
        textfont_size=12, textangle=0, textposition="outside", cliponaxis=False
    )

    return fig


def boxplot_by_bin_with_target(
    data: pl.DataFrame,
    column_to_bin: str,
    numeric_column: str,
    target: str,
    number_bins: int = 10,
) -> go.Figure:
    """Creates a plotly boxplot

    Args:
        data (pl.DataFrame): input dataframe
        column_to_bin (str): numeric column to bin
        numeric_column (str): numeric column to create a box plot from
        target (str): target column to colour a boxplot
        number_bins (int, optional): number of quantile bins to create. Defaults to 10.

    Returns:
        go.Figure: _description_
    """

    temp = data.select(
        pl.col(column_to_bin)
        .qcut(number_bins, allow_duplicates=True)
        .alias(f"{column_to_bin}_binned"),
        pl.col(column_to_bin),
        pl.col(numeric_column),
        pl.col(target),
    )

    order = (
        temp.groupby(f"{column_to_bin}_binned")
        .agg(pl.col(column_to_bin).min().alias("min"))
        .sort("min")[f"{column_to_bin}_binned"]
        .to_list()
    )

    fig = px.box(
        x=temp[f"{column_to_bin}_binned"].to_list(),
        y=temp[numeric_column].to_list(),
        color=temp[target].to_list(),
        color_discrete_sequence=px.colors.qualitative.Antique,
        log_y=True,
        category_orders={"x": order},
        labels={
            "x": "",
            "y": numeric_column,
        },
    )

    return fig
