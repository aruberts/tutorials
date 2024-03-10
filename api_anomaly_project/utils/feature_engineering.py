import numpy as np
import plotly.express as px
import polars as pl
import ppscore as pps


def aggregate_node_features(
    data: pl.DataFrame, node_features: list[str], by: str = "_id"
) -> pl.DataFrame:
    """Utility function to generate basic aggregation statistics features for node level features

    Args:
        data (pl.DataFrame): input dataframe
        node_features (list[str]): list of node features to aggregate
        by (str, optional): the graph ID column. Defaults to "_id".

    Returns:
        pl.DataFrame: dataframe with aggregated features
    """
    aggs = []
    for f in node_features:
        avg = pl.col(f).mean().alias(f"avg_{f}")
        min_val = pl.col(f).min().alias(f"min_{f}")
        max_val = pl.col(f).max().alias(f"max_{f}")
        std = pl.col(f).std().alias(f"std_{f}")
        aggs += [avg, min_val, max_val, std]
    agg_data = data.group_by(by).agg(aggs)

    return agg_data


def feature_predictive_power(
    data: pl.DataFrame, x: str, y: str, plot: bool = True
) -> np.float32:
    """Utility to calcualte predictive power of a feature and plot its relationship with the target
    Args:
        data (pl.DataFrame): input dataframe
        x (str): name of the feature
        y (str): name of the target
        plot (bool, optional): indicator whether you want to plot the relationship. Defaults to True.

    Returns:
        np.float32: predictive power score
    """
    data_pd = data.select([x, y]).to_pandas()
    score = np.float32(pps.score(data_pd, x, y)["ppscore"]).round(4)

    if plot:
        print(f"Predictive Power Score: {score}")
        fig = px.histogram(
            x=data_pd[x],
            color=data_pd[y],
            marginal="box",
            histnorm="probability",
            title=f"{x} distribution by {y}",
        )
        fig.show()

    return score


def get_graph_features(data: pl.DataFrame, node_features: bool = True) -> pl.DataFrame:
    """Pipeline function to generate graph features

    Args:
        data (pl.DataFrame): dataframe with edges 'from' and 'to'
        node_features (bool, optional): Indicator whether you want to create node level features. Defaults to True.

    Returns:
        pl.DataFrame: dataframe with engineered features
    """
    graph_features = (
        data.groupby("_id")
        .agg(pl.count().alias("n_connections"), pl.col("from"), pl.col("to"))
        .with_columns(
            pl.concat_list("from", "to")
            .list.unique()
            .list.lengths()
            .alias("n_unique_nodes")
        )
        .select(["_id", "n_connections", "n_unique_nodes"])
    )

    if node_features:
        node_features_agg = aggregate_node_features(
            data,
            node_features=[
                "global_source_degrees",
                "global_dest_degrees",
                "local_source_degrees",
                "local_dest_degrees",
            ],
            by="_id",
        )

        graph_features = graph_features.join(node_features_agg, on="_id")

    return graph_features
