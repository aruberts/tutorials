import numpy as np
import plotly.express as px
import ppscore as pps

import polars as pl


def aggregate_node_features(
    data: pl.DataFrame, node_features: list[str], by: str = "_id"
) -> pl.DataFrame:
    aggs = []
    for f in node_features:
        avg = pl.col(f).mean().alias(f"avg_{f}")
        min = pl.col(f).min().alias(f"min_{f}")
        max = pl.col(f).max().alias(f"max_{f}")
        std = pl.col(f).std().alias(f"std_{f}")
        aggs += [avg, min, max, std]
    agg_data = data.groupby(by).agg(aggs)

    return agg_data


def feature_predictive_power(
    data: pl.DataFrame, x: str, y: str, plot: bool = True
) -> np.float32:
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
