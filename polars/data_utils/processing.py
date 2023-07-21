import json
from typing import Dict, List

import polars as pl


def read_category_mappings(path: str) -> Dict[int, str]:
    with open(path, "r") as f:
        categories = json.load(f)

    id_to_category = {}
    for c in categories["items"]:
        id_to_category[int(c["id"])] = c["snippet"]["title"]

    return id_to_category


def parse_dates(date_cols: Dict[str, str]) -> List[pl.Expr]:
    expressions = []
    for date_col, fmt in date_cols.items():
        expressions.append(pl.col(date_col).str.to_date(format=fmt))

    return expressions


def map_dict_columns(
    mapping_cols: Dict[str, Dict[str | int, str | int]]
) -> List[pl.Expr]:
    expressions = []
    for col, mapping in mapping_cols.items():
        expressions.append(pl.col(col).map_dict(mapping))
    return expressions


def clean_data(
    df: pl.LazyFrame,
    date_cols_config: Dict[str, str],
    mapping_cols_config: Dict[str, Dict[str | int, str | int]],
) -> pl.LazyFrame:
    parse_dates_expressions = parse_dates(date_cols=date_cols_config)
    mapping_expressions = map_dict_columns(mapping_cols_config)

    df = df.with_columns(parse_dates_expressions + mapping_expressions)
    return df
