import mmh3
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import LongType


@F.udf(returnType=LongType())
def hash_udf(x):
    return mmh3.hash64(str(x))[0]


def hash_split(
    data: DataFrame, col: str, test_size: float = 0.2
) -> tuple[DataFrame, DataFrame]:
    data = data.withColumn("hash", hash_udf(F.col(col)))

    # 80/20 split
    train_thr = data.approxQuantile(
        "hash", probabilities=[test_size], relativeError=0.01
    )[0]
    train = data.where(F.col("hash") >= train_thr).drop("hash")
    test = data.where(F.col("hash") < train_thr).drop("hash")

    return train, test


def ip_based_split(
    data: DataFrame, col: str, test_size: float = 0.2
) -> tuple[DataFrame, DataFrame]:
    # Get list of IPs with > 20% malicious activity
    bad_ips = (
        data.groupby("source_ip")
        .agg(F.avg(F.col("is_bad")).alias("bad_avg"))
        .where(F.col("bad_avg") > 0.2)
        .select("source_ip")
        .toPandas()
        .values.ravel()
    )
    bad_ips = list(bad_ips)
    print(bad_ips)

    data = data.withColumn("ip_hash", hash_udf(F.col("source_ip")))

    # Split good IPs
    good_df = data.where(~F.col("source_ip").isin(bad_ips))
    bad_df = data.where(F.col("source_ip").isin(bad_ips))
    print("Original Sizes")
    print("Good", good_df.count())
    print("Bad", bad_df.count())

    # 80/20 split
    good_train, good_test = hash_split(good_df, col, test_size)
    print("Good data", good_train.count(), good_test.count())
    bad_train, bad_test = hash_split(bad_df, col, test_size)
    print("Bad data", bad_train.count(), bad_test.count())

    train = good_train.union(bad_train)
    test = good_test.union(bad_test)

    return train, test
