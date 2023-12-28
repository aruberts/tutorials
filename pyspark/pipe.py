import pyspark.sql.functions as F
import yaml
from hyperopt import hp
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import SparkSession

from cleaning import get_static, remove_rare_categories
from feature_engineering import generate_rolling_aggregate
from ml_prep import ip_based_split
from tuning import tune_rf

with open("gcs_config.yaml", "r") as file:
    conf = yaml.safe_load(file)

numerical_features: list[str] = conf["numerical_features"]
categorical_features: list[str] = conf["categorical_features"]

spark = SparkSession.builder.appName("LocalTest").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Read in and do some basic processing
df = (
    spark.read.option("delimiter", "|")
    .csv(conf["filepaths"], inferSchema=True, header=True)
    .withColumns(
        {
            "is_bad": F.when(F.col("label") != "Benign", 1).otherwise(0),
            "dt": F.to_timestamp(F.from_unixtime("ts")),
        }
    )
    .withColumnsRenamed(
        {
            "id.orig_h": "source_ip",
            "id.orig_p": "source_port",
            "id.resp_h": "dest_ip",
            "id.resp_p": "dest_port",
        }
    )
    .withColumns({n: F.col(n).cast("double") for n in numerical_features})
    .replace("-", None)
    .fillna(conf["na_fill_vals"])
)

# Find and drop static columns
static_numerical = get_static(df, numerical_features)
static_categorical = get_static(df, categorical_features)
numerical_features = [f for f in numerical_features if f not in static_numerical]
categorical_features = [f for f in categorical_features if f not in static_categorical]
categorical_features_indexed = [c + "_ind" for c in categorical_features]
input_features = numerical_features + categorical_features_indexed

# Process categorical
df = remove_rare_categories(
    df.drop(*static_numerical + static_categorical), categorical_features, min_count=100
)

# Feature engineering
df = df.withColumns(
    {
        "source_ip_count_last_min": generate_rolling_aggregate(
            col="source_ip", operation="count", timestamp_col="dt", window_in_minutes=1
        ),
        "source_ip_count_last_30_mins": generate_rolling_aggregate(
            col="source_ip", operation="count", timestamp_col="dt", window_in_minutes=30
        ),
        "source_port_count_last_min": generate_rolling_aggregate(
            col="source_port",
            operation="count",
            timestamp_col="dt",
            window_in_minutes=1,
        ),
        "source_port_count_last_30_mins": generate_rolling_aggregate(
            col="source_port",
            operation="count",
            timestamp_col="dt",
            window_in_minutes=30,
        ),
        "source_ip_avg_pkts_last_min": generate_rolling_aggregate(
            col="orig_pkts",
            partition_by="source_ip",
            operation="avg",
            timestamp_col="dt",
            window_in_minutes=1,
        ),
        "source_ip_avg_pkts_last_30_mins": generate_rolling_aggregate(
            col="orig_pkts",
            partition_by="source_ip",
            operation="avg",
            timestamp_col="dt",
            window_in_minutes=30,
        ),
        "source_ip_avg_bytes_last_min": generate_rolling_aggregate(
            col="orig_ip_bytes",
            partition_by="source_ip",
            operation="avg",
            timestamp_col="dt",
            window_in_minutes=1,
        ),
        "source_ip_avg_bytes_last_30_mins": generate_rolling_aggregate(
            col="orig_ip_bytes",
            partition_by="source_ip",
            operation="avg",
            timestamp_col="dt",
            window_in_minutes=30,
        ),
    }
)

if conf["random_split"]:
    df_train, df_test = df.randomSplit(weights=[0.8, 0.2], seed=200)
else:
    df_train, df_test = ip_based_split(df, "source_ip", 0.2)


df_train, df_val = df_train.randomSplit(weights=[0.8, 0.2], seed=200)

search_space = {
    "numTrees": hp.uniformint("numTrees", 10, 500),
    "maxDepth": hp.uniformint("maxDepth", 2, 10),
}

roc = BinaryClassificationEvaluator(labelCol="is_bad", metricName="areaUnderROC")

ind = StringIndexer(
    inputCols=categorical_features,
    outputCols=categorical_features_indexed,
    handleInvalid="skip",
)
va = VectorAssembler(
    inputCols=input_features, outputCol="features", handleInvalid="skip"
)

if conf["tuning_rounds"] > 0:
    print("Tuning the model for {conf['tuning_rounds']} round")
    best_params = tune_rf(
        train=df_train,
        val=df_val,
        string_indexer=ind,
        vector_assembler=va,
        evaluator=roc,
        param_grid=search_space,
        tuning_rounds=conf["tuning_rounds"],
    )
else:
    print("Skipping the tuning...")
    best_params = {"numTrees": 10, "maxDepth": 4}

best_rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="is_bad",
    numTrees=best_params["numTrees"],
    maxDepth=best_params["maxDepth"],
)

best_pipeline = Pipeline(stages=[ind, va, best_rf])
best_pipeline = best_pipeline.fit(df_train)
test_preds = best_pipeline.transform(df_test)

score = roc.evaluate(test_preds)
print("ROC AUC", score)
best_pipeline.save(conf["model_output_path"])
