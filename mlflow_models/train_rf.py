import warnings

import click
import mlflow
import pandas as pd
from category_encoders import WOEEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils.columns import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET
from utils.data_utils import load_raw_data
from utils.eval_utils import eval_and_log_metrics


@click.command(
    help="Trains RF Model"
    "The input is expected in csv format."
    "The model and its metrics are logged with mlflow."
)
@click.option("--max-depth", type=click.INT, default=5, help="Depth of the trees")
@click.option(
    "--max-features", type=click.FLOAT, default=0.1, help="Fraction of features to use"
)
@click.option(
    "--class-weight", type=click.STRING, default="balanced", help="Weight of labels"
)
@click.option(
    "--min-samples-leaf",
    type=click.INT,
    default=10,
    help="Minimum number of samples required to be at a leaf node.",
)
@click.argument("dset_name")
def run(dset_name, max_depth, max_features, class_weight, min_samples_leaf):
    warnings.filterwarnings("ignore")
    # Read data
    csv_loc = load_raw_data(dset_name, file_name="Base.csv")
    data = pd.read_csv(csv_loc)

    train, test = train_test_split(data, random_state=42)

    # The predicted column is "fraud_bool"
    train_x = train[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    train_y = train[[TARGET]]

    test_x = test[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    test_y = test[[TARGET]]

    with mlflow.start_run():
        rf_params = {
            "max_depth": max_depth,
            "max_features": max_features,
            "class_weight": class_weight if class_weight != 'None' else None,
            "min_samples_leaf": min_samples_leaf,
        }
        # Define model
        rf = RandomForestClassifier(**rf_params)
        # Define transform
        transformer = ColumnTransformer(
            transformers=[("categorical", WOEEncoder(), CATEGORICAL_FEATURES)],
            remainder='passthrough'
        )
        # Define pipeline
        pipeline = Pipeline(steps=[("prep", transformer), ("model", rf)])
        pipeline.fit(train_x, train_y)
        test_preds = pipeline.predict_proba(test_x)
        eval_and_log_metrics("test", test_y, test_preds[:, 1])
        mlflow.sklearn.log_model(pipeline, "sklearn_models", pyfunc_predict_fn='predict_proba')


if __name__ == "__main__":
   run()
