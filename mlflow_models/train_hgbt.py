import warnings

import click
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from utils.columns import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET
from utils.data_utils import load_raw_data
from utils.eval_utils import eval_and_log_metrics


@click.command(
    help="Trains HGBT Model"
    "The input is expected in csv format."
    "The model and its metrics are logged with mlflow."
)
@click.option("--max-depth", type=click.INT, default=20, help="Depth of the trees")
@click.option(
    "--max-leaf-nodes",
    type=click.INT,
    default=31,
    help="The maximum number of leaves for each tree",
)
@click.option(
    "--class-weight", type=click.STRING, default="balanced", help="Weight of labels"
)
@click.option(
    "--l2-regularization",
    type=click.FLOAT,
    default=1.0,
    help="The L2 regularization parameter",
)
@click.option(
    "--learning-rate",
    type=click.FLOAT,
    default=0.1,
    help="The learning rate, also known as shrinkage",
)
@click.argument("dset_name")
def run(
    dset_name, max_depth, max_leaf_nodes, class_weight, l2_regularization, learning_rate
):
    """
    This function trains and logs an HistGradientBoostingClassifier model on a dataset.

    :param dset_name: The name of the dataset to be used. (str)
    :param max_depth: The maximum depth of the decision tree. (int)
    :param max_leaf_nodes: The maximum number of leaf nodes in the decision tree. (int)
    :param class_weight: The weight to be given to different classes in the target column. (str or None)
    :param l2_regularization: The L2 regularization value to be used by the model. (float)
    :param learning_rate: The learning rate to be used by the model. (float)

    :returns: None

    The function starts an MLflow run and logs various metrics such as accuracy, precision, and recall.
    It also logs the trained model using the mlflow.sklearn.log_model function.
    """
    warnings.filterwarnings("ignore")
    # Read data
    csv_loc = load_raw_data(dset_name, file_name="Base.csv")
    data = pd.read_csv(csv_loc)

    # Transform categoricals into category type 
    data[CATEGORICAL_FEATURES] = data[CATEGORICAL_FEATURES].astype("category")

    # Train/test split
    train, test = train_test_split(data, random_state=42)

    # Separate X and y
    train_x = train[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    train_y = train[[TARGET]]
    test_x = test[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    test_y = test[[TARGET]]

    # Start the experiemnt
    with mlflow.start_run():
        # Pass the params into dictionary
        hgbt_params = {
            "learning_rate": learning_rate,
            "max_leaf_nodes": max_leaf_nodes,
            "max_depth": max_depth,
            "class_weight": class_weight if class_weight != "None" else None,
            "l2_regularization": l2_regularization,
        }
        # Define model
        hgbt = HistGradientBoostingClassifier(
            **hgbt_params,
            categorical_features=CATEGORICAL_FEATURES,
            max_iter=10000,
            early_stopping=True,
            validation_fraction=10
        )
        # Define transform
        transformer = ColumnTransformer(
            transformers=[("categorical", OrdinalEncoder(), CATEGORICAL_FEATURES)], # HGBT still needs this
            verbose_feature_names_out=False, # to not alter categorical names
            remainder="passthrough",
        )
        # Define pipeline
        pipeline = Pipeline(steps=[("prep", transformer), ("model", hgbt)]).set_output(
            transform="pandas"
        )
        # Fit the pipeline
        pipeline.fit(train_x, train_y)
        # Evaluate on testset
        test_preds = pipeline.predict_proba(test_x)
        eval_and_log_metrics("test", test_y, test_preds[:, 1])
        # Save the pipeline
        mlflow.sklearn.log_model(
            pipeline, "sklearn_models", pyfunc_predict_fn="predict_proba"
        )


if __name__ == "__main__":
    run()
