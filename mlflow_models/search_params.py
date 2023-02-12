import click

from hyperopt import fmin, hp, tpe
from hyperopt.pyll import scope

import mlflow.projects
from mlflow.tracking import MlflowClient


@click.command(
    help="""
    Perform hyperparameter search with Hyperopt library.
    Optimize PR AUC.
    """
)
@click.option(
    "--max-runs",
    type=click.INT,
    default=10,
    help="Maximum number of runs to evaluate."
)
@click.option(
    "--model-type",
    type=click.STRING,
    default="hgbt",
    help="Model type to tune"
)
@click.argument("training_data")
def train(training_data, max_runs, model_type):
    """
    Run hyperparameter optimization.
    """
    # create random file to store run ids of the training tasks
    tracking_client = MlflowClient()

    def new_eval(experiment_id):
        """
        Create a new eval function
        :experiment_id: Experiment id for the training run
        :return: new eval function.
        """

        def eval(params):
            """
            Train sklearn model with given parameters by invoking MLflow run.
            :param params: Parameters to the train script we optimize over
            :return: The metric value evaluated on the validation data.
            """
            with mlflow.start_run(nested=True) as child_run:
                if model_type == "rf":
                    # Params used to train RF
                    (
                        max_depth, max_features,
                        class_weight, min_samples_leaf
                    ) = params
                    # Run the training script as MLflow sub-run
                    p = mlflow.projects.run(
                        uri=".",
                        entry_point="train_rf",
                        run_id=child_run.info.run_id,
                        parameters={
                            "dset_name": training_data,
                            "max_depth": str(max_depth),
                            "max_features": str(max_features),
                            "class_weight": str(class_weight),
                            "min_samples_leaf": str(min_samples_leaf),
                        },
                        experiment_id=experiment_id,
                        synchronous=False,
                    )
                    # No idea why, but it's needed?
                    succeeded = p.wait()
                    # Log params
                    mlflow.log_params(
                        {
                            "max_depth": max_depth,
                            "max_features": max_features,
                            "class_weight": class_weight,
                            "min_samples_leaf": min_samples_leaf,
                        }
                    )
                elif model_type == "hgbt":
                    # Params used to train HGBT
                    (
                        max_depth,
                        max_leaf_nodes,
                        class_weight,
                        l2_regularization,
                        learning_rate,
                    ) = params
                    # Run the train_hgbt as sub-run
                    p = mlflow.projects.run(
                        uri=".",
                        entry_point="train_hgbt",
                        run_id=child_run.info.run_id,
                        parameters={
                            "dset_name": training_data,
                            "learning_rate": str(learning_rate),
                            "max_leaf_nodes": str(max_leaf_nodes),
                            "max_depth": str(max_depth),
                            "class_weight": str(class_weight),
                            "l2_regularization": str(l2_regularization),
                        },
                        experiment_id=experiment_id,
                        synchronous=False,
                    )
                    succeeded = p.wait()
                    mlflow.log_params(
                        {
                            "learning_rate": learning_rate,
                            "max_leaf_nodes": max_leaf_nodes,
                            "max_depth": max_depth,
                            "class_weight": class_weight,
                            "l2_regularization": l2_regularization,
                        }
                    )
                    print(succeeded)

            # Grab the test metrics from the MLflow run
            training_run = tracking_client.get_run(p.run_id)
            metrics = training_run.data.metrics
            test_prauc = metrics["test_PR_AUC"]

            return -test_prauc

        return eval

    if model_type == "rf":
        # Search space for RF
        space = [
            scope.int(hp.quniform("max_depth", 1, 30, q=1)),
            hp.uniform("max_features", 0.05, 0.8),
            hp.choice("class_weight", ["balanced", None]),
            scope.int(hp.quniform("min_samples_leaf", 5, 100, q=5)),
        ]
    elif model_type == "hgbt":
        # Search space for HGBT
        space = [
            scope.int(hp.quniform("max_depth", 1, 30, q=1)),
            scope.int(hp.quniform("max_leaf_nodes", 5, 100, q=5)),
            hp.choice("class_weight", ["balanced", None]),
            hp.uniform("l2_regularization", 0.0, 20.0),
            hp.uniform("learning_rate", 0.01, 0.1),
        ]
    else:
        raise ValueError(f"Model type {model_type} is not supported")

    # This starts the actual search_rf.py experiment run
    with mlflow.start_run() as run:
        # Get parent ID
        experiment_id = run.info.experiment_id

        # Optimisation function that takes parent id and search params as input
        best = fmin(
            fn=new_eval(experiment_id),
            space=space,
            algo=tpe.suggest,
            max_evals=max_runs,
        )
        mlflow.set_tag("best params", str(best))


if __name__ == "__main__":
    train()
