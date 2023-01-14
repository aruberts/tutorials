import catboost as cb
import click
import mlflow
import optuna
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, roc_auc_score
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

TARGET = "fraud_bool"
CATEGORICAL_FEATURES = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]
NUMERICAL_FEATURES = [
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "credit_risk_score",
    "email_is_free",
    "phone_home_valid",
    "phone_mobile_valid",
    "bank_months_count",
    "has_other_cards",
    "proposed_credit_limit",
    "foreign_request",
    "session_length_in_minutes",
    "keep_alive_session",
    "device_distinct_emails_8w",
    "month",
]

def read_cb_data(
    path: str, numeric_features: list, categorical_features: list, target_feature: str
):
    data = pd.read_parquet(path)
    dataset = cb.Pool(
        data=data[numeric_features + categorical_features],
        label=data[target_feature],
        cat_features=categorical_features,
    )
    return dataset


@click.command(
    help="""Given a path to raw csv file, this steps fills in missing data, 
    drops static columns, and splits the data into train/val/test sets"""
)
@click.option("--parent-run")
@click.option("--train-path")
@click.option("--val-path")
@click.option("--n-trials", default=10)
def tune_model(parent_run, train_path, val_path, n_trials):
    client = MlflowClient()
    experiment_name='fraud'
    try:
        experiment = client.create_experiment(experiment_name)
    except:
        experiment = client.get_experiment_by_name(experiment_name).experiment_id
    
    print(f"Experiment {experiment}")

    train_dataset = read_cb_data(
        train_path, 
        numeric_features=NUMERICAL_FEATURES, 
        categorical_features=CATEGORICAL_FEATURES, 
        target_feature=TARGET
    )
    val_dataset = read_cb_data(
        val_path, 
        numeric_features=NUMERICAL_FEATURES, 
        categorical_features=CATEGORICAL_FEATURES, 
        target_feature=TARGET
    )

    def get_objective(parent_run_id):
        # get an objective function for optuna that creates nested MLFlow runs
        def objective(trial):
            trial_run = client.create_run(
                experiment_id=experiment,
                tags={
                    MLFLOW_PARENT_RUN_ID: parent_run_id
                }
            )
            print("Trial run info run id:", trial_run.info.run_id)
            param = {
                "n_estimators": 1000,
                "objective": "Logloss",
                "subsample": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
                "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-3, 10.0),
                "learning_rate": trial.suggest_uniform("learning_rate", 0.006, 0.02),
                "colsample_bylevel": trial.suggest_float(
                    "colsample_bylevel", 0.05, 0.5
                ),
                "depth": trial.suggest_int("depth", 2, 12),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 300),
            }
            for k, v in param.items():
                client.log_param(trial_run.info.run_id, k, v)

            gbm = cb.CatBoostClassifier(**param)
            gbm.fit(train_dataset, eval_set=val_dataset, early_stopping_rounds=50)

            preds = gbm.predict_proba(val_dataset)
            ap = average_precision_score(val_dataset.get_label(), preds[:, 1])
            roc = roc_auc_score(val_dataset.get_label(), preds[:, 1])
            client.log_metric(trial_run.info.run_id, "PR AUC", ap)
            client.log_metric(trial_run.info.run_id, "ROC AUC", roc)
            return ap
    
        return objective

    with mlflow.start_run(run_name='hp_tuning') as active_run:
        print("Active run:", active_run)
        study = optuna.create_study(direction="maximize")
        study.optimize(get_objective(active_run.info.run_id), n_trials=n_trials)
        # Log all tuned params
        mlflow.log_params(study.best_trial.params)
        client.log_metric(active_run.info.run_id, "best_prauc", study.best_value)

if __name__ == "__main__":
    tune_model()
