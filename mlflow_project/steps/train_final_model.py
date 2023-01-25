import catboost as cb
import click
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from steps.tune_model import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET, read_cb_data
import mlflow

def train_model(params, train_path, val_path, test_path):
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
    test_dataset = read_cb_data(
        test_path, 
        numeric_features=NUMERICAL_FEATURES, 
        categorical_features=CATEGORICAL_FEATURES, 
        target_feature=TARGET
    )
    mlflow.set_experiment("fraud")
    experiment = mlflow.get_experiment_by_name("fraud")
    client = mlflow.tracking.MlflowClient()
    run = client.create_run(experiment.experiment_id)
    with mlflow.start_run(run_id = run.info.run_id):
        gbm = cb.CatBoostClassifier(**params)
        gbm.fit(train_dataset, eval_set=val_dataset, early_stopping_rounds=50)
        preds = gbm.predict_proba(test_dataset)
        ap = average_precision_score(test_dataset.get_label(), preds[:, 1])
        roc = roc_auc_score(test_dataset.get_label(), preds[:, 1])

        mlflow.log_metric("Test ROC AUC", roc)
        mlflow.log_metric("Test PR AUC", ap)
        mlflow.log_params(params)
        mlflow.catboost.log_model(gbm, "catboost_model")
    
    return roc, ap


if __name__ == "__main__":
    train_model()
