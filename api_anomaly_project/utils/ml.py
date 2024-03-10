import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.express as px
from optuna import create_study
from optuna.integration.mlflow import MLflowCallback
from optuna.trial import FrozenTrial
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score


def evaluate_thresholds(
    thresholds: npt.NDArray[np.float32],
    y_true: npt.NDArray[np.float32],
    y_pred_proba: npt.NDArray[np.float32],
    plot: bool = True,
) -> tuple[list[float], list[float], list[float]]:
    rcs = []
    prs = []
    f1s = []

    for t in thresholds:
        test_binary_pred = y_pred_proba[:, 1] >= t
        prs.append(precision_score(y_true, test_binary_pred))
        rcs.append(recall_score(y_true, test_binary_pred))
        f1s.append(f1_score(y_true, test_binary_pred))

    metrics_df = pd.DataFrame({"threshold": thresholds, "score": f1s, "metric": "F1"})
    metrics_df = pd.concat(
        (
            metrics_df,
            pd.DataFrame({"threshold": thresholds, "score": rcs, "metric": "Recall"}),
        )
    )
    metrics_df = pd.concat(
        (
            metrics_df,
            pd.DataFrame(
                {"threshold": thresholds, "score": prs, "metric": "Precision"}
            ),
        )
    )

    optimal_thr = thresholds[np.argmax(f1s)]
    optimal_f1 = f1s[np.argmax(f1s)]
    optimal_rc = rcs[np.argmax(f1s)]
    optimal_pr = prs[np.argmax(f1s)]

    print("Threshold with Max F1 Score: ", optimal_thr)
    print(f"F1 at threshold {optimal_thr}: {optimal_f1}")
    print(f"Recall at threshold {optimal_thr}: {optimal_rc}")
    print(f"Precision at threshold {optimal_thr}: {optimal_pr} ")

    if plot:
        fig = px.line(
            metrics_df,
            x="threshold",
            y="score",
            color="metric",
            title="Metrics per Threshold",
        )
        fig.show()

    return rcs, prs, f1s


def tune_hgbt(
    n_trials: int, mlflc: MLflowCallback, X_train: pd.DataFrame, y_train: pd.Series
) -> FrozenTrial:
    @mlflc.track_in_mlflow()
    def objective(trial):
        params = {
            "learning_rate": 0.1,
            "max_iter": trial.suggest_int("max_iter", 10, 100),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 10, 31),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "l2_regularization": trial.suggest_float("l2_regularization", 0, 10),
        }
        mlflow.set_tag("model_name", "HGBT")
        mlflow.log_params(params)

        gbt = HistGradientBoostingClassifier(**params)

        roc_auc = cross_val_score(gbt, X_train, y_train, cv=5, scoring="roc_auc").mean()
        print("ROC AUC (avg 5-fold):", roc_auc)

        return roc_auc

    study = create_study(direction="maximize", study_name="hgbt_tuning")
    study.optimize(objective, n_trials=n_trials, callbacks=[mlflc])
    return study.best_trial
