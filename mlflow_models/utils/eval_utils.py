from sklearn.metrics import average_precision_score
import mlflow

def eval_and_log_metrics(prefix, actual, pred):
    pr = average_precision_score(actual, pred)
    mlflow.log_metric("{}_PR_AUC".format(prefix), pr)
    return pr