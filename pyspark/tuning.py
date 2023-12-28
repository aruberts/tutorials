from hyperopt import STATUS_OK, Trials, fmin, tpe
from hyperopt.pyll.base import Apply
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import Evaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import DataFrame


def tune_rf(
    train: DataFrame,
    val: DataFrame,
    string_indexer: StringIndexer,
    vector_assembler: VectorAssembler,
    evaluator: Evaluator,
    param_grid: dict[str, Apply],
    tuning_rounds: int = 10,
):
    def objective(params):
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="is_bad",
            numTrees=params["numTrees"],
            maxDepth=params["maxDepth"],
        )

        pipeline = Pipeline(stages=[string_indexer, vector_assembler, rf])

        pipeline = pipeline.fit(train)
        val_df = pipeline.transform(val)

        score = evaluator.evaluate(val_df)
        return {"loss": -score, "status": STATUS_OK}

    rf_trials = Trials()

    argmin = fmin(
        fn=objective,
        space=param_grid,
        algo=tpe.suggest,
        max_evals=tuning_rounds,
        trials=rf_trials,
    )

    return argmin
