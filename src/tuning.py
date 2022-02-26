import numpy as np
import optuna

import training


def objective(
    trial: optuna.trial.Trial,
    X: np.ndarray,
    y: np.ndarray,
    pruner: bool = False,
    gpu: bool = False,
) -> np.float32:
    params = {
        "max_depth": trial.suggest_int("max_depth", 1, 9),
        "n_estimators": trial.suggest_int("n_estimators", 5, 250),
        "alpha": trial.suggest_float("alpha", 10, 100, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "lr": trial.suggest_float("lr", 1e-8, 1.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1, log=True),
        "early_stopping_rounds": trial.suggest_categorical(
            "early_stopping_rounds", [5, 10, 15, 20, 25]
        ),
        "booster": trial.suggest_categorical("booster", ["gblinear", "gbtree"]),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 5, log=True),
    }

    if gpu is True:
        params["sampling_method"] = trial.suggest_categorical(
            "sampling_method", ["uniform", "gradient_based"]
        )

    if pruner is True:
        params["callbacks"] = [
            optuna.integration.XGBoostPruningCallback(trial, "validation_1-logloss")
        ]

    _, loss = training.train(X=X, y=y, params=params, gpu=gpu)
    return loss
