from __future__ import annotations

import functools
import os
import warnings

import numpy as np
import optuna
import pandas as pd
from rich import print

import config
import training
import tuning

warnings.filterwarnings("ignore")


def make_model_dirs():
    model_dirs = [config.NO_TUNED_MODELS_DIR, config.TUNED_MODELS_DIR]
    dirs = ["roc", "feature_importance", "loss_curves"]

    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            continue

        os.mkdir(model_dir)

        for d in dirs:
            path = os.path.join(model_dir, d)
            os.mkdir(path)


def train_baseline(
    X: np.ndarray, y: np.ndarray, gpu: bool = False, save_every: int = None
) -> tuple[np.float32, np.float32]:
    params = {
        "max_depth": config.MAX_DEPTH,
        "n_estimators": config.N_ESTIMATORS,
        "alpha": config.ALPHA,
        "lr": config.LR,
        "scale_pos_weight": sum(y == 0) / sum(y == 1),
        "early_stopping_rounds": config.EARLY_STOPPING_ROUNDS,
    }

    print("Training model with no hyperparameter tuning:\n")

    roc, loss = training.train(
        X=X,
        y=y,
        params=params,
        gpu=gpu,
        save_model=True,
        directory=config.NO_TUNED_MODELS_DIR,
        save_every=save_every,
    )

    return roc, loss


def train_with_optuna(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    gpu: bool = False,
    save_every: int = None,
) -> tuple[np.float32, np.float32]:
    objective_ = functools.partial(tuning.objective, X=X, y=y, pruner=True, gpu=gpu)

    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        sampler=sampler,
    )

    print("Finding best parameters:")
    study.optimize(objective_, n_trials=n_trials, gc_after_trial=True)

    best_trial = study.best_trial

    print("\nTraining model with the best obtained hyperparameters:\n")

    roc, loss = training.train(
        X=X,
        y=y,
        params=best_trial.params,
        gpu=gpu,
        save_model=True,
        directory=config.TUNED_MODELS_DIR,
        save_every=save_every,
    )

    return roc, loss


def main():
    make_model_dirs()

    path = os.path.join(config.DATA_DIR, "final_train.csv")
    train_df = pd.read_csv(path)

    features = train_df.drop(["Survived", "PassengerId"], axis=1).values
    labels = train_df["Survived"].values

    roc, loss = train_baseline(X=features, y=labels, save_every=2)
    print(f"Baseline model: Average ROC={roc:.4f}; Average validation loss={loss:.4f}")

    roc, loss = train_with_optuna(
        X=features, y=labels, n_trials=config.N_TRIALS, save_every=2
    )
    print(
        f"Optuna-tuned model: Average ROC={roc:.4f}; Average validation loss={loss:.4f}"
    )


if __name__ == "__main__":
    main()
