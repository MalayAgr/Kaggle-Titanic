from __future__ import annotations

import gc
import glob
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn import metrics, model_selection
from tqdm import tqdm

import config

ParamDict = dict[str, Any]


class Engine:
    def __init__(self, model: xgb.XGBClassifier) -> None:
        self.model = model

    @staticmethod
    def roc(true: np.ndarray, preds: np.ndarray) -> float:
        return metrics.roc_auc_score(true, preds)

    def save_roc_curve(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        directory: str,
        filepath: str,
    ) -> None:
        preds = self.model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = metrics.roc_curve(y_val, preds)
        roc = self.roc(y_val, preds)

        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", label=f"ROC curve (roc={roc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
        plt.xlim([-0.02, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve")
        plt.legend(loc="lower right")

        path = os.path.join(directory, "roc", filepath)
        plt.savefig(path)
        plt.clf()

    def save_feature_importance_plot(self, directory: str, filepath: str) -> None:
        xgb.plot_importance(self.model)
        path = os.path.join(directory, "feature_importance", filepath)
        plt.savefig(path)
        plt.clf()

    def save_loss_curve(self, directory: str, filepath: str) -> None:
        eval_results = self.model.evals_result()

        train_loss = eval_results["validation_0"]["logloss"]
        val_loss = eval_results["validation_1"]["logloss"]

        plt.figure()
        plt.plot(train_loss, label="Training")
        plt.plot(val_loss, label="Validation")

        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()

        path = os.path.join(directory, "loss_curves", filepath)
        plt.savefig(path)
        plt.clf()

    def save_model(self, directory: str, filepath: str) -> None:
        path = os.path.join(directory, filepath)
        self.model.save_model(path)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        early_stopping_rounds: int = None,
        callbacks: list = None,
    ) -> tuple[float, float, np.float32, np.float32]:
        model = self.model

        model.fit(
            X=X_train,
            y=y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric="logloss",
            verbose=False,
            early_stopping_rounds=early_stopping_rounds,
            callbacks=callbacks,
        )

        eval_results = model.evals_result()

        train_loss = np.mean(eval_results["validation_0"]["logloss"], dtype=np.float32)
        val_loss = np.mean(eval_results["validation_1"]["logloss"], dtype=np.float32)

        preds = model.predict(X_val)
        accuracy = metrics.accuracy_score(y_val, preds)

        preds = model.predict_proba(X_val)[:, 1]
        roc = self.roc(y_val, preds)

        return accuracy, roc, train_loss, val_loss


def get_model_params(params: ParamDict, gpu: bool = False) -> ParamDict:
    model_params = {
        "max_depth": params.get("max_depth", 6),
        "n_estimators": params["n_estimators"],
        "objective": params.get("objective", "binary:logistic"),
        "alpha": params.get("alpha", 0.0),
        "gamma": params.get("gamma", 0.0),
        "learning_rate": params.get("lr", 0.3),
        "colsample_bytree": params.get("colsample_bytree", 1.0),
        "subsample": params.get("subsample", 1.0),
        "sampling_method": params.get("sampling_method", "uniform"),
        "scale_pos_weight": params.get("scale_pos_weight", 1.0),
        "use_label_encoder": False,
        "n_jobs": config.N_JOBS,
        "eval_metric": "logloss",
    }

    if gpu is True:
        model_params.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"})

    return model_params


def train(
    X: np.ndarray,
    y: np.ndarray,
    params: ParamDict,
    gpu: bool = False,
    save_model: bool = False,
    directory: str = "",
) -> tuple[np.float32, np.float32]:

    if save_model is True and not directory:
        raise ValueError("A directory is required when save_model is True.")

    cv = model_selection.StratifiedKFold(
        n_splits=config.FOLDS, shuffle=True, random_state=42
    )

    folds = cv.split(X, y)

    scores, losses = [], []

    esr = params.get("early_stopping_rounds")
    callbacks = params.get("callbacks")

    with tqdm(folds, total=config.FOLDS) as p_folds:
        for fold, (train_idx, val_idx) in enumerate(p_folds):
            p_folds.set_description(f"Fold {fold + 1}")

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            classifier = xgb.XGBClassifier(**get_model_params(params, gpu=gpu))

            engine = Engine(model=classifier)

            accuracy, roc, train_loss, val_loss = engine.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                early_stopping_rounds=esr,
                callbacks=callbacks,
            )
            gc.collect()

            if save_model is True and fold % 3 == 2:
                key = fold + 1
                engine.save_model(directory, f"model-{key}.json")
                engine.save_roc_curve(X_val, y_val, directory, f"model-{key}.png")
                engine.save_feature_importance_plot(directory, f"model-{key}.png")
                engine.save_loss_curve(directory, f"model-{key}.png")

            p_folds.set_postfix(
                roc=roc, train_ll=train_loss, val_ll=val_loss, acc=accuracy
            )

            scores.append(roc)
            losses.append(val_loss)

    return np.mean(scores), np.mean(losses)


def plot(directory: str) -> None:
    fig, axs = plt.subplots(3, 2, figsize=(20, 20), sharex=True)
    axs = axs.flatten()
    path = os.path.join(directory, "*.png")

    for img_path, ax in zip(sorted(glob.glob(path)), axs):
        basename = os.path.basename(img_path)
        filename, _ = os.path.splitext(basename)
        img = plt.imread(img_path)
        ax.set_axis_off()
        ax.imshow(img)
        ax.set_title(filename)

    for ax in axs.flat:
        if not ax.has_data():
            fig.delaxes(ax)
