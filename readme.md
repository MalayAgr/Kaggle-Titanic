# <!-- omit in toc --> Dogs vs. Cats - Kaggle

A complete pipeline for training and tuning (using [Optuna](https://optuna.org/)) an XGBoost model to predict survivability chances of passengers on the Titanic.

**Competition link**: [https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic).

## <!-- omit in toc --> Table of Contents

- [Requirements](#requirements)
- [What's included](#whats-included)
- [Model](#model)
- [Optuna Tuning](#optuna-tuning)
- [GPU Support](#gpu-support)
- [Data](#data)
- [Run](#run)

## Requirements

- Python &ge; 3.8
- XGBoost
- Scikit-learn
- Optuna
- NumPy
- Pandas
- Rich

## What's included

- [eda.ipynb](eda.ipynb) - Exploratory data analysis.
- [feature_engineering.ipynb](feature_engineering.ipynb) - Feature engineering and missing values.
- [config.py](src/config.py) - Contains static parameters and parameters for a baseline model (model without any tuning).
- [data.py](src/data.py) - Implements a `DataPreprocessor` class which implements the feature engineering in the notebook.
- [training.py](src/training.py) - Implements an `Engine` class to encapsulate training of a XGBoost model, with the ability to save ROC curves, loss curves and feature importance plots. Also includes the main training loop and a plotting function that can be used to view each type of plot in a grid.
- [tuning.py](src/tuning.py) - Implements the objective function used by Optuna.
- [main.py](src/main.py) - Main driver for the pipeline.

## Model

The parameters used by the baseline model are:

- `max_depth`: 5.
- `n_estimators`: 100.
- `alpha`: 0.3.
- `learning_rate`: 0.3.
- `scale_pos_weight`: Ratio between number of samples with class 0 and that of class 1 (before any validation split).
- `early_stopping_rounds`: 15.

All other parameters have their default values.

## Optuna Tuning

The following parameters are tuned using Optuna:

- `max_depth`
- `n_estimators`
- `alpha`
- `gamma`
- `lr`
- `colsample_bytree`
- `subsample`
- `early_stopping_rounds`
- `booster`
- `scale_pos_weight`

## GPU Support

There is optional GPU support. When GPU is enabled, an additional parameter `sampling_method` is also tuned using Optuna.

## Data

- Create a directory which matches `DATA_DIR` in [config.py](src/config.py). The default value is relative to the `src` folder.

```shell
mkdir <DATA_DIR>
```

- Download the dataset using the `kaggle` API ([guide](https://github.com/Kaggle/kaggle-api)) or from the competition page inside `DATA_DIR`:

```shell
kaggle competitions download -c titanic
```

## Run

- Clone the repository:

```shell
git clone https://github.com/MalayAgr/Kaggle-Titanic.git
```

- `cd` into the repository:

```shell
cd Kaggle-Dogs-Vs-Cats
```

- If using the default value for `TUNED_MODELS_DIR` and `NO_TUNED_MODELS_DIR`, make sure there is a directory called `models`:

```shell
mkdir models
```

- Execute the `main.py` script:

```shell
python src/main.py
```

> Note: This may not work if your `DATA_DIR` and other directory paths are relative to `src`. In that case, `cd` into the `src` folder and then run `main.py`.
