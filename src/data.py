from __future__ import annotations

import os

import numpy as np
import pandas as pd
from rich import print
from sklearn import impute, preprocessing

import config


def encode(
    values: np.ndarray, *, encoder: preprocessing.LabelEncoder = None
) -> np.ndarray:
    if encoder is None:
        encoder = preprocessing.LabelEncoder()
        encoder.fit(values)

    return encoder.fit_transform(values)


def impute_missing(values: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    scaler = preprocessing.MinMaxScaler()
    temp = scaler.fit_transform(values)
    imputer = impute.KNNImputer(n_neighbors=5)
    imputed = imputer.fit_transform(temp)
    return scaler.inverse_transform(imputed)


class DataPreprocessor:
    fmt = "[green]{}...[/green]"

    def __init__(
        self,
        train_csv: str,
        test_csv: str,
        to_be_encoded: list[str] = None,
        n_neighbors: int = 5,
        imputation_features: list[str] = None,
    ) -> None:
        self.train_csv = os.path.join(config.DATA_DIR, train_csv)
        self.test_csv = os.path.join(config.DATA_DIR, test_csv)
        self.to_be_encoded = to_be_encoded
        self._to_be_encoded = to_be_encoded.copy()
        self.n_neighbors = n_neighbors
        self.imputation_features = imputation_features

        self.tr_index: pd.Index = None
        self.te_index: pd.Index = None
        self.labels: pd.Series = None

    def combine_train_test(self):
        # Read in the two datasets
        tr_df = pd.read_csv(self.train_csv, index_col="PassengerId")
        te_df = pd.read_csv(self.test_csv, index_col="PassengerId")

        # Take out labels from training data
        self.label = tr_df["Survived"]
        tr_df = tr_df.drop("Survived", axis=1)

        # Store indices
        self.tr_index = tr_df.index
        self.te_index = te_df.index

        df = pd.concat([tr_df, te_df])

        return df

    def _add_features_msg(self) -> str:
        features = ["missing_Age", "missing_Cabin", "Alone", "Title", "Married"]

        self._to_be_encoded.extend(features)

        encoded_msg = ", ".join(f"[bold]{feature}[/bold]" for feature in features)

        features.extend(("FamilySize", "GroupSize"))

        msg = "\t[red]Added features {}. {}[/red]"
        msg = msg.format(
            ", ".join(f"[bold]{feature}[/bold]" for feature in features),
            f"{encoded_msg} will be also encoded with to_be_encoded features.",
        )
        return msg

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["missing_Age"] = df["Age"].isna()
        df["missing_Cabin"] = df["Cabin"].isna()

        df["FamilySize"] = df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

        df["Alone"] = df["FamilySize"] == 1

        df["Title"] = (
            df["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
        )

        df["GroupSize"] = df.groupby("Ticket")["Ticket"].transform("count")

        df["Married"] = df["Title"] == "Mrs"

        msg = self._add_features_msg()
        print(msg)

        return df

    def encode_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in self._to_be_encoded:
            print(f"\t[red]Encoding [bold]{column}[/bold][/red].")
            df[column] = encode(df[column].values)

        return df

    def split_df(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        tr_df = df.loc[self.tr_index, :]
        tr_df["Survived"] = self.labels
        return tr_df, df.loc[self.te_index, :]

    def impute_missing_values(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        cols = self.imputation_features
        train[cols] = impute_missing(train[cols].values, n_neighbors=self.n_neighbors)
        test[cols] = impute_missing(test[cols].values, n_neighbors=self.n_neighbors)
        return train, test

    def save_datasets(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple[str, str]:
        tr_path = os.path.join(config.DATA_DIR, "final_train.csv")
        train = train.reset_index()
        train.to_csv(tr_path, index=False)

        te_path = os.path.join(config.DATA_DIR, "final_test.csv")
        test = test.reset_index()
        test.to_csv(te_path, index=False)

        return tr_path, te_path

    def preprocess(self) -> None:
        print(self.fmt.format("Combing train and test datasets"))
        df = self.combine_train_test()

        print(self.fmt.format("Adding features"))
        df = self.add_features(df)

        print(self.fmt.format("Dropping unnecessary columns"))
        df = df.drop(["Name", "Ticket", "Cabin"], axis=1)

        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iloc[0])

        if self.to_be_encoded is not None:
            print(self.fmt.format("Encoding columns"))
            df = self.encode_cols(df)

        print(self.fmt.format("Splitting into train and test datasets"))
        train, test = self.split_df(df)

        if self.imputation_features is not None:
            print(self.fmt.format("Imputing missing values using KNNImputer"))
            train, test = self.impute_missing_values(train, test)

        tr_p, te_p = self.save_datasets(train, test)

        msg = f"Saved new datasets at [bold]{tr_p}[/bold] and [bold]{te_p}[/bold]."
        msg = self.fmt.format(msg)
        print(msg, "\n\n")
