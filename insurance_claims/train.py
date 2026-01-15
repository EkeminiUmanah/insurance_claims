import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor


RANDOM_STATE = 42


def main():
    # paths
    DATA_DIR = Path("allstate-claims-severity")
    TRAIN_PATH = DATA_DIR / "train.csv"

    ARTIFACT_DIR = Path("artifacts")
    ARTIFACT_DIR.mkdir(exist_ok=True)

    MODEL_PATH = ARTIFACT_DIR / "claim_severity_model.pkl"

    # load data
    df = pd.read_csv(TRAIN_PATH)

    target_col = "loss"
    id_col = "id"

    categorical_cols = [c for c in df.columns if c.startswith("cat")]
    numerical_cols = [c for c in df.columns if c.startswith("cont")]

    # target transformation
    df["log_loss"] = np.log1p(df[target_col])

    X = df.drop(columns=[target_col, "log_loss", id_col])
    y = df["log_loss"]

    # preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                ),
                categorical_cols,
            ),
            ("num", "passthrough", numerical_cols),
        ]
    )

    # model
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    random_state=RANDOM_STATE,
                    learning_rate=0.1,
                    max_depth=None,
                    max_leaf_nodes=63,
                    min_samples_leaf=50,
                ),
            ),
        ]
    )

    # train
    model.fit(X, y)

    # save
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
