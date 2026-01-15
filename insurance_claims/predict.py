import pickle
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# paths
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "claim_severity_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model artifact not found at {MODEL_PATH}. "
        "Run `python train.py` first to create it."
    )

# load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


app = FastAPI(title="Insurance Claim Severity Prediction API")


class ClaimFeatures(BaseModel):
    features: Dict[str, Union[float, str]]


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.get("/schema")
def schema_hint():
    """
    Lightweight hint endpoint for reviewers.
    The model expects cat1..cat116 and cont1..cont14 inside `features`.
    """
    return {
        "expected_features": {
            "categorical": "cat1..cat116",
            "continuous": "cont1..cont14",
        },
        "note": "Send a JSON body: {'features': {...}}. Do not include id/loss/log_loss."
    }


@app.post("/predict")
def predict(payload: ClaimFeatures):
    # defensive cleanup: remove training-only fields if user accidentally sends them
    payload_dict = dict(payload.features)
    payload_dict.pop("id", None)
    payload_dict.pop("loss", None)
    payload_dict.pop("log_loss", None)

    # convert input to DataFrame
    X = pd.DataFrame([payload_dict])

    # predict in log space
    log_pred = float(model.predict(X)[0])

    # convert back to original scale
    pred = float(np.expm1(log_pred))

    return {
        "predicted_claim_severity": round(pred, 4),
        "predicted_log_severity": round(log_pred, 6),
    }
