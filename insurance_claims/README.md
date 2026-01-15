# Allstate Claim Severity Prediction  
**ML Zoomcamp – Capstone Project**

---

## 1. Problem Description

Insurance companies need to estimate the **severity of claims** in order to price policies, manage reserves, and control risk.  
This project builds a machine learning model that predicts **claim severity (`loss`)** using a mix of:

- **116 categorical features** (`cat1` … `cat116`)
- **14 continuous features** (`cont1` … `cont14`)

### How the solution is used
Given claim-related feature values, the deployed API returns a predicted claim severity.  
This can be used for risk scoring, pricing, and operational decision-making.

---

## 2. Dataset

This project uses the **Allstate Claims Severity** dataset (Kaggle).

Expected local structure:
```text
allstate-claims-severity/
├── train.csv
└── test.csv
```


The dataset contains:
- `id`: unique identifier
- `loss`: target variable (claim severity)
- `cat1..cat116`: categorical features
- `cont1..cont14`: continuous features

---

## 3. Project Structure

```text
├── allstate-claims-severity/
│   ├── train.csv
│   └── test.csv
├── artifacts/
│   └── claim_severity_model.pkl
├── notebook.ipynb
├── train.py
├── predict.py
├── requirements.txt
├── requirements-dev.txt
└── README.md
```


---

## 4. Approach Overview

### Exploratory Data Analysis (EDA)
- Verified **no missing values**
- Analyzed highly skewed target distribution
- Applied **log1p transformation** to `loss`
- Explored continuous feature distributions and correlations
- Analyzed categorical feature cardinality and target behavior

### Modeling
Models trained and evaluated:
- Linear Regression (One-Hot Encoding)
- HistGradientBoostingRegressor (Ordinal Encoding)
- Tuned HistGradientBoostingRegressor (**final model**)
- RandomForestRegressor (baseline + tuning)

### Final Model
- **HistGradientBoostingRegressor**
- Ordinal encoding for categorical features
- Trained on log-transformed target
- Retrained on full dataset
- Saved as a single serialized pipeline

---

## 5. Environment Setup

### Create and activate a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Install runtime dependencies
```bash
pip install -r requirements.txt
```
### (Optional) Install development dependencies
```bash
pip install -r requirements-dev.txt
```

## 6. Reproduce Training

Train the final model and save the artifact:
```bash
python train.py
```
Expected output:
```bash
Model saved to artifacts/claim_severity_model.pkl
```

## 7. Run the Prediction API (Local)
### Start the FastAPI server
```bash
uvicorn predict:app --reload
```
Open Swagger UI:
```text
http://127.0.0.1:8000/docs
```

## 8. API Usage (IMPORTANT)

### Endpoint
POST /predict


### Request format
```json
{
  "features": {
    "cat1": "A",
    "cat2": "B",
    "cat3": "A",
    "cat4": "B",
    "cat5": "A",
    "cat6": "A",
    "cat7": "A",
    "cat8": "A",
    "cat9": "B",
    "cat10": "A",
    "cat11": "B",
    "cat12": "A",
    "cat13": "A",
    "cat14": "A",
    "cat15": "A",
    "cat16": "A",
    "cat17": "A",
    "cat18": "A",
    "cat19": "A",
    "cat20": "A",
    "cat21": "A",
    "cat22": "A",
    "cat23": "B",
    "cat24": "A",
    "cat25": "A",
    "cat26": "A",
    "cat27": "A",
    "cat28": "A",
    "cat29": "A",
    "cat30": "A",
    "cat31": "A",
    "cat32": "A",
    "cat33": "A",
    "cat34": "A",
    "cat35": "A",
    "cat36": "A",
    "cat37": "A",
    "cat38": "A",
    "cat39": "A",
    "cat40": "A",
    "cat41": "A",
    "cat42": "A",
    "cat43": "A",
    "cat44": "A",
    "cat45": "A",
    "cat46": "A",
    "cat47": "A",
    "cat48": "A",
    "cat49": "A",
    "cat50": "A",
    "cat51": "A",
    "cat52": "A",
    "cat53": "A",
    "cat54": "A",
    "cat55": "A",
    "cat56": "A",
    "cat57": "A",
    "cat58": "A",
    "cat59": "A",
    "cat60": "A",
    "cat61": "A",
    "cat62": "A",
    "cat63": "A",
    "cat64": "A",
    "cat65": "A",
    "cat66": "A",
    "cat67": "A",
    "cat68": "A",
    "cat69": "A",
    "cat70": "A",
    "cat71": "A",
    "cat72": "A",
    "cat73": "A",
    "cat74": "A",
    "cat75": "B",
    "cat76": "A",
    "cat77": "D",
    "cat78": "B",
    "cat79": "B",
    "cat80": "D",
    "cat81": "D",
    "cat82": "B",
    "cat83": "D",
    "cat84": "C",
    "cat85": "B",
    "cat86": "D",
    "cat87": "B",
    "cat88": "A",
    "cat89": "A",
    "cat90": "A",
    "cat91": "A",
    "cat92": "A",
    "cat93": "D",
    "cat94": "B",
    "cat95": "C",
    "cat96": "E",
    "cat97": "A",
    "cat98": "C",
    "cat99": "T",
    "cat100": "B",
    "cat101": "G",
    "cat102": "A",
    "cat103": "A",
    "cat104": "I",
    "cat105": "E",
    "cat106": "G",
    "cat107": "J",
    "cat108": "G",
    "cat109": "BU",
    "cat110": "BC",
    "cat111": "C",
    "cat112": "AS",
    "cat113": "S",
    "cat114": "A",
    "cat115": "O",
    "cat116": "LB",
    "cont1": 0.7263,
    "cont2": 0.245921,
    "cont3": 0.187583,
    "cont4": 0.789639,
    "cont5": 0.310061,
    "cont6": 0.718367,
    "cont7": 0.33506,
    "cont8": 0.3026,
    "cont9": 0.67135,
    "cont10": 0.8351,
    "cont11": 0.569745,
    "cont12": 0.594646,
    "cont13": 0.822493,
    "cont14": 0.714843
  }
}
```
### Important notes

* JSON must use double quotes
* Do not include loss or log_loss
* For best results, include all cat1..cat116 and cont1..cont14


## 9. Generate a Valid Request Automatically

To avoid formatting errors, generate a valid JSON payload from the dataset.

Run this in the notebook:
```python
import json

sample = X_full.iloc[0].to_dict()

# defensive cleanup (in case someone uses the wrong dataframe)
sample.pop("loss", None)
sample.pop("log_loss", None)
sample.pop("id", None)

payload = {"features": sample}

with open("sample_request.json", "w") as f:
    json.dump(payload, f, indent=2)

print("Saved sample_request.json")
```
### Test with curl
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```
Example response:
```json
{
  "predicted_claim_severity": 1881.9189
}
```

## 10. Model Evaluation Summary
Validation RMSE (log scale):
| Model                           | RMSE (log)  |
| ------------------------------- | ----------- |
| Linear Regression               | ~0.5607     |
| HistGradientBoosting (baseline) | ~0.5383     |
| HistGradientBoosting (tuned)    | **~0.5359** |
| RandomForest (tuned)            | ~0.5526     |

The tuned HistGradientBoostingRegressor achieved the best performance and was selected for deployment.

## 11. Docker (Optional)

### Build the image
```bash
docker build -t claim-severity-api .
```
### Train once on the host to create the model artifact
```bash
python train.py
```
### Run (mount artifacts so the container can load the trained model)
macOS / Linux / Git Bash:
```bash
docker run -p 8000:8000 -v "$(pwd)/artifacts:/app/artifacts" claim-severity-api
```
Windows PowerShell:
```bash
docker run -p 8000:8000 -v "${PWD}\artifacts:/app/artifacts" claim-severity-api
```
### Then open:
```text
http://127.0.0.1:8000/docs
```
If Docker is not available locally, the API can still be run using the local instructions above.

## 12. Notes on Reproducibility

* All training logic is in train.py 
* The prediction service loads the serialized pipeline 
* Dependencies are explicitly pinned in requirements.txt 
* The project can be executed end-to-end from scratch


