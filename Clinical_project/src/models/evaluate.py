import pandas as pd
from sklearn.metrics import classification_report
from joblib import load
from pathlib import Path

def evaluate_model():
    project_root = Path(__file__).resolve().parents[2]
    model = load(project_root / "models" / "model.joblib")

    X_test = pd.read_csv(project_root / "data" / "processed" / "X_test.csv")
    y_test = pd.read_csv(project_root / "data" / "processed" / "y_test.csv").squeeze("columns")

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))