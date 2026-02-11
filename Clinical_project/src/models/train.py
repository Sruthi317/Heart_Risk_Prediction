import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump
from pathlib import Path

def train_model():
    project_root = Path(__file__).resolve().parents[2]
    X_train = pd.read_csv(project_root / "data" / "processed" / "X_train.csv")
    y_train = pd.read_csv(project_root / "data" / "processed" / "y_train.csv").squeeze("columns")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True, parents=True)
    dump(model, models_dir / "model.joblib")
    print("[train] Model saved!")