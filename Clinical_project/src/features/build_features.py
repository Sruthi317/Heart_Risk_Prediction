import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

def build_features():
    # Always load .env from the project root (two levels up from this file)
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(dotenv_path=project_root / ".env")

    target = os.getenv("TARGET_COLUMN", "").strip()
    df_path = project_root / "data" / "processed" / "clean.csv"
    if not df_path.exists():
        raise FileNotFoundError(f"Processed file not found at {df_path}. Run prepare step first.")

    df = pd.read_csv(df_path)

    # Validate target column
    if not target:
        # Try a couple of common fallbacks
        candidates = [c for c in ["DEATH_EVENT", "death_event", "target", "label"] if c in df.columns]
        if candidates:
            target = candidates[0]
            print(f"[features] TARGET_COLUMN not set; using detected '{target}'")
        else:
            raise ValueError(
                "[features] TARGET_COLUMN is not set and could not be inferred.\n"
                f"Available columns: {list(df.columns)}\n"
                "Set TARGET_COLUMN in your .env file."
            )

    if target not in df.columns:
        raise ValueError(
            f"[features] TARGET_COLUMN '{target}' not found in data columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target])
    y = df[target]

    # Scale numeric features (dataset is numeric, so this is fine)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=float(os.getenv("TEST_SIZE", 0.2)),
        random_state=int(os.getenv("RANDOM_STATE", 42))
    )

    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(X_train).to_csv(out_dir / "X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(out_dir / "X_test.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)

    print("[features] Saved X_train/X_test/y_train/y_test to data/processed")