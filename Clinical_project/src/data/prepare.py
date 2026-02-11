import pandas as pd
from pathlib import Path

def load_and_clean():
    raw_dir = Path("data/raw")
    csv_path = list(raw_dir.glob("*.csv"))[0]

    df = pd.read_csv(csv_path)
    df = df.drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(processed_dir / "clean.csv", index=False)