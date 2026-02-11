from src.data.download import download_kaggle_dataset
from src.data.prepare import load_and_clean
from src.features.build_features import build_features
from src.models.train import train_model
from src.models.evaluate import evaluate_model

if __name__ == "__main__":
    # download_kaggle_dataset() 
    load_and_clean()
    build_features()
    train_model()
    evaluate_model()