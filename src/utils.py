# src/utils.py
import pandas as pd

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def save_model(model, model_path):
    """Save a trained model to disk."""
    import joblib
    joblib.dump(model, model_path)

def load_model(model_path):
    """Load a model from disk."""
    import joblib
    return joblib.load(model_path)

